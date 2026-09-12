"""FastAPI + WebSocket live loop.

PRIVACY ASSERTION is logged once at startup and is a real check, not a banner:
perception returns the number of frames it retained, and the UI shows it.
"""
from __future__ import annotations

import asyncio, email, json, logging, os, time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .scene_graph import SceneGraph
from .perception import MockPerception, PerceptionResult
from .perception.base import debug_keep_frames
from .sampler import PerturbationSampler, AXES
from .agent import LinUCB, FetchMugTask, STRATEGIES

# Readiness is scored on the robot ALONE. The deployed bandit may learn to ask
# a human, and in a hazardous room it does; asking a human is a fallback, not
# readiness. Before this, a room with more hazards scored HIGHER than a clean
# one because the human always succeeds and the formula does not charge for it.
AUTONOMOUS = tuple(s for s in STRATEGIES if s != "request_human_assist")
from .metrics import Metrics
from .providers import MockProvider

load_dotenv()
log = logging.getLogger("skopos")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)

app = FastAPI(title="Skopos")
WARM_STEPS = 120          # = metrics WINDOW; the first frame is a full window
TICK_S = 0.35             # was 0.55; step() costs 0.3 ms, the browser keeps up

# Set by POST /api/scan. New Sessions start from it instead of the mock room;
# the "reset_scene" WebSocket message clears it.
LATEST_SCAN: SceneGraph | None = None
LATEST_MAP: dict | None = None
MAX_SCAN_BYTES = 12 * 1024 * 1024


def _perception():
    if os.environ.get("ANTHROPIC_API_KEY"):
        from .perception.vlm import VLMPerception
        return VLMPerception()
    return MockPerception()


def _provider(name: str):
    if name == "reactor":
        from .providers.reactor import ReactorProvider
        return ReactorProvider()
    if name == "runware":
        from .providers.runware import RunwareProvider
        return RunwareProvider()
    return MockProvider()


@app.on_event("startup")
async def _startup() -> None:
    keep = debug_keep_frames()
    log.info("PRIVACY: frames are processed into a SceneGraph and discarded; "
             "only structured text crosses a network boundary. "
             "SKOPOS_DEBUG_KEEP_FRAMES=%s", "1 (RAW FRAMES WILL TOUCH DISK)" if keep else "0")
    if keep:
        log.warning("PRIVACY: debug frame retention is ON. Do not use on a real scan.")


class Session:
    """One browser, one room, one bandit."""

    def __init__(self, seed: int, provider_name: str):
        self.seed = seed
        self.provider_name = provider_name
        try:
            self.provider = _provider(provider_name)
        except RuntimeError as exc:
            # e.g. SKOPOS_PROVIDER=reactor with no key. Degrade loudly, not silently.
            log.error("provider %r unavailable (%s); falling back to mock", provider_name, exc)
            self.provider_name = "mock"
            self.provider = MockProvider()
            self.provider_error = str(exc)
        self.perception = MockPerception()
        self.result = self.perception.analyse([], room_id="demo-room")
        if LATEST_SCAN is not None:
            # bytes_discarded is not recoverable from the graph; the UI shows frames only.
            self.result = PerceptionResult(
                LATEST_SCAN, LATEST_SCAN.frames_seen,
                LATEST_SCAN.frames_seen if debug_keep_frames() else 0, 0)
        self.base: SceneGraph = self.result.graph
        self.sampler = PerturbationSampler(seed=seed)
        self.task = FetchMugTask(seed=seed)
        self.bandit = LinUCB(SceneGraph.CONTEXT_DIM)
        self.probe = LinUCB(SceneGraph.CONTEXT_DIM, arms=AUTONOMOUS)   # feeds readiness
        self.metrics = Metrics()
        self.metrics_autonomous = Metrics()
        self.current: SceneGraph = self.base
        self.paused = False
        self.history: list[dict] = []
        self.provider_error: str | None = getattr(self, "provider_error", None)

    def set_axis_scale(self, axis: str, value: float) -> None:
        if axis in AXES:
            self.sampler.axis_scale[axis] = max(0.0, min(1.0, value))

    def warm(self, n: int) -> None:
        """Run n attempts without rendering so the FIRST frame already carries a
        full window (WINDOW=120 -> ~40 ms of CPU). The score then means
        something immediately instead of after a minute of watching the bars
        settle. These are real simulated attempts, counted as such."""
        for _ in range(n):
            p = self.sampler.sample()
            self.current = self.sampler.apply(self.base, p)
            ctx = self.current.context_vector()
            arm = self.bandit.select(ctx)
            outcome = self.task.attempt(self.current, arm)
            self.bandit.update(arm, ctx, outcome.reward)
            self.metrics.record(outcome, p.weight)
            probe_arm = self.probe.select(ctx)
            probe = self.task.attempt(self.current, probe_arm)
            self.probe.update(probe_arm, ctx, probe.reward)
            self.metrics_autonomous.record(probe, p.weight)

    def step(self) -> dict:
        p = self.sampler.sample()
        self.current = self.sampler.apply(self.base, p)
        ctx = self.current.context_vector()
        arm = self.bandit.select(ctx)
        outcome = self.task.attempt(self.current, arm)
        self.bandit.update(arm, ctx, outcome.reward)
        # Headline readiness = the real attempt (the robot with every option,
        # including asking a human) — the number Mike was shown all day.
        self.metrics.record(outcome, p.weight)
        # Second, stricter number: the same room, robot on its own (no human
        # assist). Scored into its own Metrics so it never replaces the headline.
        probe_arm = self.probe.select(ctx)
        probe = self.task.attempt(self.current, probe_arm)
        self.probe.update(probe_arm, ctx, probe.reward)
        self.metrics_autonomous.record(probe, p.weight)
        handle = self.provider.prepare(self.current)
        report = self.metrics.report()

        frame = {
            "type": "step",
            "seed": self.seed,
            "provider": self.provider_name,
            "render": {"kind": handle.kind, "prompt": handle.prompt,
                       "detail": handle.detail, "note": handle.note},
            "perturbation": {"described": p.described,
                             "weight": round(p.weight, 4),
                             "severity": round(p.severity, 3),
                             "magnitudes": {k: round(v, 3) for k, v in p.magnitudes.items()}},
            "attempt": {"strategy": outcome.strategy, "success": outcome.success,
                        "reward": round(outcome.reward, 3),
                        "hazard_hit": outcome.hazard_hit,
                        "blamed_object": outcome.blamed_object},
            "probe": {"strategy": probe.strategy, "success": probe.success,
                      "hazard_hit": probe.hazard_hit, "blamed_object": probe.blamed_object},
            "bandit": self.bandit.scores(ctx),
            "metrics": {
                "placeholder_success_rate": report.placeholder_success_rate,
                "placeholder_hazard_contact_rate": report.placeholder_hazard_contact_rate,
                "coverage_confidence": report.coverage_confidence,
                "effective_sample_size": report.effective_sample_size,
                "attempts": report.attempts,
                "placeholder_readiness": report.placeholder_readiness,
                "placeholder_readiness_smoothed": report.placeholder_readiness_smoothed,
                "warming_up": report.warming_up,
                "placeholder_readiness_autonomous": self.metrics_autonomous.report().placeholder_readiness_smoothed,
                "state_autonomous": self.metrics_autonomous.report().state,
                "window_fill": report.window_fill,
                "placeholder_success_ci": list(report.placeholder_success_ci),
                "state": report.state,
                "blame": report.blame,
                "rolling": list(self.metrics.sparkline),
            },
            "scene_graph": self.current.to_dict(),
            "privacy": {"frames_processed": self.result.frames_processed,
                        "frames_retained": self.result.frames_retained,
                        "bytes_discarded": self.result.bytes_discarded,
                        "debug_keep_frames": debug_keep_frames()},
        }
        self.history.append(frame)
        return frame


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/config")
async def config() -> JSONResponse:
    return JSONResponse({
        "axes": list(AXES),
        "strategies": list(STRATEGIES),
        "task": FetchMugTask.name,
        "provider": os.environ.get("SKOPOS_PROVIDER", "mock"),
        "seed": int(os.environ.get("SKOPOS_SEED", "1337")),
        "debug_keep_frames": debug_keep_frames(),
    })


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    global LATEST_SCAN, LATEST_MAP
    await sock.accept()
    session = Session(int(os.environ.get("SKOPOS_SEED", "1337")),
                      os.environ.get("SKOPOS_PROVIDER", "mock"))
    # A full window of attempts before the first frame (~20 ms) so the score is
    # settled at once. Done here, not in __init__, so a bare Session() stays cold
    # for checks and tools that want to watch it learn from zero.
    await asyncio.to_thread(session.warm, WARM_STEPS)

    # Mint the Reactor session token ONCE, in a worker thread. prepare() runs
    # inside step() on the event loop; if it had to mint there, one cold
    # httpx.post(timeout=30) would freeze the pump, every inbound slider/pause/
    # record message, and every HTTP route at the same time.
    mint = getattr(session.provider, "mint_token", None)
    if mint is not None:
        try:
            await asyncio.to_thread(mint)
        except Exception as exc:  # surface it in the UI rather than a dead pane
            log.error("reactor token mint failed: %s", exc)
            await sock.send_json({"type": "error", "where": "reactor.mint_token",
                                  "message": str(exc)[:300]})

    async def pump() -> None:
        while True:
            if not session.paused:
                await sock.send_json(session.step())
            await asyncio.sleep(TICK_S)

    task = asyncio.create_task(pump())
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            kind = msg.get("type")
            if kind == "axis":
                session.set_axis_scale(msg["axis"], float(msg["value"]))
            elif kind == "pause":
                session.paused = bool(msg.get("value", True))
            elif kind == "reseed":
                session.__init__(int(msg.get("seed", session.seed)), session.provider_name)
                await asyncio.to_thread(session.warm, WARM_STEPS)
            elif kind == "reset_scene":
                LATEST_SCAN = None
                session.__init__(session.seed, session.provider_name)
                await asyncio.to_thread(session.warm, WARM_STEPS)
            elif kind == "record":
                stamp = int(time.time())
                path = RUNS / f"run-{stamp}-seed{session.seed}.json"
                path.write_text(json.dumps({
                    "seed": session.seed,
                    "provider": session.provider_name,
                    "task": FetchMugTask.name,
                    "axis_scale": session.sampler.axis_scale,
                    "frames": session.history,
                }, indent=2))
                await sock.send_json({"type": "recorded", "path": str(path.name),
                                      "frames": len(session.history)})
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        session.provider.close()


def _multipart_file(content_type: str, body: bytes) -> bytes | None:
    """The 'file' part of a multipart/form-data body, parsed in memory.

    Not Starlette's UploadFile: it spools anything over 1 MB to a temp file on
    disk, and a raw frame must never touch disk (see perception/base.py).
    """
    msg = email.message_from_bytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    for part in msg.walk():
        if part.get_param("name", header="content-disposition") == "file":
            return part.get_payload(decode=True)
    return None


def _multipart_files(content_type: str, body: bytes) -> list[bytes]:
    """Every 'file' part of a multipart body, in order, parsed in memory."""
    msg = email.message_from_bytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    return [part.get_payload(decode=True) for part in msg.walk()
            if part.get_param("name", header="content-disposition") == "file"]


class _SkipMap(Exception):
    pass


MAX_SWEEP_BYTES = 40 * 1024 * 1024
_JPEG, _PNG = b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n"
PANO_ASPECT = 2.5          # one image wider than this is a panorama, not a sweep


def _decode_rgb(frame: bytes):
    import io
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(frame)).convert("RGB"))


PANO_VFOV_DEG = 69.0   # the phone camera's long-side FOV; a pano strip's height spans roughly this


def _clamp_fov(fov: float) -> float:
    # Measured on IMG_6987: 120 deg -> 0 classified cells, 180 -> 4%, 290 -> 46%.
    # Below ~150 the floor band the depth scale is fitted on collapses, so that is the floor.
    return max(150.0, min(360.0, float(fov)))


def _derive_fov(img) -> float:
    """iPhone panoramas carry no FOV. Aspect x the camera's vertical FOV is the
    principled guess: a 4.2:1 strip -> ~290 deg."""
    h, w = img.shape[:2]
    return _clamp_fov((w / max(h, 1)) * PANO_VFOV_DEG)


async def _pano_map(img, fov: float, source: str = "given") -> dict:
    """Panorama pixels -> LATEST_MAP + runs/ cache, in a worker thread. Returns
    the response body. The caller deletes the pixels."""
    global LATEST_MAP
    from .perception.depth import HFOV_DEG, build_pano_map
    smap = await asyncio.to_thread(build_pano_map, img, fov)
    stamp = int(time.time())
    LATEST_MAP = smap.to_dict()
    (RUNS / f"map-{stamp}.json").write_text(json.dumps(LATEST_MAP))
    log.info("pano: %dx%d, assumed fov %.0f -> %d points, depth %.0f ms",
             img.shape[1], img.shape[0], fov, smap.n_points, smap.depth_ms)
    return {"n_points": smap.n_points, "depth_ms": round(smap.depth_ms, 1),
            "pano_fov_deg": round(float(smap.pano_fov_deg), 1), "pano_fov_source": source, "frames_retained": 0,
            "cached_as": f"map-{stamp}.json", "note": smap.note}


@app.post("/api/scan/pano")
async def scan_pano(request: Request, fov: float | None = None) -> JSONResponse:
    """One wide panorama (iPhone Pano) -> one centred map. ?fov= is the assumed
    horizontal sweep in degrees — the phone does not record it — clamped 90..360.
    Same privacy contract as /api/scan: parsed in memory, deleted, numbers only."""
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_SCAN_BYTES:
            return JSONResponse({"error": "upload larger than 12 MB"}, status_code=413)
        chunks.append(chunk)
    frame = _multipart_file(request.headers.get("content-type", ""), b"".join(chunks))
    del chunks
    if frame is None:
        return JSONResponse({"error": "multipart field 'file' missing"}, status_code=400)
    if not (frame[:3] == _JPEG or frame[:8] == _PNG):
        del frame
        return JSONResponse({"error": "only image/jpeg or image/png accepted"}, status_code=415)
    img = _decode_rgb(frame)
    del frame
    try:
        body = await _pano_map(img, _derive_fov(img) if fov is None else _clamp_fov(fov), "derived from aspect" if fov is None else "given")
    finally:
        del img
    return JSONResponse(body)


@app.post("/api/scan/sweep")
async def scan_sweep(request: Request, fov: float | None = None, sweep: float | None = None) -> JSONResponse:
    """Several photos of ONE spot, turning -> one fused 360-ish map.

    Same privacy contract as /api/scan: frames are parsed in memory, fused in a
    worker thread, and deleted. Only the numbers survive. Does not touch the
    scene graph — the sweep is geometry, the graph is semantics.
    A single image wider than PANO_ASPECT is routed to the panorama path."""
    global LATEST_MAP
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_SWEEP_BYTES:
            return JSONResponse({"error": "sweep larger than 40 MB"}, status_code=413)
        chunks.append(chunk)
    frames = _multipart_files(request.headers.get("content-type", ""), b"".join(chunks))
    del chunks
    frames = [f for f in frames if f and (f[:3] == _JPEG or f[:8] == _PNG)]
    if not frames:
        return JSONResponse({"error": "need at least two JPEG/PNG 'file' parts"}, status_code=400)
    from .perception.depth import fuse_sweep, HFOV_DEG
    imgs = [_decode_rgb(f) for f in frames]
    n_bytes = sum(len(f) for f in frames)
    del frames
    if len(imgs) == 1 and imgs[0].shape[1] / imgs[0].shape[0] > PANO_ASPECT:
        try:
            body = await _pano_map(imgs[0], _derive_fov(imgs[0]) if fov is None else _clamp_fov(fov), "derived from aspect" if fov is None else "given")
        finally:
            del imgs
        body["routed_to"] = "pano"
        return JSONResponse(body)
    if len(imgs) < 2:
        del imgs
        return JSONResponse({"error": "need at least two JPEG/PNG 'file' parts"}, status_code=400)
    try:
        smap = await asyncio.to_thread(fuse_sweep, imgs, HFOV_DEG, sweep)
    finally:
        del imgs
    stamp = int(time.time())
    LATEST_MAP = smap.to_dict()
    (RUNS / f"map-{stamp}.json").write_text(json.dumps(LATEST_MAP))
    log.info("sweep: %d photos, %d bytes -> %d points; yaws %s; %s",
             len(smap.yaws or []), n_bytes, smap.n_points, smap.yaws, smap.used)
    return JSONResponse({"photos": len(smap.yaws or []), "n_points": smap.n_points,
                         "yaws": smap.yaws, "sweep_deg": smap.sweep_deg, "placed_by": smap.used,
                         "depth_ms": round(smap.depth_ms, 1), "note": smap.note,
                         "frames_retained": 0, "cached_as": f"map-{stamp}.json"})


@app.get("/api/scan/status")
async def scan_status() -> JSONResponse:
    """Cheap probe: is there a map to append to? (GET routes do not answer HEAD.)"""
    m = LATEST_MAP
    if m is None:
        maps = sorted(RUNS.glob("map-*.json"), key=lambda p: p.stat().st_mtime)
        if maps:
            try:
                m = json.loads(maps[-1].read_text())
            except Exception:
                m = None
    return JSONResponse({"has_map": m is not None, "centred": bool(m and m.get("centred")),
                         "views": int(m.get("views", 1)) if m else 0})


@app.post("/api/scan/append")
async def scan_append(request: Request, fov: float | None = None) -> JSONResponse:
    """A new photo of the SAME room -> registered into the existing map by yaw
    search against the retained grid. Never creates a new space. Same privacy
    contract: parsed in memory, deleted, only numbers survive."""
    global LATEST_MAP
    if LATEST_MAP is None:
        maps = sorted(RUNS.glob("map-*.json"), key=lambda p: p.stat().st_mtime)
        if not maps:
            return JSONResponse({"error": "no existing map to append to — scan or fuse first"}, status_code=409)
        LATEST_MAP = json.loads(maps[-1].read_text())
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_SCAN_BYTES:
            return JSONResponse({"error": "upload larger than 12 MB"}, status_code=413)
        chunks.append(chunk)
    frame = _multipart_file(request.headers.get("content-type", ""), b"".join(chunks))
    del chunks
    if frame is None or not (frame[:3] == _JPEG or frame[:8] == _PNG):
        return JSONResponse({"error": "one JPEG/PNG 'file' part required"}, status_code=415)
    import io as _io
    import numpy as _np
    from PIL import Image as _Image
    from .perception.depth import build_map, build_pano_map
    from .perception.incremental import append_view
    img = _np.asarray(_Image.open(_io.BytesIO(frame)).convert("RGB"))
    del frame
    h, w = img.shape[:2]
    is_pano = (w / max(h, 1)) > 2.5
    def _work():
        m = build_pano_map(img, _derive_fov(img) if fov is None else _clamp_fov(fov)) if is_pano else build_map(img)
        return append_view(LATEST_MAP, m), m
    try:
        merged, m = await asyncio.to_thread(_work)
    finally:
        del img
    LATEST_MAP = merged
    stamp = int(time.time())
    (RUNS / f"map-{stamp}.json").write_text(json.dumps(LATEST_MAP))
    reg = merged["last_register"]
    log.info("append: %s view registered yaw=%.0f score=%.2f known %d->%d",
             "pano" if is_pano else "frame", reg["yaw_deg"], reg["score"],
             reg["cells_known_before"], reg["cells_known_after"])
    return JSONResponse({"registered": reg, "views": merged["views"], "kind": "pano" if is_pano else "frame",
                         "n_points": m.n_points, "depth_ms": round(m.depth_ms, 1),
                         "frames_retained": 0, "cached_as": f"map-{stamp}.json", "note": merged["note"]})


@app.post("/api/scan")
async def scan(request: Request, map: int = 1) -> JSONResponse:
    global LATEST_SCAN, LATEST_MAP
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_SCAN_BYTES:
            return JSONResponse({"error": "upload larger than 12 MB"}, status_code=413)
        chunks.append(chunk)
    frame = _multipart_file(request.headers.get("content-type", ""), b"".join(chunks))
    del chunks
    if frame is None:
        return JSONResponse({"error": "multipart field 'file' missing"}, status_code=400)
    if not (frame[:3] == b"\xff\xd8\xff" or frame[:8] == b"\x89PNG\r\n\x1a\n"):
        del frame
        return JSONResponse({"error": "only image/jpeg or image/png accepted"}, status_code=415)

    stamp = int(time.time())
    perception = _perception()
    # VLM does a blocking HTTP call; keep it off the event loop that pumps /ws.
    result = await asyncio.to_thread(perception.analyse, [frame], room_id=f"scan-{stamp}")
    # Spatial map from the SAME frame while we still hold it: depth -> points ->
    # occupancy, in a worker thread. Same privacy contract — only numbers survive.
    # ?map=0 skips it — the multi-upload calls this for the scene graph and then
    # /sweep for the map; building a map here just to overwrite it cost a full
    # depth pass per upload.
    try:
        if not map:
            raise _SkipMap()
        from .perception.depth import build_map
        import io as _io
        import numpy as _np
        from PIL import Image as _Image
        _img = _np.asarray(_Image.open(_io.BytesIO(frame)).convert("RGB"))
        smap = await asyncio.to_thread(build_map, _img)
        del _img
        LATEST_MAP = smap.to_dict()
        (RUNS / f"map-{stamp}.json").write_text(json.dumps(LATEST_MAP))
        log.info("spatial map: %d points, depth %.0f ms", smap.n_points, smap.depth_ms)
    except _SkipMap:
        pass                                       # keep whatever map is current
    except Exception as exc:  # additive: a scan without a map is still a scan
        log.warning("spatial map failed: %s", exc)
        LATEST_MAP = None
    del frame  # the only copy; nothing was written to disk

    graph = result.graph
    name = f"scan-{stamp}.json"
    (RUNS / name).write_text(json.dumps({
        "source": perception.name, "room_id": graph.room_id, "graph": graph.to_dict(),
        "frames_processed": result.frames_processed,
        "frames_retained": result.frames_retained,
    }, indent=2))
    LATEST_SCAN = graph
    log.info("scan: %d bytes -> %d objects via %s; frames_retained=%d; cached %s",
             result.bytes_discarded, len(graph.objects), perception.name,
             result.frames_retained, name)
    return JSONResponse({
        "scene": graph.to_dict(),
        "frames_seen": result.frames_processed,
        "frames_retained": result.frames_retained,
        "source": perception.name,
        "cached_as": name,
    })


@app.get("/api/scan/map")
async def scan_map() -> JSONResponse:
    """Latest spatial map (points + occupancy) or 404. Fetched once per scan, not per frame."""
    if LATEST_MAP is not None:
        return JSONResponse(LATEST_MAP)
    maps = sorted(RUNS.glob("map-*.json"), key=lambda p: p.stat().st_mtime)
    if not maps:
        return JSONResponse({"error": "no map yet"}, status_code=404)
    return JSONResponse(json.loads(maps[-1].read_text()))


@app.get("/api/scan/route")
async def scan_route(gx: int, gy: int) -> JSONResponse:
    """A* from the camera cell to (gx, gy) over the latest map. Real planning on
    the real grid; reports how many unknown cells the path had to cross."""
    from .perception.plan import plan_route, camera_cell
    import numpy as _np
    m = LATEST_MAP
    if m is None:
        maps = sorted(RUNS.glob("map-*.json"), key=lambda p: p.stat().st_mtime)
        if not maps:
            return JSONResponse({"error": "no map yet"}, status_code=404)
        m = json.loads(maps[-1].read_text())
    occ = _np.asarray(m["occupancy"], dtype=_np.uint8)
    start = camera_cell(occ, bool(m.get("centred")))
    r = plan_route(occ, start, (gx, gy), cell_m=float(m.get("cell_m", 0.1)))
    return JSONResponse(r.to_dict())


@app.get("/api/scan/latest")
async def scan_latest() -> JSONResponse:
    newest = max(RUNS.glob("scan-*.json"), key=lambda p: p.stat().st_mtime, default=None)
    if newest is None:
        return JSONResponse({"error": "no scan yet"}, status_code=404)
    return JSONResponse(json.loads(newest.read_text()))


@app.get("/api/runs")
async def runs() -> JSONResponse:
    return JSONResponse(sorted(p.name for p in RUNS.glob("run-*.json")))


@app.get("/api/runs/{name}")
async def run_file(name: str) -> JSONResponse:
    p = RUNS / name
    if not p.is_file() or p.suffix != ".json":
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(json.loads(p.read_text()))


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
