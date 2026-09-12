"""FastAPI + WebSocket live loop.

PRIVACY ASSERTION is logged once at startup and is a real check, not a banner:
perception returns the number of frames it retained, and the UI shows it.
"""
from __future__ import annotations

import asyncio, json, logging, os, time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .scene_graph import SceneGraph
from .perception import MockPerception
from .perception.base import debug_keep_frames
from .sampler import PerturbationSampler, AXES
from .agent import LinUCB, FetchMugTask, STRATEGIES
from .metrics import Metrics
from .providers import MockProvider

load_dotenv()
log = logging.getLogger("skopos")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)

app = FastAPI(title="Skopos")


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
        self.base: SceneGraph = self.result.graph
        self.sampler = PerturbationSampler(seed=seed)
        self.task = FetchMugTask(seed=seed)
        self.bandit = LinUCB(SceneGraph.CONTEXT_DIM)
        self.metrics = Metrics()
        self.current: SceneGraph = self.base
        self.paused = False
        self.history: list[dict] = []
        self.provider_error: str | None = getattr(self, "provider_error", None)

    def set_axis_scale(self, axis: str, value: float) -> None:
        if axis in AXES:
            self.sampler.axis_scale[axis] = max(0.0, min(1.0, value))

    def step(self) -> dict:
        p = self.sampler.sample()
        self.current = self.sampler.apply(self.base, p)
        ctx = self.current.context_vector()
        arm = self.bandit.select(ctx)
        outcome = self.task.attempt(self.current, arm)
        self.bandit.update(arm, ctx, outcome.reward)
        self.metrics.record(outcome, p.weight)
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
            "bandit": self.bandit.scores(ctx),
            "metrics": {
                "placeholder_success_rate": report.placeholder_success_rate,
                "placeholder_hazard_contact_rate": report.placeholder_hazard_contact_rate,
                "coverage_confidence": report.coverage_confidence,
                "effective_sample_size": report.effective_sample_size,
                "attempts": report.attempts,
                "placeholder_readiness": report.placeholder_readiness,
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
    await sock.accept()
    session = Session(int(os.environ.get("SKOPOS_SEED", "1337")),
                      os.environ.get("SKOPOS_PROVIDER", "mock"))

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
            await asyncio.sleep(0.55)

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
