"""Runnable self-check. `make check` or `python -m skopos.selfcheck`.

These are the properties that, if they broke, would make the demo dishonest or
dead. Not a test suite — the smallest set of asserts that fail loudly.
"""
from __future__ import annotations

import statistics

from .scene_graph import SceneGraph
from .perception import MockPerception
from .perception.base import debug_keep_frames
from .sampler import PerturbationSampler
from .sampler.perturbation import weighted_failure_rate
from .agent import LinUCB, FetchMugTask
from .metrics import Metrics


def check_privacy() -> None:
    r = MockPerception().analyse([b"x" * 4096, b"y" * 2048])
    assert r.frames_processed == 2
    assert r.frames_retained == 0, "frames retained with the debug flag off"
    assert r.privacy_ok
    assert not debug_keep_frames(), "SKOPOS_DEBUG_KEEP_FRAMES must default off"
    payload = r.graph.to_json()
    assert "pose" in payload and len(payload) < 20_000
    print("  privacy            OK  0 frames retained, payload is text only")


def check_importance_weights() -> None:
    """The weights must pull the estimate back toward the gentler prior.
    If they did not, the readiness score would report proposal severity as if it
    were household reality — the exact overclaim we promised not to make."""
    s = PerturbationSampler(seed=11)
    ps = [s.sample() for _ in range(4000)]
    sev = [p.severity for p in ps]
    w = [p.weight for p in ps]
    proposal = statistics.mean(sev)
    prior = sum(a * b for a, b in zip(w, sev)) / sum(w)
    assert prior < proposal, f"weights not correcting: prior {prior} >= proposal {proposal}"
    st = weighted_failure_rate(w, [x > 0.5 for x in sev])
    assert st["ess"] > 0
    print(f"  importance weights OK  proposal {proposal:.3f} -> prior {prior:.3f}, "
          f"ESS {st['ess']:.0f}/{st['n']}")


def check_bandit_flip() -> None:
    """The ordering of strategies must depend on the room. This IS the demo: if
    a clean room and a hazardous room pick the same arm, the bars never move."""
    def run(g: SceneGraph, n: int = 600) -> dict:
        b, t = LinUCB(SceneGraph.CONTEXT_DIM), FetchMugTask(seed=3)
        ctx = g.context_vector()
        for _ in range(n):
            a = b.select(ctx)
            b.update(a, ctx, t.attempt(g, a).reward)
        return b.scores(ctx)

    clean = MockPerception().analyse([]).graph
    clean.lighting, clean.clutter, clean.occlusion = 0.95, 0.03, 0.02
    clean.objects = [o for o in clean.objects if not o.hazards or o.label == "mug"]

    hard = MockPerception().analyse([]).graph
    hard.lighting, hard.clutter, hard.occlusion = 0.18, 0.85, 0.70

    cs, hs = run(clean), run(hard)
    cbest = max(cs, key=lambda a: cs[a]["mean"])
    hbest = max(hs, key=lambda a: hs[a]["mean"])
    assert cbest != hbest, f"ordering did not flip — both chose {cbest}"
    print(f"  bandit adaptation  OK  clean -> {cbest}, hazardous -> {hbest}")


def check_end_to_end() -> None:
    g = MockPerception().analyse([]).graph
    s, t, b, m = (PerturbationSampler(seed=5), FetchMugTask(seed=5),
                  LinUCB(SceneGraph.CONTEXT_DIM), Metrics())
    for _ in range(300):
        p = s.sample()
        v = s.apply(g, p)
        ctx = v.context_vector()
        a = b.select(ctx)
        o = t.attempt(v, a)
        b.update(a, ctx, o.reward)
        m.record(o, p.weight)
    r = m.report()
    assert 0.0 <= r.placeholder_readiness <= 100.0
    assert r.state in ("READY", "MARGINAL", "NOT READY")
    assert r.attempts == 300
    print(f"  end to end         OK  readiness {r.placeholder_readiness:.1f} ({r.state}), "
          f"ESS {r.effective_sample_size:.0f}")


def check_readiness_honest() -> None:
    """Anti-rigging. (1) The same robot in a harder room must score lower: the
    window and the smoothing may delay a difference, never flatten it. Both
    rooms run the SAME fixed strategy (wide_arc, the clean-room winner) so the
    gap measures the room, not the bandit. (With the 4-arm bandit feeding the
    metrics the hard room converged to request_human_assist and scored as high
    as the clean one; the server therefore scores an autonomous probe instead,
    see check_anti_rigging.) (2) On a fixed room the displayed score never
    moves more than MAX_STEP per attempt. (3) Hysteresis bands."""
    from .metrics.readiness import MAX_STEP, next_state

    def run(g: SceneGraph, arm: str | None, n: int = 300) -> tuple[list[float], object]:
        s, t, b, m = (PerturbationSampler(seed=7), FetchMugTask(seed=7),
                      LinUCB(SceneGraph.CONTEXT_DIM), Metrics())
        trace = []
        for _ in range(n):
            p = s.sample()
            v = s.apply(g, p)
            ctx = v.context_vector()
            a = arm or b.select(ctx)
            o = t.attempt(v, a)
            b.update(a, ctx, o.reward)
            m.record(o, p.weight)
            trace.append(m.report().placeholder_readiness_smoothed)
        return trace, m.report()

    clean = MockPerception().analyse([]).graph
    clean.lighting, clean.clutter, clean.occlusion = 0.95, 0.03, 0.02
    clean.objects = [o for o in clean.objects if not o.hazards or o.label == "mug"]
    hard = MockPerception().analyse([]).graph
    hard.lighting, hard.clutter, hard.occlusion = 0.18, 0.85, 0.70

    ct, cr = run(clean, "wide_arc")
    ht, hr = run(hard, "wide_arc")
    gap = cr.placeholder_readiness_smoothed - hr.placeholder_readiness_smoothed
    assert gap >= 10.0, f"hazardous room not >= 10 below clean: gap {gap:.1f}"

    jump = max(abs(a - b) for tr in (ct, ht) for a, b in zip(tr, tr[1:]))
    assert jump <= MAX_STEP + 1e-9, f"smoothed score jumped {jump:.2f} > {MAX_STEP}"

    lo, hi = cr.placeholder_success_ci
    assert 0.0 <= lo <= cr.placeholder_success_rate <= hi <= 1.0, f"bad CI {lo, hi}"
    assert next_state("READY", 72) == "READY" and next_state("MARGINAL", 72) == "MARGINAL"
    assert next_state("NOT READY", 48) == "NOT READY" and next_state("MARGINAL", 48) == "MARGINAL"
    assert next_state("READY", 69.9) == "MARGINAL" and next_state("NOT READY", 50.1) == "MARGINAL"
    print(f"  readiness honest   OK  wide_arc clean {cr.placeholder_readiness_smoothed:.1f} vs hard "
          f"{hr.placeholder_readiness_smoothed:.1f} (gap {gap:.1f}); "
          f"max step {jump:.2f}; ESS {cr.effective_sample_size:.0f}, fill {cr.window_fill:.2f}, "
          f"CI ±{(hi - lo) / 2:.3f}")


def check_anti_rigging() -> None:
    """Sliders cannot make a hazardous room look ready. With every axis scale at
    1.0, the real loop (server.Session.step, 300 steps) on the mock room and on
    the mock room plus two moving objects must score the second lower. Runs
    Session itself so the autonomous readiness probe is what is measured, not a
    copy of the loop. Seed 1337 is the server default. Measured over seeds
    1337, 1-9: ordering holds in 8 of 10, delta mean -5.4, worst +21.9 (ESS is
    only 13-30 in the 120-attempt window, so the room effect of two moving
    objects sits inside the estimator's noise). This is a pin, not a proof."""
    from .server import Session
    from .scene_graph import SceneObject, Pose

    def run(base: SceneGraph, n: int = 300) -> float:
        s = Session(1337, "mock")
        s.base = s.current = base
        for axis in ("object_moved", "clutter_added", "lighting"):
            s.set_axis_scale(axis, 1.0)
        assert all(v == 1.0 for v in s.sampler.axis_scale.values()), s.sampler.axis_scale
        for _ in range(n):
            frame = s.step()
        assert frame["probe"]["strategy"] != "request_human_assist"
        return s.metrics.report().placeholder_readiness_smoothed

    mock = MockPerception().analyse([]).graph
    hazard = mock.copy()
    hazard.objects += [SceneObject("k2", "dog", Pose(1.2, 1.6, 0.30), "—", ("moving",), 0.6),
                       SceneObject("k3", "robot vacuum", Pose(2.8, 2.3, 0.08), "plastic",
                                   ("moving",), 0.7)]
    a, h = run(mock), run(hazard)
    assert h < a, f"+2 moving objects did not lower readiness: {h:.1f} vs {a:.1f}"
    print(f"  anti-rigging       OK  mock room {a:.1f} vs +2 moving {h:.1f} "
          f"(delta {h - a:+.1f}), sliders all 1.0, real Session.step")


def check_naming_honesty() -> None:
    """Every outcome-model-derived metric must carry the placeholder_ prefix, and
    the words 'trained' / 'learned policy' must appear nowhere in the package."""
    import pathlib, re
    fields = {f for f in Metrics().report().__slots__ if "rate" in f or "readiness" in f}
    bad = {f for f in fields if not f.startswith(("placeholder_", "coverage", "effective"))}
    assert not bad, f"un-prefixed outcome metrics: {bad}"

    pkg = pathlib.Path(__file__).parent
    # A negated mention ("this is NOT a trained policy") is the disclaimer we
    # want, so only an un-negated line counts as an overclaim.
    NEGATED = re.compile(r"\b(not|never|isn't|is not|rather than|no)\b", re.I)
    offenders = []
    for py in pkg.rglob("*.py"):
        if py.name == "selfcheck.py":
            continue
        for n, line in enumerate(py.read_text().splitlines(), 1):
            if re.search(r"\b(trained|learned policy)\b", line, re.I) and not NEGATED.search(line):
                offenders.append(f"{py.relative_to(pkg)}:{n}: {line.strip()[:60]}")
    assert not offenders, f"un-negated overclaim: {offenders}"
    print("  naming honesty     OK  placeholder_ prefixes present, no 'trained'/'learned policy'")


def check_vlm_offline() -> None:
    """The VLM adapter must construct without touching the network and carry a
    non-empty model id. No request is made here."""
    from .perception.vlm import MODEL, VLMPerception
    VLMPerception(api_key="x")
    assert isinstance(MODEL, str) and MODEL, "SKOPOS_VLM_MODEL resolved to empty"
    print(f"  vlm offline        OK  constructs without network, model {MODEL!r}")


def check_spatial_map() -> None:
    """Depth -> points -> occupancy on a real kitchen fixture. Fails if the
    geometry is nonsense: every cell classified, points inside the stated
    extents, floor at z≈0, and the camera's own cell free (you are standing
    there)."""
    import pathlib
    import numpy as np
    from PIL import Image
    from .perception.depth import build_map, GRID_M, CELL_M
    fx = pathlib.Path(__file__).parent.parent / "static" / "fixtures" / "IMG_6978.jpg"
    img = np.asarray(Image.open(fx).convert("RGB"))
    m = build_map(img)
    n = int(GRID_M / CELL_M)
    assert m.occupancy.shape == (n, n)
    assert set(np.unique(m.occupancy)).issubset({0, 1, 2}), "unclassified cell"
    assert (m.occupancy == 1).sum() > 20, "no obstacles found in a kitchen photo"
    assert (m.occupancy == 0).sum() > 100, "no free floor found"
    p = m.points
    assert len(p) > 10_000
    assert (np.abs(p[:, 0]) <= GRID_M / 2 + 1e-6).all() and (p[:, 1] > 0).all() and (p[:, 1] <= GRID_M).all()
    assert p[:, 2].min() >= -0.3 - 1e-6 and p[:, 2].max() <= 2.6 + 1e-6
    d = m.to_dict()
    assert len(d["points"]) <= 8000 and len(d["occupancy"]) == n
    print(f"  spatial map        OK  {m.n_points:,} pts, {(m.occupancy==1).sum()} occ / "
          f"{(m.occupancy==0).sum()} free cells, depth {m.depth_ms:.0f} ms")


def check_sweep_fusion() -> None:
    """Three photos fused as a rotating sweep. Fails if yaw does not increase
    monotonically, if the camera's own cell is not free, or if the note stops
    admitting which pairs were placed by even spacing."""
    import pathlib
    import numpy as np
    from PIL import Image
    from .perception.depth import fuse_sweep, GRID_M, CELL_M
    fixtures = pathlib.Path(__file__).parent.parent / "static" / "fixtures"
    imgs = [np.asarray(Image.open(fixtures / f"IMG_69{i}.jpg").convert("RGB")) for i in (78, 79, 80)]
    m = fuse_sweep(imgs)
    assert m.centred and m.yaws is not None and len(m.yaws) == 3
    assert all(b > a for a, b in zip(m.yaws, m.yaws[1:])), f"yaw not increasing: {m.yaws}"
    assert m.yaws[-1] < 360.0
    n = int(GRID_M / CELL_M)
    assert m.occupancy.shape == (n, n)
    assert (m.occupancy == 1).sum() > 20 and (m.occupancy == 0).sum() > 100
    assert "fused" in m.note and "even spacing" in m.note
    assert len(m.to_dict()["points"]) <= 8000
    print(f"  sweep fusion       OK  3 photos, yaws {m.yaws}, {m.n_points:,} pts, "
          f"{sum(1 for u in m.used if u == 'overlap')}/{len(m.used)} pairs from overlap")


def check_panorama() -> None:
    """One iPhone panorama strip -> centred map. The phone does not record the
    sweep; build_pano_map defaults to 180 deg, and at 180 the strip's vertical
    FOV (fov*H/W = 43 deg) puts this fixture's floor band 3.6 m out, mostly
    beyond the 3 m half-grid: measured 46 free cells, under the 100 this check
    wants. So the check states its FOV: the pano is shot in portrait, its
    vertical FOV is the camera's long-side FOV (HFOV_DEG), and the sweep
    follows from the aspect — about 290 deg here."""
    import pathlib
    import numpy as np
    from PIL import Image
    from .perception.depth import build_pano_map, HFOV_DEG, GRID_M, CELL_M
    fx = pathlib.Path(__file__).parent.parent / "static" / "fixtures" / "IMG_6987.jpg"
    img = np.asarray(Image.open(fx).convert("RGB"))
    h, w = img.shape[:2]
    assert w / h > 2.5, "fixture is not a panorama strip"
    fov = HFOV_DEG * w / h
    m = build_pano_map(img, fov)
    n = int(GRID_M / CELL_M)
    assert m.centred and m.pano_fov_deg == fov
    assert m.occupancy.shape == (n, n)
    assert set(np.unique(m.occupancy)).issubset({0, 1, 2}), "unclassified cell"
    assert (m.occupancy == 1).sum() > 20, "no obstacles found in the panorama"
    assert (m.occupancy == 0).sum() > 100, "no free floor found in the panorama"
    p = m.points
    assert len(p) > 10_000
    assert (np.abs(p[:, 0]) <= GRID_M / 2 + 1e-6).all() and (np.abs(p[:, 1]) <= GRID_M / 2 + 1e-6).all()
    assert p[:, 2].min() >= -0.3 - 1e-6 and p[:, 2].max() <= 2.6 + 1e-6
    assert "Panorama" in m.note and f"{fov:g}" in m.note
    d = m.to_dict()
    assert d["pano_fov_deg"] == fov and d["centred"] and len(d["points"]) <= 8000
    print(f"  panorama           OK  {w}x{h} strip, assumed {fov:.0f} deg, {m.n_points:,} pts, "
          f"{(m.occupancy==1).sum()} occ / {(m.occupancy==0).sum()} free cells, depth {m.depth_ms:.0f} ms")


def check_route_planning() -> None:
    """A* on a hand-built grid with one wall and a gap, then on the real
    kitchen map. Fails if it walks through a wall, cuts a corner, or reports a
    length that disagrees with the path it returned."""
    import numpy as np
    from .perception.plan import plan_route, inflate
    n = 20
    occ = np.zeros((n, n), np.uint8)
    occ[10, :] = 1; occ[10, 9] = 0           # wall across row 10 with a one-cell gap at col 9
    occ[10, 8] = 0; occ[10, 10] = 0          # ...widened to 3 so inflation still leaves a way through
    r = plan_route(occ, (2, 2), (17, 17))
    assert not r.blocked and r.path[0] == (2, 2) and r.path[-1] == (17, 17)
    g = inflate(occ)
    assert all(g[y, x] != 1 for x, y in r.path), "path crosses an inflated obstacle"
    L = sum(((b[0]-a[0])**2 + (b[1]-a[1])**2) ** .5 for a, b in zip(r.path, r.path[1:])) * 0.1
    assert abs(L - r.length_m) < 1e-6
    occ2 = occ.copy(); occ2[10, :] = 1        # seal the wall completely
    assert plan_route(occ2, (2, 2), (17, 17)).blocked, "should be blocked by a sealed wall"
    # unknown is allowed but counted
    occ3 = np.full((n, n), 2, np.uint8); occ3[:, :3] = 0
    r3 = plan_route(occ3, (1, 1), (18, 18))
    assert not r3.blocked and r3.cells_unknown > 0
    print(f"  route planning     OK  gap route {len(r.path)} cells {r.length_m:.2f} m; sealed wall blocked; "
          f"unknown crossed {r3.cells_unknown}")


def main() -> int:
    print("skopos selfcheck")
    for fn in (check_privacy, check_importance_weights, check_bandit_flip,
               check_end_to_end, check_readiness_honest, check_anti_rigging,
               check_naming_honesty,
               check_spatial_map, check_vlm_offline, check_sweep_fusion,
               check_panorama, check_route_planning):
        fn()
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
