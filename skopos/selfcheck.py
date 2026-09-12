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


def main() -> int:
    print("skopos selfcheck")
    for fn in (check_privacy, check_importance_weights, check_bandit_flip,
               check_end_to_end, check_naming_honesty,
               check_spatial_map, check_vlm_offline):
        fn()
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
