"""Sample perturbations of a room. Never enumerate them.

WHY SAMPLING, NOT ENUMERATION
-----------------------------
The space of room states is a product, not a list. Nine objects that can each sit
in ten places is 10^9 rooms before you touch lighting or clutter. You cannot
store that, ship it, or test against it. What you CAN do is estimate how often a
robot fails over that space, from a few hundred draws.

THE ESTIMATOR
-------------
We want the failure rate a household would actually experience:

    F = E_p[ fail(x) ]        x ~ p, the PRIOR over how rooms really change

Rooms mostly change a little: a mug moves, a light dims. Under p, the severe
configurations that break robots are rare, so plain Monte Carlo spends nearly
every draw on easy rooms and almost never sees the failures we care about.

So we draw from a deliberately harsher PROPOSAL q, and correct the bias with an
importance weight:

    F ≈ (1/N) Σ w_i · fail(x_i),    w_i = p(x_i) / q(x_i),    x_i ~ q

This is unbiased for any q with support covering p. Severe rooms appear often
(good for the agent) but each carries w < 1 (honest for the metric).

CLOSED FORM
-----------
Each axis k draws a magnitude m ∈ [0,1) from a Beta(1, b) whose density is

    Beta(1,b)(m) = b · (1-m)^(b-1)

The prior uses b_p (large ⇒ mass near 0 ⇒ small changes are typical). The
proposal uses b_q = b_p / tilt with tilt > 1, which flattens it toward severity.
The per-axis weight ratio is therefore exactly

    p(m)/q(m) = [b_p (1-m)^(b_p-1)] / [b_q (1-m)^(b_q-1)]
              = (b_p/b_q) · (1-m)^(b_p-b_q)
              = tilt · (1-m)^(b_p - b_p/tilt)

and the sample weight is the product over independent axes. We report the
self-normalised estimate Σ w·fail / Σ w (lower variance, consistent) alongside
the effective sample size (Σw)² / Σw², which is the honest answer to "how many
draws is this really worth".
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..scene_graph import SceneGraph, SceneObject, Pose

# name -> prior concentration b_p. Higher b_p == this axis usually barely moves.
AXES: dict[str, float] = {
    "object_moved":      3.0,
    "object_removed":    6.0,
    "lighting":          4.0,
    "occlusion":         5.0,
    "clutter_added":     4.0,
    "reflective_surface": 7.0,
}


@dataclass(slots=True)
class Perturbation:
    magnitudes: dict[str, float]
    weight: float                       # p(x)/q(x)
    seed: int
    described: list[str] = field(default_factory=list)

    @property
    def severity(self) -> float:
        return sum(self.magnitudes.values()) / max(len(self.magnitudes), 1)


class PerturbationSampler:
    def __init__(self, seed: int = 1337, tilt: float = 2.0,
                 axis_scale: dict[str, float] | None = None):
        if tilt <= 1.0:
            raise ValueError("tilt must be > 1 or the proposal is not harsher than the prior")
        self.rng = random.Random(seed)
        self.seed = seed
        self.tilt = tilt
        # UI sliders scale each axis in [0,1]; 0 disables the axis entirely.
        self.axis_scale = axis_scale or {k: 1.0 for k in AXES}

    # -- draw ---------------------------------------------------------------
    def _draw_beta_1_b(self, b: float) -> float:
        """Inverse-CDF sample from Beta(1,b): F(m)=1-(1-m)^b  =>  m=1-(1-u)^(1/b)."""
        u = self.rng.random()
        return 1.0 - (1.0 - u) ** (1.0 / b)

    def sample(self) -> Perturbation:
        mags, log_w = {}, 0.0
        for axis, b_p in AXES.items():
            scale = self.axis_scale.get(axis, 1.0)
            if scale <= 0.0:
                mags[axis] = 0.0
                continue
            b_q = b_p / self.tilt
            m = self._draw_beta_1_b(b_q)
            # log ratio, computed in logs so a run of severe draws cannot underflow
            log_w += math.log(self.tilt) + (b_p - b_q) * math.log(max(1.0 - m, 1e-12))
            mags[axis] = m * scale
        return Perturbation(mags, math.exp(log_w), self.seed)

    # -- apply --------------------------------------------------------------
    def apply(self, base: SceneGraph, p: Perturbation) -> SceneGraph:
        g = base.copy()
        g.notes = "Perturbed variant. Plausible, not measured."
        desc: list[str] = []
        m = p.magnitudes

        if (k := m.get("object_moved", 0)) > 0.15:
            movable = [o for o in g.objects if o.label in ("mug", "cable", "rug", "plant pot")]
            if movable:
                o = self.rng.choice(movable)
                o.pose = Pose(o.pose.x + (self.rng.random() - 0.5) * 2.4 * k,
                              o.pose.y + (self.rng.random() - 0.5) * 2.4 * k, o.pose.z)
                o.perturbed_by = "object_moved"
                desc.append(f"{o.label} moved")

        if (k := m.get("object_removed", 0)) > 0.55:
            removable = [o for o in g.objects if o.label not in ("mug", "coffee table")]
            if removable:
                o = self.rng.choice(removable)
                g.objects.remove(o)
                desc.append(f"{o.label} removed")

        if (k := m.get("lighting", 0)) > 0.1:
            g.lighting = max(0.05, min(1.0, g.lighting - k * 0.65))
            desc.append(f"lighting {g.lighting:.2f}")

        if (k := m.get("occlusion", 0)) > 0.1:
            g.occlusion = min(1.0, g.occlusion + k * 0.7)
            desc.append(f"occlusion {g.occlusion:.2f}")

        if (k := m.get("clutter_added", 0)) > 0.2:
            g.clutter = min(1.0, g.clutter + k * 0.6)
            n = 1 + int(k * 2)
            for i in range(n):
                g.objects.append(SceneObject(
                    f"x{len(g.objects)}", "stray box",
                    Pose(self.rng.random() * 3.2, self.rng.random() * 2.6, 0.15),
                    "cardboard", ("low_contrast",), 0.6, "clutter_added"))
            desc.append(f"+{n} clutter")

        if (k := m.get("reflective_surface", 0)) > 0.5:
            g.objects.append(SceneObject(
                f"x{len(g.objects)}", "floor mirror",
                Pose(self.rng.random() * 3.2, self.rng.random() * 2.6, 0.8),
                "glass", ("reflective",), 0.7, "reflective_surface"))
            desc.append("reflective surface")

        p.described = desc or ["unchanged"]
        return g


def weighted_failure_rate(weights: list[float], failures: list[bool]) -> dict[str, float]:
    """Self-normalised importance estimate plus its effective sample size."""
    if not weights:
        return {"rate": 0.0, "ess": 0.0, "n": 0}
    sw = sum(weights)
    sw2 = sum(w * w for w in weights)
    rate = (sum(w for w, f in zip(weights, failures) if f) / sw) if sw > 0 else 0.0
    ess = (sw * sw / sw2) if sw2 > 0 else 0.0
    return {"rate": rate, "ess": ess, "n": len(weights)}
