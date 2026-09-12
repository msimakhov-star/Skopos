"""One task: fetch the mug from the table.

HONESTY NOTE — READ THIS BEFORE QUOTING ANY NUMBER
--------------------------------------------------
No robot executes anything here. `FetchMugTask.attempt` is an OUTCOME MODEL: a
hand-written, seeded function mapping (SceneGraph, strategy) -> success. It
encodes ordinary assumptions — dim rooms hurt vision, clutter blocks a direct
line, reflective surfaces confuse depth, a moving pet is unpredictable, and
asking a human is reliable but slow.

It is a stand-in for real execution, which is why every metric derived from it
carries the `placeholder_` prefix. What is genuinely real in this demo is the
bandit's adaptation: given whatever outcomes it observes, it re-ranks strategies
correctly and you can watch it do so.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..scene_graph import SceneGraph

# Per-strategy sensitivity to each hazard/condition. Hand-authored.
_PROFILE: dict[str, dict[str, float]] = {
    #                     dark  clutter occl  reflect moving fragile trip   speed
    "direct_approach":        {"dark": .35, "clutter": .55, "occl": .30, "reflect": .25,
                               "moving": .40, "fragile": .30, "trip": .45, "time": 1.0},
    "wide_arc":               {"dark": .30, "clutter": .25, "occl": .20, "reflect": .20,
                               "moving": .25, "fragile": .12, "trip": .22, "time": 1.7},
    "slow_scan_then_approach":{"dark": .12, "clutter": .30, "occl": .12, "reflect": .10,
                               "moving": .18, "fragile": .10, "trip": .20, "time": 2.4},
    "request_human_assist":   {"dark": .02, "clutter": .05, "occl": .02, "reflect": .02,
                               "moving": .03, "fragile": .02, "trip": .03, "time": 4.2},
}


@dataclass(slots=True)
class AttemptOutcome:
    strategy: str
    success: bool
    reward: float
    time_cost: float
    hazard_hit: str | None
    blamed_object: str | None


class FetchMugTask:
    name = "fetch the mug from the table"

    def __init__(self, seed: int = 1337):
        self.rng = random.Random(seed)

    def attempt(self, g: SceneGraph, strategy: str) -> AttemptOutcome:
        p = _PROFILE[strategy]
        counts = g.hazard_counts()

        conditions = {
            "dark":    max(0.0, 0.75 - g.lighting) / 0.75,
            "clutter": g.clutter,
            "occl":    g.occlusion,
            "reflect": min(counts["reflective"] / 2.0, 1.0),
            "moving":  min(counts["moving"] / 2.0, 1.0),
            "fragile": min(counts["fragile"] / 3.0, 1.0),
            "trip":    min(counts["trip"] / 2.0, 1.0),
        }

        # Independent failure chance per condition; first one to fire gets blamed.
        hazard_hit: str | None = None
        for cond, level in sorted(conditions.items(), key=lambda kv: -kv[1]):
            if level <= 0:
                continue
            if self.rng.random() < p[cond] * level:
                hazard_hit = cond
                break

        if g.by_label("mug") is None:          # the mug itself was removed
            hazard_hit, success = "mug_absent", False
        else:
            success = hazard_hit is None

        blamed = None
        if hazard_hit in ("fragile", "trip", "reflect", "moving"):
            key = {"reflect": "reflective"}.get(hazard_hit, hazard_hit)
            hit = g.with_hazard(key)
            blamed = hit[0].label if hit else None
        elif hazard_hit == "clutter":
            stray = [o for o in g.objects if o.perturbed_by == "clutter_added"]
            blamed = stray[0].label if stray else "clutter"
        elif hazard_hit in ("dark", "occl", "mug_absent"):
            blamed = {"dark": "lighting", "occl": "occlusion", "mug_absent": "mug"}[hazard_hit]

        time_cost = p["time"] * (1.0 + 0.4 * g.clutter)
        # Reward = success, penalised by hazard contact and time.
        reward = (1.0 if success else 0.0)
        if hazard_hit in ("fragile", "trip", "moving"):
            reward -= 0.5                      # physical contact is worse than a miss
        reward -= 0.12 * time_cost             # a safe-but-glacial arm should not dominate
        return AttemptOutcome(strategy, success, reward, time_cost, hazard_hit, blamed)
