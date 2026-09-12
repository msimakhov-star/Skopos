"""Contextual bandit over a fixed set of task strategies.

WHAT THIS IS, EXACTLY
---------------------
This is a CONTEXTUAL BANDIT, not a trained policy. It is the one-step case of
reinforcement learning: there is a context, an action, and an immediate reward,
and no state transition to credit. Nothing here is a neural network, nothing is
"trained", and no weights leave the process. Calling it a bandit is the precise
claim, and it is the only claim we make.

WHY LinUCB AND NOT EPSILON-GREEDY
---------------------------------
Both would work. LinUCB wins here for one concrete reason: it maintains a ridge
regression per arm, so it yields a VALUE ESTIMATE AND A CONFIDENCE INTERVAL for
every arm at every step. The demo shows four live bars that re-order when the
room changes; epsilon-greedy would give flat point estimates with no uncertainty
to display, and its exploration would look like random flicker rather than a
system that knows what it does not know. LinUCB's exploration bonus is legible:
the bar for an untried arm is wide because A_a is still near identity.

    theta_a = A_a^-1 b_a
    score_a = theta_a . x  +  alpha * sqrt(x^T A_a^-1 x)
    on reward r:  A_a += x x^T ;  b_a += r x

alpha trades exploration for exploitation. 1.1 keeps every arm alive long enough
that the bars stay meaningful on screen; at 0.6 LinUCB commits after a single pull
per arm and the demo shows four frozen bars instead of a system adapting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

STRATEGIES: tuple[str, ...] = (
    "direct_approach",
    "wide_arc",
    "slow_scan_then_approach",
    "request_human_assist",
)


@dataclass(slots=True)
class ArmState:
    name: str
    A: np.ndarray
    b: np.ndarray
    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.pulls if self.pulls else 0.0


class LinUCB:
    def __init__(self, dim: int, alpha: float = 1.1, arms: tuple[str, ...] = STRATEGIES,
                 ridge: float = 1.0):
        self.dim, self.alpha = dim, alpha
        self.arms = {a: ArmState(a, np.eye(dim) * ridge, np.zeros(dim)) for a in arms}

    def _theta(self, arm: ArmState) -> np.ndarray:
        return np.linalg.solve(arm.A, arm.b)

    def scores(self, context: list[float]) -> dict[str, dict[str, float]]:
        """Per-arm expected value, exploration bonus and UCB. This is what the
        four live bars render."""
        x = np.asarray(context, dtype=float)
        out: dict[str, dict[str, float]] = {}
        for name, arm in self.arms.items():
            Ainv_x = np.linalg.solve(arm.A, x)
            mean = float(self._theta(arm) @ x)
            bonus = float(self.alpha * math.sqrt(max(x @ Ainv_x, 0.0)))
            out[name] = {"mean": mean, "bonus": bonus, "ucb": mean + bonus,
                         "pulls": arm.pulls, "mean_reward": arm.mean_reward}
        return out

    def select(self, context: list[float]) -> str:
        s = self.scores(context)
        return max(s, key=lambda a: s[a]["ucb"])

    def update(self, arm_name: str, context: list[float], reward: float) -> None:
        x = np.asarray(context, dtype=float)
        arm = self.arms[arm_name]
        arm.A += np.outer(x, x)
        arm.b += reward * x
        arm.pulls += 1
        arm.reward_sum += reward
