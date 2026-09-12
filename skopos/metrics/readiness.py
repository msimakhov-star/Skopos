"""Rolling metrics and the readiness score.

Every number that comes from the outcome model rather than from real robot
execution is prefixed `placeholder_`. That prefix is the point: it is how you
tell, at a glance and in the UI, which figures would survive contact with a real
robot and which would not.

READINESS FORMULA (documented because it will be asked about)
------------------------------------------------------------
    readiness = 100 * (0.60 * S + 0.25 * (1 - H) + 0.15 * C)

    S = importance-weighted success rate over sampled perturbed rooms
        (self-normalised; see sampler.weighted_failure_rate)
    H = hazard-contact rate — attempts that physically touched a flagged object
        (fragile / trip / moving), which we weight separately from plain failure
        because breaking a thing is not the same as not finding it
    C = coverage confidence = min(ESS / ESS_TARGET, 1), so a score built on
        twenty effective samples cannot masquerade as one built on two hundred

Bands:  >= 75 READY   ·  45-74 MARGINAL  ·  < 45 NOT READY
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..agent.task import AttemptOutcome
from ..sampler.perturbation import weighted_failure_rate

ESS_TARGET = 40.0
CONTACT_HAZARDS = {"fragile", "trip", "moving"}


@dataclass(slots=True)
class ReadinessReport:
    placeholder_success_rate: float
    placeholder_hazard_contact_rate: float
    coverage_confidence: float
    effective_sample_size: float
    attempts: int
    placeholder_readiness: float
    state: str
    blame: dict[str, int] = field(default_factory=dict)


class Metrics:
    def __init__(self, window: int = 60):
        self.window = window
        self.recent: deque[bool] = deque(maxlen=window)
        self.weights: list[float] = []
        self.successes: list[bool] = []
        self.contacts: list[bool] = []
        self.blame: dict[str, int] = {}
        self.sparkline: deque[float] = deque(maxlen=120)

    def record(self, outcome: AttemptOutcome, weight: float) -> None:
        self.recent.append(outcome.success)
        self.weights.append(weight)
        self.successes.append(outcome.success)
        self.contacts.append(outcome.hazard_hit in CONTACT_HAZARDS)
        if not outcome.success and outcome.blamed_object:
            self.blame[outcome.blamed_object] = self.blame.get(outcome.blamed_object, 0) + 1
        self.sparkline.append(self.rolling_success_rate())

    def rolling_success_rate(self) -> float:
        return (sum(self.recent) / len(self.recent)) if self.recent else 0.0

    def report(self) -> ReadinessReport:
        fail = weighted_failure_rate(self.weights, [not s for s in self.successes])
        S = 1.0 - fail["rate"]
        hz = weighted_failure_rate(self.weights, self.contacts)
        H = hz["rate"]
        C = min(fail["ess"] / ESS_TARGET, 1.0) if fail["ess"] else 0.0
        score = 100.0 * (0.60 * S + 0.25 * (1.0 - H) + 0.15 * C)
        state = "READY" if score >= 75 else ("MARGINAL" if score >= 45 else "NOT READY")
        return ReadinessReport(
            placeholder_success_rate=S,
            placeholder_hazard_contact_rate=H,
            coverage_confidence=C,
            effective_sample_size=fail["ess"],
            attempts=len(self.weights),
            placeholder_readiness=score,
            state=state,
            blame=dict(sorted(self.blame.items(), key=lambda kv: -kv[1])[:6]),
        )
