"""Rolling metrics and the readiness score.

Every number that comes from the outcome model rather than from real robot
execution is prefixed `placeholder_`. That prefix is the point: it is how you
tell, at a glance and in the UI, which figures would survive contact with a real
robot and which would not.

READINESS FORMULA (documented because it will be asked about)
------------------------------------------------------------
    readiness = 100 * (0.60 * S + 0.25 * (1 - H) + 0.15 * C)

    S = importance-weighted success rate over sampled perturbed rooms
        (self-normalised; see sampler.weighted_failure_rate). The server
        feeds it the AUTONOMOUS probe attempt (server.Session.probe), never
        the deployed bandit's request_human_assist: a human fetching the
        mug is not the robot being ready.
    H = hazard-contact rate — attempts that physically touched a flagged object
        (fragile / trip / moving), which we weight separately from plain failure
        because breaking a thing is not the same as not finding it
    C = coverage confidence = min(ESS / ESS_TARGET, 1), so a score built on
        twenty effective samples cannot masquerade as one built on two hundred

S, H and ESS are estimated over a SLIDING WINDOW of the last WINDOW (120)
attempts, not the whole session. The score is a statement about the room as it
is now. Keeping every attempt since session start meant the bandit's early
exploration failures dragged the estimate for the rest of the session, and one
heavy importance weight against a small cumulative sum could move the score by
20+ points in a single step.

DISPLAY SMOOTHING
-----------------
    placeholder_readiness           raw score from the window, unsmoothed
    placeholder_readiness_smoothed  exponential moving average of the raw score
                                    (ALPHA = 0.15 per attempt) once attempts > WARMUP (20;
                                    before that it equals the raw score), then slew-limited
                                    to MAX_STEP = 4 points per attempt. This is
                                    the number to display; the raw one is
                                    reported next to it so smoothing cannot hide
                                    a change, only delay it.
    The smoothed score lags on purpose: a genuine 40-point change shows over
    ~10 attempts (~5.5 s at the server's 0.55 s step), not in one frame. The
    slew limit exists for the cold start: measured over 14 seeded runs, the EMA
    alone moved up to 13 points per attempt in the first four attempts (S flips
    between 0 and 1 on a one-sample window) and under 3 points after attempt 20.

STATE HYSTERESIS (on the smoothed score)
----------------------------------------
    READY      enter at >= 75, leave only below 70
    NOT READY  enter at <  45, leave only above 50
    MARGINAL   everything else
Without hysteresis the word flipped every time the score crossed 75 or 45 by a
fraction of a point.

CONFIDENCE
----------
    placeholder_success_ci = (lo, hi): normal approximation on the weighted
    success rate, S ± 1.96 * sqrt(S (1 - S) / ESS), clipped to [0, 1]; (0, 1)
    when ESS is 0. Also reported: effective_sample_size (Kish ESS of the
    window) and window_fill = attempts in window / WINDOW.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ..agent.task import AttemptOutcome
from ..sampler.perturbation import weighted_failure_rate

ESS_TARGET = 40.0
CONTACT_HAZARDS = {"fragile", "trip", "moving"}
WINDOW = 120
ALPHA = 0.15
MAX_STEP = 4.0
# Smoothing only means something once there is something to smooth. For the
# first WARMUP attempts the displayed score IS the raw estimate: seeding an EMA
# from attempt #1 and capping it at 4 points/step let one lucky first attempt
# hold the display near 65 while the raw estimate read 13 — a number that
# looked stable and was simply wrong.
WARMUP = 20
READY_ENTER, READY_LEAVE = 75.0, 70.0
NOT_READY_ENTER, NOT_READY_LEAVE = 45.0, 50.0


@dataclass(slots=True)
class ReadinessReport:
    placeholder_success_rate: float
    placeholder_success_ci: tuple[float, float]
    placeholder_hazard_contact_rate: float
    coverage_confidence: float
    effective_sample_size: float
    window_fill: float
    attempts: int
    placeholder_readiness: float
    placeholder_readiness_smoothed: float
    state: str
    blame: dict[str, int] = field(default_factory=dict)
    warming_up: bool = False

def next_state(prev: str | None, score: float) -> str:
    """Hysteresis: a band is left only once the score clears the far edge."""
    if prev == "READY" and score >= READY_LEAVE:
        return "READY"
    if prev == "NOT READY" and score <= NOT_READY_LEAVE:
        return "NOT READY"
    if score >= READY_ENTER:
        return "READY"
    if score < NOT_READY_ENTER:
        return "NOT READY"
    return "MARGINAL"


class Metrics:
    def __init__(self, window: int = WINDOW, alpha: float = ALPHA):
        self.window = window
        self.alpha = alpha
        self.attempts = 0
        self.weights: deque[float] = deque(maxlen=window)
        self.successes: deque[bool] = deque(maxlen=window)
        self.contacts: deque[bool] = deque(maxlen=window)
        self.recent: deque[bool] = deque(maxlen=60)      # unweighted, sparkline only
        self.blame: dict[str, int] = {}
        self.sparkline: deque[float] = deque(maxlen=120)
        self.smoothed: float | None = None
        self.state: str | None = None

    def record(self, outcome: AttemptOutcome, weight: float) -> None:
        self.attempts += 1
        self.recent.append(outcome.success)
        self.weights.append(weight)
        self.successes.append(outcome.success)
        self.contacts.append(outcome.hazard_hit in CONTACT_HAZARDS)
        if not outcome.success and outcome.blamed_object:
            self.blame[outcome.blamed_object] = self.blame.get(outcome.blamed_object, 0) + 1
        self.sparkline.append(self.rolling_success_rate())
        raw = self._raw()["score"]
        if self.attempts <= WARMUP or self.smoothed is None:
            self.smoothed = raw                    # warm-up: show what we actually have
        else:
            step = self.alpha * (raw - self.smoothed)
            self.smoothed += max(-MAX_STEP, min(MAX_STEP, step))
        self.state = next_state(self.state, self.smoothed)

    def rolling_success_rate(self) -> float:
        return (sum(self.recent) / len(self.recent)) if self.recent else 0.0

    def _raw(self) -> dict:
        fail = weighted_failure_rate(self.weights, [not s for s in self.successes])
        S = 1.0 - fail["rate"]
        H = weighted_failure_rate(self.weights, self.contacts)["rate"]
        ess = fail["ess"]
        C = min(ess / ESS_TARGET, 1.0) if ess else 0.0
        half = 1.96 * math.sqrt(S * (1.0 - S) / ess) if ess > 0 else 1.0
        return {"S": S, "H": H, "C": C, "ess": ess,
                "ci": (max(0.0, S - half), min(1.0, S + half)),
                "score": 100.0 * (0.60 * S + 0.25 * (1.0 - H) + 0.15 * C)}

    def report(self) -> ReadinessReport:
        r = self._raw()
        smoothed = r["score"] if self.smoothed is None else self.smoothed
        return ReadinessReport(
            placeholder_success_rate=r["S"],
            placeholder_success_ci=r["ci"],
            placeholder_hazard_contact_rate=r["H"],
            coverage_confidence=r["C"],
            effective_sample_size=r["ess"],
            window_fill=len(self.weights) / self.window,
            attempts=self.attempts,
            placeholder_readiness=r["score"],
            placeholder_readiness_smoothed=smoothed,
            warming_up=self.attempts <= WARMUP,
            state=self.state or next_state(None, smoothed),
            blame=dict(sorted(self.blame.items(), key=lambda kv: -kv[1])[:6]),
        )
