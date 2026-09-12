# Skopos

**A pre-deployment readiness check for home robots.**
A home robot should meet your home before it ships to your home.

You scan your room with a phone. Skopos extracts a semantic **scene graph** — objects,
approximate positions, materials, hazard flags — then samples perturbed variants of that
room, runs an agent attempting one household task across them, and reports where it fails.
When the room changes, the agent adapts online and you watch the recovery live.

Built at Worlds London, 12 September 2026.

## Quickstart

```bash
git clone <this repo> && cd skopos
./run.sh
open http://127.0.0.1:8000
```

No API keys required. The whole app runs end to end on the mock provider.
`make check` runs the self-check suite.

## Architecture in five bullets

- **`scene_graph.py`** — the central object. Pixels stop at perception; every other module
  depends only on this typed, readable structure.
- **`perception/`** — frames in, SceneGraph out, frames discarded. Mock by default; an
  Anthropic-vision implementation activates if `ANTHROPIC_API_KEY` is set.
- **`sampler/`** — a prior over six perturbation axes, Monte Carlo sampling with importance
  weighting so rare-but-severe rooms get scored without being enumerated.
- **`agent/`** — one task (fetch the mug), four discrete strategies, a LinUCB contextual
  bandit selecting among them from the scene-graph context vector.
- **`providers/`** — swappable renderers behind one interface: `mock` (synthetic, offline),
  `reactor` (real-time world model over WebRTC), `runware` (stub).

## What this is not — read before quoting any number

These limitations are the point, not a disclaimer.

1. **We are not training a policy.** Strategy selection is a **contextual bandit** (LinUCB) —
   the one-step case of reinforcement learning. There is a context, an action, and an
   immediate reward, and no state transition to credit. Nothing is trained; no weights
   leave the process.
2. **We are not building a digital twin.** Generated rooms are *plausible*, not accurate.
   There is no metric ground truth and no collision mesh. Positions are approximate.
3. **No robot executes anything.** `agent/task.py` is an **outcome model**: a hand-written,
   seeded function mapping (SceneGraph, strategy) → success. Every metric derived from it
   carries the **`placeholder_`** prefix so you can see instantly which figures would
   survive contact with a real robot. What *is* real is the adaptation: given whatever
   outcomes it observes, the bandit re-ranks strategies correctly, and you can watch it.
4. **The readiness score is a composite we defined**, not an industry standard. Formula below.

## Privacy

Frames are processed into a SceneGraph and then discarded. **Only structured text ever
crosses a network boundary.** This is not a policy we promise to follow — it is the only
code path that exists:

- `PerceptionResult.frames_retained` is returned by every perception implementation and
  displayed in the UI. The only way it becomes non-zero is `SKOPOS_DEBUG_KEEP_FRAMES=1`,
  which defaults off and logs a warning when on.
- The **"what leaves your device"** panel shows the exact JSON payload transmitted, next to
  the discarded frame count. Read it on screen — there are no pixels in it.
- A privacy assertion is logged once at startup.

If you wire up a cloud VLM or Runware, frames or photos do leave the device — redact
locally first. `providers/runware.py` carries that warning inline.

## The importance-weighting maths

We want the failure rate a household would actually experience, `F = E_p[fail(x)]`, where
`p` is the prior over how rooms really change. Under `p`, severe rooms are rare, so plain
Monte Carlo almost never samples the failures we care about. So we draw from a harsher
proposal `q` and correct the bias:

```
F ≈ (1/N) Σ w_i · fail(x_i),    w_i = p(x_i)/q(x_i),    x_i ~ q
```

Each axis draws a magnitude `m ∈ [0,1)` from `Beta(1, b)`, density `b(1-m)^(b-1)`. The prior
uses `b_p`, the proposal `b_q = b_p / tilt` with `tilt > 1`. So the per-axis ratio is exactly

```
p(m)/q(m) = tilt · (1-m)^(b_p − b_p/tilt)
```

and the sample weight is the product over independent axes, accumulated in logs. We report
the self-normalised estimate `Σ w·fail / Σ w` and the **effective sample size**
`(Σw)²/Σw²` — the honest answer to "how many draws is this really worth". ESS is shown in
the UI header; a run with low ESS is penalised through the coverage term below.

## The readiness score

```
readiness = 100 · (0.60·S + 0.25·(1 − H) + 0.15·C)

S = importance-weighted success rate over sampled rooms
H = hazard-contact rate (attempts that physically touched a flagged object)
C = coverage confidence = min(ESS / 40, 1)
```

`H` is separate from plain failure because breaking something is not the same as not
finding it. `C` stops a score built on twenty effective samples from masquerading as one
built on two hundred.

Bands: **≥ 75 READY · 45–74 MARGINAL · < 45 NOT READY**

## Providers

| Provider | Status | Notes |
|---|---|---|
| `mock` | complete | Synthetic schematic + degradation curve. Zero API keys. Default. |
| `reactor` | implemented | Real token exchange and session handoff. Needs `REACTOR_API_KEY`. |
| `runware` | stub | `TODO(runware)` — new-anchor image generation only. Not in the demo path. |

Reactor (`reactor.inc`) is the headline partner: real-time generative media, every major
world model behind one API, sub-50ms streaming. Python mints a session JWT; the **browser**
opens the WebRTC session itself, so the API key never reaches the client and the low-latency
path is not proxied away. See `docs/PROVIDER_SWAP.md`.

Two Reactor behaviours shape the design: an anchor image is required before `start` and
**the image wins over the prompt**, so prompt steering can change lighting, atmosphere and
introduce new entities, but cannot move furniture already in the anchor — that needs a new
anchor image. And billing runs from `ready` until terminated, *including while idle*, so the
client holds an idle kill-timer.

## Licence

MIT.
