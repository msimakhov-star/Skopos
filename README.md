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

**Pixels never reach the agent, the bandit, or any metric.** `SceneGraph.context_vector()`
is the only thing the policy ever sees: `server.py` calls it, hands the result to
`bandit.select` / `bandit.update`, and the outcome model takes `(SceneGraph, strategy)`.
There is no code path from a frame to a decision. This is not a policy we promise to
follow — it is the only code path that exists.

Image data *can* leave the device in two cases, and every byte that does is counted on
screen, in the two counters in the **"what leaves your device"** panel:

- **pixels → agent / bandit / metrics** — structurally zero. Nothing to count.
- **pixels → renderer** — the anchor image the Reactor provider requires the browser to
  upload before `start`. Bytes shown as they are sent. Zero on `mock`.

The two cases:

1. `perception/vlm.py`, active only when `ANTHROPIC_API_KEY` is set, POSTs base64 JPEG
   frames (at most six) to the Anthropic API and returns a SceneGraph. The frames are
   dropped when the call returns; `frames processed` / `frames retained` count them.
   Unset the key and the mock perception runs entirely offline.
2. The Reactor provider needs an anchor photo of the room, uploaded from the browser to
   Reactor. Counted by `pixels → renderer`. Redact locally first.

`PerceptionResult.frames_retained` is returned by every perception implementation and
displayed next to the counters. The only way it becomes non-zero is
`SKOPOS_DEBUG_KEEP_FRAMES=1`, which defaults off and logs a warning when on. The panel also
prints the exact JSON payload the policy consumes — read it on screen, there are no pixels
in it. A privacy assertion is logged once at startup. `providers/runware.py` carries a
second-vendor warning inline.

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
| `reactor` | implemented and verified live | Token exchange (max_sessions=50, 30-min session cap), browser WebRTC session in `static/reactor-client.js`, anchor upload, W/S/A/D + look, 90s idle kill. Needs `REACTOR_API_KEY`. |
| `runware` | stub | `TODO(runware)` — new-anchor image generation only. Not in the demo path. |

Reactor (`reactor.inc`) is the headline partner: real-time generative media, every major
world model behind one API, sub-50ms streaming. Python mints a session JWT (done); the
**browser** opens the WebRTC session itself (in progress), so the API key never reaches the
client and the low-latency path is not proxied away. See `docs/PROVIDER_SWAP.md`.

Two Reactor behaviours shape the design: an anchor image is required before `start` and
**the image wins over the prompt**, so prompt steering can change lighting, atmosphere and
introduce new entities, but cannot move furniture already in the anchor — that needs a new
anchor image. And billing runs from `ready` until terminated, *including while idle*, so the
client holds an idle kill-timer.

## Licence

MIT.
