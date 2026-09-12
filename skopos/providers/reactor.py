"""Reactor provider — real, not a stub.

Verified against docs.reactor.inc on 12 September 2026.

ARCHITECTURE, AND WHY
---------------------
Reactor delivers media over WebRTC. Proxying that through our Python WebSocket
would add a hop, re-encode, and throw away the sub-50ms property that is the
entire reason to use Reactor. So:

    Python  -> exchanges the API key for a short-lived session JWT
    Browser -> takes that JWT, opens the WebRTC session itself, renders to <video>

The API key never reaches the client; the JWT is scoped to one model and expires.
This is exactly the split Reactor's own docs prescribe.

TOKEN EXCHANGE (verified)
    POST https://api.reactor.inc/tokens
    header  Reactor-API-Key: rk_...
    body    {"authorization_details": [
               {"type": "session",
                "resources": {"models": {"match": ["reactor/lingbot-world-2"]}}}]}
    -> {"jwt": "..."}

MODEL: reactor/lingbot-world-2 — image-anchored navigable world, 1664x960 @48fps.
COMMANDS the browser sends: set_image (FileRef from uploadFile), set_prompt,
set_seed, set_move_longitudinal, set_move_lateral, set_look_horizontal,
set_look_vertical, start, pause, resume, reset.

TWO RULES THAT SHAPE HOW WE USE IT
  1. An image is REQUIRED before `start`, and when image and prompt disagree the
     image wins — so `set_prompt` cannot move furniture that is in the anchor
     photo. Furniture perturbations need a NEW anchor image; lighting/atmosphere
     and NEW entities (a person stepping in) do work through the prompt, because
     an event asserting an unestablished referent conjures it into frame.
  2. When all chunks of a run complete and the session is still started, the
     server begins the next run by itself. Billing runs from ready until
     terminated, INCLUDING WHILE IDLE, at $0.0070/sec ($25/hr). The browser holds
     an idle kill-timer; see static/index.html.
"""
from __future__ import annotations

import os

import httpx

from .base import RenderHandle
from ..scene_graph import SceneGraph

TOKEN_URL = "https://api.reactor.inc/tokens"
DEFAULT_MODEL = "reactor/lingbot-world-2"

# Perturbation axes that the prompt CAN express, because they add or alter
# atmosphere and entities rather than contradicting the anchor image.
PROMPTABLE = {"lighting", "occlusion", "clutter_added"}


class ReactorProvider:
    name = "reactor"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 idle_kill_seconds: int = 90):
        self.api_key = api_key or os.environ.get("REACTOR_API_KEY", "")
        self.model = model or os.environ.get("REACTOR_MODEL", DEFAULT_MODEL)
        self.idle_kill_seconds = idle_kill_seconds
        if not self.api_key:
            raise RuntimeError(
                "REACTOR_API_KEY unset. Claim credits with promo code WORLDSLONDON, "
                "then set it in .env — or run with SKOPOS_PROVIDER=mock.")
        self._jwt: str | None = None

    def mint_token(self) -> str:
        r = httpx.post(
            TOKEN_URL,
            headers={"Reactor-API-Key": self.api_key, "Content-Type": "application/json"},
            json={
                # 2h token; the demo window is ~1h and 6h is the hard ceiling.
                "expires_after": 7200,
                "authorization_details": [{
                    "type": "session",
                    "resources": {"models": {"match": [self.model]}},
                    # max_sessions defaults to 5 and NEVER refills — a single SDK
                    # retry storm on a 429 exhausts it and the token is dead.
                    # 50 leaves room for reconnects. The duration cap is a
                    # billing backstop: no session can outlive the demo by mistake.
                    "constraints": {"max_sessions": 50,
                                    "max_session_duration_seconds": 1800},
                }],
            },
            timeout=30,
        )
        r.raise_for_status()
        body = r.json()
        self._jwt = body["jwt"]
        self.expires_at = body.get("expires_at")
        return self._jwt

    def prepare(self, graph: SceneGraph) -> RenderHandle:
        jwt = self._jwt or self.mint_token()
        promptable = sorted(
            {o.perturbed_by for o in graph.objects if o.perturbed_by} & PROMPTABLE)
        needs_new_anchor = sorted(
            {o.perturbed_by for o in graph.objects if o.perturbed_by} - PROMPTABLE)
        return RenderHandle(
            kind="webrtc",
            prompt=graph.render_prompt(),
            detail={
                "jwt": jwt,
                "modelName": self.model,
                "seed": 42,
                "idleKillSeconds": self.idle_kill_seconds,
                "sdk": "https://cdn.jsdelivr.net/npm/@reactor-models/lingbot-world-2/+esm",
                "promptable_axes": promptable,
                "axes_needing_new_anchor": needs_new_anchor,
            },
            note=("Anchor image required before start; the image wins over the prompt, "
                  "so axes needing a new anchor are shown schematically, not rendered."),
        )

    def close(self) -> None:
        self._jwt = None
