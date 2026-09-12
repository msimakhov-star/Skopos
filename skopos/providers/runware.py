"""Runware fallback stub — same interface, deliberately minimal.

Runware is a many-model inference endpoint, not a real-time world model, so it
cannot serve the live navigable stream. It is here as an escape hatch for
generating a NEW ANCHOR IMAGE when a perturbation moves furniture (which the
Reactor prompt cannot do — the image wins).

TODO(runware): fill in from https://runware.ai/docs
  - base URL and auth header shape
  - the task name for image-to-image editing, and its parameter names
  - response -> image bytes mapping
Not wired into the demo path. See docs/PROVIDER_SWAP.md.

PRIVACY WARNING: sending a scan photo here ships it to a second vendor. Do the
face/screen redaction locally BEFORE any upload, or do not use this at all.
"""
from __future__ import annotations

import os

from .base import RenderHandle
from ..scene_graph import SceneGraph


class RunwareProvider:
    name = "runware"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("RUNWARE_API_KEY", "")

    def prepare(self, graph: SceneGraph) -> RenderHandle:
        return RenderHandle(
            kind="synthetic",
            prompt=graph.render_prompt(),
            detail={"objects": [], "quality": 0.5, "grain": 0.5, "sweep": 0.5},
            note="Runware adapter is a stub — see TODO(runware) in providers/runware.py.",
        )

    def close(self) -> None:
        pass
