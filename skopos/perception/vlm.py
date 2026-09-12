"""Anthropic vision perception. Used only when ANTHROPIC_API_KEY is set.

Frames are sent to the model, a SceneGraph comes back, and the frames are
dropped when this function returns. Nothing is written to disk.
"""
from __future__ import annotations

import base64, json, os

import httpx

from .base import PerceptionResult, debug_keep_frames
from ..scene_graph import SceneGraph, SceneObject, Pose, ALL_HAZARDS

# Exact Claude API id per platform.claude.com/docs/en/about-claude/models/overview
# (fetched 2026-09-12): Sonnet-class "claude-sonnet-5", Haiku-class
# "claude-haiku-4-5-20251001" (alias "claude-haiku-4-5"). Both accept image input.
MODEL = os.environ.get("SKOPOS_VLM_MODEL", "claude-sonnet-5")

SCHEMA_HINT = f"""Return ONLY JSON:
{{"floor_material": str, "lighting": 0..1, "clutter": 0..1, "occlusion": 0..1,
  "objects": [{{"label": str, "x": metres, "y": metres, "z": metres,
                "material": str, "hazards": [{list(ALL_HAZARDS)}], "confidence": 0..1}}]}}
Origin is the camera's position in the first frame, +x right, +y forward.
Positions are approximate. Flag every hazard that applies to a floor robot."""


class VLMPerception:
    name = "anthropic-vlm"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY unset — use MockPerception")

    def analyse(self, frames: list[bytes], room_id: str = "scan") -> PerceptionResult:
        total = sum(len(f) for f in frames)
        content: list[dict] = [{"type": "text", "text": SCHEMA_HINT}]
        for f in frames[:6]:
            content.append({"type": "image", "source": {
                "type": "base64",
                "media_type": "image/png" if f[:4] == b"\x89PNG" else "image/jpeg",
                "data": base64.b64encode(f).decode()}})
        r = httpx.post(
            os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com") + "/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={"model": MODEL, "max_tokens": 2000,
                  "messages": [{"role": "user", "content": content}]},
            timeout=90,
        )
        if r.status_code in (400, 404):
            raise RuntimeError(
                f"Anthropic API returned HTTP {r.status_code} for model {MODEL!r} "
                f"(set SKOPOS_VLM_MODEL to override): {r.text[:300]}")
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json()["content"])
        raw = json.loads(text[text.index("{"): text.rindex("}") + 1])

        objs = []
        for i, o in enumerate(raw.get("objects", [])):
            hz = tuple(h for h in o.get("hazards", []) if h in ALL_HAZARDS)
            objs.append(SceneObject(
                f"o{i}", str(o["label"]),
                Pose(float(o.get("x", 0)), float(o.get("y", 0)), float(o.get("z", 0))),
                str(o.get("material", "unknown")), hz, float(o.get("confidence", 0.7))))

        graph = SceneGraph(
            room_id=room_id, objects=objs,
            floor_material=str(raw.get("floor_material", "unknown")),
            lighting=float(raw.get("lighting", 0.7)),
            clutter=float(raw.get("clutter", 0.2)),
            occlusion=float(raw.get("occlusion", 0.1)),
            source=self.name, frames_seen=len(frames),
            notes="VLM estimate. Positions are approximate, not measured.")
        return PerceptionResult(graph, len(frames),
                                len(frames) if debug_keep_frames() else 0, total)
