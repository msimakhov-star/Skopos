"""Synthetic renderer. The whole app runs end to end on this with zero API keys.

Emits a description the browser canvas draws as a schematic top-down room, plus
a synthetic degradation curve so the UI has something moving even offline.
"""
from __future__ import annotations

import math

from .base import RenderHandle
from ..scene_graph import SceneGraph


class MockProvider:
    name = "mock"

    def __init__(self) -> None:
        self._t = 0

    def prepare(self, graph: SceneGraph) -> RenderHandle:
        self._t += 1
        # Visibility degrades with darkness, clutter and occlusion — the same
        # conditions the outcome model uses, so the picture matches the numbers.
        quality = max(0.05, graph.lighting * (1 - 0.5 * graph.clutter) * (1 - 0.5 * graph.occlusion))
        return RenderHandle(
            kind="synthetic",
            prompt=graph.render_prompt(),
            detail={
                "objects": [
                    {"label": o.label, "x": o.pose.x, "y": o.pose.y,
                     "hazards": list(o.hazards), "perturbed_by": o.perturbed_by}
                    for o in graph.objects
                ],
                "quality": quality,
                "grain": 1.0 - quality,
                "sweep": (math.sin(self._t / 9.0) + 1) / 2,
            },
            note="Synthetic schematic — no world model in the loop.",
        )

    def close(self) -> None:
        pass
