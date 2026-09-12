"""A hand-written SceneGraph for the demo room. No API key, no network.

These numbers describe a plausible small living room. They are hand-authored,
not measured — see the README's limitations section.
"""
from __future__ import annotations

from .base import Perception, PerceptionResult, debug_keep_frames
from ..scene_graph import SceneGraph, SceneObject, Pose

DEMO_ROOM = [
    SceneObject("t1", "coffee table", Pose(1.8, 0.9, 0.42), "oak",   (),                        0.93),
    SceneObject("m1", "mug",          Pose(1.9, 0.8, 0.47), "ceramic", ("fragile",),            0.88),
    SceneObject("s1", "sofa",         Pose(0.6, 2.4, 0.45), "fabric", (),                       0.95),
    SceneObject("r1", "rug",          Pose(1.6, 1.4, 0.01), "wool",   ("trip", "low_contrast"), 0.81),
    SceneObject("c1", "cable",        Pose(2.6, 1.7, 0.01), "rubber", ("trip",),                0.64),
    SceneObject("g1", "glass door",   Pose(3.4, 1.2, 1.00), "glass",  ("reflective",),          0.90),
    SceneObject("p1", "plant pot",    Pose(3.0, 0.3, 0.30), "terracotta", ("fragile",),         0.86),
    SceneObject("w1", "water bowl",   Pose(0.4, 0.4, 0.05), "plastic", ("spill",),              0.79),
    SceneObject("k1", "cat",          Pose(2.2, 2.1, 0.20), "—",      ("moving",),              0.55),
]


class MockPerception:
    name = "mock"

    def analyse(self, frames: list[bytes], room_id: str = "demo-room") -> PerceptionResult:
        total = sum(len(f) for f in frames)
        graph = SceneGraph(
            room_id=room_id,
            objects=[
                SceneObject(o.id, o.label, Pose(o.pose.x, o.pose.y, o.pose.z),
                            o.material, tuple(o.hazards), o.confidence)
                for o in DEMO_ROOM
            ],
            floor_material="engineered wood",
            lighting=0.72, clutter=0.28, occlusion=0.15,
            source="mock", frames_seen=len(frames),
            notes="Hand-authored demo room. Positions are approximate, not measured.",
        )
        retained = len(frames) if debug_keep_frames() else 0
        # frames go out of scope here and are never written anywhere
        return PerceptionResult(graph, len(frames), retained, total)
