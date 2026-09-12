"""The SceneGraph is the central object of Skopos.

Pixels stop at perception. Everything downstream — the perturbation sampler, the
agent, the bandit, the metrics, the world-model provider — depends only on this
structured, human-readable description. That is what makes the privacy claim
mechanical rather than aspirational: the SceneGraph is the ONLY thing that ever
crosses a network boundary, and you can read it on screen.

Coordinates are approximate, in metres, in a room-local frame with the origin at
the scanner's start position. They are NOT metric ground truth — see README.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Literal

Hazard = Literal[
    "fragile",        # breaks on contact
    "spill",          # liquid; contact is a failure even without breakage
    "reflective",     # confuses depth/vision
    "low_contrast",   # hard to segment from the floor
    "moving",         # pet, person, robot vacuum
    "trip",           # cable, rug edge
]

ALL_HAZARDS: tuple[str, ...] = (
    "fragile", "spill", "reflective", "low_contrast", "moving", "trip",
)


@dataclass(frozen=True, slots=True)
class Pose:
    """Approximate object centre, room-local metres. z is height above floor."""
    x: float
    y: float
    z: float = 0.0

    def dist(self, other: "Pose") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass(slots=True)
class SceneObject:
    id: str
    label: str
    pose: Pose
    material: str = "unknown"
    hazards: tuple[str, ...] = ()
    confidence: float = 0.8
    # Set when the sampler perturbs this object, so the UI can mark what moved.
    perturbed_by: str | None = None

    def has(self, hazard: str) -> bool:
        return hazard in self.hazards


@dataclass(slots=True)
class SceneGraph:
    """A room, as structured text. No pixels, ever."""
    room_id: str
    objects: list[SceneObject] = field(default_factory=list)
    floor_material: str = "unknown"
    lighting: float = 0.7          # 0 dark .. 1 bright
    clutter: float = 0.2           # 0 clear floor .. 1 impassable
    occlusion: float = 0.1         # 0 open sightlines .. 1 target hidden
    source: str = "mock"           # which perception produced this
    frames_seen: int = 0           # how many frames were processed AND discarded
    notes: str = ""

    # ---- lookup -----------------------------------------------------------
    def by_id(self, oid: str) -> SceneObject | None:
        return next((o for o in self.objects if o.id == oid), None)

    def by_label(self, label: str) -> SceneObject | None:
        return next((o for o in self.objects if o.label == label), None)

    def with_hazard(self, hazard: str) -> list[SceneObject]:
        return [o for o in self.objects if o.has(hazard)]

    def hazard_counts(self) -> dict[str, int]:
        return {h: len(self.with_hazard(h)) for h in ALL_HAZARDS}

    # ---- the agent's context ---------------------------------------------
    def context_vector(self) -> list[float]:
        """Dense context for the contextual bandit. Order is FIXED — the bandit's
        learned weights are indexed by position, so appending is safe and
        reordering is not."""
        c = self.hazard_counts()
        return [
            1.0,                                  # bias
            self.lighting,
            self.clutter,
            self.occlusion,
            min(len(self.objects) / 12.0, 1.0),
            min(c["fragile"] / 3.0, 1.0),
            min(c["spill"] / 2.0, 1.0),
            min(c["reflective"] / 2.0, 1.0),
            min(c["low_contrast"] / 2.0, 1.0),
            min(c["moving"] / 2.0, 1.0),
            min(c["trip"] / 2.0, 1.0),
        ]

    CONTEXT_DIM = 11

    # ---- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def copy(self) -> "SceneGraph":
        return SceneGraph(**json.loads(self.to_json()) | {
            "objects": [
                SceneObject(
                    id=o.id, label=o.label, pose=Pose(o.pose.x, o.pose.y, o.pose.z),
                    material=o.material, hazards=tuple(o.hazards),
                    confidence=o.confidence, perturbed_by=o.perturbed_by,
                )
                for o in self.objects
            ]
        })

    def render_prompt(self) -> str:
        """The scene, as a sentence a world model can render.

        Deliberately describes only what a camera would see. The Reactor prompt
        guide is explicit that when the reference image and the prompt disagree,
        the image wins — so this text describes the room rather than trying to
        rearrange it. Moving furniture requires a new anchor image.
        """
        here = [o.label for o in self.objects]
        light = "brightly lit" if self.lighting > 0.66 else (
            "dimly lit" if self.lighting < 0.33 else "evenly lit")
        floor = f"{self.floor_material} floor" if self.floor_material != "unknown" else "floor"
        base = (
            f"A {light} domestic room with a {floor}, seen from the eye level of a "
            f"small floor robot about 40 centimetres high. The room contains "
            f"{', '.join(here)}."
        )
        pins = [
            f"The world contains EXACTLY ONE {o.label} at a fixed position."
            for o in self.objects if o.has("fragile") or o.label == "mug"
        ]
        return " ".join([base, *pins])
