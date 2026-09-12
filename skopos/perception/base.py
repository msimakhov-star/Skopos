"""Perception turns pixels into a SceneGraph. Pixels stop here.

Every implementation MUST:
  - accept frames as bytes,
  - return a SceneGraph,
  - discard the frames before returning,
  - report how many frames it discarded.

The discard is not a policy we promise to follow; it is the only code path that
exists. `PerceptionResult.frames_retained` is what the UI displays, and the only
way it becomes non-zero is the explicitly-named debug flag.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from ..scene_graph import SceneGraph


def debug_keep_frames() -> bool:
    """Defaults OFF. The only switch that lets a raw frame touch disk."""
    return os.environ.get("SKOPOS_DEBUG_KEEP_FRAMES", "0") == "1"


@dataclass(slots=True)
class PerceptionResult:
    graph: SceneGraph
    frames_processed: int
    frames_retained: int          # non-zero ONLY under SKOPOS_DEBUG_KEEP_FRAMES=1
    bytes_discarded: int

    @property
    def privacy_ok(self) -> bool:
        return self.frames_retained == 0


class Perception(Protocol):
    name: str

    def analyse(self, frames: list[bytes], room_id: str) -> PerceptionResult: ...
