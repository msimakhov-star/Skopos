"""A provider renders a SceneGraph into something a human can look at.

Deliberately narrow: the provider receives a SceneGraph and returns a handle
describing how the BROWSER should obtain the media. It never touches pixels in
Python, which is why swapping mock -> Reactor changes no other module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..scene_graph import SceneGraph


@dataclass(slots=True)
class RenderHandle:
    kind: str                       # "synthetic" | "webrtc"
    prompt: str = ""
    detail: dict = field(default_factory=dict)
    note: str = ""


class Provider(Protocol):
    name: str
    def prepare(self, graph: SceneGraph) -> RenderHandle: ...
    def close(self) -> None: ...
