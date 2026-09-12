"""Route planning on the occupancy grid. A*, 8-connected.

This is the one thing a robot would actually run on the map, so it is a real
computation with a runnable check, not a drawing. Cost model, stated plainly:

  free cell      1.0 per step (diagonal 1.414)
  unknown cell   3.0 per step — allowed, because a scan never sees everything,
                 but the route reports how many it crossed so nobody mistakes
                 "planned" for "safe"
  occupied       impassable, and inflated by one cell (robot radius ~10 cm)

Grid convention (same as SpatialMap.occupancy): occ[row][col], 0 free /
1 occupied / 2 unknown. For a fused or panorama map the camera is the centre
cell; for a single forward view it is the bottom-centre cell (row 0).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

UNKNOWN_COST = 3.0
_STEPS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


@dataclass(slots=True)
class Route:
    path: list[tuple[int, int]]      # (col, row) from start to goal, inclusive; [] if blocked
    length_m: float
    cells_unknown: int
    blocked: bool
    start: tuple[int, int]
    goal: tuple[int, int]

    def to_dict(self) -> dict:
        return {"path": [list(p) for p in self.path], "length_m": round(self.length_m, 2),
                "cells_unknown": self.cells_unknown, "blocked": self.blocked,
                "start": list(self.start), "goal": list(self.goal)}


def inflate(occ: np.ndarray, r: int = 1) -> np.ndarray:
    """Grow occupied cells by r so the path keeps the robot's body off walls."""
    out = occ.copy()
    ys, xs = np.where(occ == 1)
    n = occ.shape[0]
    for y, x in zip(ys, xs):
        y0, y1, x0, x1 = max(0, y - r), min(n, y + r + 1), max(0, x - r), min(n, x + r + 1)
        block = out[y0:y1, x0:x1]
        block[block != 1] = 1
    return out


def camera_cell(occ: np.ndarray, centred: bool) -> tuple[int, int]:
    n = occ.shape[0]
    return (n // 2, n // 2) if centred else (n // 2, 0)


def plan_route(occ: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
               cell_m: float = 0.1, inflate_r: int = 1) -> Route:
    grid = inflate(occ, inflate_r)
    n = grid.shape[0]
    sx, sy = start; gx, gy = goal
    inb = lambda x, y: 0 <= x < n and 0 <= y < n
    if not (inb(sx, sy) and inb(gx, gy)) or grid[gy, gx] == 1:
        return Route([], 0.0, 0, True, start, goal)
    # the start cell is where the camera stood, so it is free by definition
    grid[sy, sx] = 0

    def step_cost(x, y, d):
        return d * (UNKNOWN_COST if grid[y, x] == 2 else 1.0)

    h = lambda x, y: math.hypot(gx - x, gy - y)
    best = {(sx, sy): 0.0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    pq = [(h(sx, sy), 0.0, sx, sy)]
    while pq:
        f, g, x, y = heapq.heappop(pq)
        if (x, y) == (gx, gy):
            path = [(x, y)]
            while (x, y) in came:
                x, y = came[(x, y)]; path.append((x, y))
            path.reverse()
            length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])) * cell_m
            unknown = sum(1 for cx, cy in path if occ[cy, cx] == 2)
            return Route(path, length, unknown, False, start, goal)
        if g > best.get((x, y), math.inf):
            continue
        for dx, dy, d in _STEPS:
            nx, ny = x + dx, y + dy
            if not inb(nx, ny) or grid[ny, nx] == 1:
                continue
            # no cutting corners through an obstacle
            if dx and dy and (grid[y, nx] == 1 or grid[ny, x] == 1):
                continue
            ng = g + step_cost(nx, ny, d)
            if ng < best.get((nx, ny), math.inf):
                best[(nx, ny)] = ng; came[(nx, ny)] = (x, y)
                heapq.heappush(pq, (ng + h(nx, ny), ng, nx, ny))
    return Route([], 0.0, 0, True, start, goal)
