"""Add a new view of the SAME room into the map that already exists.

The privacy contract says frames are discarded, so a new photo cannot be
matched against old photos — there are none. What IS retained is the
occupancy grid. So the new view is registered against the grid itself: build
its own (camera-centred) points, try every yaw, rasterise, and keep the yaw
whose obstacle cells agree most with what the map already knows. Then merge.

Honest limits: rotation only (the assumption of every map here is one spot);
a photo taken from somewhere else will still be forced onto the same origin.
The registration score is reported so a bad fit is visible, not hidden.
"""
from __future__ import annotations

import numpy as np

from .depth import SpatialMap, _occupancy, GRID_M, CELL_M, CAMERA_HEIGHT_M

YAW_STEP_DEG = 3.0


def _rotate(pts: np.ndarray, yaw_deg: float) -> np.ndarray:
    t = np.radians(yaw_deg)
    out = pts.copy()
    x, y = pts[:, 0], pts[:, 1]
    out[:, 0] = x * np.cos(t) + y * np.sin(t)
    out[:, 1] = -x * np.sin(t) + y * np.cos(t)
    return out


def _score(existing: np.ndarray, cand: np.ndarray) -> float:
    """Agreement between two centred grids where both have an opinion:
    +1 per cell both call occupied, +0.25 per cell both call free, -1 per
    cell one calls occupied and the other free. Unknown cells abstain."""
    known = (existing != 2) & (cand != 2)
    if known.sum() == 0:
        return -1.0
    e, c = existing[known], cand[known]
    both_occ = ((e == 1) & (c == 1)).sum()
    both_free = ((e == 0) & (c == 0)).sum()
    conflict = ((e == 1) & (c == 0)).sum() + ((e == 0) & (c == 1)).sum()
    return float(both_occ + 0.25 * both_free - conflict) / float(known.sum())


def register_yaw(existing_occ: np.ndarray, new_pts: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Best yaw (deg) placing new_pts onto existing_occ, its score, and the
    rasterised candidate at that yaw. Coarse search then a fine pass."""
    best = (0.0, -9.0, None)
    for yaw in np.arange(0.0, 360.0, YAW_STEP_DEG):
        occ = _occupancy(_rotate(new_pts, yaw), centred=True)
        s = _score(existing_occ, occ)
        if s > best[1]:
            best = (float(yaw), s, occ)
    y0 = best[0]
    for yaw in np.arange(y0 - YAW_STEP_DEG, y0 + YAW_STEP_DEG + 0.01, 0.5):
        occ = _occupancy(_rotate(new_pts, yaw % 360.0), centred=True)
        s = _score(existing_occ, occ)
        if s > best[1]:
            best = (float(yaw % 360.0), s, occ)
    return best


def merge(existing: np.ndarray, incoming: np.ndarray) -> np.ndarray:
    """Union of knowledge. Occupied wins over free (a wall someone saw is a
    wall); free fills unknown; unknown never overwrites anything."""
    out = existing.copy()
    out[(incoming == 0) & (out == 2)] = 0
    out[incoming == 1] = 1
    return out


def promote_to_centred(occ: np.ndarray) -> np.ndarray:
    """A single forward view keeps the camera at row 0 (bottom-centre) with
    6 m ahead. The centred frame keeps the camera at row n//2 with 3 m each
    way. Shift the rows: rows 0..n//2-1 (0-3 m ahead) move up to n//2..n-1;
    the far 3-6 m fall off (they were mostly unknown); the half behind the
    camera becomes unknown, because nobody has looked there yet. Points need
    no shift — they are camera-at-origin in both frames."""
    n = occ.shape[0]; h = n // 2
    out = np.full_like(occ, 2)
    out[h:, :] = occ[:n - h, :]
    return out


def append_view(existing: dict, new_map: SpatialMap) -> dict:
    """existing: the dict a previous scan produced. A single forward view is
    promoted to the centred frame first (see promote_to_centred), so the
    common case — scan one photo, then add another — just works.
    new_map: a SpatialMap for the new photo, camera at the origin.
    Returns a new map dict plus registration details."""
    occ_prev = np.asarray(existing["occupancy"], np.uint8)
    promoted = False
    if not existing.get("centred"):
        occ_prev = promote_to_centred(occ_prev)
        promoted = True
    pts = new_map.points
    if not new_map.centred:                      # a forward-only frame: put its camera at the origin
        pts = pts.copy(); pts[:, 1] -= 0.0       # already camera-at-origin in _points; nothing to shift
    yaw, score, occ_new = register_yaw(occ_prev, pts)
    merged = merge(occ_prev, occ_new)
    # merge point clouds for the viewer: keep the old subsample, add the new one rotated
    old_pts = np.asarray(existing.get("points", []), np.float32).reshape(-1, 6)
    new_pts = _rotate(pts, yaw)
    rng = np.random.default_rng(0)
    if len(new_pts) > 4000:
        new_pts = new_pts[rng.choice(len(new_pts), 4000, replace=False)]
    if len(old_pts) > 4000:
        old_pts = old_pts[rng.choice(len(old_pts), 4000, replace=False)]
    all_pts = np.concatenate([old_pts, new_pts]) if len(old_pts) else new_pts
    n = int(GRID_M / CELL_M)
    known_before = int((occ_prev != 2).sum()); known_after = int((merged != 2).sum())
    out = dict(existing)
    out.update({
        "points": np.round(all_pts, 2).tolist(),
        "occupancy": merged.tolist(),
        "centred": True,
        "views": int(existing.get("views", 1)) + 1,
        "last_register": {"yaw_deg": round(yaw, 1), "score": round(score, 3),
                          "cells_known_before": known_before, "cells_known_after": known_after,
                          "grid_cells": n * n, "promoted_single_view": promoted},
        "note": (existing.get("note", "") +
                 (" Single view promoted to the centred frame (far half dropped, rear half unknown)." if promoted else "") +
                 f" +1 view registered by yaw search (yaw {yaw:.0f}°, agreement {score:.2f}); "
                 "rotation-only, plausible not measured."),
    })
    return out
