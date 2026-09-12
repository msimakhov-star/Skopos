"""Depth → point cloud → occupancy map. On device, CPU, no torch.

Depth Anything V2 Small, quantised ONNX (Apache-2.0), 37 MB, ~350 ms per frame
on an M-series CPU. The model predicts RELATIVE inverse depth — larger means
closer — with no metric scale. We anchor scale with one assumption stated on
screen: the camera is ~1.4 m above the floor (a phone held at chest height),
and the floor is the dominant plane in the lower part of the frame.

What comes out:
  points      (N, 6) float32  x, y, z metres in a camera-forward frame, + r, g, b
  occupancy   (H, W)  uint8   0 free / 1 occupied / 2 unknown, top-down grid
  extents     metres per cell, grid origin

Honesty: this is monocular depth. Scale is approximate, walls bulge, thin
objects vanish. It is a plausible map, not a measured one. Say so.
"""
from __future__ import annotations

import os, urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL = MODEL_DIR / "model_quantized.onnx"
MODEL_DATA = MODEL_DIR / "model_quantized.onnx_data"
HF = "https://huggingface.co/onnx-community/depth-anything-v2-small-ONNX/resolve/main/onnx/"

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

CAMERA_HEIGHT_M = 1.4      # stated assumption, shown in the UI
HFOV_DEG = 69.0            # iPhone main camera, wide-ish; approximate
CELL_M = 0.10              # occupancy grid resolution
GRID_M = 6.0               # grid covers 6 m × 6 m in front of the camera


def ensure_model() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    for f in (MODEL, MODEL_DATA):
        if not f.exists():
            urllib.request.urlretrieve(HF + f.name, f)


_session = None
def _sess():
    global _session
    if _session is None:
        import onnxruntime as ort
        ensure_model()
        _session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    return _session


def predict_depth(img_rgb: np.ndarray, size: int = 518) -> np.ndarray:
    """img_rgb uint8 (H, W, 3) -> relative inverse depth float32 (h, w), larger = closer."""
    from PIL import Image
    h, w = img_rgb.shape[:2]
    s = size / min(w, h)
    nw, nh = int(round(w * s / 14)) * 14, int(round(h * s / 14)) * 14
    r = np.asarray(Image.fromarray(img_rgb).resize((nw, nh), Image.BICUBIC), np.float32) / 255.0
    x = ((r - MEAN) / STD).transpose(2, 0, 1)[None]
    d = _sess().run(None, {"pixel_values": x})[0][0]
    return d.astype(np.float32)


@dataclass(slots=True)
class SpatialMap:
    points: np.ndarray          # (N, 6) x y z r g b
    occupancy: np.ndarray       # (H, W) uint8
    cell_m: float
    grid_m: float
    camera_height_m: float
    depth_ms: float
    n_points: int
    note: str

    def to_dict(self, max_points: int = 8000) -> dict:
        pts = self.points
        if len(pts) > max_points:                       # subsample for the wire
            idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
            pts = pts[idx]
        return {
            "points": np.round(pts, 2).tolist(),
            "occupancy": self.occupancy.tolist(),
            "cell_m": self.cell_m, "grid_m": self.grid_m,
            "camera_height_m": self.camera_height_m,
            "depth_ms": round(self.depth_ms, 1),
            "n_points": int(self.n_points),
            "note": self.note,
        }


def build_map(img_rgb: np.ndarray) -> SpatialMap:
    import time
    t0 = time.time()
    inv = predict_depth(img_rgb)
    ms = (time.time() - t0) * 1000
    h, w = inv.shape

    # --- relative inverse depth -> approximate metric depth ------------------
    inv = np.clip(inv, 1e-3, None)
    rel = 1.0 / inv                                  # relative depth, unscaled
    # Scale anchor: the bottom 15% of rows is assumed to be floor at roughly
    # CAMERA_HEIGHT_M below the camera. For a pinhole camera looking level,
    # floor depth at pixel row v is  z = f_y * H / (v - c_y). We fit one scalar
    # so the median of the bottom band matches that geometry.
    fx = (w / 2) / np.tan(np.radians(HFOV_DEG) / 2)
    fy = fx
    cy = h / 2
    vs = np.arange(h)[:, None].repeat(w, 1).astype(np.float32)
    band = vs > h * 0.85
    expected = fy * CAMERA_HEIGHT_M / np.maximum(vs[band] - cy, 1.0)
    scale = np.median(expected) / max(np.median(rel[band]), 1e-6)
    z = rel * scale
    z = np.clip(z, 0.2, 12.0)

    # --- back-project to camera frame ------------------------------------------
    us = np.arange(w)[None, :].repeat(h, 0).astype(np.float32)
    x = (us - w / 2) / fx * z
    y = (vs - cy) / fy * z                            # +y down in image
    # colours from the resized image
    from PIL import Image
    rgb = np.asarray(Image.fromarray(img_rgb).resize((w, h)), np.uint8)
    # camera frame -> world: forward=z, right=x, up=-y ; put floor at 0
    X, Y, Z = x, z, (CAMERA_HEIGHT_M - y)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel(),
                    rgb[..., 0].ravel(), rgb[..., 1].ravel(), rgb[..., 2].ravel()], 1).astype(np.float32)
    keep = (pts[:, 1] > 0.2) & (pts[:, 1] < GRID_M) & (np.abs(pts[:, 0]) < GRID_M / 2) & (pts[:, 2] > -0.3) & (pts[:, 2] < 2.6)
    pts = pts[keep]

    # --- occupancy grid: anything between ankle and head height is an obstacle --
    n = int(GRID_M / CELL_M)
    occ = np.full((n, n), 2, np.uint8)                # unknown
    gx = ((pts[:, 0] + GRID_M / 2) / CELL_M).astype(int)
    gy = (pts[:, 1] / CELL_M).astype(int)
    ok = (gx >= 0) & (gx < n) & (gy >= 0) & (gy < n)
    gx, gy, zz = gx[ok], gy[ok], pts[ok, 2]
    floor = zz < 0.12
    obst = (zz >= 0.12) & (zz < 1.9)
    occ[gy[floor], gx[floor]] = 0
    occ[gy[obst], gx[obst]] = 1                        # obstacle wins over floor

    return SpatialMap(pts, occ, CELL_M, GRID_M, CAMERA_HEIGHT_M, ms, len(pts),
                      "Monocular depth, scale anchored on a 1.4 m camera height. Plausible, not measured.")
