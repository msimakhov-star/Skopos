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
    yaws: list | None = None
    confs: list | None = None
    used: list | None = None
    centred: bool = False

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
            "centred": bool(self.centred),
            "yaws": self.yaws,
        }


def _points(img_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """One image -> (N,6) points in the camera-forward frame, plus depth ms."""
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
    fx = (max(w, h) / 2) / np.tan(np.radians(HFOV_DEG) / 2)   # HFOV is quoted on the long side
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
    return pts[keep], ms


def _occupancy(pts: np.ndarray, centred: bool = False) -> np.ndarray:
    """Points -> grid. centred=True puts the camera at the grid centre (for a
    fused 360 map); False keeps it at the bottom-centre (single forward view)."""
    n = int(GRID_M / CELL_M)
    occ = np.full((n, n), 2, np.uint8)                # unknown
    gx = ((pts[:, 0] + GRID_M / 2) / CELL_M).astype(int)
    gy = (((pts[:, 1] + GRID_M / 2) if centred else pts[:, 1]) / CELL_M).astype(int)
    ok = (gx >= 0) & (gx < n) & (gy >= 0) & (gy < n)
    gx, gy, zz = gx[ok], gy[ok], pts[ok, 2]
    floor = zz < 0.12
    obst = (zz >= 0.12) & (zz < 1.9)
    occ[gy[floor], gx[floor]] = 0
    occ[gy[obst], gx[obst]] = 1                        # obstacle wins over floor
    return occ


def build_map(img_rgb: np.ndarray) -> SpatialMap:
    pts, ms = _points(img_rgb)
    return SpatialMap(pts, _occupancy(pts), CELL_M, GRID_M, CAMERA_HEIGHT_M, ms, len(pts),
                      "Monocular depth, scale anchored on a 1.4 m camera height. Plausible, not measured.")


# ---------------------------------------------------------------------------
# Multi-photo fusion — a rotating sweep from ONE spot
# ---------------------------------------------------------------------------

def _yaw_between(a: np.ndarray, b: np.ndarray, fx: float) -> tuple[float, float]:
    """How far the camera turned between photo a and photo b, in degrees, from
    the horizontal shift that best aligns them (phase correlation on a
    downscaled grey strip). Returns (yaw_deg, confidence 0..1). Pure rotation
    about the vertical axis is assumed; a step sideways will read as a turn."""
    from PIL import Image
    def strip(x):
        g = np.asarray(Image.fromarray(x).convert("L").resize((256, 192)), np.float32)
        g = g[48:144]                                   # middle band: least floor/ceiling
        g -= g.mean(); g /= (g.std() + 1e-6)
        return g
    A, B = strip(a), strip(b)
    # 1-D correlation along x on the column-averaged profile + 2-D check
    FA, FB = np.fft.fft2(A), np.fft.fft2(B)
    R = FA * np.conj(FB); R /= (np.abs(R) + 1e-9)
    corr = np.fft.ifft2(R).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dx = peak[1] if peak[1] <= 128 else peak[1] - 256
    conf = float((corr.max() - corr.mean()) / (corr.std() + 1e-9)) / 12.0   # ~1 at a clean peak
    # pixels at 256-wide -> pixels at the image's own width -> degrees
    scale = a.shape[1] / 256.0
    yaw = float(np.degrees(np.arctan((dx * scale) / fx)))
    return yaw, min(max(conf, 0.0), 1.0)


def fuse_sweep(images: list[np.ndarray], hfov_deg: float = HFOV_DEG) -> SpatialMap:
    """Fuse a rotating sweep of photos into one 360-ish map, camera at centre.

    Yaw between consecutive photos is estimated from image overlap; when the
    estimate is unconfident (little or no overlap) the photo is placed by
    even spacing, 360/N. Both cases are recorded in the note so a bad seam is
    reported, never smoothed over.
    """
    if not images:
        raise ValueError("no images")
    long_side = max(images[0].shape[:2])
    fx = (long_side / 2) / np.tan(np.radians(hfov_deg) / 2)
    even = 360.0 / len(images)
    yaws, confs, used = [0.0], [], []
    for i in range(1, len(images)):
        y, c = _yaw_between(images[i - 1], images[i], fx)
        # a turn between overlapping shots must be positive and smaller than the FOV
        ok = c > 0.35 and 3.0 < abs(y) < hfov_deg
        step = abs(y) if ok else even
        yaws.append(yaws[-1] + step); confs.append(round(c, 2)); used.append("overlap" if ok else "even")
    all_pts, total_ms = [], 0.0
    for img, yaw in zip(images, yaws):
        pts, ms = _points(img); total_ms += ms
        t = np.radians(yaw)
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = x * np.cos(t) + y * np.sin(t)      # rotate about the vertical axis
        pts[:, 1] = -x * np.sin(t) + y * np.cos(t)
        all_pts.append(pts)
    pts = np.concatenate(all_pts)
    inside = (np.abs(pts[:, 0]) < GRID_M / 2) & (np.abs(pts[:, 1]) < GRID_M / 2)
    pts = pts[inside]
    n_over = used.count("overlap")
    note = (f"{len(images)} photos fused as a rotating sweep from one spot (assumed stationary). "
            f"Yaw from overlap for {n_over}/{len(used)} pairs, even spacing for the rest. "
            "Monocular depth, 1.4 m camera height. Plausible, not measured.")
    m = SpatialMap(pts, _occupancy(pts, centred=True), CELL_M, GRID_M, CAMERA_HEIGHT_M,
                   total_ms, len(pts), note)
    m.yaws = [round(v, 1) for v in yaws]; m.confs = confs; m.used = used; m.centred = True
    return m
