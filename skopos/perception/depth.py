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


# Inference input size (short side). Measured on an M-series CPU, IMG_6978:
#   518 -> 985 ms (429 obstacle cells) | 392 -> 380 ms (348) | 308 -> 227 ms (328)
# 392 keeps most of the structure at 2.6x the speed; 518 is one env var away.
DEPTH_SIZE = int(os.environ.get("SKOPOS_DEPTH_SIZE", "392"))
DEPTH_MAX_LONG = 1400   # a 4:1 panorama at short-side 392 would be 1568 wide; cap it


def predict_depth(img_rgb: np.ndarray, size: int | None = None) -> np.ndarray:
    """img_rgb uint8 (H, W, 3) -> relative inverse depth float32 (h, w), larger = closer."""
    from PIL import Image
    h, w = img_rgb.shape[:2]
    size = size or DEPTH_SIZE
    s = size / min(w, h)
    s = min(s, DEPTH_MAX_LONG / max(w, h))            # long-side cap (panoramas)
    nw, nh = max(14, int(round(w * s / 14)) * 14), max(14, int(round(h * s / 14)) * 14)
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
    sweep_deg: float | None = None
    pano_fov_deg: float | None = None   # set only by build_pano_map

    def to_dict(self, max_points: int = 6000) -> dict:
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
            "yaw_source": self.used,
            "sweep_deg": self.sweep_deg,
            "pano_fov_deg": self.pano_fov_deg,
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

def _strip(x: np.ndarray) -> np.ndarray:
    from PIL import Image
    g = np.asarray(Image.fromarray(x).convert("L").resize((256, 192)), np.float32)[48:144]
    g -= g.mean(); g /= (g.std() + 1e-6)
    return g


def _yaw_phase(A: np.ndarray, B: np.ndarray, px_scale: float, fx: float) -> tuple[float, float]:
    """Phase correlation. Returns (yaw_deg, peak sharpness 0..1)."""
    R = np.fft.fft2(A) * np.conj(np.fft.fft2(B)); R /= (np.abs(R) + 1e-9)
    corr = np.fft.ifft2(R).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dx = peak[1] if peak[1] <= 128 else peak[1] - 256
    conf = float((corr.max() - corr.mean()) / (corr.std() + 1e-9)) / 12.0
    return float(np.degrees(np.arctan((dx * px_scale) / fx))), min(max(conf, 0.0), 1.0)


def _yaw_ncc(A: np.ndarray, B: np.ndarray, px_scale: float, fx: float) -> tuple[float, float]:
    """Brute-force normalised cross-correlation over horizontal shifts. Returns
    (yaw_deg, best NCC -1..1). Independent of the FFT estimator on purpose."""
    best, bd = -2.0, 0
    for dx in range(-120, 121):
        x, y = (A[:, dx:], B[:, :256 - dx]) if dx >= 0 else (A[:, :256 + dx], B[:, -dx:])
        if x.shape[1] < 40:
            continue
        c = float((x * y).mean())
        if c > best:
            best, bd = c, dx
    return float(np.degrees(np.arctan((bd * px_scale) / fx))), best


NCC_MIN = 0.40          # below this the "match" is texture noise (measured 0.07-0.46 on plain walls)
AGREE_DEG = 8.0         # the two estimators must land within this of each other


def _yaw_between(a: np.ndarray, b: np.ndarray, fx: float) -> dict:
    """How far the camera turned between two photos, and whether that number
    deserves to be believed. Two independent estimators must agree in sign and
    within AGREE_DEG, and the NCC must clear NCC_MIN; otherwise the pair is
    reported as unmeasurable and the caller falls back to even spacing.

    Measured on the venue fixtures (plain walls, small tilt changes): the two
    estimators disagreed on 6 of 8 pairs and the two agreements had NCC 0.19
    and 0.07 — noise agreeing with noise. A cleverer matcher here would only
    produce confident garbage; a stricter gate is the honest fix."""
    A, B = _strip(a), _strip(b)
    px_scale = a.shape[1] / 256.0
    y1, c1 = _yaw_phase(A, B, px_scale, fx)
    y2, c2 = _yaw_ncc(A, B, px_scale, fx)
    ok = (np.sign(y1) == np.sign(y2)) and abs(y1 - y2) <= AGREE_DEG and c2 >= NCC_MIN and abs(y2) > 3.0
    return {"yaw": y2 if ok else None, "phase": round(y1, 1), "ncc": round(y2, 1),
            "phase_conf": round(c1, 2), "ncc_score": round(c2, 2), "accepted": bool(ok)}


def fuse_sweep(images: list[np.ndarray], hfov_deg: float = HFOV_DEG,
               total_sweep_deg: float | None = None) -> SpatialMap:
    """Fuse a rotating sweep of photos into one map, camera at the centre.

    Yaw between consecutive photos is used only when two independent estimators
    agree (see _yaw_between) AND the sign matches the sequence's majority — a
    real turn does not reverse. Every other pair is placed by even spacing:
    across `total_sweep_deg` if the user stated the arc they turned, else
    assuming a full 360 degree turn. Which rule placed each pair is recorded.
    """
    if not images:
        raise ValueError("no images")
    long_side = max(images[0].shape[:2])
    fx = (long_side / 2) / np.tan(np.radians(hfov_deg) / 2)
    n = len(images)
    if total_sweep_deg is not None:
        total_sweep_deg = max(30.0, min(360.0, float(total_sweep_deg)))
        even = total_sweep_deg / max(n - 1, 1)
    else:
        even = 360.0 / n
    ests = [_yaw_between(images[i - 1], images[i], fx) for i in range(1, n)]
    signs = [np.sign(e["yaw"]) for e in ests if e["accepted"]]
    majority = float(np.sign(sum(signs))) if signs else 0.0
    yaws, used, confs = [0.0], [], []
    for e in ests:
        ok = e["accepted"] and (majority == 0.0 or np.sign(e["yaw"]) == majority)
        step = abs(e["yaw"]) if ok else even
        yaws.append(yaws[-1] + step)
        used.append("overlap" if ok else ("even:stated" if total_sweep_deg is not None else "even:360"))
        confs.append(e["ncc_score"])
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
    arc = (f"across a stated {total_sweep_deg:.0f} deg arc" if total_sweep_deg is not None
           else "assuming a full 360 deg turn")
    note = (f"{n} photos fused as a rotating sweep from one spot (assumed stationary), {arc}. "
            f"Yaw measured from overlap for {n_over}/{len(used)} pairs; the rest placed by even spacing. "
            "Monocular depth, 1.4 m camera height. Plausible, not measured.")
    m = SpatialMap(pts, _occupancy(pts, centred=True), CELL_M, GRID_M, CAMERA_HEIGHT_M,
                   total_ms, len(pts), note)
    m.yaws = [round(v, 1) for v in yaws]; m.confs = confs; m.used = used; m.centred = True
    m.sweep_deg = total_sweep_deg
    return m


# ---------------------------------------------------------------------------
# One panorama strip (iPhone "Pano" mode) — treated as a cylindrical projection
# ---------------------------------------------------------------------------

def build_pano_map(img_rgb: np.ndarray, pano_fov_deg: float = 180.0) -> SpatialMap:
    """One wide panorama -> centred map, camera at the grid centre.

    Geometry (cylindrical, camera level, FOV assumed — iPhone does not record it):
      column u -> bearing   theta = (u/W - 0.5) * pano_fov
      row    v -> elevation phi   = (0.5 - v/H) * vfov,  vfov = pano_fov * H/W
      d = horizontal range along that bearing (the cylinder radius), so
      x = d sin(theta), y = d cos(theta), z = CAMERA_HEIGHT_M + d tan(phi).
    Scale anchor is the same idea as build_map — the bottom 15% of rows is
    floor — adapted to this geometry: a floor pixel at elevation phi < 0 sits at
    d = CAMERA_HEIGHT_M / tan(-phi); one scalar fits the band's median to that.
    """
    import time
    fov = float(pano_fov_deg)
    t0 = time.time()
    inv = predict_depth(img_rgb)
    ms = (time.time() - t0) * 1000
    h, w = inv.shape
    vfov = fov * h / w
    # Explicit ufunc calls, not `a * b`, wherever a local array is an operand
    # and is used again afterwards: on Python 3.14 (LOAD_FAST_BORROW) NumPy
    # 2.2's temporary elision sees a refcount of 1 on a plain local and writes
    # the product into its buffer, so `X = d * s` silently made X *be* d.
    us = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
    vs = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
    theta = np.radians((np.divide(us, w) - 0.5) * fov)
    phi = np.radians((0.5 - np.divide(vs, h)) * vfov)

    rel = np.divide(1.0, np.clip(inv, 1e-3, None))
    band = vs > h * 0.85                              # floor band; phi < 0 there
    expected = CAMERA_HEIGHT_M / np.tan(-phi[band])
    scale = np.median(expected) / max(np.median(rel[band]), 1e-6)
    d = np.clip(np.multiply(rel, scale), 0.2, 12.0)

    from PIL import Image
    rgb = np.asarray(Image.fromarray(img_rgb).resize((w, h)), np.uint8)
    X = np.multiply(d, np.sin(theta))
    Y = np.multiply(d, np.cos(theta))
    Z = CAMERA_HEIGHT_M + np.multiply(d, np.tan(phi))
    assert X is not d and Y is not d, "operand buffer reused"
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel(),
                    rgb[..., 0].ravel(), rgb[..., 1].ravel(), rgb[..., 2].ravel()], 1).astype(np.float32)
    keep = ((np.abs(pts[:, 0]) < GRID_M / 2) & (np.abs(pts[:, 1]) < GRID_M / 2)
            & (pts[:, 2] > -0.3) & (pts[:, 2] < 2.6))
    pts = pts[keep]
    note = (f"Panorama (cylindrical, assumed {fov:g} deg FOV), monocular depth, "
            "1.4 m camera height. Plausible, not measured.")
    m = SpatialMap(pts, _occupancy(pts, centred=True), CELL_M, GRID_M, CAMERA_HEIGHT_M,
                   ms, len(pts), note)
    m.centred = True; m.pano_fov_deg = fov
    return m
