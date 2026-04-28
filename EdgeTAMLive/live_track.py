#!/usr/bin/env python3
"""
Real-time surgical tool segmentation and 6-DoF pose with EdgeTAM and SAM3D.

Pipeline
--------
1. Frames are captured (camera index, video path, or synthetic still → MP4),
   resized to ``TARGET_SIZE``, and normalized for the EdgeTAM video predictor.
2. The user places seed clicks (monotonic IDs; **ID1 is always scissors**).
   EdgeTAM propagates instance masks through the stream.
3. SAM3D (fal.ai) turns each seed mask into a mesh GLB. Vertices are rescaled to
   metres using measured length for ID1 (11.75 cm) or ``CLASS_EXTENTS_M`` for
   other classes so OpenCV PnP ``tvec`` is in real-world metres.
4. :class:`MeshPoseEstimator` aligns a PCA-derived model contour to each mask
   contour (phase/reversal search, axis sign bootstrap, optional Kalman filter).

Outputs
-------
Live window with mask overlay, pose axes, and COM trails; optional MP4;
``posesN.csv`` in the current working directory (Euler ZYX degrees + translation m).

Environment
-----------
``FAL_KEY`` must be set for SAM3D mesh bootstrap. Optional: ``pyorbbecsdk`` for
factory colour intrinsics on Orbbec devices.

Usage::

    .venv/bin/python live_track.py                        # camera 0
    .venv/bin/python live_track.py --camera 1
    .venv/bin/python live_track.py --no-half
    .venv/bin/python live_track.py --output out.mp4

Keybindings (seed window)
-------------------------
Enter / Return    Confirm seeds and start tracking
Backspace         Undo last point
c                 Clear all points
q / ESC           Quit without starting
"""

import argparse
import csv
import itertools
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import json
from urllib.request import urlretrieve
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import trimesh  # type: ignore
except Exception:
    trimesh = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SIZE = (640, 360)
EDGETAM_REPO = Path(__file__).parent / "EdgeTAM"
CHECKPOINT = EDGETAM_REPO / "checkpoints" / "edgetam.pt"
MODEL_CFG = "configs/edgetam.yaml"

# ImageNet normalization (matches EdgeTAM / SAM2 training).
_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_IMG_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

# ---------------------------------------------------------------------------
# Calibration: ID1 is always scissors, measured at 11.75 cm.
# All other objects are normalised using CLASS_EXTENTS as a best-guess
# until per-object measured sizes are added here.
# ---------------------------------------------------------------------------
OBJECT_ID_LENGTH_M: dict[int, float] = {
    1: 0.1175,   # scissors — measured ground truth (11.75 cm)
}

CLASS_EXTENTS_M: dict[str, float] = {
    "scissors": 0.1175,
    "scalpel":  0.160,
    "tweezers": 0.130,
    "bottle":   0.220,
    "bag":      0.280,
    "tool":     0.150,
}

# ---------------------------------------------------------------------------
# EdgeTAM predictor
# ---------------------------------------------------------------------------

def _load_predictor(device: str):
    """Insert local ``EdgeTAM`` repo on ``sys.path`` and build the video predictor."""
    repo = str(EDGETAM_REPO.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore
    return build_sam2_video_predictor(MODEL_CFG, str(CHECKPOINT), device=device)


def _autocast_config(device: str, use_half: bool) -> tuple[str, torch.dtype, bool]:
    """Return (device_type, dtype, enabled) for ``torch.autocast``."""
    device_type = device.split(":")[0]
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float16
    enabled = use_half and device_type != "cpu"
    return device_type, dtype, enabled


def choose_device(arg: str) -> str:
    """Resolve ``auto`` to cuda, mps, or cpu in that order of preference."""
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray, rotate_180: bool) -> np.ndarray:
    """Optional 180° rotation (Orbbec colour) then resize to ``TARGET_SIZE``."""
    if rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def detect_orbbec_camera(camera_id: int) -> bool:
    """Return True if ffmpeg's AVFoundation listing names this index as Orbbec."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=3.0, check=False,
        )
    except Exception:
        return False
    listing = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for line in listing.splitlines():
        m = re.search(r"\[(\d+)\]\s+(.+)$", line.strip())
        if not m:
            continue
        if int(m.group(1)) == camera_id:
            return "orbbec" in m.group(2).lower()
    return False


def _build_repeated_video_from_image(
    image_path: Path, out_path: Path, num_frames: int = 100, fps: float = 30.0,
) -> Path:
    """Encode ``num_frames`` copies of one resized frame as a synthetic MP4."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    frame = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps), (frame.shape[1], frame.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for: {out_path}")
    try:
        for _ in range(max(1, int(num_frames))):
            writer.write(frame)
    finally:
        writer.release()
    return out_path

# ---------------------------------------------------------------------------
# Point-picking UI
# ---------------------------------------------------------------------------

def pick_points_live(
    provider: "LiveFrameProvider", stop_flag: threading.Event
) -> tuple[list[tuple[int, float, float]], np.ndarray | None]:
    """
    Interactive OpenCV window to collect positive seed points.

    Each click appends ``(object_id, x, y)`` with IDs ``1..N`` in order.
    The first click freezes the background frame used as the EdgeTAM seed.

    Returns:
        Empty list and None if cancelled; else ``(points, seed_frame_bgr)``.
    """
    win = "Select EdgeTAM points  (ID1 = scissors)"
    points: list[tuple[int, float, float]] = []
    frozen_frame: np.ndarray | None = None

    def draw() -> np.ndarray:
        base = frozen_frame if frozen_frame is not None else provider.get_raw(-1)
        vis = base.copy() if base is not None else np.zeros(
            (TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8)
        for obj_id, px_f, py_f in points:
            px, py = int(px_f), int(py_f)
            label = f"ID{obj_id}" + (" (scissors 11.75cm)" if obj_id == 1 else "")
            cv2.circle(vis, (px, py), 6, (0, 255, 255), -1)
            cv2.circle(vis, (px, py), 9, (255, 255, 255), 1)
            cv2.putText(vis, label, (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(
            vis,
            "Click: add (ID1=scissors) | Backspace: undo | c: clear | Enter: start",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2,
        )
        return vis

    def on_mouse(event, x, y, flags, param):
        nonlocal frozen_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            if frozen_frame is None:
                latest = provider.get_raw(-1)
                if latest is not None:
                    frozen_frame = latest.copy()
            points.append((len(points) + 1, float(x), float(y)))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    cancelled = False
    while True:
        if stop_flag.is_set():
            cancelled = True
            break
        cv2.imshow(win, draw())
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and points:
            break
        elif k in (8, 127) and points:
            points.pop()
        elif k == ord("c"):
            points.clear()
            frozen_frame = None
        elif k in (ord("q"), 27):
            cancelled = True
            break
    cv2.destroyWindow(win)
    if cancelled:
        return [], None
    seed_frame = frozen_frame if frozen_frame is not None else provider.get_raw(-1)
    return points, (None if seed_frame is None else seed_frame.copy())

# ---------------------------------------------------------------------------
# Visualization (mask tint, stable colours per object id)
# ---------------------------------------------------------------------------

def point_color(obj_id: int) -> tuple[int, int, int]:
    """Deterministic saturated BGR colour for ``obj_id`` (for trails / overlays)."""
    hue = (obj_id * 47 + 20) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _mask_to_2d_bool(m: np.ndarray, fh: int, fw: int) -> np.ndarray:
    """Collapse arbitrary mask rank to ``(fh, fw)`` boolean; nearest resize if needed."""
    x = np.squeeze(np.asarray(m, dtype=np.float32))
    while x.ndim > 2:
        x = x[0]
    if x.ndim != 2:
        return np.zeros((fh, fw), dtype=bool)
    if x.shape != (fh, fw):
        x = cv2.resize(x, (fw, fh), interpolation=cv2.INTER_NEAREST)
    return x > 0.0


def overlay_masks(frame, obj_ids, masks, alpha=0.45):
    """Alpha-blend each instance mask onto ``frame`` using :func:`point_color`."""
    vis = frame.copy().astype(np.float32)
    fh, fw = frame.shape[:2]
    masks_np = masks.detach().cpu().numpy()
    for i in range(min(len(obj_ids), masks_np.shape[0])):
        binm = _mask_to_2d_bool(masks_np[i], fh, fw)
        if not np.any(binm):
            continue
        c = np.array(point_color(int(obj_ids[i])), dtype=np.float32)
        vis[binm] = vis[binm] * (1 - alpha) + c * alpha
    return vis.astype(np.uint8)

# ---------------------------------------------------------------------------
# Camera intrinsics (file, Orbbec SDK, or generic FOV fallback)
# ---------------------------------------------------------------------------

def _estimate_intrinsics_from_cap(
    cap: cv2.VideoCapture, target_w: int, target_h: int
) -> np.ndarray:
    """Approximate K from cap native resolution and fixed 79°×62° FOV (last resort)."""
    native_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    native_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if native_w <= 0 or native_h <= 0:
        native_w, native_h = float(target_w), float(target_h)
    hfov_rad = np.radians(79.0)
    vfov_rad = np.radians(62.0)
    fx = (native_w / (2.0 * np.tan(hfov_rad / 2.0))) * (target_w / native_w)
    fy = (native_h / (2.0 * np.tan(vfov_rad / 2.0))) * (target_h / native_h)
    return np.array(
        [[fx, 0.0, target_w / 2.0], [0.0, fy, target_h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rescale_intrinsics(K, src_w, src_h, dst_w, dst_h):
    """Scale fx, fy, cx, cy when the calibration resolution differs from the target."""
    sx = float(dst_w) / max(float(src_w), 1e-9)
    sy = float(dst_h) / max(float(src_h), 1e-9)
    K2 = np.asarray(K, dtype=np.float64).copy()
    K2[0, 0] *= sx; K2[1, 1] *= sy; K2[0, 2] *= sx; K2[1, 2] *= sy
    return K2


def _load_intrinsics_from_file(path, target_w, target_h):
    """Load 3×3 K (and optional distortion) from ``.npz``; rescale if width/height present."""
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"Failed to load intrinsics file '{path}': {e}"); return None
    K = None
    for kname in ("K", "camera_matrix", "intrinsics"):
        if kname in data:
            cand = np.asarray(data[kname], dtype=np.float64)
            if cand.shape == (3, 3):
                K = cand; break
    if K is None:
        print(f"Intrinsics file '{path}' missing 3x3 matrix."); return None
    dist = np.zeros((4, 1), dtype=np.float64)
    for dname in ("dist", "dist_coeffs", "distortion"):
        if dname in data:
            dist = np.asarray(data[dname], dtype=np.float64).reshape(-1, 1); break
    sw = sh = None
    if "width" in data and "height" in data:
        sw, sh = float(np.asarray(data["width"]).flat[0]), float(np.asarray(data["height"]).flat[0])
    elif "image_width" in data and "image_height" in data:
        sw, sh = float(np.asarray(data["image_width"]).flat[0]), float(np.asarray(data["image_height"]).flat[0])
    if sw and sh:
        K = _rescale_intrinsics(K, sw, sh, target_w, target_h)
    return K.astype(np.float64), dist.astype(np.float64)


def _try_read_orbbec_intrinsics(target_w, target_h):
    """Read colour intrinsics from pyorbbecsdk default profile, scaled to target size."""
    try:
        from pyorbbecsdk import Config, OBSensorType, Pipeline  # type: ignore
    except Exception:
        return None
    pipeline = None
    try:
        pipeline = Pipeline()
        pl = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if pl is None: return None
        cp = pl.get_default_video_stream_profile()
        if cp is None: return None
        nw, nh = float(cp.get_width()), float(cp.get_height())
        intr = None
        if hasattr(cp, "get_intrinsic"): intr = cp.get_intrinsic()
        elif hasattr(cp, "get_camera_intrinsic"): intr = cp.get_camera_intrinsic()
        if intr is None: return None
        fx = float(getattr(intr, "fx", 0.0)); fy = float(getattr(intr, "fy", 0.0))
        if fx <= 0 or fy <= 0: return None
        cx = float(getattr(intr, "cx", nw / 2.0)); cy = float(getattr(intr, "cy", nh / 2.0))
        K = _rescale_intrinsics(
            np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64), nw, nh, target_w, target_h)
        dist = np.zeros((4, 1), dtype=np.float64)
        vals = [float(getattr(intr, n)) for n in ("k1","k2","p1","p2","k3","k4","k5","k6") if hasattr(intr, n)]
        if vals: dist = np.array(vals, np.float64).reshape(-1,1)
        return K, dist
    except Exception:
        return None
    finally:
        if pipeline:
            try:
                pipeline.stop()
            except Exception:
                pass


def _resolve_pose_intrinsics(cap, target_w, target_h, intrinsics_file, try_orbbec):
    """Prefer ``.npz`` file, then Orbbec SDK, else FOV-based estimate; returns (K, dist, tag)."""
    if intrinsics_file:
        loaded = _load_intrinsics_from_file(intrinsics_file, target_w, target_h)
        if loaded is not None:
            return loaded[0], loaded[1], f"file:{intrinsics_file}"
        print("Falling back — intrinsics file could not be used.")
    if try_orbbec:
        sdk = _try_read_orbbec_intrinsics(target_w, target_h)
        if sdk is not None:
            return sdk[0], sdk[1], "orbbec_sdk"
    K = _estimate_intrinsics_from_cap(cap, target_w, target_h)
    return K, np.zeros((4, 1), dtype=np.float64), "fov_estimate"

# ---------------------------------------------------------------------------
# Mesh scale: SAM3D meshes are unitless; rescale to metres before PnP
# ---------------------------------------------------------------------------

def _normalize_mesh_to_known_length(
    verts: np.ndarray,
    obj_id: int,
    object_class: str,
) -> np.ndarray:
    """
    Rescale mesh vertices so the longest axis equals the known physical length.

    Priority:
      1. OBJECT_ID_LENGTH_M[obj_id]  — measured ground truth (ID1 = 11.75 cm)
      2. CLASS_EXTENTS_M[object_class]
      3. CLASS_EXTENTS_M["tool"]

    SAM3D produces meshes in arbitrary normalised units (~0–1).  Without this
    rescaling the PnP tvec Z has no physical meaning and reprojection fails.
    """
    if obj_id in OBJECT_ID_LENGTH_M:
        target_m = OBJECT_ID_LENGTH_M[obj_id]
        src = f"ID{obj_id} measured ({target_m*100:.2f} cm)"
    else:
        target_m = CLASS_EXTENTS_M.get(object_class, CLASS_EXTENTS_M["tool"])
        src = f"class={object_class} ({target_m*100:.1f} cm)"

    longest = float((verts.max(0) - verts.min(0)).max())
    if longest < 1e-9:
        return verts
    scale = target_m / longest
    print(f"  [mesh scale] {src}  raw longest={longest:.4f}  scale×{scale:.5f}")
    return verts * scale

# ---------------------------------------------------------------------------
# Dense contour ↔ model PnP, temporal smoothing, calibration prior
# ---------------------------------------------------------------------------

def _reg_sign_from_state(state: dict) -> np.ndarray:
    """Per-axis flips (+1/−1) chosen at bootstrap so mesh axes match the mask."""
    s = state.get("reg_sign")
    if s is None:
        return np.ones(3, dtype=np.float64)
    return np.asarray(s, dtype=np.float64).reshape(3)


# Samples per edge when densifying the model polygon for PnP correspondence.
PNP_PER_EDGE = 8
# Default 1D Kalman variances for rvec (3) + tvec (3) element-wise filtering.
KALMAN_PROCESS_VAR = 2e-4
KALMAN_MEAS_VAR = 4e-3
# Extra cost terms when choosing contour phase/reversal vs previous frame.
PNP_ROT_SMOOTH_W = 0.03
PNP_TRANS_SMOOTH_W = 8.0
PNP_SHIFT_PENALTY = 0.75


def _sample_poly_perimeter(poly: np.ndarray, n: int) -> np.ndarray:
    """Uniformly sample ``n`` points along closed polygon edge length."""
    D = int(poly.shape[1])
    if poly.shape[0] < 2:
        return np.repeat(poly[:1].astype(np.float64), n, axis=0)
    lens = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    L = float(lens.sum())
    out = np.zeros((n, D), dtype=np.float64)
    if L < 1e-12:
        return np.repeat(poly[:1].astype(np.float64), n, axis=0)
    cum = np.zeros(len(poly) + 1, dtype=np.float64)
    for i in range(len(poly)):
        cum[i + 1] = cum[i] + lens[i]
    for i in range(n):
        t = ((i + 0.5) / n) * L
        for k in range(len(poly)):
            if cum[k + 1] >= t - 1e-15:
                u = (t - cum[k]) / (lens[k] + 1e-12)
                out[i] = poly[k] * (1.0 - u) + poly[(k + 1) % len(poly)] * u
                break
    return out


def _sample_mask_contour(mask_u8: np.ndarray, n: int) -> np.ndarray | None:
    """Largest external contour of binary mask, uniformly resampled to ``n`` points."""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cnt is None or len(cnt) < 4:
        return None
    poly = cnt.reshape(-1, 2).astype(np.float64)
    d = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    L = float(d.sum())
    if L < 1e-9:
        return None
    out = np.zeros((n, 2), dtype=np.float64)
    cum = np.zeros(len(poly) + 1, dtype=np.float64)
    for i in range(len(poly)):
        cum[i + 1] = cum[i] + d[i]
    for i in range(n):
        t = ((i + 0.5) / n) * L
        k = int(np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(poly) - 1))
        u = (t - cum[k]) / (d[k] + 1e-12)
        out[i] = poly[k] * (1.0 - u) + poly[(k + 1) % len(poly)] * u
    return out


def _rotation_delta_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    """Geodesic angle in degrees between two Rodrigues rotation vectors."""
    Ra, _ = cv2.Rodrigues(rvec_a.astype(np.float64))
    Rb, _ = cv2.Rodrigues(rvec_b.astype(np.float64))
    R = Ra @ Rb.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))))


class KalmanScalar:
    """Scalar Kalman filter (constant velocity implicit in random-walk process noise)."""

    def __init__(self, process_var: float, meas_var: float):
        self.q = max(float(process_var), 1e-12)
        self.r = max(float(meas_var), 1e-12)
        self.x: float | None = None
        self.p: float = 1.0

    def filter(self, z: float) -> float:
        z = float(z)
        if self.x is None:
            self.x = z
            self.p = 1.0
            return z
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1.0 - k) * self.p
        return self.x


def _apply_kalman_pose_filter(state, rv, tv, process_var, meas_var):
    """Six independent scalar filters on ``rvec`` and ``tvec`` components; stateful."""
    filters = state.get("kalman_filters")
    if filters is None or len(filters) != 6:
        filters = [KalmanScalar(process_var, meas_var) for _ in range(6)]
        state["kalman_filters"] = filters
    vec = np.concatenate([rv.reshape(3), tv.reshape(3)]).astype(np.float64)
    out = np.array([filters[i].filter(float(vec[i])) for i in range(6)])
    return out[:3].reshape(3, 1), out[3:].reshape(3, 1)

# ---------------------------------------------------------------------------
# MeshPoseEstimator: metric mesh + dense PnP + optional scissors prior
# ---------------------------------------------------------------------------

class MeshPoseEstimator:
    """
    Build a planar PnP model from a mesh (PCA, convex hull in XY), then each frame:

    - Erode the mask, sample the largest contour, and match it to the model polygon
      with phase/reversal search and optional previous-pose regularization.
    - On first solve, search all eight axis sign flips and keep the best score
      (reprojection + scissors upright prior for ``obj_id == 1``).
    - Apply per-component Kalman smoothing to the winning ``rvec``/``tvec``.
    """

    def __init__(self, mesh, obj_id: int = 0, object_class: str = "tool"):
        self.obj_id = obj_id
        self.object_class = object_class

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)

        # PCA-align: longest axis → axis 0  (original logic, single eigh call)
        cen = verts.mean(0)
        v   = verts - cen
        cov = (v.T @ v) / len(v)
        eigvals, evecs = np.linalg.eigh(cov)
        order   = np.argsort(eigvals)[::-1]        # descending
        aligned = v @ evecs[:, order]

        # ── Scale to known physical length ────────────────────────────
        aligned = _normalize_mesh_to_known_length(aligned, obj_id, object_class)

        self.mesh_vertices = aligned.astype(np.float64)
        self.mesh_faces    = faces

        hP = (aligned[:, 0].max() - aligned[:, 0].min()) / 2
        hS = (aligned[:, 1].max() - aligned[:, 1].min()) / 2
        hT = (aligned[:, 2].max() - aligned[:, 2].min()) / 2
        self.hP, self.hS, self.hT = hP, hS, hT

        # PnP model points: use a PCA-plane convex hull so non-rectangular
        # objects (e.g. scissors) align better than a fixed quad.
        plane_xy = aligned[:, :2].astype(np.float32)
        hull = cv2.convexHull(plane_xy.reshape(-1, 1, 2)).reshape(-1, 2)
        if hull.shape[0] >= 3:
            peri = float(cv2.arcLength(hull.reshape(-1, 1, 2), True))
            eps = 0.01 * peri
            approx = cv2.approxPolyDP(hull.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
            poly_xy = approx if approx.shape[0] >= 3 else hull
            self.model_pts = np.column_stack(
                [poly_xy[:, 0], poly_xy[:, 1], np.zeros(poly_xy.shape[0], dtype=np.float32)]
            ).astype(np.float64)
        else:
            self.model_pts = np.array(
                [[-hP, -hS, 0.0], [hP, -hS, 0.0], [hP, hS, 0.0], [-hP, hS, 0.0]],
                dtype=np.float64,
            )

        # Axis display points
        ax = hP * 1.0
        ay = hS * 1.5
        az = max(hT * 5.0, hP * 0.35)
        self.axis_pts = np.array(
            [[0.0,0.0,0.0],[ax,0.0,0.0],[0.0,ay,0.0],[0.0,0.0,az]],
            dtype=np.float64,
        )
        # Estimate semantic endpoints on principal axis for calibration prior:
        # handle/rings end has larger cross-sectional radius than tip end.
        x = aligned[:, 0]
        yz_r = np.linalg.norm(aligned[:, 1:3], axis=1)
        x_min, x_max = float(x.min()), float(x.max())
        band = max(1e-6, 0.20 * (x_max - x_min))
        low_mask = x <= (x_min + band)
        high_mask = x >= (x_max - band)
        r_low = float(np.mean(yz_r[low_mask])) if np.any(low_mask) else 0.0
        r_high = float(np.mean(yz_r[high_mask])) if np.any(high_mask) else 0.0
        if r_low >= r_high:
            self.handle_pt = np.array([x_min, 0.0, 0.0], dtype=np.float64)
            self.tip_pt = np.array([x_max, 0.0, 0.0], dtype=np.float64)
        else:
            self.handle_pt = np.array([x_max, 0.0, 0.0], dtype=np.float64)
            self.tip_pt = np.array([x_min, 0.0, 0.0], dtype=np.float64)
        self._pnp_n = max(4 * PNP_PER_EDGE, int(self.model_pts.shape[0]) * max(2, PNP_PER_EDGE // 2))

    # ------------------------------------------------------------------
    def _calibration_orientation_penalty(self, rvec, tvec, K, dist, sign=None) -> float:
        """
        Prior for calibration object (ID1 = scissors):
        - scissors should be upright in the image
        - tip should point upward
        - handle/rings should stay below tip
        """
        if int(self.obj_id) != 1:
            return 0.0
        s = np.ones(3, dtype=np.float64) if sign is None else np.asarray(sign, dtype=np.float64).reshape(3)
        calib_pts = np.vstack([self.handle_pt, self.tip_pt]).astype(np.float64) * s.reshape(1, 3)
        proj, _ = cv2.projectPoints(calib_pts, rvec, tvec, K, dist)
        pts = proj.reshape(-1, 2)
        handle = pts[0]
        tip = pts[1]
        v = tip - handle
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return 1e6
        ux, uy = float(v[0] / n), float(v[1] / n)
        # Prefer vertical orientation (small |ux|) and tip-up (negative uy),
        # and strongly reject flipped pose where tip is below handle.
        vertical_pen = abs(ux)
        tip_up_pen = max(0.0, uy)
        hard_flip_pen = 100.0 if float(tip[1]) >= float(handle[1]) else 0.0
        return hard_flip_pen + 4.0 * vertical_pen + 8.0 * tip_up_pen

    # ------------------------------------------------------------------
    def _pnp_best_contour(
        self, model_corners4, mask_bool, K, dist,
        prev_rvec=None, prev_tvec=None, prev_phase=None, prev_reverse=None,
    ):
        """
        Match uniformly sampled model boundary to mask contour; search phase/reversal.

        Returns ``(rvec, tvec, mean_reproj_err, phase, reverse)`` or Nones if PnP fails.
        """
        n = int(self._pnp_n)
        img_n = _sample_mask_contour(mask_bool.astype(np.uint8) * 255, n)
        if img_n is None:
            return None, None, np.inf, None, False
        model_n0 = _sample_poly_perimeter(model_corners4.astype(np.float64), n)

        step = max(1, n // 8)
        if prev_phase is not None and prev_reverse is not None:
            phases = [int(prev_phase) % n,
                      (int(prev_phase) - step) % n,
                      (int(prev_phase) + step) % n,
                      (int(prev_phase) - 2 * step) % n,
                      (int(prev_phase) + 2 * step) % n]
            candidates = [(p, bool(prev_reverse)) for p in phases]
            candidates += [(int(prev_phase) % n, not bool(prev_reverse))]
        else:
            candidates = [(int(phase), rev)
                          for rev in (False, True)
                          for phase in range(0, n, step)]

        best_rv, best_tv, best_err = None, None, np.inf
        best_phase, best_reverse = None, False
        best_score = np.inf

        for phase, rev in candidates:
            model_seq = model_n0[::-1] if rev else model_n0
            model_n   = np.roll(model_seq, int(phase), axis=0)

            if prev_rvec is not None and prev_tvec is not None:
                ok, rv, tv = cv2.solvePnP(
                    model_n, img_n, K, dist,
                    prev_rvec.astype(np.float64), prev_tvec.astype(np.float64),
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
                )
            else:
                ok, rv, tv = cv2.solvePnP(
                    model_n, img_n, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)

            if not ok:
                ok2, rv2, tv2 = cv2.solvePnP(model_n, img_n, K, dist, flags=cv2.SOLVEPNP_EPNP)
                if not ok2:
                    continue
                ok3, rv, tv = cv2.solvePnP(
                    model_n, img_n, K, dist, rv2, tv2,
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok3:
                    rv, tv = rv2, tv2

            proj, _ = cv2.projectPoints(model_n, rv, tv, K, dist)
            err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_n, axis=1)))
            score = err
            if prev_rvec is not None and prev_tvec is not None:
                d_rot = _rotation_delta_deg(rv, prev_rvec)
                z_ref = max(abs(float(prev_tvec[2, 0])), 1e-6)
                d_t   = float(np.linalg.norm(tv.reshape(3) - prev_tvec.reshape(3)) / z_ref)
                score += PNP_ROT_SMOOTH_W * d_rot + PNP_TRANS_SMOOTH_W * d_t
            if prev_phase is not None:
                dp = min(abs(int(phase) - int(prev_phase)), n - abs(int(phase) - int(prev_phase)))
                score += 0.01 * float(dp)
            if prev_reverse is not None and bool(rev) != bool(prev_reverse):
                score += PNP_SHIFT_PENALTY
            if score < best_score:
                best_score = score
                best_err, best_rv, best_tv = err, rv, tv
                best_phase, best_reverse = int(phase), bool(rev)

        return best_rv, best_tv, best_err, best_phase, best_reverse

    # ------------------------------------------------------------------
    def estimate_pose(
        self, mask_bool, K, dist, state,
        kalman_process_var=KALMAN_PROCESS_VAR,
        kalman_meas_var=KALMAN_MEAS_VAR,
    ) -> dict:
        """
        Update ``state`` with ``rvec``, ``tvec``, contour phase, and registration sign.

        Returns the same dict reference (possibly unchanged if the mask is empty).
        """
        if state is None:
            state = {}
        if not np.any(mask_bool):
            return state

        clean = cv2.erode(mask_bool.astype(np.uint8) * 255, np.ones((3,3), np.uint8)) > 0
        ys, xs = np.where(clean)
        if len(xs) < 20:
            return state

        prev_rv = state.get("rvec_raw")
        if prev_rv is None:
            prev_rv = state.get("rvec")
        prev_tv = state.get("tvec_raw")
        if prev_tv is None:
            prev_tv = state.get("tvec")

        if "reg_sign" not in state:
            # One-time bootstrap: find best axis-sign combination
            best = (np.inf, None, None, None, None, None)
            for bits in range(8):
                s = np.array([-1.0 if (bits >> i) & 1 else 1.0 for i in range(3)], np.float64)
                rv_i, tv_i, err_i, ph_i, rev_i = self._pnp_best_contour(
                    self.model_pts * s, clean, K, dist,
                    prev_rvec=prev_rv, prev_tvec=prev_tv,
                    prev_phase=state.get("contour_phase"),
                    prev_reverse=state.get("contour_reverse"),
                )
                if rv_i is None:
                    continue
                sign_score = float(err_i) + self._calibration_orientation_penalty(
                    rv_i, tv_i, K, dist, sign=s
                )
                if sign_score < float(best[0]):
                    best = (sign_score, rv_i, tv_i, ph_i, rev_i, s)
            if best[1] is None:
                return state
            _, rv, tv, phase, reverse, s_use = best
            state["reg_sign"] = np.asarray(s_use, dtype=np.float64)
        else:
            s = _reg_sign_from_state(state)
            rv, tv, _err, phase, reverse = self._pnp_best_contour(
                self.model_pts * s, clean, K, dist,
                prev_rvec=prev_rv, prev_tvec=prev_tv,
                prev_phase=state.get("contour_phase"),
                prev_reverse=state.get("contour_reverse"),
            )

        if rv is None or tv is None:
            return state
        if phase is not None:
            state["contour_phase"] = int(phase)
        state["contour_reverse"] = bool(reverse)
        state["rvec_raw"] = rv.copy()
        state["tvec_raw"] = tv.copy()

        rv, tv = _apply_kalman_pose_filter(state, rv, tv, kalman_process_var, kalman_meas_var)
        state["rvec"] = rv
        state["tvec"] = tv
        return state

# ---------------------------------------------------------------------------
# SAM3D (fal.ai): mask → GLB mesh → MeshPoseEstimator per object id
# ---------------------------------------------------------------------------

def _bootstrap_sam3d_estimators(
    seed_frame_bgr: np.ndarray,
    ids: list[int],
    masks_np: np.ndarray,
    output_dir: Path,
    fal_model: str,
    object_classes: dict[int, str],
) -> tuple[dict[int, MeshPoseEstimator], dict[int, dict]]:
    """
    Upload seed image and per-object masks, run SAM3D, download GLBs, build estimators.

    Requires ``FAL_KEY``, ``fal_client``, and ``trimesh``. Writes ``sam3d_bootstrap.json``
    under ``output_dir`` when any object succeeds.
    """
    if trimesh is None:
        print("`trimesh` not installed."); return {}, {}
    if not os.environ.get("FAL_KEY"):
        print("FAL_KEY not set."); return {}, {}
    try:
        import fal_client  # type: ignore
    except Exception as e:
        print(f"`fal-client` import failed: {e}"); return {}, {}

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "seed_frame.png"
    cv2.imwrite(str(seed_path), seed_frame_bgr)
    image_url = fal_client.upload_file(str(seed_path))
    print(f"SAM3D bootstrap: uploaded seed frame.")

    estimators: dict[int, MeshPoseEstimator] = {}
    payload:    dict[int, dict]               = {}
    fh, fw = seed_frame_bgr.shape[:2]

    for i in range(min(len(ids), masks_np.shape[0])):
        oid  = int(ids[i])
        binm = _mask_to_2d_bool(masks_np[i], fh, fw).astype(np.uint8)
        if not np.any(binm):
            print(f"[ID{oid}] Empty mask; skipping SAM3D."); continue

        mask_path = output_dir / f"object_{oid}_mask.png"
        cv2.imwrite(str(mask_path), binm * 255)
        try:
            mask_url = fal_client.upload_file(str(mask_path))
            result   = fal_client.subscribe(
                fal_model,
                arguments={"image_url": image_url, "mask_urls": [mask_url], "seed": 42},
                with_logs=True,
            )
            model_glb = result.get("model_glb", {}) if isinstance(result, dict) else {}
            glb_url   = model_glb.get("url")
            if not isinstance(glb_url, str) or not glb_url:
                print(f"[ID{oid}] SAM3D returned no model_glb."); continue

            glb_path = output_dir / f"object_{oid}.glb"
            urlretrieve(glb_url, glb_path)
            mesh = trimesh.load(str(glb_path), force="mesh")
            cls  = object_classes.get(oid, "tool")
            # Pass obj_id so scissors (ID1) use the measured 11.75 cm
            estimators[oid] = MeshPoseEstimator(mesh, obj_id=oid, object_class=cls)
            payload[oid]    = {"mask_path": str(mask_path), "glb_path": str(glb_path)}
            print(f"[ID{oid}] SAM3D GLB ready: {glb_path.name}  class={cls}")
        except Exception as e:
            print(f"[ID{oid}] SAM3D bootstrap failed: {e}")

    if payload:
        (output_dir / "sam3d_bootstrap.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    return estimators, payload

# ---------------------------------------------------------------------------
# On-screen pose axes, HUD, and alignment debug (mesh vs mask IoU)
# ---------------------------------------------------------------------------

def _draw_pose_axes(vis, state, K, dist, axis_pts, obj_id):
    """Draw capped XYZ arrows in image space from ``axis_pts`` and current pose."""
    max_arrow_px = 40.0
    rvec = state.get("rvec"); tvec = state.get("tvec")
    if rvec is None or tvec is None:
        return
    s = _reg_sign_from_state(state)
    proj, _ = cv2.projectPoints((axis_pts * s).astype(np.float64), rvec, tvec, K, dist)
    pts2d = proj.reshape(-1, 2)
    fh, fw = vis.shape[:2]

    def clip_pt(p):
        return (int(np.clip(p[0], 0, fw-1)), int(np.clip(p[1], 0, fh-1)))

    def cap(o, t):
        v = t - o; n = float(np.linalg.norm(v))
        return o if n < 1e-9 else (o + v * (max_arrow_px / n) if n > max_arrow_px else t)

    o = pts2d[0].astype(np.float64)
    origin = clip_pt(o)
    x_tip  = clip_pt(cap(o, pts2d[1]))
    y_tip  = clip_pt(cap(o, pts2d[2]))
    z_tip  = clip_pt(cap(o, pts2d[3]))

    cv2.arrowedLine(vis, origin, x_tip, (0,   0, 220), 2, tipLength=0.20, line_type=cv2.LINE_AA)
    cv2.arrowedLine(vis, origin, y_tip, (0, 200,   0), 2, tipLength=0.20, line_type=cv2.LINE_AA)
    cv2.arrowedLine(vis, origin, z_tip, (220, 80,  0), 2, tipLength=0.20, line_type=cv2.LINE_AA)
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, "X", (x_tip[0]+4, x_tip[1]+4), fnt, 0.50, (0,   0, 220), 1, cv2.LINE_AA)
    cv2.putText(vis, "Y", (y_tip[0]+4, y_tip[1]+4), fnt, 0.50, (0, 200,   0), 1, cv2.LINE_AA)
    cv2.putText(vis, "Z", (z_tip[0]+4, z_tip[1]+4), fnt, 0.50, (220, 80,  0), 1, cv2.LINE_AA)


def _draw_pose_hud(vis, pose_states):
    """Overlay Euler + translation text for each tracked object id."""
    if not pose_states:
        return
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    for row, oid in enumerate(sorted(pose_states)):
        st   = pose_states.get(oid, {})
        rvec = st.get("rvec")
        tvec = st.get("tvec")
        if rvec is None or tvec is None:
            continue
        R, _ = cv2.Rodrigues(rvec)
        sy = float(np.sqrt(R[0,0]**2 + R[1,0]**2))
        if sy > 1e-6:
            rx = np.degrees(np.arctan2(float(R[2,1]), float(R[2,2])))
            ry = np.degrees(np.arctan2(-float(R[2,0]), sy))
            rz = np.degrees(np.arctan2(float(R[1,0]), float(R[0,0])))
        else:
            rx = np.degrees(np.arctan2(-float(R[1,2]), float(R[1,1])))
            ry = np.degrees(np.arctan2(-float(R[2,0]), sy)); rz = 0.0
        tv  = np.asarray(tvec, dtype=np.float64).reshape(-1)
        tag = " (scissors)" if oid == 1 else ""
        text = f"ID{oid}{tag}  R({rx:.0f},{ry:.0f},{rz:.0f})  T({tv[0]:.3f},{tv[1]:.3f},{tv[2]:.3f}m)"
        yy = 18 + row * 18
        cv2.putText(vis, text, (12, yy), fnt, 0.46, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(vis, text, (12, yy), fnt, 0.46, (0,  0,  0  ), 1, cv2.LINE_AA)


def _draw_registration_debug(canvas, mask_bool, state, est, K, dist, obj_id):
    """Project mesh faces, SAM mask, PnP polygon, contour samples, and IoU for tuning."""
    rvec = state.get("rvec"); tvec = state.get("tvec")
    if rvec is None or tvec is None or not np.any(mask_bool):
        return
    fh, fw = canvas.shape[:2]
    pred_mask = np.zeros((fh, fw), dtype=np.uint8)
    s = _reg_sign_from_state(state)
    proj_mesh, _ = cv2.projectPoints(est.mesh_vertices * s, rvec, tvec, K, dist)
    pts2d = proj_mesh.reshape(-1, 2)
    overlay = canvas.copy()
    for f in est.mesh_faces:
        poly_f = pts2d[f].astype(np.float32)
        # Skip faces wildly off-screen (prevents fillConvexPoly explosion)
        if (poly_f[:,0].min() > 3*fw or poly_f[:,0].max() < -2*fw or
                poly_f[:,1].min() > 3*fh or poly_f[:,1].max() < -2*fh):
            continue
        poly = np.clip(poly_f, [-fw,-fh], [2*fw,2*fh]).astype(np.int32)
        cv2.fillConvexPoly(overlay, poly, (180,130,255), lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [poly], True, (255,0,255), 1, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(pred_mask, poly, 255, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0, dst=canvas)

    sam_color = np.zeros_like(canvas)
    mu8 = mask_bool.astype(np.uint8) * 255
    sam_color[:,:,0] = mu8; sam_color[:,:,1] = mu8
    cv2.addWeighted(sam_color, 0.30, canvas, 1.0, 0.0, dst=canvas)

    # Bounding box only around on-screen vertices
    vis_pts = pts2d[(pts2d[:,0]>=0)&(pts2d[:,0]<fw)&(pts2d[:,1]>=0)&(pts2d[:,1]<fh)]
    if len(vis_pts):
        cv2.rectangle(canvas,
                      (int(vis_pts[:,0].min()), int(vis_pts[:,1].min())),
                      (int(vis_pts[:,0].max()), int(vis_pts[:,1].max())),
                      (0,255,255), 2, lineType=cv2.LINE_AA)

    proj, _ = cv2.projectPoints(est.model_pts * s, rvec, tvec, K, dist)
    poly = np.round(proj.reshape(-1, 2)).astype(np.int32)
    if poly.shape[0] >= 3:
        cv2.polylines(canvas, [poly], True, (255,255,255), 2, lineType=cv2.LINE_AA)

    n = int(est._pnp_n)
    mask_pts = _sample_mask_contour(mu8, n)
    mesh_pts = _sample_mask_contour(pred_mask, n)
    if mask_pts is not None:
        for p in mask_pts:
            cv2.circle(canvas, (int(round(p[0])), int(round(p[1]))), 2, (255,255,0), -1, cv2.LINE_AA)
    if mesh_pts is not None:
        for p in mesh_pts:
            cv2.circle(canvas, (int(round(p[0])), int(round(p[1]))), 2, (255,0,255), -1, cv2.LINE_AA)

    pred_b = pred_mask > 0
    inter  = float(np.logical_and(pred_b, mask_bool).sum())
    union  = float(np.logical_or(pred_b, mask_bool).sum())
    iou    = inter / union if union > 0.0 else 0.0
    ys, xs = np.where(mask_bool)
    tx, ty = (int(xs.mean()), int(ys.mean())) if len(xs) else (12, 24)
    text   = f"ID{obj_id} IoU={iou:.3f}"
    cv2.putText(canvas, text, (tx+8,ty+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (tx+8,ty+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# LiveFrameProvider: bridge OpenCV capture to EdgeTAM tensor "video" API
# ---------------------------------------------------------------------------

class LiveFrameProvider:
    """
    Feeds EdgeTAM ``__getitem__(idx)`` with normalized CHW tensors.

    **Live camera:** a daemon thread calls :meth:`capture_next` in a loop (discards
    buffered frames, keeps latest). ``__getitem__`` returns the cached tensor for any
    ``idx`` (small LRU cache of raw BGR for ``get_raw``).

    **Sequential file:** ``capture_next`` advances one frame per call; ``__getitem__``
    blocks until that index exists. ``len()`` reflects ``max_frames`` for synthetic clips.
    """

    def __init__(self, cap, image_size, rotate_180,
                 sequential_mode=False, max_frames=None):
        self.cap            = cap
        self.image_size     = image_size
        self.rotate_180     = rotate_180
        self.sequential_mode = bool(sequential_mode)
        self.max_frames     = None if max_frames is None else int(max_frames)
        self._cache: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
        self._latest_tensor: torch.Tensor | None = None
        self._latest_raw:   np.ndarray   | None = None
        self._lock = threading.Lock()
        self._seq_next_idx = 0
        self._eof = False

    def _encode(self, frame):
        """BGR frame → preprocessed BGR ``TARGET_SIZE`` + normalized square RGB tensor."""
        frame = preprocess(frame, self.rotate_180)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t     = torch.from_numpy(
            cv2.resize(rgb, (self.image_size, self.image_size))
        ).float().div(255.0).permute(2, 0, 1)
        return (t - _IMG_MEAN) / _IMG_STD, frame

    def capture_next(self) -> bool:
        """Grab one frame from ``cap``, update latest tensor/raw; False on EOF/failure."""
        if self.sequential_mode:
            if self.max_frames is not None and self._seq_next_idx >= self.max_frames:
                self._eof = True
                return False
            ok, frame = self.cap.read()
            if not ok:
                self._eof = True
                return False
            t, raw = self._encode(frame)
            with self._lock:
                idx = self._seq_next_idx
                self._seq_next_idx += 1
                self._latest_tensor = t
                self._latest_raw = raw
                self._cache[idx] = (t, raw)
            return True
        frame = None
        for _ in range(4):
            ok, f = self.cap.read()
            if ok: frame = f
        if frame is None: return False
        t, raw = self._encode(frame)
        with self._lock:
            self._latest_tensor = t
            self._latest_raw = raw
        return True

    def __len__(self):
        if self.sequential_mode and self.max_frames is not None:
            return max(1, int(self.max_frames))
        return 1_000_000

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Normalized CHW tensor for frame index ``idx`` (blocks until available)."""
        if self.sequential_mode:
            while True:
                with self._lock:
                    entry = self._cache.get(int(idx))
                    if entry is not None: return entry[0]
                    if self._eof:
                        if self._latest_tensor is not None: return self._latest_tensor
                        raise IndexError("No frames available.")
                if not self.capture_next(): time.sleep(0.001)
        while True:
            with self._lock:
                if self._latest_tensor is not None:
                    if idx not in self._cache:
                        self._cache[idx] = (self._latest_tensor, self._latest_raw)
                        if len(self._cache) > 32:
                            del self._cache[min(self._cache)]
                    return self._cache[idx][0]
            time.sleep(0.001)

    def get_raw(self, idx: int) -> np.ndarray:
        """BGR uint8 at ``TARGET_SIZE`` for visualization / CSV alignment to frame ``idx``."""
        with self._lock:
            entry = self._cache.get(idx)
            return entry[1] if entry is not None else self._latest_raw

# ---------------------------------------------------------------------------
# CLI entry: wiring predictor, SAM3D, propagate loop, CSV / video export
# ---------------------------------------------------------------------------

def _pose_to_euler_zyx_deg(rvec):
    """Euler angles in degrees (same convention as HUD) from a Rodrigues vector."""
    R, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(R[0,0]**2 + R[1,0]**2))
    if sy > 1e-6:
        rx = float(np.degrees(np.arctan2(float(R[2,1]), float(R[2,2]))))
        ry = float(np.degrees(np.arctan2(-float(R[2,0]), sy)))
        rz = float(np.degrees(np.arctan2(float(R[1,0]), float(R[0,0]))))
    else:
        rx = float(np.degrees(np.arctan2(-float(R[1,2]), float(R[1,1]))))
        ry = float(np.degrees(np.arctan2(-float(R[2,0]), sy))); rz = 0.0
    return rx, ry, rz


def _next_pose_csv_path(base_dir: Path) -> Path:
    """First non-existing ``poses{i}.csv`` in ``base_dir`` (typically cwd)."""
    i = 1
    while True:
        p = base_dir / f"poses{i}.csv"
        if not p.exists(): return p
        i += 1


def run(args) -> None:
    """
    End-to-end live demo: load EdgeTAM, capture, pick seeds, SAM3D meshes, track.

    Exits early if checkpoint missing, device is CPU-only, capture fails, the user
    cancels seeding, or SAM3D returns no estimators.
    """
    device = choose_device(args.device)
    if device == "cpu":
        print("GPU required. Use --device mps or --device cuda."); return
    print(f"Device: {device}")
    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT}"); return

    print("Loading EdgeTAM …")
    predictor = _load_predictor(device)

    source = (args.camera or "").strip()
    use_synthetic = (source == "") or source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    generated_video_path: Path | None = None

    if use_synthetic:
        image_path = Path(source) if source else Path(args.synthetic_image)
        if not image_path.is_absolute():
            image_path = (Path(__file__).parent / image_path).resolve()
        generated_video_path = (Path(__file__).parent / args.synthetic_video_name).resolve()
        _build_repeated_video_from_image(
            image_path, generated_video_path,
            args.synthetic_frames, args.synthetic_fps)
        source_to_open = str(generated_video_path)
        print(f"Synthetic video: {image_path.name} → {generated_video_path.name}")
    else:
        source_to_open = source

    cap = cv2.VideoCapture(source_to_open)
    if not cap.isOpened():
        print(f"Could not open: {source_to_open}"); return

    is_camera_index = source_to_open.isdigit()
    if is_camera_index:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    rotate_180 = detect_orbbec_camera(int(source_to_open)) if is_camera_index else False
    sequential_mode = not is_camera_index

    K_cam, dist_cam, intr_src = _resolve_pose_intrinsics(
        cap, TARGET_SIZE[0], TARGET_SIZE[1],
        args.intrinsics_file, not args.no_orbbec_intrinsics)
    print(f"Intrinsics ({intr_src}): "
          f"fx={K_cam[0,0]:.1f} fy={K_cam[1,1]:.1f} "
          f"cx={K_cam[0,2]:.1f} cy={K_cam[1,2]:.1f}")

    # Object class map — ID1 hardcoded as scissors
    object_classes: dict[int, str] = {1: "scissors"}
    for i, cls in enumerate(args.object_classes or []):
        oid = i + 1
        if oid not in object_classes:        # don't override ID1
            object_classes[oid] = cls.lower()

    image_size     = predictor.image_size
    provider       = LiveFrameProvider(
        cap, image_size, rotate_180,
        sequential_mode=sequential_mode,
        max_frames=args.synthetic_frames if generated_video_path else None,
    )
    if not provider.capture_next():
        print("No frame from camera."); cap.release(); return

    stop_flag = threading.Event()

    def _capture_loop():
        while not stop_flag.is_set():
            if not provider.capture_next(): stop_flag.set()

    if not sequential_mode:
        threading.Thread(target=_capture_loop, daemon=True).start()

    points, seed_frame = pick_points_live(provider, stop_flag)
    if not points:
        print("No points selected. Exiting.")
        stop_flag.set(); cap.release(); return
    if seed_frame is None:
        print("No seed frame. Exiting.")
        stop_flag.set(); cap.release(); return

    tmp = tempfile.mkdtemp(prefix="edgetam_live_")
    cv2.imwrite(os.path.join(tmp, "000000.jpg"), seed_frame)
    et_state = predictor.init_state(tmp, async_loading_frames=False)
    shutil.rmtree(tmp, ignore_errors=True)

    et_state["images"]     = provider
    et_state["num_frames"] = len(provider) if sequential_mode else 1_000_000

    _seed_ids: list[int] = [int(p[0]) for p in points]
    _seed_masks_np = np.zeros((0, TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.float32)

    for obj_id, x, y in points:
        _out = predictor.add_new_points_or_box(
            et_state, frame_idx=0, obj_id=int(obj_id),
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        if _out is not None and len(_out) >= 3:
            raw = _out[1]
            _seed_ids      = [int(v) for v in (raw.tolist() if hasattr(raw,"tolist") else raw)]
            _seed_masks_np = _out[2].detach().cpu().numpy()

    prop_stream  = predictor.propagate_in_video(et_state)
    first_packet = None
    if _seed_masks_np.shape[0] == 0:
        try:
            first_packet   = next(prop_stream)
            raw            = first_packet[1]
            _seed_ids      = [int(v) for v in (raw.tolist() if hasattr(raw,"tolist") else raw)]
            _seed_masks_np = first_packet[2].detach().cpu().numpy()
        except StopIteration:
            print("No propagated masks. Exiting.")
            stop_flag.set(); cap.release(); return

    print("Bootstrapping SAM3D on initial frozen frame (blocking)…")
    mesh_estimators, _ = _bootstrap_sam3d_estimators(
        seed_frame_bgr=seed_frame.copy(),
        ids=_seed_ids,
        masks_np=_seed_masks_np,
        output_dir=Path(args.sam3d_output_dir),
        fal_model=args.sam3d_model,
        object_classes=object_classes,
    )
    if not mesh_estimators:
        print("SAM3D produced no estimators. Exiting.")
        stop_flag.set(); cap.release(); return
    print("SAM3D ready: " + ", ".join(f"ID{k}" for k in sorted(mesh_estimators)))

    writer = None
    if args.output:
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"),
            30.0, (TARGET_SIZE[0], TARGET_SIZE[1]))

    csv_path = _next_pose_csv_path(Path.cwd())
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_idx","time_s","object_id","class",
                         "rx_deg","ry_deg","rz_deg","tx_m","ty_m","tz_m"])
    print(f"Pose CSV: {csv_path}")

    com_trails:  dict[int, list[tuple[int,int]]] = {}
    pose_states: dict[int, dict]                 = {}
    MAX_TRAIL = 60

    # Init pose from seed masks
    for i in range(min(len(_seed_ids), _seed_masks_np.shape[0])):
        oid = int(_seed_ids[i])
        est = mesh_estimators.get(oid)
        if est is None: continue
        binm = _mask_to_2d_bool(_seed_masks_np[i], TARGET_SIZE[1], TARGET_SIZE[0])
        if np.any(binm):
            pose_states[oid] = est.estimate_pose(
                binm, K_cam, dist_cam, pose_states.get(oid),
                kalman_process_var=args.kalman_process_var,
                kalman_meas_var=args.kalman_meas_var,
            )

    fps_t0, fps_frames = time.perf_counter(), 0
    ac_device, ac_dtype, ac_enabled = _autocast_config(device, args.half)
    show_debug = bool(args.align_debug_out) or bool(args.overlay_frames_dir)

    print("Live tracking started. Press 'q' or ESC to stop.")
    try:
        with torch.autocast(device_type=ac_device, dtype=ac_dtype, enabled=ac_enabled):
            stream_iter = (prop_stream if first_packet is None
                           else itertools.chain([first_packet], prop_stream))
            for fi, obj_ids, masks in stream_iter:
                frame = provider.get_raw(fi)
                if frame is None: continue

                debug_canvas = frame.copy() if show_debug else None
                ids = [int(x) for x in (obj_ids.tolist() if hasattr(obj_ids,"tolist") else obj_ids)]

                vis      = overlay_masks(frame, ids, masks, alpha=args.alpha)
                fh, fw   = frame.shape[:2]
                masks_np = masks.detach().cpu().numpy()

                for i in range(min(len(ids), masks_np.shape[0])):
                    oid  = ids[i]
                    binm = _mask_to_2d_bool(masks_np[i], fh, fw)
                    if not np.any(binm): continue

                    ys, xs = np.where(binm)
                    cx, cy = int(xs.mean()), int(ys.mean())
                    trail  = com_trails.setdefault(oid, [])
                    trail.append((cx, cy))
                    if len(trail) > MAX_TRAIL:
                        com_trails[oid] = trail[-MAX_TRAIL:]

                    est = mesh_estimators.get(oid)
                    if est is None: continue

                    pose_states[oid] = est.estimate_pose(
                        binm, K_cam, dist_cam, pose_states.get(oid),
                        kalman_process_var=args.kalman_process_var,
                        kalman_meas_var=args.kalman_meas_var,
                    )
                    _draw_pose_axes(vis, pose_states[oid], K_cam, dist_cam, est.axis_pts, oid)
                    if debug_canvas is not None:
                        _draw_registration_debug(
                            debug_canvas, binm, pose_states[oid], est, K_cam, dist_cam, oid)

                for oid, trail in com_trails.items():
                    col = point_color(oid)
                    for j in range(1, len(trail)):
                        cv2.line(vis, trail[j-1], trail[j], col, 1, lineType=cv2.LINE_AA)
                    if trail:
                        lp = trail[-1]
                        cv2.circle(vis, lp, 5, col, -1)
                        cv2.circle(vis, lp, 7, (255,255,255), 1)
                        cv2.putText(vis, f"ID{oid}", (lp[0]+8, lp[1]-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

                _draw_pose_hud(vis, pose_states)

                if fi == 0:
                    for oid, px, py in points:
                        cv2.circle(vis, (int(px), int(py)), 5, (0,255,255), -1)
                        cv2.putText(vis, f"ID{oid}", (int(px)+8, int(py)-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

                cv2.imshow("EdgeTAM Live", vis)
                if writer: writer.write(vis)
                if debug_canvas is not None:
                    if args.align_debug_out:
                        cv2.imwrite(args.align_debug_out, debug_canvas)
                    if args.overlay_frames_dir:
                        od = Path(args.overlay_frames_dir)
                        od.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(od / f"frame_{int(fi):06d}.png"), debug_canvas)

                t_s = float(time.perf_counter())
                for oid in sorted(pose_states):
                    st   = pose_states.get(oid, {})
                    rvec = st.get("rvec"); tvec = st.get("tvec")
                    if rvec is None or tvec is None: continue
                    rx, ry, rz = _pose_to_euler_zyx_deg(rvec)
                    tv = np.asarray(tvec, dtype=np.float64).reshape(-1)
                    if tv.size < 3: continue
                    csv_writer.writerow([
                        int(fi), t_s, oid, object_classes.get(oid, "tool"),
                        rx, ry, rz, tv[0], tv[1], tv[2],
                    ])

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27): break
                if stop_flag.is_set() and not sequential_mode: break

                fps_frames += 1
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    print(f"FPS: {fps_frames / (now - fps_t0):.2f}")
                    fps_t0, fps_frames = now, 0
    finally:
        stop_flag.set()
        cap.release()
        if writer: writer.release()
        csv_file.close()
        cv2.destroyAllWindows()
        print(f"Done. Poses saved to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EdgeTAM live surgical tool tracking with SAM3D mesh bootstrap. "
            "ID1 is scissors (11.75 cm metric reference); use --object-classes for other IDs."
        ),
    )
    parser.add_argument("--camera", type=str, default="")
    parser.add_argument("--device", default="auto", choices=["auto","cuda","mps","cpu"])
    parser.add_argument("--object-classes", nargs="*", default=[],
                        help="Classes for ID2, ID3 … in order (ID1 is always scissors). "
                             "e.g. --object-classes scalpel bottle bag")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--kalman-process-var", type=float, default=KALMAN_PROCESS_VAR)
    parser.add_argument("--kalman-meas-var",    type=float, default=KALMAN_MEAS_VAR)
    parser.add_argument("--half", action="store_true", default=True)
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--output", default="")
    parser.add_argument("--sam3d-model",      default="fal-ai/sam-3/3d-objects")
    parser.add_argument("--sam3d-output-dir", default=str(Path(__file__).parent / "sam3d_live_bootstrap"))
    parser.add_argument("--intrinsics-file",  default="")
    parser.add_argument("--no-orbbec-intrinsics", action="store_true")
    parser.add_argument("--align-debug-out",  default="")
    parser.add_argument(
        "--overlay-frames-dir",
        default="",
        help="Optional directory to export per-frame debug overlays. Disabled by default.",
    )
    parser.add_argument("--synthetic-image",  default=str(Path(__file__).parent / "testing.jpg"))
    parser.add_argument("--synthetic-frames", type=int,   default=100)
    parser.add_argument("--synthetic-fps",    type=float, default=30.0)
    parser.add_argument("--synthetic-video-name", default="testing_100frames.mp4")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()