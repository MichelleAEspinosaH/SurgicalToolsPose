#!/usr/bin/env python3
"""
Real-time surgical tool tracking with EdgeTAM.

Opens the Orbbec RGB camera (or any camera index), lets you click seed
points on the first frame, then streams live masks + mesh-based 6DoF overlays.

Usage:
    .venv/bin/python live_track.py                        # camera 0, default settings
    .venv/bin/python live_track.py --camera 1             # different camera index
    .venv/bin/python live_track.py --kalman-process-var 2e-4   # smoother pose tracks
    .venv/bin/python live_track.py --no-half              # disable half-precision
    .venv/bin/python live_track.py --output out.mp4       # save output video
Keybindings:
    Enter      Confirm seed points and start tracking
    q / ESC    Quit
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
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

_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_IMG_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

# ---------------------------------------------------------------------------
# EdgeTAM loader
# ---------------------------------------------------------------------------

def _load_predictor(device: str):
    repo = str(EDGETAM_REPO.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore
    return build_sam2_video_predictor(MODEL_CFG, str(CHECKPOINT), device=device)


def _autocast_config(device: str, use_half: bool) -> tuple[str, torch.dtype, bool]:
    device_type = device.split(":")[0]
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float16
    enabled = use_half and device_type != "cpu"
    return device_type, dtype, enabled


def choose_device(arg: str) -> str:
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
    if rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def detect_orbbec_camera(camera_id: int) -> bool:
    """Best-effort camera-name lookup on macOS via ffmpeg/AVFoundation."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except Exception:
        return False

    listing = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for line in listing.splitlines():
        m = re.search(r"\[(\d+)\]\s+(.+)$", line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        name = m.group(2).strip()
        if idx == camera_id:
            return "orbbec" in name.lower()
    return False

# ---------------------------------------------------------------------------
# Point-picking UI
# ---------------------------------------------------------------------------

def pick_points_live(
    provider: "LiveFrameProvider", stop_flag: threading.Event
) -> tuple[list[tuple[int, float, float]], np.ndarray | None]:
    win = "Select EdgeTAM points"
    points: list[tuple[int, float, float]] = []
    frozen_frame: np.ndarray | None = None

    def draw() -> np.ndarray:
        base = frozen_frame if frozen_frame is not None else provider.get_raw(-1)
        if base is None:
            vis = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8)
        else:
            vis = base.copy()
        for obj_id, px_f, py_f in points:
            px, py = int(px_f), int(py_f)
            cv2.circle(vis, (px, py), 6, (0, 255, 255), -1)
            cv2.circle(vis, (px, py), 9, (255, 255, 255), 1)
            cv2.putText(vis, f"ID{obj_id}", (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(
            vis,
            "Left click: add point (freezes frame on first click) | Backspace: undo | c: clear | Enter: start",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
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
    # Seed frame must match the point-picking image exactly.
    seed_frame = frozen_frame if frozen_frame is not None else provider.get_raw(-1)
    return points, (None if seed_frame is None else seed_frame.copy())

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def point_color(obj_id: int) -> tuple[int, int, int]:
    hue = (obj_id * 47 + 20) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _mask_to_2d_bool(m: np.ndarray, fh: int, fw: int) -> np.ndarray:
    x = np.squeeze(np.asarray(m, dtype=np.float32))
    while x.ndim > 2:
        x = x[0]
    if x.ndim != 2:
        return np.zeros((fh, fw), dtype=bool)
    if x.shape != (fh, fw):
        x = cv2.resize(x, (fw, fh), interpolation=cv2.INTER_NEAREST)
    return x > 0.0


def overlay_masks(frame, obj_ids, masks, alpha=0.45):
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


def _order_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    center = corners.mean(axis=0)
    ang = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    ordered = corners[np.argsort(ang)]
    # Rotate order so first corner is top-most, then left-most.
    idx0 = int(np.argmin(ordered[:, 1] * 10000.0 + ordered[:, 0]))
    return np.roll(ordered, -idx0, axis=0)


def _estimate_intrinsics_from_cap(
    cap: cv2.VideoCapture, target_w: int, target_h: int
) -> np.ndarray:
    """
    Estimate camera intrinsic matrix without an external calibration file.

    Queries the VideoCapture for its native frame dimensions, then applies the
    Orbbec IR FOV spec (H=79°, V=62°). Focal lengths are computed on the native
    sensor size and scaled to the resized inference resolution (TARGET_SIZE).
    """
    native_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    native_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if native_w <= 0 or native_h <= 0:
        native_w, native_h = float(target_w), float(target_h)

    # Use separate focal lengths from spec FOV:
    # fx = (w/2)/tan(hfov/2), fy = (h/2)/tan(vfov/2), then scale to target size.
    hfov_rad = np.radians(79.0)
    vfov_rad = np.radians(62.0)
    fx_native = native_w / (2.0 * np.tan(hfov_rad / 2.0))
    fy_native = native_h / (2.0 * np.tan(vfov_rad / 2.0))
    fx = fx_native * (target_w / native_w)
    fy = fy_native * (target_h / native_h)

    return np.array(
        [[fx, 0.0, target_w / 2.0], [0.0, fy, target_h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rescale_intrinsics(
    K: np.ndarray,
    src_w: float,
    src_h: float,
    dst_w: int,
    dst_h: int,
) -> np.ndarray:
    """Rescale intrinsic matrix when image resolution changes."""
    sx = float(dst_w) / max(float(src_w), 1e-9)
    sy = float(dst_h) / max(float(src_h), 1e-9)
    K2 = np.asarray(K, dtype=np.float64).copy()
    K2[0, 0] *= sx  # fx
    K2[1, 1] *= sy  # fy
    K2[0, 2] *= sx  # cx
    K2[1, 2] *= sy  # cy
    return K2


def _load_intrinsics_from_file(
    path: str,
    target_w: int,
    target_h: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load camera intrinsics from .npz.

    Supported keys:
      - K / camera_matrix / intrinsics  (3x3)
      - dist / dist_coeffs / distortion (optional)
      - width,height or image_width,image_height (optional, for rescaling)
    """
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"Failed to load intrinsics file '{path}': {e}")
        return None

    K = None
    for kname in ("K", "camera_matrix", "intrinsics"):
        if kname in data:
            cand = np.asarray(data[kname], dtype=np.float64)
            if cand.shape == (3, 3):
                K = cand
                break
    if K is None:
        print(
            f"Intrinsics file '{path}' is missing a 3x3 matrix "
            f"(expected one of: K, camera_matrix, intrinsics)."
        )
        return None

    dist = np.zeros((4, 1), dtype=np.float64)
    for dname in ("dist", "dist_coeffs", "distortion"):
        if dname in data:
            d = np.asarray(data[dname], dtype=np.float64).reshape(-1, 1)
            if d.size > 0:
                dist = d
            break

    src_w = src_h = None
    if "width" in data and "height" in data:
        src_w = float(np.asarray(data["width"]).reshape(-1)[0])
        src_h = float(np.asarray(data["height"]).reshape(-1)[0])
    elif "image_width" in data and "image_height" in data:
        src_w = float(np.asarray(data["image_width"]).reshape(-1)[0])
        src_h = float(np.asarray(data["image_height"]).reshape(-1)[0])

    if src_w is not None and src_h is not None and src_w > 0 and src_h > 0:
        K = _rescale_intrinsics(K, src_w, src_h, target_w, target_h)

    return K.astype(np.float64), dist.astype(np.float64)


def _try_read_orbbec_intrinsics(
    target_w: int,
    target_h: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Try reading Orbbec color intrinsics via pyorbbecsdk.

    Returns (K, dist) in TARGET_SIZE scale when available, else None.
    """
    try:
        from pyorbbecsdk import Config, OBSensorType, Pipeline  # type: ignore
    except Exception:
        return None

    pipeline = None
    try:
        pipeline = Pipeline()
        config = Config()
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is None:
            return None
        color_profile = profile_list.get_default_video_stream_profile()
        if color_profile is None:
            return None

        native_w = float(color_profile.get_width())
        native_h = float(color_profile.get_height())

        intr = None
        if hasattr(color_profile, "get_intrinsic"):
            intr = color_profile.get_intrinsic()
        elif hasattr(color_profile, "get_camera_intrinsic"):
            intr = color_profile.get_camera_intrinsic()

        if intr is None:
            return None

        fx = float(getattr(intr, "fx", 0.0))
        fy = float(getattr(intr, "fy", 0.0))
        cx = float(getattr(intr, "cx", native_w / 2.0))
        cy = float(getattr(intr, "cy", native_h / 2.0))
        if fx <= 0 or fy <= 0:
            return None

        K_native = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        K = _rescale_intrinsics(K_native, native_w, native_h, target_w, target_h)

        # Distortion is optional in SDK wrappers; use zeros if unavailable.
        dist = np.zeros((4, 1), dtype=np.float64)
        if hasattr(intr, "coeffs"):
            coeffs = np.asarray(getattr(intr, "coeffs"), dtype=np.float64).reshape(-1, 1)
            if coeffs.size > 0:
                dist = coeffs
        else:
            vals = []
            for name in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"):
                if hasattr(intr, name):
                    vals.append(float(getattr(intr, name)))
            if vals:
                dist = np.asarray(vals, dtype=np.float64).reshape(-1, 1)
        return K, dist
    except Exception:
        return None
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass


def _resolve_pose_intrinsics(
    cap: cv2.VideoCapture,
    target_w: int,
    target_h: int,
    intrinsics_file: str,
    try_orbbec_intrinsics: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Resolve pose intrinsics in priority order:
      1) user-provided intrinsics file
      2) Orbbec SDK-reported intrinsics (if enabled/available)
      3) FOV-based estimate fallback
    """
    if intrinsics_file:
        loaded = _load_intrinsics_from_file(intrinsics_file, target_w, target_h)
        if loaded is not None:
            return loaded[0], loaded[1], f"file:{intrinsics_file}"
        print("Falling back because intrinsics file could not be used.")

    if try_orbbec_intrinsics:
        sdk_loaded = _try_read_orbbec_intrinsics(target_w, target_h)
        if sdk_loaded is not None:
            return sdk_loaded[0], sdk_loaded[1], "orbbec_sdk"

    K = _estimate_intrinsics_from_cap(cap, target_w, target_h)
    dist = np.zeros((4, 1), dtype=np.float64)
    return K, dist, "fov_estimate"

def _mesh_projection_iou(
    mask_bool: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    """IoU between binary mask and mesh silhouette (filled projected faces)."""
    fh, fw = mask_bool.shape[:2]
    pred_mask = np.zeros((fh, fw), dtype=np.uint8)
    proj_mesh, _ = cv2.projectPoints(verts.astype(np.float64), rvec, tvec, K, dist)
    pts2d = proj_mesh.reshape(-1, 2)
    for f in faces:
        poly = np.round(pts2d[f]).astype(np.int32)
        cv2.fillConvexPoly(pred_mask, poly, 255, lineType=cv2.LINE_AA)
    pred_b = pred_mask > 0
    sam_b = mask_bool
    inter = float(np.logical_and(pred_b, sam_b).sum())
    union = float(np.logical_or(pred_b, sam_b).sum())
    return inter / union if union > 0.0 else 0.0


def _reg_sign_from_state(state: dict) -> np.ndarray:
    s = state.get("reg_sign")
    if s is None:
        return np.ones(3, dtype=np.float64)
    return np.asarray(s, dtype=np.float64).reshape(3)


# Number of uniform samples on each rectangle edge (total = 4 * PNP_PER_EDGE).
# Dense PnP anchors pose better than 4 corners alone.
PNP_PER_EDGE = 8
KALMAN_PROCESS_VAR = 2e-4
KALMAN_MEAS_VAR = 4e-3
PNP_ROT_SMOOTH_W = 0.03
PNP_TRANS_SMOOTH_W = 8.0
PNP_SHIFT_PENALTY = 0.75


def _sample_quad_perimeter(corners4: np.ndarray, n: int) -> np.ndarray:
    """Uniform arc-length samples on a closed quadrilateral (first vertex not repeated).

    corners4: (4, D) in cyclic order. Returns (n, D).
    """
    D = int(corners4.shape[1])
    lens = np.linalg.norm(np.roll(corners4, -1, axis=0) - corners4, axis=1)
    L = float(lens.sum())
    out = np.zeros((n, D), dtype=np.float64)
    if L < 1e-12:
        return np.repeat(corners4[:1].astype(np.float64), n, axis=0)
    cum = np.zeros(5, dtype=np.float64)
    for i in range(4):
        cum[i + 1] = cum[i] + lens[i]
    for i in range(n):
        t = ((i + 0.5) / n) * L
        for k in range(4):
            if cum[k + 1] >= t - 1e-15:
                denom = lens[k] + 1e-12
                u = (t - cum[k]) / denom
                out[i] = corners4[k] * (1.0 - u) + corners4[(k + 1) % 4] * u
                break
    return out


def _sample_mask_contour(mask_u8: np.ndarray, n: int) -> np.ndarray | None:
    """Uniformly sample n points along the largest contour of a binary mask."""
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
        k = int(np.searchsorted(cum, t, side="right") - 1)
        k = int(np.clip(k, 0, len(poly) - 1))
        seg = d[k] + 1e-12
        u = (t - cum[k]) / seg
        out[i] = poly[k] * (1.0 - u) + poly[(k + 1) % len(poly)] * u
    return out


def _rotation_delta_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    """Geodesic angular difference between two Rodrigues rotations."""
    Ra, _ = cv2.Rodrigues(rvec_a.astype(np.float64))
    Rb, _ = cv2.Rodrigues(rvec_b.astype(np.float64))
    R = Ra @ Rb.T
    tr = float(np.trace(R))
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


class KalmanScalar:
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
        # Predict: x_k|k-1 = x_k-1, P_k|k-1 = P_k-1 + Q
        self.p += self.q
        # Update with measurement z_k
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1.0 - k) * self.p
        return self.x


def _apply_kalman_pose_filter(
    state: dict,
    rv: np.ndarray,
    tv: np.ndarray,
    process_var: float,
    meas_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    filters = state.get("kalman_filters")
    if filters is None or len(filters) != 6:
        filters = [
            KalmanScalar(process_var, meas_var),
            KalmanScalar(process_var, meas_var),
            KalmanScalar(process_var, meas_var),
            KalmanScalar(process_var, meas_var),
            KalmanScalar(process_var, meas_var),
            KalmanScalar(process_var, meas_var),
        ]
        state["kalman_filters"] = filters
    vec = np.concatenate([rv.reshape(3), tv.reshape(3)]).astype(np.float64)
    out = np.empty_like(vec)
    for i in range(6):
        out[i] = filters[i].filter(float(vec[i]))
    return out[:3].reshape(3, 1), out[3:].reshape(3, 1)


# ---------------------------------------------------------------------------
# Mesh-based 6DoF pose estimator
# ---------------------------------------------------------------------------

class MeshPoseEstimator:
    """
    6DoF pose estimator for an elongated surgical tool given its 3D mesh.

    Vertices live in a PCA-aligned frame (same order as columns of
    ``mesh_vertices``):
        axis 0  (red)   — primary / length
        axis 1  (green) — secondary / width
        axis 2  (blue)  — tertiary / thickness

    PnP matches the mask ``minAreaRect`` to the model midplane rectangle using
    many uniformly sampled perimeter points (see ``PNP_PER_EDGE``), seeded by
    IPPE on four corners then refined with ``SOLVEPNP_ITERATIVE``.
    """

    def __init__(self, mesh):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        cen = verts.mean(0)
        v = verts - cen
        cov = (v.T @ v) / len(v)
        eigvals, evecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]   # longest axis first
        aligned = v @ evecs[:, order]
        self.mesh_vertices = aligned.astype(np.float64)
        self.mesh_faces = faces

        # Half-extents: 0=primary(length), 1=secondary(width), 2=tertiary(thickness)
        hP = (aligned[:, 0].max() - aligned[:, 0].min()) / 2
        hS = (aligned[:, 1].max() - aligned[:, 1].min()) / 2
        hT = (aligned[:, 2].max() - aligned[:, 2].min()) / 2
        self.hP, self.hS, self.hT = hP, hS, hT
        self.extents = np.array([2 * hP, 2 * hS, 2 * hT])

        # ── PnP model points (same 3D basis as mesh_vertices) ────────────
        # Midplane at thickness=0; corners span primary (±hP) × secondary (±hS).
        self.model_pts = np.array(
            [
                [-hP, -hS, 0.0],
                [hP, -hS, 0.0],
                [hP, hS, 0.0],
                [-hP, hS, 0.0],
            ],
            dtype=np.float64,
        )

        # ── Axis display points (origin + tips along mesh axes 0,1,2) ─────
        ax = hP * 1.0
        ay = hS * 1.5
        az = max(hT * 5.0, hP * 0.35)
        self.axis_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [ax, 0.0, 0.0],
                [0.0, ay, 0.0],
                [0.0, 0.0, az],
            ],
            dtype=np.float64,
        )
        self._pnp_n = 4 * PNP_PER_EDGE

    # ------------------------------------------------------------------
    def _pnp_best_pts(
        self,
        model_pts: np.ndarray,
        img_corners: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """
        Try all 4 cyclic shifts of model_pts → img_corners correspondences
        with IPPE + ITERATIVE.  Return (rvec, tvec, mean reprojection error).
        """
        best_rv, best_tv, best_err = None, None, np.inf
        for shift in range(4):
            pts = np.roll(model_pts, shift, axis=0)
            for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
                ok, rv, tv = cv2.solvePnP(pts, img_corners, K, dist, flags=flag)
                if not ok:
                    continue
                proj, _ = cv2.projectPoints(pts, rv, tv, K, dist)
                err = float(
                    np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_corners, axis=1))
                )
                if err < best_err:
                    best_err, best_rv, best_tv = err, rv, tv
        return best_rv, best_tv, best_err

    # ------------------------------------------------------------------
    def _pnp_best_dense(
        self,
        model_corners4: np.ndarray,
        img_corners4: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        prev_rvec: np.ndarray | None = None,
        prev_tvec: np.ndarray | None = None,
        prev_shift: int | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float, int | None]:
        """
        Many-point PnP on uniformly sampled mask/model rectangle perimeters.
        Solves directly on all sampled correspondences (32 by default).
        """
        n = self._pnp_n
        img_n = _sample_quad_perimeter(img_corners4.astype(np.float64), n)
        model_n0 = _sample_quad_perimeter(model_corners4.astype(np.float64), n)
        step = n // 4
        best_rv, best_tv, best_err, best_shift = None, None, np.inf, None
        best_score = np.inf
        for shift in range(4):
            model_n = np.roll(model_n0, shift * step, axis=0)
            ok, rv, tv = cv2.solvePnP(
                model_n,
                img_n,
                K,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                ok_epnp, rv_epnp, tv_epnp = cv2.solvePnP(
                    model_n,
                    img_n,
                    K,
                    dist,
                    flags=cv2.SOLVEPNP_EPNP,
                )
                if not ok_epnp:
                    continue
                ok_refine, rv, tv = cv2.solvePnP(
                    model_n,
                    img_n,
                    K,
                    dist,
                    rv_epnp,
                    tv_epnp,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok_refine:
                    rv, tv = rv_epnp, tv_epnp
            proj, _ = cv2.projectPoints(model_n, rv, tv, K, dist)
            err = float(
                np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_n, axis=1))
            )
            score = err
            if prev_rvec is not None and prev_tvec is not None:
                d_rot = _rotation_delta_deg(rv, prev_rvec)
                z_ref = max(abs(float(prev_tvec[2, 0])), 1e-6)
                d_t = float(np.linalg.norm(tv.reshape(3) - prev_tvec.reshape(3)) / z_ref)
                score += PNP_ROT_SMOOTH_W * d_rot + PNP_TRANS_SMOOTH_W * d_t
            if prev_shift is not None and shift != int(prev_shift):
                score += PNP_SHIFT_PENALTY
            if score < best_score:
                best_score = score
                best_err, best_rv, best_tv, best_shift = err, rv, tv, shift
        return best_rv, best_tv, best_err, best_shift

    # ------------------------------------------------------------------
    def estimate_pose(
        self,
        mask_bool: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        state: dict | None,
        kalman_process_var: float = KALMAN_PROCESS_VAR,
        kalman_meas_var: float = KALMAN_MEAS_VAR,
    ) -> dict:
        """
        Estimate 6DoF pose from a binary mask.  Updates and returns the
        per-object state dict (keys: 'rvec', 'tvec', optional 'reg_sign').

        PCA eigenvectors are sign-ambiguous; the first successful solve tests
        all eight ``(±1,±1,±1)`` axis flips on model/mesh, picks the pose with
        best mask–mesh projection IoU, then fixes ``reg_sign`` for later frames.
        """
        if state is None:
            state = {}
        if not np.any(mask_bool):
            return state

        clean = cv2.erode(mask_bool.astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
        ys, xs = np.where(clean)
        if len(xs) < 20:
            return state

        p32 = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        rect = cv2.minAreaRect(p32.reshape(-1, 1, 2))
        img_corners = _order_corners_clockwise(cv2.boxPoints(rect).astype(np.float64))

        if "reg_sign" not in state:
            best_key: tuple[float, float] = (-1.0, np.inf)  # (iou, repro_err) maximize iou then min err
            best_rv, best_tv, best_s, best_shift = None, None, None, None
            for bits in range(8):
                s = np.array(
                    [
                        -1.0 if (bits >> 0) & 1 else 1.0,
                        -1.0 if (bits >> 1) & 1 else 1.0,
                        -1.0 if (bits >> 2) & 1 else 1.0,
                    ],
                    dtype=np.float64,
                )
                model_s = self.model_pts * s
                verts_s = self.mesh_vertices * s
                rv, tv, err, shift = self._pnp_best_dense(model_s, img_corners, K, dist)
                if rv is None:
                    continue
                iou = _mesh_projection_iou(
                    mask_bool, verts_s, self.mesh_faces, rv, tv, K, dist
                )
                key = (iou, err)
                if key[0] > best_key[0] or (
                    key[0] == best_key[0] and key[1] < best_key[1]
                ):
                    best_key = (key[0], key[1])
                    best_rv, best_tv, best_s = rv, tv, s.copy()
                    best_shift = shift
            if best_rv is None or best_s is None:
                return state
            state["reg_sign"] = best_s
            rv, tv = best_rv, best_tv
            state["pnp_shift"] = 0 if best_shift is None else int(best_shift)
        else:
            s = _reg_sign_from_state(state)
            model_s = self.model_pts * s
            prev_rv = state.get("rvec_raw")
            prev_tv = state.get("tvec_raw")
            if prev_rv is None:
                prev_rv = state.get("rvec")
            if prev_tv is None:
                prev_tv = state.get("tvec")
            rv, tv, _, shift = self._pnp_best_dense(
                model_s,
                img_corners,
                K,
                dist,
                prev_rvec=prev_rv,
                prev_tvec=prev_tv,
                prev_shift=state.get("pnp_shift"),
            )
            if rv is None:
                return state
            if shift is not None:
                state["pnp_shift"] = int(shift)

        state["rvec_raw"] = rv.copy()
        state["tvec_raw"] = tv.copy()

        rv, tv = _apply_kalman_pose_filter(
            state,
            rv,
            tv,
            kalman_process_var,
            kalman_meas_var,
        )
        state["rvec"] = rv
        state["tvec"] = tv
        return state


def _load_mesh_estimators() -> dict[int, MeshPoseEstimator]:
    """Load per-object GLB meshes with fixed mapping: ID1/2/3 -> object_0/1/2.glb."""
    estimators: dict[int, MeshPoseEstimator] = {}
    if trimesh is None:
        return estimators
    base = Path(__file__).parent
    id_to_glb = {
        1: base / "object_0.glb",
        2: base / "object_1.glb",
        3: base / "object_2.glb",
    }
    for obj_id, p in id_to_glb.items():
        if not p.exists():
            print(f"Missing mesh for ID{obj_id}: {p.name}")
            continue
        try:
            mesh = trimesh.load(str(p), force="mesh")
            estimators[obj_id] = MeshPoseEstimator(mesh)
        except Exception as e:
            print(f"Failed to load {p.name} for ID{obj_id}: {e}")
    return estimators


def _draw_pose_axes(
    vis: np.ndarray,
    state: dict,
    K: np.ndarray,
    dist: np.ndarray,
    axis_pts: np.ndarray,
    obj_id: int,
) -> None:
    """Project and draw X(red)/Y(green)/Z(blue) pose axes."""
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None:
        return

    s = _reg_sign_from_state(state)
    axis_use = (axis_pts * s).astype(np.float64)
    proj, _ = cv2.projectPoints(axis_use, rvec, tvec, K, dist)
    pts2d = proj.reshape(-1, 2)

    fh, fw = vis.shape[:2]

    def clip_pt(p: np.ndarray) -> tuple[int, int]:
        return (int(np.clip(p[0], 0, fw - 1)), int(np.clip(p[1], 0, fh - 1)))

    origin = clip_pt(pts2d[0])
    x_tip  = clip_pt(pts2d[1])
    y_tip  = clip_pt(pts2d[2])
    z_tip  = clip_pt(pts2d[3])

    cv2.arrowedLine(vis, origin, x_tip, (0, 0, 220),   2, tipLength=0.20, line_type=cv2.LINE_AA)
    cv2.arrowedLine(vis, origin, y_tip, (0, 200, 0),   2, tipLength=0.20, line_type=cv2.LINE_AA)
    cv2.arrowedLine(vis, origin, z_tip, (220, 80, 0),  2, tipLength=0.20, line_type=cv2.LINE_AA)

    fnt = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, "X", (x_tip[0] + 4, x_tip[1] + 4), fnt, 0.50, (0, 0, 220),  1, cv2.LINE_AA)
    cv2.putText(vis, "Y", (y_tip[0] + 4, y_tip[1] + 4), fnt, 0.50, (0, 200, 0),  1, cv2.LINE_AA)
    cv2.putText(vis, "Z", (z_tip[0] + 4, z_tip[1] + 4), fnt, 0.50, (220, 80, 0), 1, cv2.LINE_AA)


def _draw_pose_hud(vis: np.ndarray, pose_states: dict[int, dict]) -> None:
    """Draw fixed-corner R/T readouts for all objects."""
    if not pose_states:
        return
    x0, y0 = 12, 18
    line_h = 18
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    for row, oid in enumerate(sorted(pose_states)):
        st = pose_states.get(oid, {})
        rvec = st.get("rvec")
        tvec = st.get("tvec_cal")
        if tvec is None:
            tvec = st.get("tvec")
        if rvec is None or tvec is None:
            continue
        # Euler angles (ZYX) for readout
        R, _ = cv2.Rodrigues(rvec)
        sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        if sy > 1e-6:
            rx = np.degrees(np.arctan2(float(R[2, 1]), float(R[2, 2])))
            ry = np.degrees(np.arctan2(-float(R[2, 0]), sy))
            rz = np.degrees(np.arctan2(float(R[1, 0]), float(R[0, 0])))
        else:
            rx = np.degrees(np.arctan2(-float(R[1, 2]), float(R[1, 1])))
            ry = np.degrees(np.arctan2(-float(R[2, 0]), sy))
            rz = 0.0
        tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
        text = f"ID{oid}  R({rx:.0f},{ry:.0f},{rz:.0f})  T({tx:.1f},{ty:.1f},{tz:.1f})"
        yy = y0 + row * line_h
        cv2.putText(vis, text, (x0, yy), fnt, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, text, (x0, yy), fnt, 0.48, (0, 0, 0), 1, cv2.LINE_AA)


def _update_translation_calibration_from_surface(
    state: dict,
    surface_distance_cm: float,
    ema_alpha: float = 0.15,
) -> None:
    """
    Build a calibrated translation readout from known camera->surface distance.

    This keeps raw ``tvec`` untouched for rendering/projection consistency and
    stores a calibrated copy in ``tvec_cal`` for HUD/CSV readouts.
    """
    if surface_distance_cm <= 0:
        return
    tvec = state.get("tvec")
    if tvec is None:
        return

    tz_raw = abs(float(tvec[2]))
    if tz_raw < 1e-9:
        return

    scale_now = float(surface_distance_cm) / tz_raw
    prev = state.get("tvec_cal_scale")
    if prev is None:
        scale = scale_now
    else:
        scale = (1.0 - ema_alpha) * float(prev) + ema_alpha * scale_now

    state["tvec_cal_scale"] = scale
    state["tvec_cal"] = np.asarray(tvec, dtype=np.float64) * scale


def _pose_to_euler_zyx_deg(rvec: np.ndarray) -> tuple[float, float, float]:
    R, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        rx = float(np.degrees(np.arctan2(float(R[2, 1]), float(R[2, 2]))))
        ry = float(np.degrees(np.arctan2(-float(R[2, 0]), sy)))
        rz = float(np.degrees(np.arctan2(float(R[1, 0]), float(R[0, 0]))))
    else:
        rx = float(np.degrees(np.arctan2(-float(R[1, 2]), float(R[1, 1]))))
        ry = float(np.degrees(np.arctan2(-float(R[2, 0]), sy)))
        rz = 0.0
    return rx, ry, rz


def _next_pose_csv_path(base_dir: Path) -> Path:
    """Return first available posesN.csv path in base_dir."""
    i = 1
    while True:
        p = base_dir / f"poses{i}.csv"
        if not p.exists():
            return p
        i += 1


def _draw_registration_debug(
    canvas: np.ndarray,
    mask_bool: np.ndarray,
    state: dict,
    est: MeshPoseEstimator,
    K: np.ndarray,
    dist: np.ndarray,
    obj_id: int,
) -> None:
    """Visualize SAM mask vs projected GLB registration for one object.

    Note: live pose is rigid (rotation + translation). No non-rigid stretch
    is applied in this real-time path.
    """
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None or not np.any(mask_bool):
        return

    fh, fw = canvas.shape[:2]
    pred_mask = np.zeros((fh, fw), dtype=np.uint8)

    # Project and draw full registered mesh faces as a wireframe/fill overlay.
    s = _reg_sign_from_state(state)
    verts = est.mesh_vertices * s
    faces = est.mesh_faces
    proj_mesh, _ = cv2.projectPoints(verts, rvec, tvec, K, dist)
    pts2d = proj_mesh.reshape(-1, 2)
    overlay = canvas.copy()
    for f in faces:
        poly = np.round(pts2d[f]).astype(np.int32)
        cv2.fillConvexPoly(overlay, poly, (180, 130, 255), lineType=cv2.LINE_AA)  # light magenta fill
        cv2.polylines(canvas, [poly], True, (255, 0, 255), 1, lineType=cv2.LINE_AA)  # magenta edges
        cv2.fillConvexPoly(pred_mask, poly, 255, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0, dst=canvas)

    # Draw SAM mask overlay in cyan.
    sam_mask_u8 = (mask_bool.astype(np.uint8) * 255)
    sam_color = np.zeros_like(canvas)
    sam_color[:, :, 1] = sam_mask_u8
    sam_color[:, :, 0] = sam_mask_u8
    cv2.addWeighted(sam_color, 0.30, canvas, 1.0, 0.0, dst=canvas)

    # 2D min-area rectangle around the full projected mesh (mesh footprint in image).
    rect_mesh = cv2.minAreaRect(pts2d.astype(np.float32))
    box_mesh = cv2.boxPoints(rect_mesh).astype(np.int32)
    cv2.polylines(canvas, [box_mesh], True, (0, 255, 255), 2, lineType=cv2.LINE_AA)  # yellow = mesh 2D bbox

    # Model midplane quad used by PnP (not the same as full mesh silhouette).
    proj, _ = cv2.projectPoints(est.model_pts * s, rvec, tvec, K, dist)
    poly = np.round(proj.reshape(-1, 2)).astype(np.int32)
    if poly.shape[0] >= 3:
        cv2.polylines(canvas, [poly], True, (255, 255, 255), 2, lineType=cv2.LINE_AA)  # white = PnP quad

    # Visualize 32 contour points on object mask and projected mesh silhouette.
    n = int(est._pnp_n)
    mask_pts = _sample_mask_contour((mask_bool.astype(np.uint8) * 255), n)
    mesh_pts = _sample_mask_contour(pred_mask, n)
    if mask_pts is not None and mesh_pts is not None:
        # Cyan = object mask contour samples, Magenta = projected mesh contour samples.
        for p in mask_pts:
            u, v = int(round(float(p[0]))), int(round(float(p[1])))
            cv2.circle(canvas, (u, v), 2, (255, 255, 0), -1, lineType=cv2.LINE_AA)
        for p in mesh_pts:
            u, v = int(round(float(p[0]))), int(round(float(p[1])))
            cv2.circle(canvas, (u, v), 2, (255, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"Contour pts={n}",
            (12, 20 + 18 * max(0, obj_id - 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Contour pts={n}",
            (12, 20 + 18 * max(0, obj_id - 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # IoU between projected model polygon and SAM mask.
    pred_b = pred_mask > 0
    sam_b = mask_bool
    inter = float(np.logical_and(pred_b, sam_b).sum())
    union = float(np.logical_or(pred_b, sam_b).sum())
    iou = inter / union if union > 0.0 else 0.0

    ys, xs = np.where(mask_bool)
    if len(xs) > 0:
        tx, ty = int(xs.mean()), int(ys.mean())
    else:
        tx, ty = 12, 24
    text = f"ID{obj_id} reg IoU={iou:.3f} (rigid)"
    cv2.putText(canvas, text, (tx + 8, ty + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (tx + 8, ty + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Live camera frame provider
# ---------------------------------------------------------------------------

class LiveFrameProvider:
    """Always-latest frame provider for real-time inference.

    The capture thread continuously overwrites a single 'latest' slot.
    When propagate_in_video requests frame idx, it receives whatever the
    newest camera frame is at that moment — no queue ever builds up.
    Each idx is cached once so get_raw(idx) returns the same frame the
    model saw.
    """

    def __init__(self, cap: cv2.VideoCapture, image_size: int, rotate_180: bool):
        self.cap = cap
        self.image_size = image_size
        self.rotate_180 = rotate_180
        # Rolling cache: model_idx -> (tensor, raw). Bounded to last 32 frames.
        self._cache: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
        self._latest_tensor: torch.Tensor | None = None
        self._latest_raw: np.ndarray | None = None
        self._lock = threading.Lock()

    def _encode(self, frame: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        frame = preprocess(frame, self.rotate_180)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(
            cv2.resize(rgb, (self.image_size, self.image_size))
        ).float().div(255.0).permute(2, 0, 1)
        return (t - _IMG_MEAN) / _IMG_STD, frame

    def capture_next(self) -> bool:
        # Drain any buffered camera frames so we always get the newest one.
        frame = None
        for _ in range(4):
            ok, f = self.cap.read()
            if ok:
                frame = f
        if frame is None:
            return False
        t, raw = self._encode(frame)
        with self._lock:
            self._latest_tensor = t
            self._latest_raw = raw
        return True

    def __len__(self):
        return 1_000_000  # always tell SAM2 there are more frames

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Block until the camera has produced at least one frame.
        while True:
            with self._lock:
                if self._latest_tensor is not None:
                    if idx not in self._cache:
                        # Snapshot the current latest for this model index.
                        self._cache[idx] = (self._latest_tensor, self._latest_raw)
                        # Evict entries older than 32 frames to bound memory.
                        if len(self._cache) > 32:
                            oldest = min(self._cache)
                            del self._cache[oldest]
                    return self._cache[idx][0]
            time.sleep(0.001)

    def get_raw(self, idx: int) -> np.ndarray:
        with self._lock:
            entry = self._cache.get(idx)
            return entry[1] if entry is not None else self._latest_raw

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args) -> None:
    device = choose_device(args.device)
    print(f"Device: {device}")

    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT}")
        print("Expected: EdgeTAM/checkpoints/edgetam.pt")
        return

    print("Loading EdgeTAM …")
    predictor = _load_predictor(device)
    mesh_estimators = _load_mesh_estimators()
    if trimesh is None:
        print("`trimesh` not installed; mesh-based pose overlays unavailable.")
    elif mesh_estimators:
        loaded = ", ".join(f"ID{k}->object_{k-1}" for k in sorted(mesh_estimators))
        print(f"Pose estimators loaded for {loaded}")
    else:
        print("No mesh files found; pose overlays disabled.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize camera-side frame queue
    rotate_180 = detect_orbbec_camera(args.camera)
    print(f"Camera {args.camera}: {'Orbbec detected, rotating 180°' if rotate_180 else 'non-Orbbec, no rotation'}")

    # Resolve intrinsics for pose (file > Orbbec SDK > FOV estimate fallback).
    K_cam, dist_cam, intr_src = _resolve_pose_intrinsics(
        cap,
        TARGET_SIZE[0],
        TARGET_SIZE[1],
        args.intrinsics_file,
        not args.no_orbbec_intrinsics,
    )
    print(
        f"Camera intrinsics ({intr_src}): "
        f"fx={K_cam[0, 0]:.1f}  fy={K_cam[1, 1]:.1f}  "
        f"cx={K_cam[0, 2]:.1f}  cy={K_cam[1, 2]:.1f}"
    )
    if dist_cam.size > 0:
        dflat = dist_cam.reshape(-1)
        preview = ", ".join(f"{x:.4g}" for x in dflat[:5])
        print(f"Distortion coeffs ({len(dflat)}): [{preview}{' ...' if len(dflat) > 5 else ''}]")

    image_size = predictor.image_size
    provider = LiveFrameProvider(cap, image_size, rotate_180)

    if not provider.capture_next():
        print("No frame from camera.")
        cap.release()
        return

    stop_flag = threading.Event()

    def _capture_loop():
        while not stop_flag.is_set():
            if not provider.capture_next():
                stop_flag.set()

    threading.Thread(target=_capture_loop, daemon=True).start()

    points, seed_frame = pick_points_live(provider, stop_flag)
    if not points:
        print("No points selected. Exiting.")
        stop_flag.set()
        cap.release()
        return
    if seed_frame is None:
        print("No seed frame available. Exiting.")
        stop_flag.set()
        cap.release()
        return

    # Init EdgeTAM state from single-frame temp folder, then swap in live provider
    tmp = tempfile.mkdtemp(prefix="edgetam_live_")
    cv2.imwrite(os.path.join(tmp, "000000.jpg"), seed_frame)
    state = predictor.init_state(tmp, async_loading_frames=False)
    shutil.rmtree(tmp, ignore_errors=True)

    state["images"] = provider
    state["num_frames"] = 1_000_000

    for obj_id, x, y in points:
        predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=int(obj_id),
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

    writer = None
    if args.output:
        h, w = TARGET_SIZE[1], TARGET_SIZE[0]
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    csv_path = _next_pose_csv_path(Path.cwd())
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    t_cols = ["tx", "ty", "tz"]
    if args.surface_distance_cm > 0:
        t_cols = ["tx_cm", "ty_cm", "tz_cm"]
    csv_writer.writerow(["frame_idx", "time_s", "object_id", "rx_deg", "ry_deg", "rz_deg", *t_cols])
    print(f"Pose CSV export: {csv_path}")
    if args.surface_distance_cm > 0:
        print(
            f"Translation calibration enabled: using camera->surface distance "
            f"{args.surface_distance_cm:.2f} cm for T readouts/CSV."
        )

    com_trails: dict[int, list[tuple[int, int]]] = {}  # obj_id -> list of (x, y) COM positions
    pose_states: dict[int, dict] = {}          # per-obj state for MeshPoseEstimator
    fps_t0 = time.perf_counter()
    fps_frames = 0
    ac_device, ac_dtype, ac_enabled = _autocast_config(device, args.half)

    print("Live tracking started. Press 'q' or ESC to stop.")
    try:
        with torch.autocast(device_type=ac_device, dtype=ac_dtype, enabled=ac_enabled):
            for fi, obj_ids, masks in predictor.propagate_in_video(state):
                frame = provider.get_raw(fi)
                debug_canvas = frame.copy() if args.align_debug_out else None
                if hasattr(obj_ids, "tolist"):
                    ids = [int(x) for x in obj_ids.tolist()]
                else:
                    ids = [int(x) for x in obj_ids]

                vis = overlay_masks(frame, ids, masks, alpha=args.alpha)
                fh, fw = frame.shape[:2]
                masks_np = masks.detach().cpu().numpy()
                for i in range(min(len(ids), masks_np.shape[0])):
                    oid = ids[i]
                    binm = _mask_to_2d_bool(masks_np[i], fh, fw)

                    if np.any(binm):
                        ys, xs = np.where(binm)
                        cx, cy = int(xs.mean()), int(ys.mean())
                        if oid not in com_trails:
                            com_trails[oid] = []
                        com_trails[oid].append((cx, cy))

                        est = mesh_estimators.get(oid)
                        if est is not None:
                            # Full 6DoF pose estimation + XYZ axis overlay
                            pose_states[oid] = est.estimate_pose(
                                binm,
                                K_cam,
                                dist_cam,
                                pose_states.get(oid),
                                kalman_process_var=args.kalman_process_var,
                                kalman_meas_var=args.kalman_meas_var,
                            )
                            if args.surface_distance_cm > 0:
                                _update_translation_calibration_from_surface(
                                    pose_states[oid], args.surface_distance_cm
                                )
                            _draw_pose_axes(
                                vis, pose_states[oid], K_cam, dist_cam,
                                est.axis_pts, oid,
                            )
                            if debug_canvas is not None:
                                _draw_registration_debug(
                                    debug_canvas, binm, pose_states[oid], est, K_cam, dist_cam, oid
                                )
                        else:
                            continue

                # Draw COM trails
                for oid, trail in com_trails.items():
                    col = point_color(oid)
                    for j in range(1, len(trail)):
                        cv2.line(vis, trail[j - 1], trail[j], col, 1, lineType=cv2.LINE_AA)
                    if trail:
                        cv2.circle(vis, trail[-1], 5, col, -1)
                        cv2.circle(vis, trail[-1], 7, (255, 255, 255), 1)
                        cv2.putText(
                            vis,
                            f"ID{oid}",
                            (trail[-1][0] + 8, trail[-1][1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            vis,
                            f"ID{oid}",
                            (trail[-1][0] + 8, trail[-1][1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            col,
                            1,
                            cv2.LINE_AA,
                        )

                _draw_pose_hud(vis, pose_states)

                if fi == 0:
                    for oid, px, py in points:
                        cv2.circle(vis, (int(px), int(py)), 5, (0, 255, 255), -1)
                        cv2.putText(vis, f"ID{oid}", (int(px) + 8, int(py) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                cv2.imshow("EdgeTAM Live", vis)
                if writer is not None:
                    writer.write(vis)
                if debug_canvas is not None:
                    cv2.imwrite(args.align_debug_out, debug_canvas)
                t_s = float(time.perf_counter())
                for oid in sorted(pose_states):
                    st = pose_states.get(oid, {})
                    rvec = st.get("rvec")
                    tvec = st.get("tvec_cal")
                    if tvec is None:
                        tvec = st.get("tvec")
                    if rvec is None or tvec is None:
                        continue
                    rx, ry, rz = _pose_to_euler_zyx_deg(rvec)
                    tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
                    csv_writer.writerow([int(fi), t_s, int(oid), rx, ry, rz, tx, ty, tz])

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if stop_flag.is_set():
                    break
                fps_frames += 1
                now = time.perf_counter()
                dt = now - fps_t0
                if dt >= 1.0:
                    print(f"FPS: {fps_frames / dt:.2f}")
                    fps_t0 = now
                    fps_frames = 0
    finally:
        stop_flag.set()
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time EdgeTAM tracking on Orbbec RGB stream.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--alpha", type=float, default=0.45,
                        help="Mask overlay alpha")
    parser.add_argument("--kalman-process-var", type=float, default=KALMAN_PROCESS_VAR,
                        help="Kalman process variance Q (higher = more responsive, less smooth)")
    parser.add_argument("--kalman-meas-var", type=float, default=KALMAN_MEAS_VAR,
                        help="Kalman measurement variance R (higher = smoother, slower)")
    parser.add_argument("--half", action="store_true", default=True,
                        help="Use half-precision autocast (default: on)")
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--output", default="",
                        help="Optional path to save output video (e.g. out.mp4)")
    parser.add_argument(
        "--intrinsics-file",
        default="",
        help=(
            "Optional .npz intrinsics file for pose calibration. "
            "Expected keys: K (or camera_matrix/intrinsics), optional dist/dist_coeffs, "
            "optional width,height."
        ),
    )
    parser.add_argument(
        "--no-orbbec-intrinsics",
        action="store_true",
        help="Disable attempting to read Orbbec SDK color intrinsics before fallback estimate.",
    )
    parser.add_argument(
        "--surface-distance-cm",
        type=float,
        default=0.0,
        help=(
            "Optional known camera-to-surface distance in cm. "
            "When > 0, T(x,y,z) HUD/CSV are scaled to this reference."
        ),
    )
    parser.add_argument(
        "--align-debug-out",
        default="",
        help="Optional path to save registration-debug image (SAM mask vs projected GLB) each frame.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
