#!/usr/bin/env python3
"""
Real-time surgical tool tracking with EdgeTAM.

Opens the Orbbec RGB camera (or any camera index), lets you click seed
points on the first frame, then streams live masks + rigidly registered solid mesh overlays.

Usage:
    .venv/bin/python live_track.py                        # camera 0, default settings
    .venv/bin/python live_track.py --camera 1             # different camera index
    .venv/bin/python live_track.py --no-half              # disable half-precision
    .venv/bin/python live_track.py --output out.mp4       # save output video
Keybindings:
    Enter      Confirm seed points and start tracking
    q / ESC    Quit
"""

import argparse
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

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

try:
    from scipy.optimize import minimize  # type: ignore
except Exception:
    minimize = None

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


def get_mask_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def overlay_masks(frame, obj_ids, masks, alpha=1.0):
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
    Fixed approximate Orbbec intrinsics (no checkerboard/depth calibration).

    Baseline is defined for 640x360 and scaled linearly for other target sizes:
      fx=388, fy=300, cx=320, cy=180.
    """
    _ = cap  # kept for signature compatibility
    base_w, base_h = 640.0, 360.0
    base_fx, base_fy = 388.0, 300.0
    base_cx, base_cy = 320.0, 180.0

    sx = float(target_w) / base_w
    sy = float(target_h) / base_h
    fx = base_fx * sx
    fy = base_fy * sy
    cx = base_cx * sx
    cy = base_cy * sy

    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
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


# Legacy PnP pose path removed; rigid SE(3) mesh↔mask IoU registration below.
PNP_PER_EDGE = 8
_REG_NEIGHBORHOOD_W = 0.02
_REG_MAXITER_FIRST = 40
_REG_MAXITER_TRACK = 22
# Conic-section refinement: mask ≈ ellipse (plane cut); mesh ≈ slender “cone”.
# Grid half-range (rad) for tilt about camera X/Y; dz steps along optical axis.
_CONIC_TILT_HALF_RAD = 0.26
_CONIC_GRID_TILT = 7
_CONIC_DZ_STEPS = 5
_CONIC_DZ_FRAC_OF_EXTENT = 0.12


class ToolMesh:
    """PCA-aligned tool mesh for projection and rigid registration (no PnP pose)."""

    def __init__(self, mesh):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        cen = verts.mean(0)
        v = verts - cen
        cov = (v.T @ v) / len(v)
        eigvals, evecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        aligned = v @ evecs[:, order]
        self.mesh_vertices = aligned.astype(np.float64)
        self.mesh_faces = faces

        hP = (aligned[:, 0].max() - aligned[:, 0].min()) / 2
        hS = (aligned[:, 1].max() - aligned[:, 1].min()) / 2
        hT = (aligned[:, 2].max() - aligned[:, 2].min()) / 2
        self.hP, self.hS, self.hT = hP, hS, hT
        self.extents = np.array([2 * hP, 2 * hS, 2 * hT], dtype=np.float64)

        self.model_pts = np.array(
            [
                [-hP, -hS, 0.0],
                [hP, -hS, 0.0],
                [hP, hS, 0.0],
                [-hP, hS, 0.0],
            ],
            dtype=np.float64,
        )
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
        self._pnp_n = max(32, 4 * PNP_PER_EDGE)


def _bits_from_reg_sign(s: np.ndarray) -> int:
    s = np.asarray(s, dtype=np.float64).reshape(3)
    bits = 0
    for j in range(3):
        if s[j] < 0:
            bits |= 1 << j
    return bits


def _reg_sign_bits_from_index(bits: int) -> np.ndarray:
    return np.array(
        [
            -1.0 if (bits >> 0) & 1 else 1.0,
            -1.0 if (bits >> 1) & 1 else 1.0,
            -1.0 if (bits >> 2) & 1 else 1.0,
        ],
        dtype=np.float64,
    )


def _mask_ellipse_params(mask_bool: np.ndarray) -> dict | None:
    """Fit OpenCV ellipse to largest mask contour; return center, semi-axes, angle, eccentricity."""
    m = (mask_bool.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cnt is None or len(cnt) < 5:
        return None
    (ex, ey), (ma_ax, mi_ax), ang = cv2.fitEllipse(cnt)
    maj = float(max(ma_ax, mi_ax) * 0.5)
    minr = float(min(ma_ax, mi_ax) * 0.5)
    ecc = float(np.sqrt(max(0.0, 1.0 - (minr / max(maj, 1e-9)) ** 2)))
    # OpenCV: ``angle`` is for the *first* tuple axis (``ma_ax``). If ``mi_ax > ma_ax``,
    # the longer physical axis is perpendicular to ``angle`` in the image plane.
    major_axis_deg = float(ang if ma_ax >= mi_ax else ang + 90.0)
    return {
        "center": (float(ex), float(ey)),
        "semi_major": maj,
        "semi_minor": minr,
        "angle_deg": float(ang),
        "major_axis_deg": major_axis_deg,
        "eccentricity": ecc,
    }


def _skew_sym(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]],
        dtype=np.float64,
    )


def _rotmat_align_unit_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R with R @ a_hat ≈ b_hat (OpenCV object→camera column convention)."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-12 or bn < 1e-12:
        return np.eye(3, dtype=np.float64)
    a = a / an
    b = b / bn
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-10:
        if c > 0.999999:
            return np.eye(3, dtype=np.float64)
        t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(a, t))) > 0.85:
            t = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(a, t)
        axis /= float(np.linalg.norm(axis) + 1e-12)
        r180 = (np.pi * axis).reshape(3, 1)
        R180, _ = cv2.Rodrigues(r180.astype(np.float64))
        return R180.astype(np.float64)
    axis = (v / s).astype(np.float64)
    theta = float(np.arctan2(s, c))
    K = _skew_sym(axis)
    return (np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)).astype(np.float64)


def _mask_major_axis_unit_cam(
    K: np.ndarray,
    center_xy: tuple[float, float],
    major_axis_deg: float,
    z_depth: float,
    half_span_px: float = 80.0,
) -> np.ndarray:
    """Unit direction in camera frame along the mask ellipse **long** axis (two-point back-project)."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = max(float(z_depth), 1e-4)
    th = np.deg2rad(float(major_axis_deg))
    du, dv = float(np.cos(th)), float(np.sin(th))
    ex, ey = float(center_xy[0]), float(center_xy[1])
    p1 = np.array([ex + half_span_px * du, ey + half_span_px * dv], dtype=np.float64)
    p0 = np.array([ex - half_span_px * du, ey - half_span_px * dv], dtype=np.float64)

    def _bp(u: float, v: float) -> np.ndarray:
        return np.array([(u - cx) / fx * z, (v - cy) / fy * z, z], dtype=np.float64)

    d = _bp(float(p1[0]), float(p1[1])) - _bp(float(p0[0]), float(p0[1]))
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return (d / n).astype(np.float64)


def _p0_major_axis_seed(
    est: ToolMesh,
    mask_bool: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    s: np.ndarray,
    t_init: np.ndarray,
    z0: float,
) -> np.ndarray | None:
    """
    Seed [rvec; tvec]: align the mesh PCA longest axis (±X in signed object frame)
    with the 2D ellipse major axis (back-projected to camera 3D), trying both flips.
    """
    ell = _mask_ellipse_params(mask_bool)
    if ell is None:
        return None
    maj_deg = float(ell.get("major_axis_deg", ell["angle_deg"]))
    target = _mask_major_axis_unit_cam(K, ell["center"], maj_deg, z0)
    sx = float(np.asarray(s, dtype=np.float64).reshape(3)[0])
    tv = np.asarray(t_init, dtype=np.float64).reshape(3, 1)
    best_p0: np.ndarray | None = None
    best_iou = -1.0
    for flip in (1.0, -1.0):
        d_obj = np.array([sx * flip, 0.0, 0.0], dtype=np.float64)
        dn = float(np.linalg.norm(d_obj))
        if dn < 1e-12:
            continue
        d_obj = d_obj / dn
        R_align = _rotmat_align_unit_vectors(d_obj, target)
        # Roll about mesh long axis (object +X): fixes ~90° silhouette ambiguity after axis match.
        for k in range(4):
            psi = float(k) * (0.5 * np.pi)
            r_tw, _ = cv2.Rodrigues(np.array([[psi], [0.0], [0.0]], dtype=np.float64))
            R = (R_align @ r_tw).astype(np.float64)
            rvec, _ = cv2.Rodrigues(R)
            rv = rvec.astype(np.float64)
            verts_s = est.mesh_vertices * s.reshape(1, 3)
            iou = _mesh_projection_iou(mask_bool, verts_s, est.mesh_faces, rv, tv, K, dist)
            p_try = np.concatenate([rv.reshape(3), tv.reshape(3)])
            if iou > best_iou:
                best_iou = iou
                best_p0 = p_try
    return best_p0


def _register_rigid_se3_mesh_to_mask(
    est: ToolMesh,
    mask_bool: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    state: dict | None,
) -> dict:
    """Rigid SE(3): maximize mask IoU vs solid projected mesh (Nelder–Mead)."""
    if state is None:
        state = {}
    if minimize is None or not np.any(mask_bool):
        if minimize is None and not getattr(_register_rigid_se3_mesh_to_mask, "_warned", False):
            print("scipy not installed; install scipy for rigid mesh–mask registration.")
            setattr(_register_rigid_se3_mesh_to_mask, "_warned", True)
        return state

    ys, xs = np.where(mask_bool)
    my, mx = float(ys.mean()), float(xs.mean())
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    span = float(np.linalg.norm(est.extents))
    Z0 = float(np.clip(span * 1.4, 0.12, 3.5))
    t_init = np.array([(mx - cx) / fx * Z0, (my - cy) / fy * Z0, Z0], dtype=np.float64)

    prev_r = state.get("rvec")
    prev_t = state.get("tvec")
    if prev_r is not None and prev_t is not None:
        p_base = np.concatenate(
            [
                np.asarray(prev_r, dtype=np.float64).reshape(3),
                np.asarray(prev_t, dtype=np.float64).reshape(3),
            ]
        )
    else:
        p_base = np.concatenate([np.zeros(3, dtype=np.float64), t_init])

    locked = state.get("reg_sign") is not None
    bits_list = [_bits_from_reg_sign(_reg_sign_from_state(state))] if locked else list(range(8))
    max_iter = _REG_MAXITER_TRACK if locked else _REG_MAXITER_FIRST

    best_iou = -1.0
    best_rv = None
    best_tv = None
    best_s: np.ndarray | None = None

    for bits in bits_list:
        s = _reg_sign_bits_from_index(int(bits))
        verts_s = est.mesh_vertices * s.reshape(1, 3)
        p0 = p_base.astype(np.float64).copy()
        if prev_r is None:
            p0[3:] = t_init
            seed = _p0_major_axis_seed(est, mask_bool, K, dist, s, t_init, Z0)
            if seed is not None:
                p0 = seed.astype(np.float64)

        def _obj(p: np.ndarray, p0_ref=p0, vs=verts_s) -> float:
            rv = np.asarray(p[:3], dtype=np.float64).reshape(3, 1)
            tv = np.asarray(p[3:], dtype=np.float64).reshape(3, 1)
            iou = _mesh_projection_iou(mask_bool, vs, est.mesh_faces, rv, tv, K, dist)
            reg = float(_REG_NEIGHBORHOOD_W) * float(np.sum((p - p0_ref) ** 2))
            return float(-iou + reg)

        try:
            res = minimize(
                _obj,
                p0,
                method="Nelder-Mead",
                options={"maxiter": int(max_iter), "xatol": 2e-3, "fatol": 2e-3},
            )
            p_opt = np.asarray(res.x, dtype=np.float64)
        except Exception:
            p_opt = p0

        rv = p_opt[:3].reshape(3, 1)
        tv = p_opt[3:].reshape(3, 1)
        iou = _mesh_projection_iou(mask_bool, verts_s, est.mesh_faces, rv, tv, K, dist)
        if iou > best_iou:
            best_iou, best_rv, best_tv, best_s = iou, rv.copy(), tv.copy(), s.copy()

    if best_rv is None or best_s is None:
        return state

    state["reg_sign"] = best_s
    state["rvec_raw"] = best_rv.copy()
    state["tvec_raw"] = best_tv.copy()
    state["rvec"] = best_rv
    state["tvec"] = best_tv
    return state


def _mesh_cone_aperture_rad(est: ToolMesh) -> float:
    """Crude semi-aperture: arctan(in-plane radius / thickness) for a cone-like tool."""
    r_in = float(np.hypot(est.hP, est.hS))
    h = max(float(est.hT), 1e-6)
    return float(np.arctan2(r_in, h))


def _R_cam_tilt_xyz(ax: float, ay: float, az: float) -> np.ndarray:
    """Rz(az) Ry(ay) Rx(ax) for small camera-frame tilts (radians)."""
    Rx, _ = cv2.Rodrigues(np.array([ax, 0.0, 0.0], dtype=np.float64).reshape(3, 1))
    Ry, _ = cv2.Rodrigues(np.array([0.0, ay, 0.0], dtype=np.float64).reshape(3, 1))
    Rz, _ = cv2.Rodrigues(np.array([0.0, 0.0, az], dtype=np.float64).reshape(3, 1))
    return (Rz @ Ry @ Rx).astype(np.float64)


def _compose_camera_wobble(
    rvec: np.ndarray,
    tvec: np.ndarray,
    ax: float,
    ay: float,
    az: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply camera-left rotation and axial shift: X' = R_w (R X + t) + (0,0,dz)."""
    Rm, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    Rw = _R_cam_tilt_xyz(ax, ay, az)
    R2 = Rw @ Rm
    t2 = Rw @ t + np.array([[0.0], [0.0], [dz]], dtype=np.float64)
    r2, _ = cv2.Rodrigues(R2)
    return r2.astype(np.float64), t2.astype(np.float64)


def _refine_conic_section_overlap(
    est: ToolMesh,
    mask_bool: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    state: dict,
) -> None:
    """
    Conic-section view: the 2D mask is summarized as an ellipse (eccentricity +
    axes from ``cv2.fitEllipse``); the tool is treated as a slender cone by a
    simple aperture heuristic. We search small tilts about camera X/Y (slice
    plane / viewing obliquity) and a shift along optical Z so the *solid*
    projection overlaps the mask most (same IoU as the main rigid objective).
    """
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None or not np.any(mask_bool):
        return

    ell = _mask_ellipse_params(mask_bool)
    ecc = float(ell["eccentricity"]) if ell is not None else 0.0
    state["mask_ellipse_ecc"] = ecc
    if ell is not None:
        state["mask_ellipse_axes"] = (ell["semi_major"], ell["semi_minor"])
        state["mask_ellipse_angle_deg"] = ell["angle_deg"]
        state["mask_ellipse_major_axis_deg"] = float(ell.get("major_axis_deg", ell["angle_deg"]))

    s = _reg_sign_from_state(state)
    verts_s = est.mesh_vertices * s.reshape(1, 3)

    base_iou = _mesh_projection_iou(mask_bool, verts_s, est.mesh_faces, rvec, tvec, K, dist)
    alpha = _mesh_cone_aperture_rad(est)
    state["mesh_cone_aperture_rad"] = alpha
    # Wider tilt search when the mask ellipse is more eccentric (sharper “slice”).
    half = float(_CONIC_TILT_HALF_RAD) * (0.65 + 0.55 * min(ecc, 0.98))
    ax_vals = np.linspace(-half, half, int(_CONIC_GRID_TILT))
    ay_vals = np.linspace(-half, half, int(_CONIC_GRID_TILT))
    span = float(np.linalg.norm(est.extents))
    dz_half = float(_CONIC_DZ_FRAC_OF_EXTENT) * span
    dz_vals = np.linspace(-dz_half, dz_half, int(_CONIC_DZ_STEPS))

    best_iou = base_iou
    best_ax = best_ay = best_az = 0.0
    best_dz = 0.0
    for ax in ax_vals:
        for ay in ay_vals:
            for dz in dz_vals:
                rv2, tv2 = _compose_camera_wobble(rvec, tvec, float(ax), float(ay), 0.0, float(dz))
                iou = _mesh_projection_iou(mask_bool, verts_s, est.mesh_faces, rv2, tv2, K, dist)
                if iou > best_iou:
                    best_iou = iou
                    best_ax, best_ay, best_az = float(ax), float(ay), 0.0
                    best_dz = float(dz)

    if best_iou > base_iou + 1e-6:
        rv_f, tv_f = _compose_camera_wobble(rvec, tvec, best_ax, best_ay, best_az, best_dz)
        state["rvec_raw"] = rv_f.copy()
        state["tvec_raw"] = tv_f.copy()
        state["rvec"] = rv_f
        state["tvec"] = tv_f
        state["conic_tilt_xy_rad"] = (best_ax, best_ay)
        state["conic_tilt_az_rad"] = best_az
        state["conic_dz"] = best_dz
        state["conic_iou"] = best_iou
    else:
        state["conic_tilt_xy_rad"] = (0.0, 0.0)
        state["conic_tilt_az_rad"] = 0.0
        state["conic_dz"] = 0.0
        state["conic_iou"] = base_iou


def _load_tool_meshes() -> dict[int, ToolMesh]:
    """Load per-object GLB meshes: ID1/2/3 -> object_0/1/2.glb."""
    meshes: dict[int, ToolMesh] = {}
    if trimesh is None:
        return meshes
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
            meshes[obj_id] = ToolMesh(mesh)
        except Exception as e:
            print(f"Failed to load {p.name} for ID{obj_id}: {e}")
    return meshes


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


def load_glb_points(glb_path, n_points=2000):
    if trimesh is None:
        raise ImportError("`trimesh` is required for load_glb_points().")
    mesh = trimesh.load(glb_path, force="mesh")
    # Normalize to unit scale centered at origin.
    mesh.apply_translation(-mesh.centroid)
    scale = float(mesh.scale)
    if abs(scale) > 1e-12:
        mesh.apply_scale(1.0 / scale)
    points, _ = trimesh.sample.sample_surface(mesh, int(n_points))
    return points  # (N, 3) array


def _render_glb_points(points: np.ndarray, angle_rad: float, size: int = 512) -> np.ndarray:
    """Render sampled GLB surface points in a simple rotating white point-cloud view."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return canvas

    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    R_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    pts_r = pts @ R_y.T

    # Orthographic projection with fixed margin.
    xy = pts_r[:, :2]
    half = max(float(np.max(np.abs(xy))), 1e-6)
    scale = (size * 0.45) / half
    u = (xy[:, 0] * scale + size * 0.5).astype(np.int32)
    v = (xy[:, 1] * scale + size * 0.5).astype(np.int32)

    keep = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    u, v = u[keep], v[keep]
    canvas[v, u] = (255, 255, 255)

    cv2.putText(canvas, "GLB sampled points", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _start_interactive_glb_3d_view(points: np.ndarray) -> bool:
    """
    Open an interactive 3D scatter window for GLB points.
    User can rotate/pan/zoom with the mouse in the Matplotlib window.
    """
    if plt is None:
        return False
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return False
    try:
        fig = plt.figure("GLB Points 3D")
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="white", s=1, depthshade=False)
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.set_title("GLB Points 3D (drag to orbit)", color="white")
        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.set_zlabel("Z", color="white")
        ax.tick_params(colors="white")
        ax.set_box_aspect((1, 1, 1))
        lim = float(np.max(np.abs(pts))) if pts.size > 0 else 1.0
        lim = max(lim, 1e-3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        plt.show(block=False)
        plt.pause(0.001)
        return True
    except Exception:
        return False


def _init_alignment_3d_view():
    """Create interactive 3D view for mesh-vs-mask alignment points."""
    if plt is None:
        return None
    try:
        fig = plt.figure("Mask-Mesh Alignment 3D")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.set_title("Mask-Mesh Alignment 3D", color="white")
        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.set_zlabel("Z", color="white")
        ax.tick_params(colors="white")
        ax.set_box_aspect((1, 1, 1))
        plt.show(block=False)
        plt.pause(0.001)
        return {"fig": fig, "ax": ax}
    except Exception:
        return None


def _build_alignment_points_3d(
    est: "ToolMesh",
    state: dict,
    mask_bool: np.ndarray,
    K: np.ndarray,
    n_mask_pts: int = 200,
    n_mesh_pts: int = 1200,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build 3D point sets in the **OpenCV camera frame** (same as ``solvePnP``).

    **Magenta** is the **PnP midplane rectangle** (``model_pts`` perimeter at
    ``z_obj = 0``), not arbitrary mesh vertices. Full meshes extend off that
    plane (handles, thickness); plotting them against a **midplane-only** lift
    of the 2D silhouette often looks like two scissors rotated ~90° in the
    same plane even when the pose is fine.

    **Cyan**: each mask contour pixel defines a camera ray; intersect with the
    same midplane ``n·P = n·t`` with ``n = R[:, 2]``.
    """
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None or not np.any(mask_bool):
        return None

    s = _reg_sign_from_state(state)
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(1, 3)

    model_s = (np.asarray(est.model_pts, dtype=np.float64) * s.reshape(1, 3))
    n_plane = max(int(n_mesh_pts), int(n_mask_pts), int(est._pnp_n), 32)
    perim_obj = _sample_quad_perimeter(model_s, n_plane)
    mesh_cam = (perim_obj @ R.T) + t

    mask_pts = _sample_mask_contour(mask_bool.astype(np.uint8) * 255, int(n_mask_pts))
    if mask_pts is None or mask_pts.shape[0] == 0:
        return None

    # Midplane z_obj = 0  ⟺  n·P = n·t  in camera frame, n = third column of R.
    n = np.asarray(R[:, 2], dtype=np.float64).reshape(3)
    t3 = t.reshape(3)
    nt = float(np.dot(n, t3))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u, v = mask_pts[:, 0], mask_pts[:, 1]
    dx = (u - cx) / max(fx, 1e-9)
    dy = (v - cy) / max(fy, 1e-9)
    dz = np.ones_like(dx, dtype=np.float64)
    dirs = np.column_stack([dx, dy, dz])
    nd = dirs @ n
    eps = 1e-9
    lam = np.full(nd.shape[0], np.nan, dtype=np.float64)
    hit = np.abs(nd) > eps
    lam[hit] = nt / nd[hit]
    hit &= lam > 0
    mask_cam = dirs * lam[:, np.newaxis]
    mask_cam = mask_cam[hit]
    if mask_cam.shape[0] < 8:
        # Plane nearly edge-on to view — fall back to single depth (degraded viz).
        z = float(np.median(mesh_cam[:, 2]))
        z = max(z, 1e-3)
        mask_cam = np.column_stack(
            [dx * z, dy * z, np.full_like(dx, z, dtype=np.float64)]
        )
    return mesh_cam, mask_cam


def _extract_mask_keypoints(mask_bool: np.ndarray) -> np.ndarray | None:
    """Extract 4 ordered 2D keypoints from mask via min-area rectangle corners."""
    if not np.any(mask_bool):
        return None
    ys, xs = np.where(mask_bool)
    if len(xs) < 8:
        return None
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    box = cv2.boxPoints(rect).astype(np.float64)
    return _order_corners_clockwise(box)


def _project_mesh_keypoints(
    est: "ToolMesh", state: dict, K: np.ndarray, dist: np.ndarray
) -> np.ndarray | None:
    """Project 3D model keypoints (model quad corners) into image."""
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None:
        return None
    s = _reg_sign_from_state(state)
    model = (est.model_pts * s).astype(np.float64)
    proj, _ = cv2.projectPoints(model, rvec, tvec, K, dist)
    return _order_corners_clockwise(proj.reshape(-1, 2).astype(np.float64))


def _match_keypoints_temporal(
    mask_kps: np.ndarray,
    mesh_kps: np.ndarray,
    prev_perm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match mask corners to projected mesh corners by minimum total distance.
    Uses brute-force permutation search (N is tiny) + small temporal penalty.
    Returns (mesh_kps_reordered, best_perm).
    """
    n = int(min(len(mask_kps), len(mesh_kps)))
    if n <= 1:
        perm = np.arange(n, dtype=np.int32)
        return mesh_kps[:n].copy(), perm
    mk = np.asarray(mask_kps[:n], dtype=np.float64)
    pk = np.asarray(mesh_kps[:n], dtype=np.float64)
    import itertools

    best_perm = np.arange(n, dtype=np.int32)
    best_cost = float("inf")
    prev = None if prev_perm is None else np.asarray(prev_perm, dtype=np.int32).reshape(-1)
    for p in itertools.permutations(range(n)):
        perm = np.asarray(p, dtype=np.int32)
        d = np.linalg.norm(mk - pk[perm], axis=1)
        cost = float(np.sum(d))
        if prev is not None and len(prev) == n:
            # Stabilize label identities across frames.
            cost += 8.0 * float(np.sum(perm != prev))
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return pk[best_perm], best_perm


def _draw_keypoint_alignment(
    frame: np.ndarray,
    mask_kps: np.ndarray,
    mesh_kps: np.ndarray,
    oid: int,
) -> np.ndarray:
    """Visualize matched 2D mask keypoints and projected 3D keypoints."""
    out = frame.copy()
    n = min(len(mask_kps), len(mesh_kps))
    for i in range(n):
        mk = tuple(np.round(mask_kps[i]).astype(int))
        pk = tuple(np.round(mesh_kps[i]).astype(int))
        cv2.circle(out, mk, 5, (255, 255, 0), -1, lineType=cv2.LINE_AA)   # mask kp: cyan
        cv2.circle(out, pk, 5, (255, 0, 255), -1, lineType=cv2.LINE_AA)   # mesh kp: magenta
        cv2.line(out, mk, pk, (0, 255, 255), 1, lineType=cv2.LINE_AA)      # match line: yellow
        cv2.putText(out, str(i), (mk[0] + 5, mk[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(out, str(i), (pk[0] + 5, pk[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    d = np.linalg.norm(mask_kps[:n] - mesh_kps[:n], axis=1)
    err = float(np.mean(d)) if d.size else 0.0
    cv2.putText(
        out,
        f"ID{oid} keypoint err(px): {err:.2f}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _sample_projected_mesh_contour(
    est: "ToolMesh",
    state: dict,
    K: np.ndarray,
    dist: np.ndarray,
    fh: int,
    fw: int,
    n: int = 64,
) -> np.ndarray | None:
    """Sample contour points from projected mesh silhouette."""
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None:
        return None
    s = _reg_sign_from_state(state)
    verts = est.mesh_vertices * s
    proj_mesh, _ = cv2.projectPoints(verts, rvec, tvec, K, dist)
    pts2d = proj_mesh.reshape(-1, 2)
    pred_mask = np.zeros((fh, fw), dtype=np.uint8)
    for f in est.mesh_faces:
        poly = np.round(pts2d[f]).astype(np.int32)
        cv2.fillConvexPoly(pred_mask, poly, 255, lineType=cv2.LINE_AA)
    return _sample_mask_contour(pred_mask, int(n))


def _match_dense_contours(
    mask_pts: np.ndarray,
    mesh_pts: np.ndarray,
    prev_shift: int | None = None,
    prev_rev: bool | None = None,
) -> tuple[np.ndarray, int, bool, float]:
    """
    Dense contour matching with free cyclic shift and direction flip.
    This avoids rigid index-order pairing artifacts.
    """
    m = np.asarray(mask_pts, dtype=np.float64)
    q0 = np.asarray(mesh_pts, dtype=np.float64)
    n = int(min(len(m), len(q0)))
    m = m[:n]
    q0 = q0[:n]
    best = q0.copy()
    best_shift = 0
    best_rev = False
    best_cost = float("inf")
    for rev in (False, True):
        q = q0[::-1] if rev else q0
        for shift in range(n):
            q_s = np.roll(q, shift, axis=0)
            d = np.linalg.norm(m - q_s, axis=1)
            cost = float(np.mean(d))
            if prev_shift is not None:
                ds = abs(int(shift) - int(prev_shift))
                ds = min(ds, n - ds)
                cost += 0.2 * float(ds)
            if prev_rev is not None and bool(rev) != bool(prev_rev):
                cost += 3.0
            if cost < best_cost:
                best_cost = cost
                best = q_s
                best_shift = int(shift)
                best_rev = bool(rev)
    return best, best_shift, best_rev, best_cost


def _draw_dense_alignment(
    frame: np.ndarray,
    mask_pts: np.ndarray,
    mesh_pts_aligned: np.ndarray,
    oid: int,
    mean_err: float,
) -> np.ndarray:
    """Draw dense 2D-3D contour correspondence lines."""
    out = frame.copy()
    n = min(len(mask_pts), len(mesh_pts_aligned))
    m = np.asarray(mask_pts[:n], dtype=np.float64)
    q = np.asarray(mesh_pts_aligned[:n], dtype=np.float64)
    step = max(1, n // 32)
    for i in range(0, n, step):
        mk = tuple(np.round(m[i]).astype(int))
        pk = tuple(np.round(q[i]).astype(int))
        cv2.circle(out, mk, 2, (255, 255, 0), -1, lineType=cv2.LINE_AA)   # mask: cyan
        cv2.circle(out, pk, 2, (255, 0, 255), -1, lineType=cv2.LINE_AA)   # mesh: magenta
        cv2.line(out, mk, pk, (0, 255, 255), 1, lineType=cv2.LINE_AA)     # match line
    cv2.putText(
        out,
        f"ID{oid} dense contour err(px): {float(mean_err):.2f}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _render_projected_mesh_mask(
    est: "ToolMesh",
    state: dict,
    K: np.ndarray,
    dist: np.ndarray,
    fh: int,
    fw: int,
) -> np.ndarray | None:
    """Render projected mesh as a filled binary mask."""
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None:
        return None
    s = _reg_sign_from_state(state)
    verts = est.mesh_vertices * s
    proj_mesh, _ = cv2.projectPoints(verts, rvec, tvec, K, dist)
    pts2d = proj_mesh.reshape(-1, 2)
    pred_mask = np.zeros((fh, fw), dtype=np.uint8)
    for f in est.mesh_faces:
        poly = np.round(pts2d[f]).astype(np.int32)
        cv2.fillConvexPoly(pred_mask, poly, 255, lineType=cv2.LINE_AA)
    return pred_mask


def _mask_iou_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float]:
    """Compute IoU and Dice between binary masks."""
    a = np.asarray(mask_a, dtype=np.uint8) > 0
    b = np.asarray(mask_b, dtype=np.uint8) > 0
    inter = float(np.logical_and(a, b).sum())
    a_sum = float(a.sum())
    b_sum = float(b.sum())
    union = float(np.logical_or(a, b).sum())
    iou = inter / union if union > 0.0 else 0.0
    dice = (2.0 * inter) / (a_sum + b_sum) if (a_sum + b_sum) > 0.0 else 0.0
    return iou, dice


def _draw_solid_alignment(
    frame: np.ndarray,
    mask_u8: np.ndarray,
    mesh_mask_u8: np.ndarray,
    oid: int,
    iou: float,
    dice: float,
) -> np.ndarray:
    """Visualize solid-vs-solid alignment (mask fill vs projected mesh fill)."""
    out = frame.copy()
    mask_b = (mask_u8 > 0)
    mesh_b = (mesh_mask_u8 > 0)
    inter_b = np.logical_and(mask_b, mesh_b)
    only_mask = np.logical_and(mask_b, np.logical_not(mesh_b))
    only_mesh = np.logical_and(mesh_b, np.logical_not(mask_b))

    # Color coding: intersection=white, only mask=cyan, only mesh=magenta.
    out[inter_b] = (255, 255, 255)
    out[only_mask] = (255, 255, 0)
    out[only_mesh] = (255, 0, 255)

    cv2.putText(
        out,
        f"ID{oid} solid alignment  IoU={iou:.3f}  Dice={dice:.3f}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _update_alignment_3d_view(view, mesh_cam: np.ndarray, mask_cam: np.ndarray, oid: int, fi: int) -> None:
    """Refresh interactive 3D alignment figure for one object."""
    if view is None or plt is None:
        return
    ax = view["ax"]
    ax.cla()
    ax.set_facecolor("black")
    ax.set_title(f"Mask-Mesh Alignment 3D | ID{oid} | frame {int(fi)}", color="white")
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z", color="white")
    ax.tick_params(colors="white")
    ax.set_box_aspect((1, 1, 1))

    # PnP midplane quad (magenta) vs silhouette lifted to that plane (cyan).
    ax.scatter(
        mesh_cam[:, 0],
        mesh_cam[:, 1],
        mesh_cam[:, 2],
        c="#ff66ff",
        s=2,
        depthshade=False,
        label="PnP midplane (model quad)",
    )
    ax.scatter(
        mask_cam[:, 0],
        mask_cam[:, 1],
        mask_cam[:, 2],
        c="#66ffff",
        s=7,
        depthshade=False,
        label="mask silhouette → midplane",
    )
    ax.legend(loc="upper right", facecolor="black", edgecolor="white", labelcolor="white")

    all_pts = np.vstack([mesh_cam, mask_cam])
    c = all_pts.mean(axis=0)
    span = np.max(np.ptp(all_pts, axis=0))
    span = max(float(span), 1e-3)
    h = span * 0.6
    ax.set_xlim(c[0] - h, c[0] + h)
    ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_zlim(c[2] - h, c[2] + h)


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


def _draw_registration_debug(
    canvas: np.ndarray,
    mask_bool: np.ndarray,
    state: dict,
    est: ToolMesh,
    K: np.ndarray,
    dist: np.ndarray,
    obj_id: int,
) -> None:
    """Solid projected mesh vs filled SAM mask (rigid registration debug)."""
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None or not np.any(mask_bool):
        return

    fh, fw = canvas.shape[:2]
    pred_mask = np.zeros((fh, fw), dtype=np.uint8)

    s = _reg_sign_from_state(state)
    verts = est.mesh_vertices * s.reshape(1, 3)
    faces = est.mesh_faces
    proj_mesh, _ = cv2.projectPoints(verts, rvec, tvec, K, dist)
    pts2d = proj_mesh.reshape(-1, 2)
    mesh_layer = np.zeros_like(canvas)
    for f in faces:
        poly = np.round(pts2d[f]).astype(np.int32)
        cv2.fillConvexPoly(mesh_layer, poly, (120, 40, 200), lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(pred_mask, poly, 255, lineType=cv2.LINE_AA)
    cv2.addWeighted(mesh_layer, 0.82, canvas, 0.18, 0.0, dst=canvas)

    # Filled mask (opaque cyan tint inside mask, BGR).
    canvas_f = canvas.astype(np.float64)
    cy = np.array([255.0, 255.0, 0.0], dtype=np.float64)
    m = mask_bool
    canvas_f[m] = canvas_f[m] * 0.15 + cy * 0.85
    canvas[:] = np.clip(canvas_f, 0.0, 255.0).astype(np.uint8)

    # 2D min-area rectangle around the full projected mesh (mesh footprint in image).
    rect_mesh = cv2.minAreaRect(pts2d.astype(np.float32))
    box_mesh = cv2.boxPoints(rect_mesh).astype(np.int32)
    cv2.polylines(canvas, [box_mesh], True, (0, 255, 255), 2, lineType=cv2.LINE_AA)  # yellow = mesh 2D bbox

    # Visualize contour samples on mask and projected mesh silhouette.
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
    mesh_estimators = _load_tool_meshes()
    if trimesh is None:
        print("`trimesh` not installed; 3D mesh overlay window unavailable.")
    elif mesh_estimators:
        loaded = ", ".join(f"ID{k}->object_{k-1}" for k in sorted(mesh_estimators))
        print(f"Mesh overlay estimators loaded for {loaded}")
    else:
        print("No mesh files found; 3D mesh overlay window disabled.")

    # Optional secondary window: sampled GLB point-cloud preview.
    glb_points = None
    if trimesh is not None:
        base = Path(__file__).parent
        glb_candidates = [base / "object_0.glb", base / "object_1.glb", base / "object_2.glb"]
        glb_path = next((p for p in glb_candidates if p.exists()), None)
        if glb_path is not None:
            try:
                glb_points = load_glb_points(str(glb_path), n_points=2000)
                print(f"GLB points window enabled: {glb_path.name} ({len(glb_points)} pts)")
            except Exception as e:
                print(f"GLB points preview disabled ({glb_path.name} load failed): {e}")
    glb_3d_ok = False
    if glb_points is not None:
        glb_3d_ok = _start_interactive_glb_3d_view(glb_points)
        if glb_3d_ok:
            print("Interactive GLB 3D window enabled (mouse orbit/pan/zoom).")
        elif plt is None:
            print("Interactive GLB 3D window unavailable (matplotlib not installed).")
    alignment_3d_view = _init_alignment_3d_view()
    if alignment_3d_view is not None:
        print("Interactive mask-mesh alignment 3D window enabled.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize camera-side frame queue
    rotate_180 = detect_orbbec_camera(args.camera)
    print(f"Camera {args.camera}: {'Orbbec detected, rotating 180°' if rotate_180 else 'non-Orbbec, no rotation'}")

    K_cam = _estimate_intrinsics_from_cap(cap, TARGET_SIZE[0], TARGET_SIZE[1])
    dist_cam = np.zeros((1, 5), dtype=np.float64)
    print(
        f"Camera intrinsics (fixed): "
        f"fx={K_cam[0, 0]:.1f}  fy={K_cam[1, 1]:.1f}  "
        f"cx={K_cam[0, 2]:.1f}  cy={K_cam[1, 2]:.1f}"
    )

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
    # Pose estimation disabled.
    # csv_path = Path.cwd() / "poses_live.csv"
    # csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    # csv_writer = csv.writer(csv_file)
    # t_cols = ["tx", "ty", "tz"]
    # if args.surface_distance_cm > 0:
    #     t_cols = ["tx_cm", "ty_cm", "tz_cm"]
    # csv_writer.writerow(["frame_idx", "time_s", "object_id", "rx_deg", "ry_deg", "rz_deg", *t_cols])
    # print(f"Pose CSV export: {csv_path}")
    # if args.surface_distance_cm > 0:
    #     print(
    #         f"Translation calibration enabled: using camera->surface distance "
    #         f"{args.surface_distance_cm:.2f} cm for T readouts/CSV."
    #     )

    com_trails: dict[int, list[tuple[int, int]]] = {}  # obj_id -> list of (x, y) COM positions
    pose_states: dict[int, dict] = {}          # per-obj rigid registration (rvec, tvec, reg_sign)
    fps_t0 = time.perf_counter()
    fps_frames = 0
    ac_device, ac_dtype, ac_enabled = _autocast_config(device, args.half)

    print("Live tracking started. Press 'q' or ESC to stop.")
    try:
        with torch.autocast(device_type=ac_device, dtype=ac_dtype, enabled=ac_enabled):
            for fi, obj_ids, masks in predictor.propagate_in_video(state):
                frame = provider.get_raw(fi)
                mesh_overlay = frame.copy()
                align_payload = None
                keypoint_align_vis = frame.copy()
                if hasattr(obj_ids, "tolist"):
                    ids = [int(x) for x in obj_ids.tolist()]
                else:
                    ids = [int(x) for x in obj_ids]

                vis = overlay_masks(frame, ids, masks, alpha=args.alpha)
                fh, fw = frame.shape[:2]
                mask_points_vis = np.zeros((fh, fw, 3), dtype=np.uint8)
                masks_np = masks.detach().cpu().numpy()
                for i in range(min(len(ids), masks_np.shape[0])):
                    oid = ids[i]
                    binm = _mask_to_2d_bool(masks_np[i], fh, fw)

                    if np.any(binm):
                        cnt = get_mask_contour(binm)
                        if cnt is not None:
                            cv2.drawContours(vis, [cnt], -1, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.drawContours(mask_points_vis, [cnt], -1, (255, 255, 255), 1, cv2.LINE_AA)
                            # Show selected contour points used for 2D mask geometry.
                            pts = cnt.reshape(-1, 2)
                            if len(pts) > 0:
                                step = max(1, len(pts) // 250)
                                for p in pts[::step]:
                                    cv2.circle(
                                        mask_points_vis,
                                        (int(p[0]), int(p[1])),
                                        1,
                                        (255, 255, 255),
                                        -1,
                                        lineType=cv2.LINE_AA,
                                    )
                        ys, xs = np.where(binm)
                        cx, cy = int(xs.mean()), int(ys.mean())
                        if oid not in com_trails:
                            com_trails[oid] = []
                        com_trails[oid].append((cx, cy))

                        est = mesh_estimators.get(oid)
                        if est is not None:
                            pose_states[oid] = _register_rigid_se3_mesh_to_mask(
                                est,
                                binm,
                                K_cam,
                                dist_cam,
                                pose_states.get(oid),
                            )
                            _refine_conic_section_overlap(
                                est, binm, K_cam, dist_cam, pose_states[oid]
                            )
                            _draw_registration_debug(
                                mesh_overlay, binm, pose_states[oid], est, K_cam, dist_cam, oid
                            )
                            if align_payload is None:
                                align_pts = _build_alignment_points_3d(est, pose_states[oid], binm, K_cam)
                                if align_pts is not None:
                                    align_payload = (oid, align_pts[0], align_pts[1])
                            mask_u8 = (binm.astype(np.uint8) * 255)
                            mesh_mask_u8 = _render_projected_mesh_mask(
                                est, pose_states[oid], K_cam, dist_cam, fh, fw
                            )
                            if mesh_mask_u8 is not None:
                                iou, dice = _mask_iou_dice(mask_u8, mesh_mask_u8)
                                keypoint_align_vis = _draw_solid_alignment(
                                    keypoint_align_vis, mask_u8, mesh_mask_u8, oid, iou, dice
                                )

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

                # _draw_pose_hud(vis, pose_states)

                if fi == 0:
                    for oid, px, py in points:
                        cv2.circle(vis, (int(px), int(py)), 5, (0, 255, 255), -1)
                        cv2.putText(vis, f"ID{oid}", (int(px) + 8, int(py) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                cv2.imshow("EdgeTAM Live", vis)
                cv2.imshow("2D-3D Solid Alignment", keypoint_align_vis)
                cv2.imshow("2D Mask Points", mask_points_vis)
                if mesh_estimators:
                    cv2.imshow("3D Mesh Overlay", mesh_overlay)
                if glb_points is not None:
                    glb_vis = _render_glb_points(glb_points, angle_rad=0.02 * float(fi), size=512)
                    cv2.imshow("GLB Points", glb_vis)
                if glb_3d_ok and plt is not None:
                    plt.pause(0.001)
                if alignment_3d_view is not None and align_payload is not None:
                    oid_a, mesh_cam_pts, mask_cam_pts = align_payload
                    _update_alignment_3d_view(alignment_3d_view, mesh_cam_pts, mask_cam_pts, oid_a, int(fi))
                    if plt is not None:
                        plt.pause(0.001)
                if writer is not None:
                    writer.write(vis)
                # if debug_canvas is not None:
                #     cv2.imwrite(args.align_debug_out, debug_canvas)
                # t_s = float(time.perf_counter())
                # for oid in sorted(pose_states):
                #     st = pose_states.get(oid, {})
                #     rvec = st.get("rvec")
                #     tvec = st.get("tvec_cal")
                #     if tvec is None:
                #         tvec = st.get("tvec")
                #     if rvec is None or tvec is None:
                #         continue
                #     rx, ry, rz = _pose_to_euler_zyx_deg(rvec)
                #     tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
                #     csv_writer.writerow([int(fi), t_s, int(oid), rx, ry, rz, tx, ty, tz])

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
        # if 'csv_file' in locals():
        #     csv_file.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time EdgeTAM tracking on Orbbec RGB stream.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Mask overlay alpha (1.0 = fully filled mask tint)")
    parser.add_argument("--half", action="store_true", default=True,
                        help="Use half-precision autocast (default: on)")
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--output", default="",
                        help="Optional path to save output video (e.g. out.mp4)")
    # Pose estimation / calibration flags disabled.
    # parser.add_argument(
    #     "--intrinsics-file",
    #     default="",
    #     help=(
    #         "Optional .npz intrinsics file for pose calibration. "
    #         "Expected keys: K (or camera_matrix/intrinsics), optional dist/dist_coeffs, "
    #         "optional width,height."
    #     ),
    # )
    # parser.add_argument(
    #     "--no-orbbec-intrinsics",
    #     action="store_true",
    #     help="Disable attempting to read Orbbec SDK color intrinsics before fallback estimate.",
    # )
    # parser.add_argument(
    #     "--surface-distance-cm",
    #     type=float,
    #     default=0.0,
    #     help=(
    #         "Optional known camera-to-surface distance in cm. "
    #         "When > 0, T(x,y,z) HUD/CSV are scaled to this reference."
    #     ),
    # )
    # parser.add_argument(
    #     "--align-debug-out",
    #     default="",
    #     help="Optional path to save registration-debug image (SAM mask vs projected GLB) each frame.",
    # )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
