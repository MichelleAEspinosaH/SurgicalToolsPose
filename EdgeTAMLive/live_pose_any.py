#!/usr/bin/env python3
"""
Live surgical tool tracking: EdgeTAM seeds → fal SAM3D GLBs → dense PnP 6DoF pose.

See EdgeTAMLive/README.md and ../README.md for setup, calibration (cm), and CLI.

Pipeline:
  1. Click seed points on the live camera feed.
  2. EdgeTAM computes instance masks on the seed frame.
  3. Confirm masks; fal SAM3D generates a GLB mesh per object.
  4. MeshPoseEstimator (PCA-aligned, dense-perimeter minAreaRect PnP) estimates
     6DoF pose each frame from the EdgeTAM mask, smoothed with Kalman filters.
  5. XYZ axes are projected onto the live video; HUD shows rotation (deg) and
     translation (cm) when checkerboard calibration is loaded.

Units:
  - Checkerboard calibration object points: cm (--checkerboard-square-cm 2).
  - SAM3D GLBs: fal normalizes longest axis to 1 m; pipeline converts mesh to cm.
  - PnP tvec from the mesh is in mm; HUD and CSV multiply by 0.1 for cm display.

Usage:
    cd EdgeTAMLive
    python live_pose_any.py --calibrate-only --camera 0
    python live_pose_any.py --show-calibration
    python live_pose_any.py --camera 0
"""

from __future__ import annotations

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

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
CHECKPOINT   = EDGETAM_REPO / "checkpoints" / "edgetam.pt"
MODEL_CFG    = "configs/edgetam.yaml"

_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_IMG_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

PNP_PER_EDGE       = 8
KALMAN_PROCESS_VAR = 5e-3
KALMAN_MEAS_VAR    = 1e-3
PNP_ROT_SMOOTH_W   = 0.03
PNP_TRANS_SMOOTH_W = 8.0
PNP_SHIFT_PENALTY  = 0.75

_MAX_COM_TRAIL = 60

# MRPT 9×7 checkerboard — count squares on the print, OpenCV uses inner corners:
#   9×7 squares  →  --checkerboard-cols 8  --checkerboard-rows 6
# https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf
CHECKERBOARD_INNER_COLS = 8
CHECKERBOARD_INNER_ROWS = 6
CHECKERBOARD_SQUARE_CM  = 2.0   # 2 cm squares (20 mm at 1:1 print); object points in cm
# SAM3D mesh PnP tvec is in mm; checkerboard cal and HUD/CSV use cm.
PNP_TVEC_MM_TO_CM = 0.1
DEFAULT_CALIBRATION_NPZ = Path(__file__).resolve().parent / "camera_calibration.npz"

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
    dtype   = torch.bfloat16 if device_type == "cuda" else torch.float16
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
            return "orbbec" in m.group(2).strip().lower()
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
        vis  = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8) if base is None else base.copy()
        for obj_id, px_f, py_f in points:
            px, py = int(px_f), int(py_f)
            cv2.circle(vis, (px, py), 6, (0, 255, 255), -1)
            cv2.circle(vis, (px, py), 9, (255, 255, 255), 1)
            cv2.putText(vis, f"ID{obj_id}", (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis,
            "Left click: add point (freezes on first click) | Backspace: undo | c: clear | Enter: start",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
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
            cancelled = True; break
        cv2.imshow(win, draw())
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and points:
            break
        elif k in (8, 127) and points:
            points.pop()
        elif k == ord("c"):
            points.clear(); frozen_frame = None
        elif k in (ord("q"), 27):
            cancelled = True; break
    cv2.destroyWindow(win)
    if cancelled:
        return [], None
    seed_frame = frozen_frame if frozen_frame is not None else provider.get_raw(-1)
    return points, (None if seed_frame is None else seed_frame.copy())

# ---------------------------------------------------------------------------
# Colour / mask helpers
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
    ang    = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    ordered = corners[np.argsort(ang)]
    idx0    = int(np.argmin(ordered[:, 1] * 10000.0 + ordered[:, 0]))
    return np.roll(ordered, -idx0, axis=0)

# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def _estimate_intrinsics_from_cap(cap: cv2.VideoCapture, target_w: int, target_h: int) -> np.ndarray:
    native_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    native_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if native_w <= 0 or native_h <= 0:
        native_w, native_h = float(target_w), float(target_h)
    fx = (native_w / (2.0 * np.tan(np.radians(79.0) / 2.0))) * (target_w / native_w)
    fy = (native_h / (2.0 * np.tan(np.radians(62.0) / 2.0))) * (target_h / native_h)
    return np.array(
        [[fx, 0.0, target_w / 2.0], [0.0, fy, target_h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rescale_intrinsics(K, src_w, src_h, dst_w, dst_h):
    sx = float(dst_w) / max(float(src_w), 1e-9)
    sy = float(dst_h) / max(float(src_h), 1e-9)
    K2 = np.asarray(K, dtype=np.float64).copy()
    K2[0, 0] *= sx; K2[1, 1] *= sy; K2[0, 2] *= sx; K2[1, 2] *= sy
    return K2


def _load_intrinsics_from_file(path, target_w, target_h):
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
        print(f"Intrinsics file '{path}' missing a 3x3 matrix."); return None
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
    if src_w and src_h and src_w > 0 and src_h > 0:
        K = _rescale_intrinsics(K, src_w, src_h, target_w, target_h)
    return K.astype(np.float64), dist.astype(np.float64)


def _read_calibration_meta(path: Path) -> dict | None:
    """Load summary fields from a checkerboard .npz (or None if unreadable)."""
    try:
        data = np.load(path, allow_pickle=False)
    except Exception:
        return None
    meta: dict = {"path": str(path.resolve())}
    if "reprojection_error" in data:
        meta["rms_px"] = float(np.asarray(data["reprojection_error"]).reshape(-1)[0])
    if "num_samples" in data:
        meta["num_samples"] = int(np.asarray(data["num_samples"]).reshape(-1)[0])
    if "checkerboard_cols" in data and "checkerboard_rows" in data:
        meta["inner_cols"] = int(np.asarray(data["checkerboard_cols"]).reshape(-1)[0])
        meta["inner_rows"] = int(np.asarray(data["checkerboard_rows"]).reshape(-1)[0])
    if "square_cm" in data:
        meta["square_cm"] = float(np.asarray(data["square_cm"]).reshape(-1)[0])
    elif "square_mm" in data:
        meta["square_cm"] = float(np.asarray(data["square_mm"]).reshape(-1)[0]) / 10.0
        meta["legacy_square_mm"] = True
    if "K" in data:
        K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
        meta["fx"] = float(K[0, 0])
        meta["fy"] = float(K[1, 1])
    if "width" in data and "height" in data:
        meta["width"] = int(np.asarray(data["width"]).reshape(-1)[0])
        meta["height"] = int(np.asarray(data["height"]).reshape(-1)[0])
    return meta


def _rms_quality_label(rms_px: float) -> str:
    if rms_px < 0.5:
        return "good"
    if rms_px < 1.0:
        return "acceptable"
    return "poor — recalibrate"


def _print_calibration_summary(path: Path) -> bool:
    """Print RMS, board size, and intrinsics from a saved calibration file."""
    meta = _read_calibration_meta(path)
    if meta is None:
        print(f"Could not read calibration: {path}")
        return False
    print(f"Calibration file: {meta['path']}")
    if "rms_px" in meta:
        print(
            f"  RMS reprojection error: {meta['rms_px']:.4f} px "
            f"({_rms_quality_label(meta['rms_px'])})"
        )
    if "num_samples" in meta:
        print(f"  Captured views: {meta['num_samples']}")
    if "inner_cols" in meta and "inner_rows" in meta:
        ic, ir = meta["inner_cols"], meta["inner_rows"]
        print(
            f"  Board: {ic}×{ir} inner corners "
            f"({ic + 1}×{ir + 1} squares)"
        )
    if "square_cm" in meta:
        note = " (legacy file used mm → converted to cm)" if meta.get("legacy_square_mm") else ""
        print(f"  Square size: {meta['square_cm']:g} cm{note}")
    if "fx" in meta and "fy" in meta:
        print(f"  fx={meta['fx']:.1f}  fy={meta['fy']:.1f}")
    if "width" in meta and "height" in meta:
        print(f"  Resolution: {meta['width']}×{meta['height']}")
    return True


def _load_metric_context(intrinsics_file: str, intr_src: str) -> dict:
    """Return whether to show metric cm poses (checkerboard .npz)."""
    ctx = {
        "use_metric": False,
        "square_cm": CHECKERBOARD_SQUARE_CM,
        "source": "",
    }
    path_str = intrinsics_file
    if not path_str and intr_src.startswith("file:"):
        path_str = intr_src.split(":", 1)[1]
    if not path_str:
        return ctx
    path = Path(path_str).expanduser()
    if not path.is_file():
        return ctx
    try:
        data = np.load(path, allow_pickle=False)
    except Exception:
        return ctx
    if "square_cm" in data or "square_mm" in data or "checkerboard_cols" in data or "checkerboard_rows" in data:
        ctx["use_metric"] = True
        ctx["source"] = str(path)
        if "square_cm" in data:
            ctx["square_cm"] = float(np.asarray(data["square_cm"]).reshape(-1)[0])
        elif "square_mm" in data:
            ctx["square_cm"] = float(np.asarray(data["square_mm"]).reshape(-1)[0]) / 10.0
    return ctx


def _try_read_orbbec_intrinsics(target_w, target_h):
    try:
        from pyorbbecsdk import OBSensorType, Pipeline  # type: ignore
    except Exception:
        return None
    pipeline = None
    try:
        pipeline = Pipeline()
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
        K_native = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        K = _rescale_intrinsics(K_native, native_w, native_h, target_w, target_h)
        dist = np.zeros((4, 1), dtype=np.float64)
        if hasattr(intr, "coeffs"):
            coeffs = np.asarray(getattr(intr, "coeffs"), dtype=np.float64).reshape(-1, 1)
            if coeffs.size > 0:
                dist = coeffs
        else:
            vals = [float(getattr(intr, n)) for n in ("k1","k2","p1","p2","k3","k4","k5","k6") if hasattr(intr, n)]
            if vals:
                dist = np.asarray(vals, dtype=np.float64).reshape(-1, 1)
        return K, dist
    except Exception:
        return None
    finally:
        if pipeline is not None:
            try: pipeline.stop()
            except Exception: pass


def _resolve_pose_intrinsics(cap, target_w, target_h, intrinsics_file, try_orbbec):
    if intrinsics_file:
        loaded = _load_intrinsics_from_file(intrinsics_file, target_w, target_h)
        if loaded is not None:
            return loaded[0], loaded[1], f"file:{intrinsics_file}"
        print("Falling back because intrinsics file could not be used.")
    if try_orbbec:
        sdk = _try_read_orbbec_intrinsics(target_w, target_h)
        if sdk is not None:
            return sdk[0], sdk[1], "orbbec_sdk"
    K = _estimate_intrinsics_from_cap(cap, target_w, target_h)
    return K, np.zeros((4, 1), dtype=np.float64), "fov_estimate"


# ---------------------------------------------------------------------------
# Checkerboard calibration  (OpenCV calibrateCamera)
# ---------------------------------------------------------------------------

def _checkerboard_object_points(cols: int, rows: int, square_cm: float) -> np.ndarray:
    """3-D corner coordinates on the Z=0 board plane (cm), float32 for OpenCV."""
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = (grid * float(square_cm)).astype(np.float32)
    return objp


def _find_chessboard_corners(gray: np.ndarray, pattern_size: tuple[int, int]):
    """Return sub-pixel inner corners; tries findChessboardCornersSB when available."""
    base_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = base_flags
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            sb_flags |= cv2.CALIB_CB_EXHAUSTIVE
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, sb_flags)
        if found:
            return True, corners.reshape(-1, 1, 2).astype(np.float32)
    fast_flags = base_flags | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, pattern_size, fast_flags)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, pattern_size, base_flags)
    if not found:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def _detect_checkerboard(
    gray: np.ndarray,
    cols: int,
    rows: int,
    locked_pattern: tuple[int, int] | None,
) -> tuple[bool, np.ndarray | None, tuple[int, int] | None]:
    """
    Detect inner corners. On first success, lock (cols, rows) or rotated (rows, cols)
    so every captured sample uses the same object-point grid.
    """
    candidates: list[tuple[int, int]] = []
    if locked_pattern is not None:
        candidates = [locked_pattern]
    else:
        primary = (cols, rows)
        if primary not in candidates:
            candidates.append(primary)
        alt = (rows, cols)
        if alt != primary and alt not in candidates:
            candidates.append(alt)
    for pattern in candidates:
        found, corners = _find_chessboard_corners(gray, pattern)
        if found:
            return True, corners, pattern
    return False, None, locked_pattern


def run_checkerboard_calibration(args) -> bool:
    """
    Interactive checkerboard capture → cv2.calibrateCamera → .npz save.

    Frames are preprocessed exactly like the live pose pipeline (resize to
    TARGET_SIZE, optional Orbbec 180° rotation) so K/dist match PnP input.

    Defaults match the MRPT 9×7 PDF (8×6 inner corners, 2 cm / 20 mm squares).
    """
    cols = int(args.checkerboard_cols)
    rows = int(args.checkerboard_rows)
    if cols < 3 or rows < 3:
        print("Checkerboard must have at least 3×3 inner corners.")
        return False

    square_cm = float(args.checkerboard_square_cm)
    min_samples = max(3, int(args.calibration_min_samples))
    target_w, target_h = TARGET_SIZE
    out_path = Path(args.calibration_out).expanduser().resolve()
    squares_x = cols + 1
    squares_y = rows + 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return False
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    rotate_180 = detect_orbbec_camera(args.camera)
    print(
        f"Checkerboard calibration  camera={args.camera}  "
        f"inner corners={cols}×{rows}  ({squares_x}×{squares_y} squares)  "
        f"square={square_cm:g} cm  resolution={target_w}×{target_h}"
    )
    print("Board: MRPT 9×7 @ 2 cm (20 mm) squares — print PDF at 1:1 scale on A4.")
    if rotate_180:
        print("Orbbec detected — frames rotated 180° (same as live tracking).")

    locked_pattern: tuple[int, int] | None = None
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    win = "Checkerboard calibration"

    def _active_objp(pattern: tuple[int, int]) -> np.ndarray:
        pc, pr = pattern
        return _checkerboard_object_points(pc, pr, square_cm)

    def _draw_panel(frame: np.ndarray, found: bool, corners, pattern) -> np.ndarray:
        vis = frame.copy()
        if found and corners is not None and pattern is not None:
            cv2.drawChessboardCorners(vis, pattern, corners, found)
        n = len(objpoints)
        status = "DETECTED — press SPACE to capture" if found else "Searching for board…"
        color = (0, 220, 0) if found else (0, 140, 255)
        pat_txt = (
            f"{pattern[0]}×{pattern[1]} inner corners"
            if pattern is not None
            else f"{cols}×{rows} or {rows}×{cols} inner corners"
        )
        lines = [
            f"Samples: {n}/{min_samples}  ({status})",
            f"Board: {pat_txt}, {square_cm:g} cm squares",
            "SPACE: capture   U: undo last   C/Enter: calibrate   Q/ESC: cancel",
        ]
        y = 24
        for line in lines:
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color if y == 24 else (230, 230, 230), 1)
            y += 22
        return vis

    cv2.namedWindow(win)
    calibrated = False
    active_cols, active_rows = cols, rows
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                print("Camera read failed.")
                break
            frame = preprocess(raw, rotate_180)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners, pattern = _detect_checkerboard(gray, cols, rows, locked_pattern)
            if found and pattern is not None and locked_pattern is None:
                locked_pattern = pattern
                active_cols, active_rows = pattern
                print(
                    f"Board detected: {active_cols}×{active_rows} inner corners "
                    f"({active_cols + 1}×{active_rows + 1} squares)."
                )
            cv2.imshow(win, _draw_panel(frame, found, corners, locked_pattern))
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                print("Calibration cancelled.")
                break
            if key in (8, ord("u"), ord("U")) and objpoints:
                objpoints.pop()
                imgpoints.pop()
                print(f"Removed last sample ({len(objpoints)} remaining).")
                continue
            if key == ord(" ") and found and corners is not None and locked_pattern is not None:
                objpoints.append(_active_objp(locked_pattern))
                imgpoints.append(corners)
                print(f"Captured sample {len(objpoints)}.")
                continue
            if key in (ord("c"), ord("C"), 13, 10) and len(objpoints) >= min_samples:
                calibrated = True
                break
    finally:
        cap.release()
        cv2.destroyWindow(win)

    if not calibrated or len(objpoints) < min_samples:
        if len(objpoints) > 0:
            print(f"Need at least {min_samples} samples (have {len(objpoints)}).")
        elif locked_pattern is None:
            print(
                "No board detected. Check: 1:1 print scale, flat board, lighting, "
                f"and --checkerboard-cols/--checkerboard-rows (expected {cols}×{rows} "
                f"inner corners for MRPT 9×7)."
            )
        return False

    print(f"Calibrating from {len(objpoints)} views …")
    obj_cv = [np.ascontiguousarray(o, dtype=np.float32) for o in objpoints]
    img_cv = [np.ascontiguousarray(i, dtype=np.float32) for i in imgpoints]
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_cv,
        img_cv,
        (target_w, target_h),
        None,
        None,
    )
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        K=K.astype(np.float64),
        dist=dist,
        width=np.int32(target_w),
        height=np.int32(target_h),
        reprojection_error=np.float64(rms),
        checkerboard_cols=np.int32(active_cols),
        checkerboard_rows=np.int32(active_rows),
        checkerboard_squares_x=np.int32(active_cols + 1),
        checkerboard_squares_y=np.int32(active_rows + 1),
        square_cm=np.float64(square_cm),
        num_samples=np.int32(len(objpoints)),
        source=np.array("mrpt_9x7_2cm"),
    )
    print(f"Saved calibration: {out_path}")
    _print_calibration_summary(out_path)
    print(f"  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}  dist={dist.ravel()}")
    print(f"\nInspect anytime:\n  python live_pose_any.py --show-calibration")
    print(f"Next run:\n  python live_pose_any.py --camera {args.camera}")
    return True

# ---------------------------------------------------------------------------
# Pose math utilities
# ---------------------------------------------------------------------------

def _mesh_projection_iou(mask_bool, verts, faces, rvec, tvec, K, dist) -> float:
    fh, fw = mask_bool.shape[:2]
    pred_mask = np.zeros((fh, fw), dtype=np.uint8)
    proj_mesh, _ = cv2.projectPoints(verts.astype(np.float64), rvec, tvec, K, dist)
    pts2d = proj_mesh.reshape(-1, 2)
    for f in faces:
        poly = np.round(pts2d[f]).astype(np.int32)
        cv2.fillConvexPoly(pred_mask, poly, 255, lineType=cv2.LINE_AA)
    pred_b = pred_mask > 0
    inter  = float(np.logical_and(pred_b, mask_bool).sum())
    union  = float(np.logical_or(pred_b, mask_bool).sum())
    return inter / union if union > 0.0 else 0.0


def _reg_sign_from_state(state: dict) -> np.ndarray:
    s = state.get("reg_sign")
    if s is None:
        return np.ones(3, dtype=np.float64)
    return np.asarray(s, dtype=np.float64).reshape(3)


def _sample_quad_perimeter(corners4: np.ndarray, n: int) -> np.ndarray:
    D    = int(corners4.shape[1])
    lens = np.linalg.norm(np.roll(corners4, -1, axis=0) - corners4, axis=1)
    L    = float(lens.sum())
    out  = np.zeros((n, D), dtype=np.float64)
    if L < 1e-12:
        return np.repeat(corners4[:1].astype(np.float64), n, axis=0)
    cum = np.zeros(5, dtype=np.float64)
    for i in range(4):
        cum[i + 1] = cum[i] + lens[i]
    for i in range(n):
        t = ((i + 0.5) / n) * L
        for k in range(4):
            if cum[k + 1] >= t - 1e-15:
                u = (t - cum[k]) / (lens[k] + 1e-12)
                out[i] = corners4[k] * (1.0 - u) + corners4[(k + 1) % 4] * u
                break
    return out


def _sample_mask_contour(mask_u8: np.ndarray, n: int) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cnt is None or len(cnt) < 4:
        return None
    poly = cnt.reshape(-1, 2).astype(np.float64)
    d    = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    L    = float(d.sum())
    if L < 1e-9:
        return None
    cum = np.zeros(len(poly) + 1, dtype=np.float64)
    for i in range(len(poly)):
        cum[i + 1] = cum[i] + d[i]
    out = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        t = ((i + 0.5) / n) * L
        k = int(np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(poly) - 1))
        u = (t - cum[k]) / (d[k] + 1e-12)
        out[i] = poly[k] * (1.0 - u) + poly[(k + 1) % len(poly)] * u
    return out


def _rotation_delta_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    Ra, _ = cv2.Rodrigues(rvec_a.astype(np.float64))
    Rb, _ = cv2.Rodrigues(rvec_b.astype(np.float64))
    R  = Ra @ Rb.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))))

# ---------------------------------------------------------------------------
# Quaternion helpers  (prevent axis-flip ambiguity in rvec tracking)
# ---------------------------------------------------------------------------

def _rvec_to_quat(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues vector → unit quaternion [w, x, y, z]."""
    R, _ = cv2.Rodrigues(rvec.reshape(3).astype(np.float64))
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25 / s, (m[2,1]-m[1,2])*s, (m[0,2]-m[2,0])*s, (m[1,0]-m[0,1])*s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
        return np.array([(m[2,1]-m[1,2])/s, 0.25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
        return np.array([(m[0,2]-m[2,0])/s, (m[0,1]+m[1,0])/s, 0.25*s, (m[1,2]+m[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
        return np.array([(m[1,0]-m[0,1])/s, (m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, 0.25*s])


def _quat_to_rvec(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] → Rodrigues vector."""
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)

# ---------------------------------------------------------------------------
# Kalman filter  (constant-velocity model per scalar)
# ---------------------------------------------------------------------------

class KalmanScalar:
    """1-D constant-velocity Kalman filter. State = [position, velocity]."""
    def __init__(self, process_var: float, meas_var: float):
        self.q = max(float(process_var), 1e-12)
        self.r = max(float(meas_var),    1e-12)
        self.x: float | None = None
        self.v: float = 0.0
        self.P = np.eye(2, dtype=np.float64)

    def filter(self, z: float) -> float:
        z = float(z)
        if self.x is None:
            self.x = z; self.v = 0.0; self.P = np.eye(2, dtype=np.float64); return z
        x_p = self.x + self.v
        v_p = self.v
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.array([[self.q, 0.0], [0.0, self.q * 0.1]])
        P_p = F @ self.P @ F.T + Q
        S   = P_p[0, 0] + self.r
        K0  = P_p[0, 0] / S
        K1  = P_p[1, 0] / S
        inn = z - x_p
        self.x = x_p + K0 * inn
        self.v = v_p + K1 * inn
        self.P = (np.eye(2) - np.outer([K0, K1], [1.0, 0.0])) @ P_p
        return self.x


def _apply_kalman_pose_filter(state, rv, tv, process_var, meas_var):
    """Track rotation as quaternion (with sign continuity) + translation."""
    q_new = _rvec_to_quat(rv)
    q_prev = state.get("_kf_quat_prev")
    if q_prev is not None and float(np.dot(q_new, q_prev)) < 0.0:
        q_new = -q_new

    filters = state.get("kalman_filters")
    if filters is None or len(filters) != 7:
        filters = [KalmanScalar(process_var, meas_var) for _ in range(7)]
        state["kalman_filters"] = filters

    vec = np.concatenate([q_new, tv.reshape(3)]).astype(np.float64)
    out = np.array([filters[i].filter(float(vec[i])) for i in range(7)])

    q_filt = out[:4] / (np.linalg.norm(out[:4]) + 1e-12)
    state["_kf_quat_prev"] = q_filt.copy()
    return _quat_to_rvec(q_filt).reshape(3, 1), out[4:].reshape(3, 1)

# ---------------------------------------------------------------------------
# MeshPoseEstimator  —  dense minAreaRect PnP  (from live_track.py)
# ---------------------------------------------------------------------------

# fal SAM3D GLBs (glTF): longest axis is normalized to 1 m — not real tool size.
SAM3D_NORMALIZED_AXIS_M = 1.0
SAM3D_METRES_SPAN_MAX = 3.0   # raw GLB max extent ≤ this → treat as metres
SAM3D_CM_SPAN_MIN = 15.0      # repaired / already ×100 → max extent in cm


def _convert_mesh_to_cm(mesh) -> tuple:
    """
    Put SAM3D GLB vertices in centimetres for checkerboard-calibrated PnP.

    fal SAM3D outputs meshes with the longest axis at 1 m (glTF metres).
    span ≤ 3 m → ×100.  Already-converted repaired caches (~100 cm) are left
    as-is.  Legacy mm caches (span > 250) are ×0.1.
    """
    mesh = mesh.copy()
    span = float(np.ptp(np.asarray(mesh.vertices, dtype=np.float64), axis=0).max())
    if span <= SAM3D_METRES_SPAN_MAX:
        mesh.apply_scale(100.0)
        return (
            mesh, 100.0,
            f"SAM3D 1 m axis → cm (raw extent {span:.4g} m, now {span * 100:.1f} cm)",
        )
    if span > 250.0:
        mesh.apply_scale(0.1)
        return mesh, 0.1, f"legacy mm→cm (span {span:.1f})"
    if span >= SAM3D_CM_SPAN_MIN:
        return mesh, 1.0, f"already cm (extent {span:.1f} cm)"
    return mesh, 1.0, f"unchanged (extent {span:.2g}, assumed cm)"


class MeshPoseEstimator:
    """
    6DoF pose estimator for a surgical tool given its 3D mesh.

    Uses minAreaRect on the mask for 2D correspondences matched to the
    model rectangle with dense perimeter sampling + ITERATIVE PnP.
    Axis-sign ambiguity resolved once by IoU bootstrap. Pose smoothed
    per-component with 6 independent Kalman filters.
    """

    def __init__(self, mesh):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces,    dtype=np.int32)
        cen   = verts.mean(0)
        v     = verts - cen
        cov   = (v.T @ v) / len(v)
        eigvals, evecs = np.linalg.eigh(cov)
        order   = np.argsort(eigvals)[::-1]
        aligned = v @ evecs[:, order]
        self.mesh_vertices = aligned.astype(np.float64)
        self.mesh_faces    = faces

        hP = (aligned[:, 0].max() - aligned[:, 0].min()) / 2
        hS = (aligned[:, 1].max() - aligned[:, 1].min()) / 2
        hT = (aligned[:, 2].max() - aligned[:, 2].min()) / 2
        self.hP, self.hS, self.hT = hP, hS, hT
        self.extents = np.array([2 * hP, 2 * hS, 2 * hT])

        self.model_pts = np.array(
            [[-hP, -hS, 0.0], [hP, -hS, 0.0], [hP, hS, 0.0], [-hP, hS, 0.0]],
            dtype=np.float64,
        )

        ax = hP * 1.0
        ay = hS * 1.5
        az = max(hT * 5.0, hP * 0.35)
        self.axis_pts = np.array(
            [[0.0, 0.0, 0.0], [ax, 0.0, 0.0], [0.0, ay, 0.0], [0.0, 0.0, az]],
            dtype=np.float64,
        )
        self._pnp_n = 4 * PNP_PER_EDGE

    def _pnp_best_pts(self, model_pts, img_corners, K, dist):
        best_rv, best_tv, best_err = None, None, np.inf
        for shift in range(4):
            pts = np.roll(model_pts, shift, axis=0)
            for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
                ok, rv, tv = cv2.solvePnP(pts, img_corners, K, dist, flags=flag)
                if not ok:
                    continue
                proj, _ = cv2.projectPoints(pts, rv, tv, K, dist)
                err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_corners, axis=1)))
                if err < best_err:
                    best_err, best_rv, best_tv = err, rv, tv
        return best_rv, best_tv, best_err

    def _pnp_best_dense(self, model_corners4, img_corners4, K, dist,
                        prev_rvec=None, prev_tvec=None, prev_shift=None,
                        img_pts_n=None):
        n        = self._pnp_n
        img_n    = img_pts_n if img_pts_n is not None else _sample_quad_perimeter(img_corners4.astype(np.float64), n)
        model_n0 = _sample_quad_perimeter(model_corners4.astype(np.float64), n)
        step     = n // 4
        best_rv, best_tv, best_err, best_shift = None, None, np.inf, None
        best_score = np.inf
        for shift in range(4):
            model_n = np.roll(model_n0, shift * step, axis=0)
            ok, rv, tv = cv2.solvePnP(model_n, img_n, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                ok2, rv2, tv2 = cv2.solvePnP(model_n, img_n, K, dist, flags=cv2.SOLVEPNP_EPNP)
                if not ok2:
                    continue
                ok3, rv, tv = cv2.solvePnP(model_n, img_n, K, dist, rv2, tv2,
                                            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok3:
                    rv, tv = rv2, tv2
            proj, _ = cv2.projectPoints(model_n, rv, tv, K, dist)
            err   = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_n, axis=1)))
            score = err
            if prev_rvec is not None and prev_tvec is not None:
                d_rot = _rotation_delta_deg(rv, prev_rvec)
                z_ref = max(abs(float(prev_tvec[2, 0])), 1e-6)
                d_t   = float(np.linalg.norm(tv.reshape(3) - prev_tvec.reshape(3)) / z_ref)
                score += PNP_ROT_SMOOTH_W * d_rot + PNP_TRANS_SMOOTH_W * d_t
            if prev_shift is not None and shift != int(prev_shift):
                score += PNP_SHIFT_PENALTY
            if score < best_score:
                best_score = score
                best_err, best_rv, best_tv, best_shift = err, rv, tv, shift
        return best_rv, best_tv, best_err, best_shift

    def estimate_pose(self, mask_bool, K, dist, state,
                      kalman_process_var=KALMAN_PROCESS_VAR,
                      kalman_meas_var=KALMAN_MEAS_VAR) -> dict:
        if state is None:
            state = {}
        if not np.any(mask_bool):
            return state

        # # Step 1: Erosion — commented out
        # clean = cv2.erode(mask_bool.astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
        # ys, xs = np.where(clean)
        # if len(xs) < 20:
        #     return state

        # # Step 2: minAreaRect — commented out
        # p32     = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        # rect    = cv2.minAreaRect(p32.reshape(-1, 1, 2))
        # img_corners = _order_corners_clockwise(cv2.boxPoints(rect).astype(np.float64))

        mask_u8 = mask_bool.astype(np.uint8) * 255
        img_n = _sample_mask_contour(mask_u8, self._pnp_n)
        if img_n is None:
            return state

        if "reg_sign" not in state:
            best_key = (-1.0, np.inf)
            best_rv, best_tv, best_s, best_shift = None, None, None, None
            for bits in range(8):
                s = np.array([-1.0 if (bits >> i) & 1 else 1.0 for i in range(3)], dtype=np.float64)
                rv, tv, err, shift = self._pnp_best_dense(self.model_pts * s, None, K, dist,
                                                          img_pts_n=img_n)
                if rv is None:
                    continue
                iou = _mesh_projection_iou(mask_bool, self.mesh_vertices * s,
                                           self.mesh_faces, rv, tv, K, dist)
                if iou > best_key[0] or (iou == best_key[0] and err < best_key[1]):
                    best_key = (iou, err)
                    best_rv, best_tv, best_s = rv, tv, s.copy()
                    best_shift = shift
            if best_rv is None or best_s is None:
                return state
            state["reg_sign"] = best_s
            rv, tv = best_rv, best_tv
            state["pnp_shift"] = 0 if best_shift is None else int(best_shift)
        else:
            s = _reg_sign_from_state(state)
            _rv = state.get("rvec_raw"); prev_rv = _rv if _rv is not None else state.get("rvec")
            _tv = state.get("tvec_raw"); prev_tv = _tv if _tv is not None else state.get("tvec")
            rv, tv, _, shift = self._pnp_best_dense(
                self.model_pts * s, None, K, dist,
                prev_rvec=prev_rv, prev_tvec=prev_tv,
                prev_shift=state.get("pnp_shift"),
                img_pts_n=img_n,
            )
            if rv is None:
                return state
            if shift is not None:
                state["pnp_shift"] = int(shift)

        state["rvec_raw"] = rv.copy()
        state["tvec_raw"] = tv.copy()
        rv, tv = _apply_kalman_pose_filter(state, rv, tv, kalman_process_var, kalman_meas_var)
        state["rvec"] = rv
        state["tvec"] = tv
        return state

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _draw_pose_axes(vis, state, K, dist, axis_pts, obj_id) -> None:
    """Project and draw X(red)/Y(green)/Z(orange) pose axes onto vis."""
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None:
        return
    s = _reg_sign_from_state(state)
    proj, _ = cv2.projectPoints((axis_pts * s).astype(np.float64), rvec, tvec, K, dist)
    pts2d = proj.reshape(-1, 2)
    fh, fw = vis.shape[:2]

    MAX_ARROW_PX = 40.0

    def clip_pt(p):
        return (int(np.clip(p[0], 0, fw - 1)), int(np.clip(p[1], 0, fh - 1)))

    def cap(o_f, t_f):
        v = t_f - o_f
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return o_f
        return o_f + v * (MAX_ARROW_PX / n) if n > MAX_ARROW_PX else t_f

    o_f    = pts2d[0].astype(np.float64)
    origin = clip_pt(o_f)
    x_tip  = clip_pt(cap(o_f, pts2d[1].astype(np.float64)))
    y_tip  = clip_pt(cap(o_f, pts2d[2].astype(np.float64)))
    z_tip  = clip_pt(cap(o_f, pts2d[3].astype(np.float64)))

    cv2.arrowedLine(vis, origin, x_tip, (0,   0, 220), 2, tipLength=0.20, line_type=cv2.LINE_AA)
    cv2.arrowedLine(vis, origin, y_tip, (0, 200,   0), 2, tipLength=0.20, line_type=cv2.LINE_AA)
    cv2.arrowedLine(vis, origin, z_tip, (220, 80,  0), 2, tipLength=0.20, line_type=cv2.LINE_AA)
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, "X", (x_tip[0] + 4, x_tip[1] + 4), fnt, 0.50, (0,   0, 220), 1, cv2.LINE_AA)
    cv2.putText(vis, "Y", (y_tip[0] + 4, y_tip[1] + 4), fnt, 0.50, (0, 200,   0), 1, cv2.LINE_AA)
    cv2.putText(vis, "Z", (z_tip[0] + 4, z_tip[1] + 4), fnt, 0.50, (220, 80,  0), 1, cv2.LINE_AA)


def _draw_pose_hud(vis: np.ndarray, pose_states: dict, use_metric: bool = False) -> None:
    if not pose_states:
        return
    x0, y0, line_h = 12, 18, 18
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    for row, oid in enumerate(sorted(pose_states)):
        st   = pose_states.get(oid, {})
        rvec = st.get("rvec")
        if rvec is None:
            continue
        if use_metric and st.get("tvec_cm") is not None:
            tv3 = np.asarray(st["tvec_cm"], dtype=np.float64).reshape(3)
            t_label = "cm"
        else:
            tvec = st.get("tvec_cal") or st.get("tvec")
            if tvec is None:
                continue
            tv3 = np.asarray(tvec, dtype=np.float64).reshape(3)
            t_label = "cm" if st.get("tvec_cal") is not None else "u"
        euler = st.get("euler_deg")
        if euler is None:
            rx, ry, rz = _pose_to_euler_zyx_deg(rvec)
        else:
            rx, ry, rz = (float(euler[0]), float(euler[1]), float(euler[2]))
        tx, ty, tz = float(tv3[0]), float(tv3[1]), float(tv3[2])
        text = f"ID{oid}  R({rx:.0f},{ry:.0f},{rz:.0f})deg  T({tx:.1f},{ty:.1f},{tz:.1f}){t_label}"
        yy   = y0 + row * line_h
        cv2.putText(vis, text, (x0, yy), fnt, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, text, (x0, yy), fnt, 0.48, (0, 0, 0), 1, cv2.LINE_AA)


def _sam3d_registration_depth_scale(
    est: "MeshPoseEstimator",
    mask_bool: np.ndarray,
    K: np.ndarray,
    tvec,
) -> float:
    """
    Correct PnP depth when SAM3D normalizes longest axis to 1 m.

    Uses pinhole consistency on the seed mask: Z_cm = f * W_mesh_cm / L_px,
    then depth_scale = Z_cm / tz_pnp so HUD depth matches apparent size.
    """
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    tz = abs(float(tvec[2]))
    if tz < 1e-9:
        return 1.0
    L_px = _mask_major_axis_px(mask_bool)
    if L_px < 1e-3:
        return 1.0
    f = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
    w_cm = float(np.max(est.extents))
    if w_cm < 1e-6:
        return 1.0
    z_pinhole = f * w_cm / L_px
    return z_pinhole / tz


def _mask_major_axis_px(mask_bool: np.ndarray) -> float:
    ys, xs = np.where(mask_bool)
    if xs.size < 4:
        return 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
    _, (w, h), _ = cv2.minAreaRect(pts)
    return float(max(w, h))


def _compute_cm_scale(
    mask_bool: np.ndarray,
    K: np.ndarray,
    tvec,
    *,
    surface_distance_cm: float,
    tool_width_cm: float,
    mesh_in_cm: bool,
    registration_depth_scale: float = 1.0,
) -> float | None:
    """
    Scale PnP tvec to centimetres for HUD / CSV.

    SAM3D mesh solvePnP returns tvec in millimetres; checkerboard calibration
    uses centimetres. With mesh_in_cm, registration_depth_scale (pinhole fit on
    the seed mask) is multiplied by PNP_TVEC_MM_TO_CM. CLI overrides
    (surface distance, tool width) already match cm against mm depth.
    """
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    tz = max(abs(float(t[2])), 1e-9)
    if surface_distance_cm > 0:
        return float(surface_distance_cm) / tz
    if tool_width_cm > 0:
        f = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
        L_px = _mask_major_axis_px(mask_bool)
        if L_px < 1e-3:
            return None
        return (f * float(tool_width_cm) / L_px) / tz
    if mesh_in_cm:
        return max(float(registration_depth_scale), 1e-9) * PNP_TVEC_MM_TO_CM
    return None


def _update_metric_pose_display(
    state: dict,
    mask_bool: np.ndarray,
    K: np.ndarray,
    *,
    surface_distance_cm: float,
    tool_width_cm: float,
    use_metric: bool,
    mesh_in_cm: bool,
    ema_alpha: float = 0.12,
) -> None:
    """Populate state['tvec_cm'] and state['euler_deg'] for HUD / CSV."""
    if not use_metric:
        return
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None:
        return
    scale_now = _compute_cm_scale(
        mask_bool, K, tvec,
        surface_distance_cm=surface_distance_cm,
        tool_width_cm=tool_width_cm,
        mesh_in_cm=mesh_in_cm,
        registration_depth_scale=float(state.get("registration_depth_scale", 1.0)),
    )
    if scale_now is None:
        return
    prev = state.get("cm_scale")
    scale = scale_now if prev is None else (1.0 - ema_alpha) * float(prev) + ema_alpha * scale_now
    state["cm_scale"] = scale
    state["tvec_cm"] = np.asarray(tvec, dtype=np.float64).reshape(3) * scale
    state["euler_deg"] = _pose_to_euler_zyx_deg(rvec)


def _update_translation_calibration_from_surface(state, surface_distance_cm, ema_alpha=0.15):
    if surface_distance_cm <= 0:
        return
    tvec = state.get("tvec")
    if tvec is None:
        return
    tz_raw = abs(float(tvec[2]))
    if tz_raw < 1e-9:
        return
    scale_now = float(surface_distance_cm) / tz_raw
    prev  = state.get("tvec_cal_scale")
    scale = scale_now if prev is None else (1.0 - ema_alpha) * float(prev) + ema_alpha * scale_now
    state["tvec_cal_scale"] = scale
    state["tvec_cal"] = np.asarray(tvec, dtype=np.float64) * scale


def _pose_to_euler_zyx_deg(rvec) -> tuple[float, float, float]:
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
    i = 1
    while True:
        p = base_dir / f"poses{i}.csv"
        if not p.exists():
            return p
        i += 1


def _draw_registration_debug(canvas, mask_bool, state, est, K, dist, obj_id) -> None:
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None or not np.any(mask_bool):
        return
    fh, fw    = canvas.shape[:2]
    pred_mask = np.zeros((fh, fw), dtype=np.uint8)
    s = _reg_sign_from_state(state)
    proj_mesh, _ = cv2.projectPoints(est.mesh_vertices * s, rvec, tvec, K, dist)
    pts2d   = proj_mesh.reshape(-1, 2)
    overlay = canvas.copy()
    for f in est.mesh_faces:
        poly = np.round(pts2d[f]).astype(np.int32)
        cv2.fillConvexPoly(overlay, poly, (180, 130, 255), lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [poly], True, (255, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(pred_mask, poly, 255, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0, dst=canvas)
    mu8       = mask_bool.astype(np.uint8) * 255
    sam_color = np.zeros_like(canvas)
    sam_color[:, :, 0] = mu8; sam_color[:, :, 1] = mu8
    cv2.addWeighted(sam_color, 0.30, canvas, 1.0, 0.0, dst=canvas)
    box_mesh = cv2.boxPoints(cv2.minAreaRect(pts2d.astype(np.float32))).astype(np.int32)
    cv2.polylines(canvas, [box_mesh], True, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    proj, _ = cv2.projectPoints(est.model_pts * s, rvec, tvec, K, dist)
    poly    = np.round(proj.reshape(-1, 2)).astype(np.int32)
    if poly.shape[0] >= 3:
        cv2.polylines(canvas, [poly], True, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    n        = int(est._pnp_n)
    mask_pts = _sample_mask_contour(mu8, n)
    mesh_pts = _sample_mask_contour(pred_mask, n)
    if mask_pts is not None and mesh_pts is not None:
        for p in mask_pts:
            cv2.circle(canvas, (int(round(float(p[0]))), int(round(float(p[1])))), 2, (255, 255, 0), -1)
        for p in mesh_pts:
            cv2.circle(canvas, (int(round(float(p[0]))), int(round(float(p[1])))), 2, (255, 0, 255), -1)
    inter = float(np.logical_and(pred_mask > 0, mask_bool).sum())
    union = float(np.logical_or(pred_mask > 0, mask_bool).sum())
    iou   = inter / union if union > 0.0 else 0.0
    ys, xs = np.where(mask_bool)
    tx, ty = (int(xs.mean()), int(ys.mean())) if len(xs) > 0 else (12, 24)
    text = f"ID{obj_id} reg IoU={iou:.3f}"
    cv2.putText(canvas, text, (tx + 8, ty + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (tx + 8, ty + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Live camera frame provider
# ---------------------------------------------------------------------------

class LiveFrameProvider:
    """Always-latest frame provider: capture thread overwrites a single slot."""

    def __init__(self, cap: cv2.VideoCapture, image_size: int, rotate_180: bool):
        self.cap        = cap
        self.image_size = image_size
        self.rotate_180 = rotate_180
        self._cache: dict = {}
        self._latest_tensor = None
        self._latest_raw: np.ndarray | None = None
        self._lock = threading.Lock()

    def _encode(self, frame):
        frame = preprocess(frame, self.rotate_180)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t     = torch.from_numpy(
            cv2.resize(rgb, (self.image_size, self.image_size))
        ).float().div(255.0).permute(2, 0, 1)
        return (t - _IMG_MEAN) / _IMG_STD, frame

    def capture_next(self) -> bool:
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
            self._latest_raw    = raw
        return True

    def __len__(self):
        return 1_000_000

    def __getitem__(self, idx: int):
        while True:
            with self._lock:
                if self._latest_tensor is not None:
                    if idx not in self._cache:
                        self._cache[idx] = (self._latest_tensor, self._latest_raw)
                        if len(self._cache) > 32:
                            del self._cache[min(self._cache)]
                    return self._cache[idx][0]
            time.sleep(0.001)

    def get_raw(self, idx: int) -> np.ndarray | None:
        with self._lock:
            entry = self._cache.get(idx)
            return entry[1] if entry is not None else self._latest_raw

# ---------------------------------------------------------------------------
# fal SAM3D GLB generation
# ---------------------------------------------------------------------------

def _fal_set_status(
    status: dict[int, str] | None,
    lock: threading.Lock | None,
    oid: int | None,
    text: str,
) -> None:
    if status is None or oid is None:
        return
    if lock is not None:
        with lock:
            status[oid] = text
    else:
        status[oid] = text


def _fal_download_glb(
    fal_model,
    seed,
    image_url,
    mask_path,
    glb_out,
    *,
    oid: int | None = None,
    status: dict[int, str] | None = None,
    status_lock: threading.Lock | None = None,
) -> tuple[bool, str, dict]:
    timings = {"upload_s": 0.0, "infer_s": 0.0, "download_s": 0.0, "total_s": 0.0}
    t_total0 = time.perf_counter()
    try:
        import fal_client  # type: ignore
    except Exception as e:
        _fal_set_status(status, status_lock, oid, f"ERR: {e}")
        return False, f"fal_client import failed: {e}", timings
    try:
        _fal_set_status(status, status_lock, oid, "uploading mask…")
        t0 = time.perf_counter()
        mask_url = fal_client.upload_file(str(mask_path))
        timings["upload_s"] = time.perf_counter() - t0

        _fal_set_status(status, status_lock, oid, "SAM3D running…")
        t0 = time.perf_counter()
        result = fal_client.subscribe(
            fal_model,
            arguments={"image_url": image_url, "mask_urls": [mask_url], "seed": int(seed)},
            with_logs=False,
        )
        timings["infer_s"] = time.perf_counter() - t0
    except Exception as e:
        _fal_set_status(status, status_lock, oid, f"ERR: {str(e)[:40]}")
        timings["total_s"] = time.perf_counter() - t_total0
        return False, str(e), timings
    if not isinstance(result, dict):
        _fal_set_status(status, status_lock, oid, "ERR: unexpected result")
        timings["total_s"] = time.perf_counter() - t_total0
        return False, "unexpected fal result type", timings
    model_glb = result.get("model_glb") or {}
    url = model_glb.get("url") if isinstance(model_glb, dict) else None
    if not isinstance(url, str) or not url:
        _fal_set_status(status, status_lock, oid, "ERR: no GLB URL")
        timings["total_s"] = time.perf_counter() - t_total0
        return False, "no model_glb.url in fal response", timings
    try:
        _fal_set_status(status, status_lock, oid, "downloading GLB…")
        t0 = time.perf_counter()
        glb_out.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, str(glb_out))
        timings["download_s"] = time.perf_counter() - t0
    except Exception as e:
        _fal_set_status(status, status_lock, oid, f"ERR: {str(e)[:40]}")
        timings["total_s"] = time.perf_counter() - t_total0
        return False, f"download failed: {e}", timings
    timings["total_s"] = time.perf_counter() - t_total0
    ok = glb_out.is_file()
    _fal_set_status(status, status_lock, oid, "done" if ok else "ERR: file missing")
    return ok, "ok" if ok else "glb not written", timings


def _format_fal_result_log(oid: int, ok: bool, msg: str, timings: dict) -> str:
    if ok and timings:
        return (
            f"  ID{oid} fal: OK — upload {timings['upload_s']:.1f}s, "
            f"infer {timings['infer_s']:.1f}s, download {timings['download_s']:.1f}s "
            f"(total {timings['total_s']:.1f}s)"
        )
    return f"  ID{oid} fal: FAIL — {msg}"


def _wait_fal_progress_ui(
    seed_frame,
    seed_mask_bool,
    futures_map,
    fal_status: dict[int, str],
    status_lock: threading.Lock,
    win_name,
) -> dict:
    results: dict = {}

    def draw():
        vis = seed_frame.copy().astype(np.float32)
        for oid in sorted(seed_mask_bool.keys()):
            c = np.array(point_color(int(oid)), dtype=np.float32)
            vis[seed_mask_bool[oid]] = vis[seed_mask_bool[oid]] * 0.35 + c * 0.65
        vis = vis.astype(np.uint8)
        y = 26
        cv2.putText(vis, "fal SAM3D (parallel per ID)", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 22
        with status_lock:
            status_snapshot = dict(fal_status)
        for oid in sorted(seed_mask_bool.keys()):
            line = status_snapshot.get(oid, "queued…")
            cv2.putText(vis, f"ID{oid}: {line}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 255), 2)
            y += 20
        return vis

    cv2.namedWindow(win_name)
    pending = set(futures_map.keys())
    while pending:
        for fut in list(pending):
            if fut.done():
                oid = futures_map[fut]
                try:
                    ok, msg, timings = fut.result()
                except Exception as e:
                    ok, msg, timings = False, str(e), {}
                results[oid] = (ok, msg, timings)
                with status_lock:
                    fal_status[oid] = "done" if ok else f"ERR: {msg[:40]}"
                pending.discard(fut)
        cv2.imshow(win_name, draw())
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break
    for fut, oid in futures_map.items():
        if oid not in results:
            if fut.done():
                try:
                    ok, msg, timings = fut.result()
                except Exception as e:
                    ok, msg, timings = False, str(e), {}
                results[oid] = (ok, msg, timings)
            else:
                results[oid] = (False, "interrupted", {})
    cv2.destroyWindow(win_name)
    return results


def _confirm_seed_masks_ui(seed_frame, seed_mask_bool, win_name="Confirm seed masks") -> bool:
    ids = sorted(seed_mask_bool.keys())
    if not ids:
        return False
    vis = seed_frame.copy().astype(np.float32)
    for oid in ids:
        mb = seed_mask_bool.get(oid)
        if mb is None or not np.any(mb):
            continue
        c = np.array(point_color(int(oid)), dtype=np.float32)
        vis[mb] = vis[mb] * 0.35 + c * 0.65
        ys, xs = np.where(mb)
        if xs.size > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            for thick, col in [(2, (255, 255, 255)), (1, (0, 0, 0))]:
                cv2.putText(vis, f"ID{oid}", (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, thick, cv2.LINE_AA)
    vis_u8 = vis.astype(np.uint8)
    cv2.namedWindow(win_name)
    while True:
        panel = vis_u8.copy()
        cv2.putText(panel, "Confirm masks before SAM3D upload",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "Enter / Y: continue    R / N / ESC: abort",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 255), 2, cv2.LINE_AA)
        cv2.imshow(win_name, panel)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10, ord("y"), ord("Y")):
            cv2.destroyWindow(win_name); return True
        if k in (ord("r"), ord("R"), ord("n"), ord("N"), ord("q"), 27):
            cv2.destroyWindow(win_name); return False

# ---------------------------------------------------------------------------
# GLB loader → MeshPoseEstimator
# ---------------------------------------------------------------------------

def _load_mesh_and_register(oid, glb_path, mb, K, dist, mesh_in_cm: bool = False) -> tuple:
    if trimesh is None:
        print("trimesh required — pip install trimesh")
        return oid, None, {}

    repaired_path = glb_path.with_name(glb_path.stem + "_repaired.glb")
    use_cache = (
        repaired_path.exists()
        and repaired_path.stat().st_mtime >= glb_path.stat().st_mtime
    )
    load_path = repaired_path if use_cache else glb_path

    try:
        loaded = trimesh.load(str(load_path), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not parts:
                raise ValueError("no mesh geometry in scene")
            mesh = trimesh.util.concatenate(parts)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise ValueError(f"unsupported mesh type: {type(loaded)}")

        if mesh_in_cm:
            mesh, _, cm_reason = _convert_mesh_to_cm(mesh)
            print(f"[ID{oid}] mesh units: {cm_reason}")

        if not use_cache:
            try:
                from trimesh import repair as _r
                if hasattr(_r, "fill_holes"):
                    _r.fill_holes(mesh)
            except Exception:
                pass
            try:
                mesh.export(str(repaired_path))
            except Exception:
                pass
    except Exception as e:
        print(f"Failed to load GLB for ID{oid}: {e}")
        return oid, None, {}

    est = MeshPoseEstimator(mesh)
    st: dict = {}
    if np.any(mb):
        st = est.estimate_pose(mb, K, dist, st)
        if mesh_in_cm and st.get("tvec") is not None:
            ds = _sam3d_registration_depth_scale(est, mb, K, st["tvec"])
            st["registration_depth_scale"] = ds
            tz_mm = abs(float(np.asarray(st["tvec"], dtype=np.float64).reshape(3)[2]))
            tz_cm = tz_mm * PNP_TVEC_MM_TO_CM
            print(
                f"[ID{oid}] SAM3D depth scale: {ds:.3f}  "
                f"(seed tz={tz_cm:.1f} cm, raw {tz_mm:.1f} mm → ~{tz_cm * ds:.1f} cm after pinhole fit)"
            )
    print(f"[ID{oid}] estimator ready  pnp_n={est._pnp_n}"
          + ("  (repaired cache)" if use_cache else ""))
    return oid, est, st

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args) -> None:
    device = choose_device(args.device)
    print(f"Device: {device}")

    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT}"); return
    if trimesh is None:
        print("Install trimesh: pip install trimesh"); return
    if not os.environ.get("FAL_KEY"):
        print("Set FAL_KEY environment variable."); return

    print("Loading EdgeTAM …")
    predictor = _load_predictor(device)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}"); return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    rotate_180 = detect_orbbec_camera(args.camera)
    print(f"Camera {args.camera}: {'Orbbec detected, rotating 180°' if rotate_180 else 'non-Orbbec'}")

    K_cam, dist_cam, intr_src = _resolve_pose_intrinsics(
        cap, TARGET_SIZE[0], TARGET_SIZE[1],
        args.intrinsics_file, not args.no_orbbec_intrinsics,
    )
    metric_ctx = _load_metric_context(args.intrinsics_file, intr_src)
    print(f"Intrinsics ({intr_src}): fx={K_cam[0,0]:.1f}  fy={K_cam[1,1]:.1f}  "
          f"cx={K_cam[0,2]:.1f}  cy={K_cam[1,2]:.1f}")
    if metric_ctx["use_metric"]:
        sq = metric_ctx["square_cm"]
        print(
            f"Pose display: rotation (deg), translation (cm) — "
            f"checkerboard cal ({sq:g} cm squares); SAM3D mesh scaled to cm for PnP"
        )
        if args.surface_distance_cm > 0 or args.tool_width_cm > 0:
            print("  Optional scale override: --surface-distance-cm / --tool-width-cm")
    elif args.surface_distance_cm > 0:
        print(f"Pose display: translation scaled to cm via surface distance {args.surface_distance_cm:g} cm")

    image_size = predictor.image_size
    provider   = LiveFrameProvider(cap, image_size, rotate_180)
    if not provider.capture_next():
        print("No frame from camera."); cap.release(); return

    stop_flag = threading.Event()

    def _capture_loop():
        while not stop_flag.is_set():
            if not provider.capture_next():
                stop_flag.set()

    threading.Thread(target=_capture_loop, daemon=True).start()

    points, seed_frame = pick_points_live(provider, stop_flag)
    if not points or seed_frame is None:
        print("No seed selection; exiting.")
        stop_flag.set(); cap.release(); return

    fh, fw = seed_frame.shape[:2]

    # Start background fal seed-frame upload immediately.
    work_dir = Path(args.glb_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    seed_png = work_dir / "seed_frame.png"
    cv2.imwrite(str(seed_png), seed_frame)
    _seed_upload_ex  = None
    seed_url_future  = None
    try:
        import fal_client as _fal_pre  # type: ignore
        _seed_upload_ex = ThreadPoolExecutor(max_workers=1)
        seed_url_future = _seed_upload_ex.submit(_fal_pre.upload_file, str(seed_png))
        print("fal: seed frame upload started in background …")
    except Exception as _pre_e:
        print(f"fal: background seed upload skipped ({_pre_e})")

    # EdgeTAM init
    tmp = tempfile.mkdtemp(prefix="edgetam_pose_any_")
    cv2.imwrite(os.path.join(tmp, "000000.jpg"), seed_frame)
    state = predictor.init_state(tmp, async_loading_frames=False)
    shutil.rmtree(tmp, ignore_errors=True)
    state["images"]     = provider
    state["num_frames"] = 1_000_000

    for obj_id, x, y in points:
        predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=int(obj_id),
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

    ac_device, ac_dtype, ac_enabled = _autocast_config(device, args.half)

    # Compute seed masks from frame 0.
    seed_masks_raw: dict[int, np.ndarray] = {}
    print("EdgeTAM: resolving seed masks (frame 0)…")
    t0 = time.perf_counter()
    with torch.autocast(device_type=ac_device, dtype=ac_dtype, enabled=ac_enabled):
        gen0 = predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=0)
        _, obj_ids0, masks0 = next(gen0)
        ids0      = [int(x) for x in (obj_ids0.tolist() if hasattr(obj_ids0, "tolist") else obj_ids0)]
        masks_np0 = masks0.detach().cpu().numpy()
        for i in range(min(len(ids0), masks_np0.shape[0])):
            seed_masks_raw[ids0[i]] = _mask_to_2d_bool(masks_np0[i], fh, fw)
    print(f"Timing: EdgeTAM seed masks = {time.perf_counter()-t0:.3f}s")

    # User confirms masks before fal upload.
    print("Please confirm seed masks before sending to SAM3D …")
    if not _confirm_seed_masks_ui(seed_frame, seed_masks_raw):
        print("Aborted."); stop_flag.set(); cap.release(); cv2.destroyAllWindows(); return

    # Write mask PNGs.
    t_3d0 = time.perf_counter()
    for oid, mb in seed_masks_raw.items():
        cv2.imwrite(str(work_dir / f"mask_{oid}.png"), mb.astype(np.uint8) * 255)

    import fal_client  # type: ignore
    if seed_url_future is not None:
        print("fal: collecting background seed upload …")
        image_url = seed_url_future.result()
        if _seed_upload_ex is not None:
            _seed_upload_ex.shutdown(wait=False)
    else:
        print("Uploading seed image to fal …")
        image_url = fal_client.upload_file(str(seed_png))

    # Download GLBs in parallel (per-ID status + timing).
    fal_status: dict[int, str] = {int(oid): "queued…" for oid in seed_masks_raw}
    fal_status_lock = threading.Lock()
    futures_map: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, len(seed_masks_raw))) as ex:
        for oid in sorted(seed_masks_raw.keys()):
            fut = ex.submit(
                _fal_download_glb,
                args.fal_model, args.seed, image_url,
                work_dir / f"mask_{oid}.png", work_dir / f"object_{oid}.glb",
                oid=int(oid),
                status=fal_status,
                status_lock=fal_status_lock,
            )
            futures_map[fut] = oid
        fal_results = _wait_fal_progress_ui(
            seed_frame, seed_masks_raw, futures_map,
            fal_status, fal_status_lock, "fal SAM3D",
        )

    for oid in sorted(fal_results.keys()):
        ok, msg, timings = fal_results[oid]
        print(_format_fal_result_log(int(oid), ok, msg, timings))
    print(f"Timing: fal SAM3D wall = {time.perf_counter()-t_3d0:.3f}s")

    # Load GLBs + initial pose in parallel.
    mesh_estimators: dict[int, MeshPoseEstimator] = {}
    pose_states: dict[int, dict] = {}
    print("Loading GLBs + initial pose (parallel) …")
    t_reg0 = time.perf_counter()
    n_glb  = sum(1 for oid in seed_masks_raw if (work_dir / f"object_{oid}.glb").is_file())
    glb_futs: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, n_glb)) as reg_ex:
        for oid in sorted(seed_masks_raw.keys()):
            glb_path = work_dir / f"object_{oid}.glb"
            if not glb_path.is_file():
                print(f"Missing GLB for ID{oid}, skipping."); continue
            mb  = seed_masks_raw.get(oid, np.zeros((fh, fw), dtype=bool))
            fut = reg_ex.submit(
                _load_mesh_and_register, oid, glb_path, mb, K_cam, dist_cam,
                metric_ctx["use_metric"],
            )
            glb_futs[fut] = oid
        for fut in as_completed(glb_futs):
            oid, est, st = fut.result()
            if est is not None:
                mesh_estimators[oid] = est
                pose_states[oid]     = st
                if metric_ctx["use_metric"]:
                    mb = seed_masks_raw.get(oid)
                    if mb is not None and np.any(mb):
                        _update_metric_pose_display(
                            pose_states[oid], mb, K_cam,
                            surface_distance_cm=args.surface_distance_cm,
                            tool_width_cm=args.tool_width_cm,
                            use_metric=True,
                            mesh_in_cm=True,
                        )
    print(f"Timing: load + seed pose = {time.perf_counter()-t_reg0:.3f}s")

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (fw, fh))

    csv_path   = _next_pose_csv_path(Path.cwd())
    csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    use_metric_csv = metric_ctx["use_metric"] or args.surface_distance_cm > 0
    if use_metric_csv:
        t_cols = ["tx_cm", "ty_cm", "tz_cm"]
    else:
        t_cols = ["tx", "ty", "tz"]
    csv_writer.writerow(["frame_idx", "time_s", "object_id", "rx_deg", "ry_deg", "rz_deg", *t_cols])
    print(f"Pose CSV: {csv_path}")

    com_trails: dict[int, list[tuple[int, int]]] = {}
    fps_t0, fps_frames = time.perf_counter(), 0

    print("Live tracking + pose. Press q / ESC to quit.")
    try:
        with torch.autocast(device_type=ac_device, dtype=ac_dtype, enabled=ac_enabled):
            for fi, obj_ids, masks in predictor.propagate_in_video(state, start_frame_idx=1):
                frame = provider.get_raw(fi)
                if frame is None:
                    frame = seed_frame
                debug_canvas = frame.copy() if args.align_debug_out else None

                ids      = [int(x) for x in (obj_ids.tolist() if hasattr(obj_ids, "tolist") else obj_ids)]
                vis      = overlay_masks(frame, ids, masks, alpha=args.alpha)
                masks_np = masks.detach().cpu().numpy()

                for i in range(min(len(ids), masks_np.shape[0])):
                    oid  = ids[i]
                    binm = _mask_to_2d_bool(masks_np[i], fh, fw)
                    if not np.any(binm):
                        continue

                    ys, xs = np.where(binm)
                    cx, cy = int(xs.mean()), int(ys.mean())
                    com_trails.setdefault(oid, []).append((cx, cy))

                    est = mesh_estimators.get(oid)
                    if est is not None:
                        pose_states[oid] = est.estimate_pose(
                            binm, K_cam, dist_cam, pose_states.get(oid),
                            kalman_process_var=args.kalman_process_var,
                            kalman_meas_var=args.kalman_meas_var,
                        )
                        if metric_ctx["use_metric"]:
                            _update_metric_pose_display(
                                pose_states[oid], binm, K_cam,
                                surface_distance_cm=args.surface_distance_cm,
                                tool_width_cm=args.tool_width_cm,
                                use_metric=True,
                                mesh_in_cm=True,
                            )
                        elif args.surface_distance_cm > 0:
                            _update_translation_calibration_from_surface(
                                pose_states[oid], args.surface_distance_cm)
                        _draw_pose_axes(vis, pose_states[oid], K_cam, dist_cam, est.axis_pts, oid)
                        if debug_canvas is not None:
                            _draw_registration_debug(
                                debug_canvas, binm, pose_states[oid], est, K_cam, dist_cam, oid)
                    else:
                        continue

                for oid, trail in com_trails.items():
                    mt = max(8, int(args.max_trail))
                    if len(trail) > mt:
                        com_trails[oid] = trail[-mt:]
                    trail = com_trails[oid]
                    col   = point_color(oid)
                    for j in range(1, len(trail)):
                        cv2.line(vis, trail[j - 1], trail[j], col, 1, lineType=cv2.LINE_AA)
                    if trail:
                        cv2.circle(vis, trail[-1], 5, col, -1)
                        cv2.circle(vis, trail[-1], 7, (255, 255, 255), 1)
                        cv2.putText(vis, f"ID{oid}", (trail[-1][0] + 8, trail[-1][1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(vis, f"ID{oid}", (trail[-1][0] + 8, trail[-1][1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

                _draw_pose_hud(vis, pose_states, use_metric=metric_ctx["use_metric"])

                if fi == 0:
                    for oid, px, py in points:
                        cv2.circle(vis, (int(px), int(py)), 5, (0, 255, 255), -1)
                        cv2.putText(vis, f"ID{oid}", (int(px) + 8, int(py) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                cv2.imshow("EdgeTAM + 6DoF Pose", vis)
                if writer is not None:
                    writer.write(vis)
                if debug_canvas is not None:
                    cv2.imwrite(args.align_debug_out, debug_canvas)

                t_s = float(time.perf_counter())
                for oid in sorted(pose_states):
                    st   = pose_states.get(oid, {})
                    rvec = st.get("rvec")
                    if rvec is None:
                        continue
                    if use_metric_csv and st.get("tvec_cm") is not None:
                        tv3 = np.asarray(st["tvec_cm"], dtype=np.float64).reshape(3)
                    else:
                        _tc = st.get("tvec_cal")
                        tvec = _tc if _tc is not None else st.get("tvec")
                        if tvec is None:
                            continue
                        tv3 = np.asarray(tvec, dtype=np.float64).reshape(3)
                    rx, ry, rz = _pose_to_euler_zyx_deg(rvec)
                    csv_writer.writerow([int(fi), t_s, int(oid), rx, ry, rz,
                                         float(tv3[0]), float(tv3[1]), float(tv3[2])])

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("Exiting."); break
                if stop_flag.is_set():
                    print("Camera stalled."); break

                fps_frames += 1
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    print(f"FPS: {fps_frames / (now - fps_t0):.2f}")
                    fps_t0, fps_frames = now, 0
    finally:
        stop_flag.set()
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()
        cv2.destroyAllWindows()


def _apply_calibration_defaults(args) -> None:
    """Use saved checkerboard .npz when present; prefer it over Orbbec/FOV."""
    calib_path = Path(args.calibration_out).expanduser().resolve()
    if args.intrinsics_file:
        return
    if args.calibrate_checkerboard:
        return
    if calib_path.is_file():
        args.intrinsics_file = str(calib_path)
        args.no_orbbec_intrinsics = True
        print("Using checkerboard calibration:")
        _print_calibration_summary(calib_path)


def main() -> None:
    default_calib = str(DEFAULT_CALIBRATION_NPZ)
    parser = argparse.ArgumentParser(description="EdgeTAM + fal SAM3D + dense PnP 6DoF pose.")
    parser.add_argument("--camera",    type=int,   default=0)
    parser.add_argument("--device",    default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--alpha",     type=float, default=0.45)
    parser.add_argument("--half",      action="store_true", default=True)
    parser.add_argument("--no-half",   dest="half", action="store_false")
    parser.add_argument("--output",    default="")
    parser.add_argument("--kalman-process-var", type=float, default=KALMAN_PROCESS_VAR)
    parser.add_argument("--kalman-meas-var",    type=float, default=KALMAN_MEAS_VAR)
    parser.add_argument("--max-trail", type=int,   default=_MAX_COM_TRAIL)
    parser.add_argument("--fal-model", default="fal-ai/sam-3/3d-objects")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--glb-dir",
        default=str(Path(__file__).resolve().parent / "sam3d_live_objects"),
        help="Directory for seed PNG, masks, and downloaded GLBs.")
    parser.add_argument("--intrinsics-file", default="",
        help="Optional .npz with K (3x3), optional dist, optional width/height.")
    parser.add_argument("--show-calibration", action="store_true",
        help="Print RMS and metadata from --calibration-out .npz, then exit.")
    parser.add_argument("--calibrate-only", action="store_true",
        help="Run interactive checkerboard calibration and exit (no tracking).")
    parser.add_argument("--calibrate-checkerboard", action="store_true",
        help="Run checkerboard calibration before live tracking.")
    parser.add_argument("--calibration-out", default=default_calib,
        help="Path to save/load checkerboard calibration .npz.")
    parser.add_argument("--checkerboard-cols", type=int, default=CHECKERBOARD_INNER_COLS,
        help="Inner corners along board width (MRPT 9×7 PDF → 8).")
    parser.add_argument("--checkerboard-rows", type=int, default=CHECKERBOARD_INNER_ROWS,
        help="Inner corners along board height (MRPT 9×7 PDF → 6).")
    parser.add_argument("--checkerboard-square-cm", type=float, default=CHECKERBOARD_SQUARE_CM,
        help="Physical square size in cm (MRPT 9×7 PDF → 2 cm / 20 mm, print at 1:1 on A4).")
    parser.add_argument("--calibration-min-samples", type=int, default=15,
        help="Minimum captured views before calibrateCamera runs.")
    parser.add_argument("--no-orbbec-intrinsics", action="store_true",
        help="Skip Orbbec SDK intrinsics lookup.")
    parser.add_argument("--surface-distance-cm", type=float, default=0.0,
        help="Known camera→object distance in cm; overrides auto cm scale when set.")
    parser.add_argument("--tool-width-cm", type=float, default=0.0,
        help="Known tool width in cm for metric translation (optional).")
    parser.add_argument("--align-debug-out", default="",
        help="Optional path to save registration debug image each frame.")
    args = parser.parse_args()

    if args.show_calibration:
        calib_path = Path(args.calibration_out).expanduser().resolve()
        sys.exit(0 if _print_calibration_summary(calib_path) else 1)

    if args.calibrate_only:
        sys.exit(0 if run_checkerboard_calibration(args) else 1)

    if args.calibrate_checkerboard:
        if not run_checkerboard_calibration(args):
            sys.exit(1)
        if not args.intrinsics_file:
            args.intrinsics_file = str(Path(args.calibration_out).expanduser().resolve())
        args.no_orbbec_intrinsics = True

    _apply_calibration_defaults(args)
    run(args)


if __name__ == "__main__":
    main()
