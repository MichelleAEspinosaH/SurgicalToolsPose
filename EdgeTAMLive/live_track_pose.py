#!/usr/bin/env python3
"""
Real-time surgical tool segmentation and 6-DoF pose with EdgeTAM and SAM3D.

Pipeline
--------
1. Frames are captured from a live camera index,
   resized to ``TARGET_SIZE``, and normalized for the EdgeTAM video predictor.
2. The user places seed clicks (monotonic IDs; **ID1 is always scissors**).
   EdgeTAM propagates instance masks through the stream.
3. ID1 (scissors) loads a fixed local mesh from ``scissors.glb``.
   Other IDs use SAM3D (fal.ai) from their seed masks.
4. Per-object PoseEstimator aligns the 3D model silhouette to the 2D mask
   contour via PnP (phase search + Kalman smoothing) and draws XYZ axes.

Usage::

    .venv/bin/python live_track_pose.py               # camera 0
    .venv/bin/python live_track_pose.py --camera 1
    .venv/bin/python live_track_pose.py --no-half
    .venv/bin/python live_track_pose.py --output out.mp4

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
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlretrieve
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import minimize

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
SCISSORS_GLB = Path(__file__).parent / "scissors.glb"

_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_IMG_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

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

# ---------------------------------------------------------------------------
# Point-picking UI
# ---------------------------------------------------------------------------

def pick_points_live(
    provider: "LiveFrameProvider", stop_flag: threading.Event
) -> tuple[list[tuple[int, float, float]], np.ndarray | None]:
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
# Visualization
# ---------------------------------------------------------------------------

def point_color(obj_id: int) -> tuple[int, int, int]:
    hue = (obj_id * 47 + 20) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
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

# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def _estimate_intrinsics_from_cap(cap, target_w, target_h):
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
            try: pipeline.stop()
            except Exception: pass


def _resolve_pose_intrinsics(cap, target_w, target_h, intrinsics_file, try_orbbec):
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
# SAM3D bootstrap: ID1 from local scissors.glb; others via SAM3D
# (unchanged — produces GLB files and loaded trimesh objects)
# ---------------------------------------------------------------------------


def _bootstrap_sam3d_meshes(
    seed_frame_bgr: np.ndarray,
    ids: list[int],
    masks_np: np.ndarray,
    output_dir: Path,
    fal_model: str,
    object_classes: dict[int, str],
) -> dict[int, object]:
    """
    Load per-object trimesh.Trimesh instances:
    - ID1 always uses local ``SCISSORS_GLB``.
    - Other IDs: upload masks in parallel, submit all SAM3D jobs concurrently,
      then collect results — total time ≈ slowest single object, not sum.
    """
    if trimesh is None:
        print("`trimesh` not installed."); return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    meshes: dict[int, object] = {}
    fh, fw = seed_frame_bgr.shape[:2]

    if 1 in {int(v) for v in ids}:
        if not SCISSORS_GLB.exists():
            print(f"[ID1] Local scissors GLB not found: {SCISSORS_GLB}")
        else:
            try:
                meshes[1] = trimesh.load(str(SCISSORS_GLB), force="mesh")
                print(f"[ID1] Loaded local scissors GLB: {SCISSORS_GLB.name}")
            except Exception as e:
                print(f"[ID1] Failed to load local scissors GLB: {e}")

    # Collect non-ID1 objects that have valid masks
    non_id1 = {}   # oid -> binary mask uint8
    for i in range(min(len(ids), masks_np.shape[0])):
        oid = int(ids[i])
        if oid == 1:
            continue
        binm = _mask_to_2d_bool(masks_np[i], fh, fw).astype(np.uint8)
        if np.any(binm):
            non_id1[oid] = binm
        else:
            print(f"[ID{oid}] Empty mask; skipping SAM3D.")

    if not non_id1:
        return meshes

    # If all non-ID1 GLBs already exist on disk, skip network entirely
    all_cached = all((output_dir / f"object_{oid}.glb").exists() for oid in non_id1)
    if all_cached:
        print(f"All GLBs found on disk — skipping SAM3D.")
        for oid, binm in non_id1.items():
            glb_path = output_dir / f"object_{oid}.glb"
            try:
                meshes[oid] = trimesh.load(str(glb_path), force="mesh")
                print(f"[ID{oid}] Loaded existing GLB: {glb_path.name}")
            except Exception as e:
                print(f"[ID{oid}] Failed to load GLB: {e}")
        return meshes

    if not os.environ.get("FAL_KEY"):
        print("FAL_KEY not set for non-ID1 SAM3D bootstrap.")
        return meshes
    try:
        import fal_client  # type: ignore
    except Exception as e:
        print(f"`fal-client` import failed: {e}"); return meshes

    # Upload seed frame once (only needed for objects missing a local GLB)
    seed_path = output_dir / "seed_frame.png"
    cv2.imwrite(str(seed_path), seed_frame_bgr)
    image_url = fal_client.upload_file(str(seed_path))
    missing = [oid for oid in non_id1 if not (output_dir / f"object_{oid}.glb").exists()]
    print(f"SAM3D bootstrap: uploaded seed frame, submitting {len(missing)} object(s) in parallel…")

    def _process_one(oid: int, binm: np.ndarray):
        """Load from disk if GLB exists; otherwise call SAM3D."""
        glb_path = output_dir / f"object_{oid}.glb"
        if glb_path.exists():
            mesh = trimesh.load(str(glb_path), force="mesh")
            cls  = object_classes.get(oid, "tool")
            print(f"[ID{oid}] Loaded existing GLB: {glb_path.name}  class={cls}")
            return oid, mesh
        # No local GLB — call SAM3D
        mask_path = output_dir / f"object_{oid}_mask.png"
        cv2.imwrite(str(mask_path), binm * 255)
        mask_url = fal_client.upload_file(str(mask_path))
        result   = fal_client.subscribe(
            fal_model,
            arguments={"image_url": image_url, "mask_urls": [mask_url], "seed": 42},
            with_logs=False,
        )
        model_glb = result.get("model_glb", {}) if isinstance(result, dict) else {}
        glb_url   = model_glb.get("url")
        if not isinstance(glb_url, str) or not glb_url:
            raise RuntimeError("SAM3D returned no model_glb")
        urlretrieve(glb_url, glb_path)
        mesh = trimesh.load(str(glb_path), force="mesh")
        cls  = object_classes.get(oid, "tool")
        print(f"[ID{oid}] SAM3D GLB ready: {glb_path.name}  class={cls}")
        return oid, mesh

    # Fan out: all objects submitted simultaneously
    with ThreadPoolExecutor(max_workers=len(non_id1)) as pool:
        futures = {pool.submit(_process_one, oid, binm): oid
                   for oid, binm in non_id1.items()}
        for fut in as_completed(futures):
            oid = futures[fut]
            try:
                oid, mesh = fut.result()
                meshes[oid] = mesh
            except Exception as e:
                print(f"[ID{oid}] SAM3D bootstrap failed: {e}")

    return meshes

# ---------------------------------------------------------------------------
# Step 1: Extract 2D mask contour
# ---------------------------------------------------------------------------

def get_mask_contour(mask: np.ndarray) -> np.ndarray | None:
    """Return the largest external contour of a binary mask, or None."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 4:
        return None
    return cnt


def _sample_contour_uniform(cnt: np.ndarray, n: int) -> np.ndarray:
    """Uniformly resample a contour to exactly n points by arc length (vectorised)."""
    poly = cnt.reshape(-1, 2).astype(np.float64)
    d    = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    L    = float(d.sum())
    if L < 1e-9:
        return np.zeros((n, 2), dtype=np.float64)
    cum    = np.empty(len(poly) + 1); cum[0] = 0.0; np.cumsum(d, out=cum[1:])
    t_vals = ((np.arange(n) + 0.5) / n) * L
    k      = np.clip(np.searchsorted(cum, t_vals, side="right") - 1, 0, len(poly) - 1)
    u      = (t_vals - cum[k]) / (d[k] + 1e-12)
    return poly[k] * (1 - u[:, None]) + poly[(k + 1) % len(poly)] * u[:, None]

# ---------------------------------------------------------------------------
# Step 2: Load GLB and build a normalized 3D model
# ---------------------------------------------------------------------------

def load_glb_mesh(mesh_raw, obj_id: int, object_class: str):
    """
    Center mesh at origin, PCA-align (longest axis → X), and scale to
    known physical length. Returns aligned trimesh.Trimesh.
    """
    verts = np.asarray(mesh_raw.vertices, dtype=np.float64)
    faces = np.asarray(mesh_raw.faces, dtype=np.int32)

    # Center
    verts -= verts.mean(axis=0)

    # PCA-align: longest axis → axis 0
    cov = (verts.T @ verts) / len(verts)
    eigvals, evecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    verts = verts @ evecs[:, order]

    # Scale to known physical length (percentile-robust)
    target_m = OBJECT_ID_LENGTH_M.get(obj_id,
               CLASS_EXTENTS_M.get(object_class, CLASS_EXTENTS_M["tool"]))
    p95 = np.percentile(verts, 95, axis=0)
    p5  = np.percentile(verts,  5, axis=0)
    extent = float((p95 - p5).max())
    if extent > 1e-9:
        verts *= (target_m / extent)

    aligned = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return aligned

# ---------------------------------------------------------------------------
# Step 3: Camera intrinsics → see _resolve_pose_intrinsics above
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Kalman scalar filter for temporal smoothing
# ---------------------------------------------------------------------------

class KalmanScalar:
    """Random-walk scalar Kalman filter."""

    def __init__(self, process_var: float = 2e-4, meas_var: float = 4e-3):
        self.q = max(float(process_var), 1e-12)
        self.r = max(float(meas_var), 1e-12)
        self.x: float | None = None
        self.p: float = 1.0

    def update(self, z: float) -> float:
        z = float(z)
        if self.x is None:
            self.x = z; self.p = 1.0; return z
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1 - k) * self.p
        return self.x

# ---------------------------------------------------------------------------
# Step 4: Pose estimator — silhouette contour PnP with phase search
# ---------------------------------------------------------------------------

_N_CONTOUR_PTS = 64   # uniform samples on model and mask contours
_PHASE_STEP    = 8    # coarse phase search granularity on first frame


def _build_model_contour(mesh) -> np.ndarray:
    """
    Project mesh vertices onto XY plane, compute convex hull, return Nx3 model
    polygon (z=0) uniformly sampled to _N_CONTOUR_PTS points.
    """
    verts2d = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float32)[:, :2])
    hull = cv2.convexHull(verts2d.reshape(-1, 1, 2)).reshape(-1, 2)
    # Uniformly resample hull
    cnt = hull.reshape(-1, 1, 2).astype(np.float64)
    poly = _sample_contour_uniform(cnt, _N_CONTOUR_PTS)
    # Lift to 3D (z = 0 in model space)
    model3d = np.column_stack([poly, np.zeros(_N_CONTOUR_PTS)])
    return model3d.astype(np.float64)


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → unit quaternion [w,x,y,z]."""
    R = np.asarray(R, dtype=np.float64)
    t = float(np.trace(R))
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s; x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s; z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s; x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s; z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s; x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _unwrap_quat_hemisphere(prev_q: np.ndarray | None, q: np.ndarray) -> np.ndarray:
    """Flip q → -q if it is in the opposite hemisphere from prev_q (removes π jumps)."""
    if prev_q is not None and float(np.dot(q, prev_q)) < 0.0:
        q = -q
    return q / (np.linalg.norm(q) + 1e-12)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w,x,y,z] → 3×3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


_MAX_ROT_JUMP_DEG  = 30.0   # hard-reject PnP solutions that jump more than this
_ROT_SMOOTH_W      = 0.20   # per-degree penalty for rotation change between frames
_TRANS_SMOOTH_W    = 12.0   # penalty for normalised translation change
_REVERSAL_PENALTY  = 8.0    # heavy cost for flipping contour direction
_PHASE_PENALTY     = 0.05   # per-step cost for phase drift


class PoseEstimator:
    """
    6-DoF pose from 2D mask silhouette and 3D GLB model.

    Smoothing is done in quaternion space (not rvec) to avoid the Rodrigues
    wrap-around artefact.  The correct order is:
      raw PnP  →  R  →  quat  →  hemisphere-align  →  Kalman(quat+tvec)  →  output
    """

    def __init__(self, mesh, obj_id: int = 0, object_class: str = "tool"):
        self.obj_id = obj_id
        self.object_class = object_class
        self.mesh = mesh
        self.model_contour = _build_model_contour(mesh)
        v = np.asarray(mesh.vertices, dtype=np.float64)
        mn, mx = v.min(axis=0), v.max(axis=0)
        self.bbox_extents = float((mx - mn).max())
        # Kalman on quaternion (4) + tvec (3) = 7 scalars
        self._kf = [KalmanScalar(process_var=5e-5, meas_var=2e-2) for _ in range(7)]
        self._prev_q:    np.ndarray | None = None   # last smoothed quat
        self._prev_tv:   np.ndarray | None = None   # last smoothed tvec
        self._prev_rvec: np.ndarray | None = None   # last smoothed rvec (for PnP init)
        self._prev_phase: int | None = None
        self._prev_rev:   bool | None = None

    # ------------------------------------------------------------------
    def _solve_pnp(self, model3d, img2d, K, dist, init_rv=None, init_tv=None):
        """Try iterative PnP with previous-pose init, fall back to EPnP."""
        if init_rv is not None and init_tv is not None:
            ok, rv, tv = cv2.solvePnP(
                model3d, img2d, K, dist,
                init_rv.astype(np.float64), init_tv.astype(np.float64),
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            ok, rv, tv = cv2.solvePnP(model3d, img2d, K, dist,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            ok2, rv2, tv2 = cv2.solvePnP(model3d, img2d, K, dist,
                                          flags=cv2.SOLVEPNP_EPNP)
            if ok2:
                ok, rv, tv = cv2.solvePnP(model3d, img2d, K, dist, rv2, tv2,
                                           useExtrinsicGuess=True,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    rv, tv = rv2, tv2
            else:
                return None, None, np.inf
        proj, _ = cv2.projectPoints(model3d, rv, tv, K, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img2d, axis=1)))
        return rv, tv, err

    # ------------------------------------------------------------------
    def _rot_delta_deg(self, rv: np.ndarray) -> float:
        """Geodesic angle in degrees between rv and the last accepted pose."""
        if self._prev_rvec is None:
            return 0.0
        Ra, _ = cv2.Rodrigues(rv.astype(np.float64))
        Rb, _ = cv2.Rodrigues(self._prev_rvec.astype(np.float64))
        return float(np.degrees(
            np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1.0) * 0.5, -1.0, 1.0))
        ))

    # ------------------------------------------------------------------
    def _phase_search(self, img2d, K, dist):
        """
        Try phase/reversal candidates and return (rvec, tvec, phase, reverse)
        for the lowest-scoring alignment.
        """
        n    = _N_CONTOUR_PTS
        step = _PHASE_STEP

        if self._prev_phase is not None:
            phases = [(self._prev_phase + d) % n
                      for d in (0, step, -step, 2*step, -2*step)]
            candidates = [(p, self._prev_rev) for p in phases]
            # Allow reversal flip only as a last resort
            candidates += [(self._prev_phase, not self._prev_rev)]
        else:
            candidates = [(p, r)
                          for r in (False, True)
                          for p in range(0, n, step)]

        best_rv, best_tv, best_score = None, None, np.inf
        best_phase, best_rev = None, False

        for phase, rev in candidates:
            seq     = self.model_contour[::-1] if rev else self.model_contour
            model3d = np.roll(seq, int(phase), axis=0)

            rv, tv, err = self._solve_pnp(
                model3d, img2d, K, dist,
                self._prev_rvec, self._prev_tv,
            )
            if rv is None:
                continue

            score = err

            if self._prev_rvec is not None and self._prev_tv is not None:
                d_rot = self._rot_delta_deg(rv)
                z_ref = max(abs(float(self._prev_tv[2])), 1e-6)
                d_t   = float(np.linalg.norm(
                    tv.reshape(3) - self._prev_tv.reshape(3)) / z_ref)
                score += _ROT_SMOOTH_W * d_rot + _TRANS_SMOOTH_W * d_t

            if self._prev_phase is not None:
                dp = min(abs(phase - self._prev_phase),
                         n - abs(phase - self._prev_phase))
                score += _PHASE_PENALTY * float(dp)

            if self._prev_rev is not None and bool(rev) != self._prev_rev:
                score += _REVERSAL_PENALTY

            if score < best_score:
                best_score = score
                best_rv, best_tv = rv, tv
                best_phase, best_rev = int(phase), bool(rev)

        return best_rv, best_tv, best_phase, best_rev

    # ------------------------------------------------------------------
    def estimate(self, mask_bool: np.ndarray, K: np.ndarray, dist: np.ndarray) -> dict | None:
        """
        Estimate pose from a boolean mask.
        Returns dict with ``rvec``, ``tvec``, ``R``, or None on failure.
        """
        clean = cv2.erode(mask_bool.astype(np.uint8) * 255,
                          np.ones((3, 3), np.uint8)) > 0
        if clean.sum() < 20:
            return None

        cnt = get_mask_contour(clean)
        if cnt is None:
            return None
        img2d = _sample_contour_uniform(cnt, _N_CONTOUR_PTS)

        rv, tv, phase, rev = self._phase_search(img2d, K, dist)
        if rv is None or tv is None:
            return None

        # Hard-reject: if this frame's raw PnP jumps too far, fall back to
        # the Kalman prediction (previous smoothed pose) so a bad contour
        # match doesn't make the axes spin.
        if self._prev_q is not None and self._rot_delta_deg(rv) > _MAX_ROT_JUMP_DEG:
            if self._prev_rvec is not None and self._prev_tv is not None:
                rv = self._prev_rvec.copy()
                tv = self._prev_tv.copy()
            else:
                return None

        self._prev_phase = phase
        self._prev_rev   = rev

        # ── Correct smoothing order ──────────────────────────────────────
        # 1. raw rvec → R → quaternion
        R_raw, _ = cv2.Rodrigues(rv.astype(np.float64))
        q_raw = _R_to_quat(R_raw)

        # 2. hemisphere-align BEFORE Kalman so the filter never averages
        #    across a π discontinuity
        q_raw = _unwrap_quat_hemisphere(self._prev_q, q_raw)

        # 3. Kalman on quaternion (4) + tvec (3)
        meas = np.concatenate([q_raw, tv.reshape(3)])
        smoothed = np.array([self._kf[i].update(float(meas[i])) for i in range(7)])
        q_s  = smoothed[:4]
        tv_s = smoothed[4:].reshape(3, 1)

        # 4. Re-normalise and store
        q_s = q_s / (np.linalg.norm(q_s) + 1e-12)
        self._prev_q  = q_s.copy()
        self._prev_tv = tv_s.copy()

        # 5. Convert back to R, rvec
        R_final = _quat_to_R(q_s)
        rvec_s, _ = cv2.Rodrigues(R_final)
        self._prev_rvec = rvec_s.copy()

        return {"rvec": rvec_s, "tvec": tv_s, "R": R_final}

# ---------------------------------------------------------------------------
# Step 5: Draw 3D axes onto the frame
# ---------------------------------------------------------------------------

def draw_pose_live(
    frame: np.ndarray,
    mask_bool: np.ndarray,
    est: "PoseEstimator",
    last_R: np.ndarray,
    last_tz: float,
    K: np.ndarray,
    dist: np.ndarray,
    obj_id: int,
    obj_class: str = "",
) -> None:
    """
    Project the 3D model bounding box using live mask position + PnP rotation.

    Translation: derived each frame from the mask COM (XY) + last PnP depth (Z)
                 so the box tracks the object with zero lag laterally.
    Rotation:    last_R from the PnP background thread.
    Box corners: fixed in model space (est.bbox_corners), sized to the real
                 physical object — no per-frame rescaling.
    Axes:        anchored to bbox corner 0, capped at 60 px.
    """
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return

    fh, fw = frame.shape[:2]
    cx_2d, cy_2d = float(xs.mean()), float(ys.mean())
    Z = max(float(last_tz), 0.05)

    # Back-project mask COM to 3D using last known depth
    tx = (cx_2d - K[0, 2]) / K[0, 0] * Z
    ty = (cy_2d - K[1, 2]) / K[1, 1] * Z
    tvec_live = np.array([[tx], [ty], [Z]], dtype=np.float64)
    rvec_live, _ = cv2.Rodrigues(last_R)

    fnt = cv2.FONT_HERSHEY_SIMPLEX

    def _clamp(p):
        return (int(np.clip(p[0], 0, fw - 1)), int(np.clip(p[1], 0, fh - 1)))

    # ── Axes from model origin, capped at 60 px ──────────────────────
    length = est.bbox_extents * 0.6
    axes_pts = np.array([
        [0.0, 0.0, 0.0],
        [length, 0.0, 0.0],
        [0.0, length, 0.0],
        [0.0, 0.0, length],
    ], dtype=np.float64)
    proj_a, _ = cv2.projectPoints(axes_pts, rvec_live, tvec_live, K, dist)
    a2d = proj_a.reshape(-1, 2)

    o = a2d[0].astype(np.float64)

    def _cap60(tip: np.ndarray) -> tuple:
        v = tip - o; n = float(np.linalg.norm(v))
        pt = o + v * (60.0 / n) if n > 60.0 else tip
        return _clamp(pt)

    origin = _clamp(o)
    xc = _cap60(a2d[1]); yc = _cap60(a2d[2]); zc = _cap60(a2d[3])

    cv2.arrowedLine(frame, origin, xc, (0,   0, 220), 2, tipLength=0.20, line_type=cv2.LINE_AA)
    cv2.arrowedLine(frame, origin, yc, (0, 200,   0), 2, tipLength=0.20, line_type=cv2.LINE_AA)
    cv2.arrowedLine(frame, origin, zc, (220, 80,   0), 2, tipLength=0.20, line_type=cv2.LINE_AA)
    cv2.putText(frame, "X", (xc[0]+4, xc[1]+4), fnt, 0.5, (0,   0, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "Y", (yc[0]+4, yc[1]+4), fnt, 0.5, (0, 200,   0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Z", (zc[0]+4, zc[1]+4), fnt, 0.5, (220,  80,  0), 1, cv2.LINE_AA)

    # ── HUD ─────────────────────────────────────────────────────────
    sy = float(np.sqrt(last_R[0, 0]**2 + last_R[1, 0]**2))
    if sy > 1e-6:
        rx = np.degrees(np.arctan2(float(last_R[2, 1]), float(last_R[2, 2])))
        ry = np.degrees(np.arctan2(-float(last_R[2, 0]), sy))
        rz = np.degrees(np.arctan2(float(last_R[1, 0]), float(last_R[0, 0])))
    else:
        rx = np.degrees(np.arctan2(-float(last_R[1, 2]), float(last_R[1, 1])))
        ry = np.degrees(np.arctan2(-float(last_R[2, 0]), sy)); rz = 0.0
    tag  = f" ({obj_class})" if obj_class else ""
    text = (f"ID{obj_id}{tag}  R({rx:.0f},{ry:.0f},{rz:.0f})"
            f"  T({tx:.3f},{ty:.3f},{Z:.3f}m)")
    y_hud = 18 + (obj_id - 1) * 18
    cv2.putText(frame, text, (12, y_hud), fnt, 0.46, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (12, y_hud), fnt, 0.46, (0,   0,   0  ), 1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# LiveFrameProvider
# ---------------------------------------------------------------------------

class LiveFrameProvider:
    def __init__(self, cap, image_size, rotate_180,
                 sequential_mode=False, max_frames=None):
        self.cap             = cap
        self.image_size      = image_size
        self.rotate_180      = rotate_180
        self.sequential_mode = bool(sequential_mode)
        self.max_frames      = None if max_frames is None else int(max_frames)
        self._cache: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
        self._latest_tensor: torch.Tensor | None = None
        self._latest_raw:   np.ndarray   | None = None
        self._lock = threading.Lock()
        self._seq_next_idx = 0
        self._eof = False

    def _encode(self, frame):
        frame = preprocess(frame, self.rotate_180)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t     = torch.from_numpy(
            cv2.resize(rgb, (self.image_size, self.image_size))
        ).float().div(255.0).permute(2, 0, 1)
        return (t - _IMG_MEAN) / _IMG_STD, frame

    def capture_next(self) -> bool:
        if self.sequential_mode:
            if self.max_frames is not None and self._seq_next_idx >= self.max_frames:
                self._eof = True; return False
            ok, frame = self.cap.read()
            if not ok:
                self._eof = True; return False
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
        with self._lock:
            entry = self._cache.get(idx)
            return entry[1] if entry is not None else self._latest_raw

# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def _next_pose_csv_path(base_dir: Path) -> Path:
    i = 1
    while True:
        p = base_dir / f"poses{i}.csv"
        if not p.exists(): return p
        i += 1

# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(args) -> None:
    device = choose_device(args.device)
    if device == "cpu":
        print("GPU required. Use --device mps or --device cuda."); return
    print(f"Device: {device}")
    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT}"); return

    print("Loading EdgeTAM …")
    predictor = _load_predictor(device)

    camera_id = int(args.camera)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Could not open camera index: {camera_id}"); return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    rotate_180 = detect_orbbec_camera(camera_id)

    # Camera intrinsics
    K_cam, dist_cam, intr_src = _resolve_pose_intrinsics(
        cap, TARGET_SIZE[0], TARGET_SIZE[1],
        getattr(args, "intrinsics_file", ""),
        not getattr(args, "no_orbbec_intrinsics", False),
    )
    print(f"Intrinsics ({intr_src}): "
          f"fx={K_cam[0,0]:.1f} fy={K_cam[1,1]:.1f} "
          f"cx={K_cam[0,2]:.1f} cy={K_cam[1,2]:.1f}")

    object_classes: dict[int, str] = {1: "scissors"}
    for i, cls in enumerate(args.object_classes or []):
        oid = i + 1
        if oid not in object_classes:
            object_classes[oid] = cls.lower()

    image_size = predictor.image_size
    provider   = LiveFrameProvider(cap, image_size, rotate_180)
    if not provider.capture_next():
        print("No frame from camera."); cap.release(); return

    stop_flag    = threading.Event()
    capture_pause = threading.Event()   # set → capture thread sleeps
    _CAPTURE_MAX_RETRIES = 30           # ~3 s of consecutive failures before giving up

    def _capture_loop():
        failures = 0
        while not stop_flag.is_set():
            if capture_pause.is_set():
                time.sleep(0.05)
                failures = 0            # don't penalise paused periods
                continue
            if provider.capture_next():
                failures = 0
            else:
                failures += 1
                if failures >= _CAPTURE_MAX_RETRIES:
                    print("Camera lost — stopping.")
                    stop_flag.set()
                else:
                    time.sleep(0.1)

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
    et_state["num_frames"] = 1_000_000

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

    # Free GPU memory before the blocking SAM3D network call so jetsam
    # doesn't kill the process under MPS memory pressure.
    capture_pause.set()   # also pause here — GPU flush + re-init drops frames
    del et_state
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

    # --- SAM3D bootstrap (GLB acquisition — unchanged) -------------------
    # Pause the capture thread so camera read failures during the blocking
    # network call don't trigger the "camera lost" stop condition.
    capture_pause.set()
    print("Bootstrapping SAM3D on initial frozen frame (blocking)…")
    raw_meshes = _bootstrap_sam3d_meshes(
        seed_frame_bgr=seed_frame.copy(),
        ids=_seed_ids,
        masks_np=_seed_masks_np,
        output_dir=Path(args.sam3d_output_dir),
        fal_model=args.sam3d_model,
        object_classes=object_classes,
    )
    capture_pause.clear()   # resume capture before tracking starts
    if not raw_meshes:
        print("SAM3D produced no meshes. Exiting.")
        stop_flag.set(); cap.release(); return
    print("SAM3D ready: " + ", ".join(f"ID{k}" for k in sorted(raw_meshes)))

    # Restore propagation state after clearing GPU memory
    tmp2 = tempfile.mkdtemp(prefix="edgetam_live2_")
    cv2.imwrite(os.path.join(tmp2, "000000.jpg"), seed_frame)
    et_state = predictor.init_state(tmp2, async_loading_frames=False)
    shutil.rmtree(tmp2, ignore_errors=True)
    et_state["images"]     = provider
    et_state["num_frames"] = 1_000_000
    for obj_id, x, y in points:
        predictor.add_new_points_or_box(
            et_state, frame_idx=0, obj_id=int(obj_id),
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
    prop_stream  = predictor.propagate_in_video(et_state)
    first_packet = None

    # --- Build PoseEstimators from the acquired GLBs ----------------------
    pose_estimators: dict[int, PoseEstimator] = {}
    for oid, mesh_raw in raw_meshes.items():
        cls = object_classes.get(oid, "tool")
        aligned_mesh = load_glb_mesh(mesh_raw, oid, cls)
        pose_estimators[oid] = PoseEstimator(aligned_mesh, obj_id=oid, object_class=cls)
        print(f"  PoseEstimator built for ID{oid} ({cls})")

    # --- CSV output -------------------------------------------------------
    csv_path = _next_pose_csv_path(Path.cwd())
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_idx", "time_s", "object_id", "class",
                         "rx_deg", "ry_deg", "rz_deg", "tx_m", "ty_m", "tz_m"])
    print(f"Pose CSV: {csv_path}")

    writer = None
    if args.output:
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"),
            30.0, (TARGET_SIZE[0], TARGET_SIZE[1]))

    com_trails:  dict[int, list[tuple[int,int]]] = {}
    pose_results: dict[int, dict]               = {}   # oid -> latest pose (written by worker)
    pose_lock    = threading.Lock()
    MAX_TRAIL    = 60

    # Pose estimation runs in a background thread so it never blocks the
    # display loop.  The queue holds at most 1 item (latest masks); if the
    # worker is still busy the main loop drops the frame and moves on.
    pose_queue: queue.Queue = queue.Queue(maxsize=1)

    def _estimate_one(oid: int, binm: np.ndarray):
        est = pose_estimators.get(oid)
        if est is None:
            return oid, None
        return oid, est.estimate(binm, K_cam, dist_cam)

    def _pose_worker():
        # One persistent thread pool — avoids spawn overhead each frame.
        n_workers = max(1, len(pose_estimators))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            while not stop_flag.is_set():
                try:
                    item = pose_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                _fh, _fw, masks_batch = item
                # All objects estimated in parallel (cv2 releases the GIL)
                futures = {pool.submit(_estimate_one, oid, binm): oid
                           for oid, binm in masks_batch}
                for fut in as_completed(futures):
                    oid, pose = fut.result()
                    if pose is not None:
                        with pose_lock:
                            pose_results[oid] = pose

    pose_thread = threading.Thread(target=_pose_worker, daemon=True)
    pose_thread.start()

    fps_t0, fps_frames = time.perf_counter(), 0
    ac_device, ac_dtype, ac_enabled = _autocast_config(device, args.half)

    print("Live tracking started. Press 'q' or ESC to stop.")
    try:
        with torch.autocast(device_type=ac_device, dtype=ac_dtype, enabled=ac_enabled):
            stream_iter = (prop_stream if first_packet is None
                           else itertools.chain([first_packet], prop_stream))
            for fi, obj_ids, masks in stream_iter:
                frame = provider.get_raw(fi)
                if frame is None: continue

                ids = [int(x) for x in (obj_ids.tolist() if hasattr(obj_ids,"tolist") else obj_ids)]
                vis = overlay_masks(frame, ids, masks, alpha=args.alpha)
                fh, fw = frame.shape[:2]
                masks_np = masks.detach().cpu().numpy()

                masks_batch = []
                for i in range(min(len(ids), masks_np.shape[0])):
                    oid  = ids[i]
                    binm = _mask_to_2d_bool(masks_np[i], fh, fw)
                    if not np.any(binm): continue

                    # COM trail (cheap — stays in main loop)
                    ys, xs = np.where(binm)
                    cx, cy = int(xs.mean()), int(ys.mean())
                    trail  = com_trails.setdefault(oid, [])
                    trail.append((cx, cy))
                    if len(trail) > MAX_TRAIL:
                        com_trails[oid] = trail[-MAX_TRAIL:]

                    if oid in pose_estimators:
                        masks_batch.append((oid, binm))

                # Send masks to pose worker (drop frame if worker is busy)
                if masks_batch:
                    try:
                        pose_queue.put_nowait((fh, fw, masks_batch))
                    except queue.Full:
                        pass

                # Draw: position from live mask COM, orientation from last PnP R
                with pose_lock:
                    current_poses = dict(pose_results)
                for oid, pose in current_poses.items():
                    binm_live = _mask_to_2d_bool(
                        masks_np[ids.index(oid)] if oid in ids else
                        np.zeros((1,), dtype=np.float32), fh, fw
                    ) if oid in ids else None
                    if binm_live is None or not np.any(binm_live):
                        continue
                    est = pose_estimators.get(oid)
                    if est is None:
                        continue
                    last_tz = float(np.asarray(pose["tvec"]).reshape(-1)[2])
                    draw_pose_live(
                        vis, binm_live, est,
                        pose["R"], last_tz,
                        K_cam, dist_cam, oid,
                        object_classes.get(oid, ""),
                    )

                # COM trails
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

                # Seed point dots on frame 0
                if fi == 0:
                    for oid, px, py in points:
                        cv2.circle(vis, (int(px), int(py)), 5, (0,255,255), -1)
                        cv2.putText(vis, f"ID{oid}", (int(px)+8, int(py)-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

                cv2.imshow("EdgeTAM Live", vis)
                if writer: writer.write(vis)

                # CSV write — rotation from PnP, translation from live mask COM
                t_s = float(time.perf_counter())
                with pose_lock:
                    poses_for_csv = dict(pose_results)
                for oid, pose in poses_for_csv.items():
                    R = pose.get("R")
                    if R is None or oid not in ids: continue
                    i_oid = ids.index(oid)
                    binm_c = _mask_to_2d_bool(masks_np[i_oid], fh, fw)
                    if not np.any(binm_c): continue
                    ys_c, xs_c = np.where(binm_c)
                    cx_c, cy_c = float(xs_c.mean()), float(ys_c.mean())
                    Z_c = max(float(np.asarray(pose["tvec"]).reshape(-1)[2]), 0.05)
                    tx_c = (cx_c - K_cam[0,2]) / K_cam[0,0] * Z_c
                    ty_c = (cy_c - K_cam[1,2]) / K_cam[1,1] * Z_c
                    sy = float(np.sqrt(R[0,0]**2 + R[1,0]**2))
                    if sy > 1e-6:
                        rx = np.degrees(np.arctan2(float(R[2,1]), float(R[2,2])))
                        ry = np.degrees(np.arctan2(-float(R[2,0]), sy))
                        rz = np.degrees(np.arctan2(float(R[1,0]), float(R[0,0])))
                    else:
                        rx = np.degrees(np.arctan2(-float(R[1,2]), float(R[1,1])))
                        ry = np.degrees(np.arctan2(-float(R[2,0]), sy)); rz = 0.0
                    csv_writer.writerow([
                        int(fi), t_s, oid, object_classes.get(oid, "tool"),
                        rx, ry, rz, tx_c, ty_c, Z_c,
                    ])

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27): break
                if stop_flag.is_set(): break

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
        description="EdgeTAM live tracking + silhouette PnP pose from SAM3D GLBs."
    )
    parser.add_argument("--camera",         type=int,   default=0)
    parser.add_argument("--device",         default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--object-classes", nargs="*",  default=[],
                        help="Classes for ID2, ID3 … e.g. scalpel bottle bag")
    parser.add_argument("--alpha",          type=float, default=0.45)
    parser.add_argument("--half",           action="store_true", default=True)
    parser.add_argument("--no-half",        dest="half", action="store_false")
    parser.add_argument("--output",         default="")
    parser.add_argument("--sam3d-model",    default="fal-ai/sam-3/3d-objects")
    parser.add_argument("--sam3d-output-dir",
                        default=str(Path(__file__).parent / "sam3d_live_bootstrap"))
    parser.add_argument("--intrinsics-file", default="",
                        help="Path to .npz with camera K (and optional dist).")
    parser.add_argument("--no-orbbec-intrinsics", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
