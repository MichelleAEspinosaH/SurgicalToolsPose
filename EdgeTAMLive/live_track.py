#!/usr/bin/env python3
"""
Real-time surgical tool tracking with EdgeTAM.

Opens the Orbbec RGB camera (or any camera index), lets you click seed
points on the first frame, then streams live masks + oriented 3D cubes.

Usage:
    .venv/bin/python live_track.py                        # camera 0, default settings
    .venv/bin/python live_track.py --camera 1             # different camera index
    .venv/bin/python live_track.py --one-euro-beta 0.08   # stronger adaptive smoothing
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
# import urllib.request  # (used by SAM3D / fal-ai worker — commented out)
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None
try:
    import trimesh  # type: ignore
except Exception:
    trimesh = None

# try:
#     import fal_client  # type: ignore  # pip install fal-client
# except ImportError:
#     fal_client = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SIZE = (640, 360)
EDGETAM_REPO = Path(__file__).parent / "EdgeTAM"
CHECKPOINT = EDGETAM_REPO / "checkpoints" / "edgetam.pt"
MODEL_CFG = "configs/edgetam.yaml"
# SAM3D_ENDPOINT = "fal-ai/sam-3/3d-objects"  # (fal-ai — commented out)

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


def _load_tool_mesh_dims() -> dict[int, np.ndarray]:
    """Load per-object mesh extents (x, y, z) for object IDs 1..3."""
    dims: dict[int, np.ndarray] = {}
    if trimesh is None:
        return dims

    base = Path(__file__).parent
    for obj_id in (1, 2, 3):
        stem = f"object_{obj_id - 1}"
        candidates = [base / f"{stem}.ply", base / f"{stem}.glb"]
        for p in candidates:
            if not p.exists():
                continue
            try:
                mesh = trimesh.load(str(p), force="mesh")
                ext = np.asarray(getattr(mesh, "extents", None), dtype=np.float64)
                if ext.shape == (3,) and np.all(ext > 0):
                    dims[obj_id] = ext
                    break
            except Exception:
                continue
    return dims


def _order_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    center = corners.mean(axis=0)
    ang = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    ordered = corners[np.argsort(ang)]
    # Rotate order so first corner is top-most, then left-most.
    idx0 = int(np.argmin(ordered[:, 1] * 10000.0 + ordered[:, 0]))
    return np.roll(ordered, -idx0, axis=0)


def _camera_matrix_for_frame(width: int, height: int) -> np.ndarray:
    # Fallback intrinsics used only by the cube overlay path (no cap available).
    # Assumes 65° horizontal FOV — covers most webcams and phone cameras.
    f = width / (2.0 * np.tan(np.radians(65.0) / 2.0))
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _estimate_intrinsics_from_cap(
    cap: cv2.VideoCapture, target_w: int, target_h: int
) -> np.ndarray:
    """
    Estimate camera intrinsic matrix without an external calibration file.

    Queries the VideoCapture for its native frame dimensions, then applies a
    65° horizontal FOV assumption — a reasonable default for most webcams and
    smartphone wide cameras in video mode.  The focal length is scaled to
    match the resized inference resolution (TARGET_SIZE), so this stays
    consistent regardless of native resolution or whether the camera is an
    Orbbec, iPhone, or standard webcam.
    """
    native_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    native_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if native_w <= 0 or native_h <= 0:
        native_w, native_h = float(target_w), float(target_h)

    # fx = (native_w / 2) / tan(hfov/2), then scale down to target resolution.
    hfov_rad = np.radians(65.0)
    fx_native = native_w / (2.0 * np.tan(hfov_rad / 2.0))
    f = fx_native * (target_w / native_w)

    return np.array(
        [[f, 0.0, target_w / 2.0], [0.0, f, target_h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _draw_cube_from_faces(vis: np.ndarray, front_i: np.ndarray, back_i: np.ndarray, obj_id: int) -> None:
    col = point_color(obj_id)
    col_back = (max(0, col[0] - 80), max(0, col[1] - 80), max(0, col[2] - 80))
    cv2.polylines(vis, [back_i], True, col_back, 2, cv2.LINE_AA)
    cv2.polylines(vis, [front_i], True, col, 2, cv2.LINE_AA)
    for i in range(4):
        cv2.line(vis, tuple(front_i[i]), tuple(back_i[i]), col, 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        f"CUBE{obj_id}",
        tuple(front_i[0] + np.array([4, -4], dtype=np.int32)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        col,
        2,
        lineType=cv2.LINE_AA,
    )


def _draw_3d_cube_from_mask(
    vis: np.ndarray,
    mask_bool: np.ndarray,
    obj_id: int,
    state: dict | None,
    mesh_dims: np.ndarray | None = None,
    smooth_alpha: float = 0.8,
) -> dict | None:
    if not np.any(mask_bool):
        return state

    if state is None:
        state = {}

    clean = cv2.erode(mask_bool.astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
    ys, xs = np.where(clean)
    if len(xs) < 20:
        return state

    p32 = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    rect = cv2.minAreaRect(p32.reshape(-1, 1, 2))
    img_corners = _order_corners_clockwise(cv2.boxPoints(rect).astype(np.float64))

    if mesh_dims is not None and mesh_dims.shape == (3,):
        sx, sy, sz = [float(v) for v in mesh_dims]
    else:
        rw, rh = rect[1]
        sx, sy, sz = max(1.0, float(rw)), max(1.0, float(rh)), 0.35 * max(1.0, min(float(rw), float(rh)))

    hw, hh, hd = 0.5 * sx, 0.5 * sy, 0.5 * sz
    obj_front = np.array(
        [[-hw, -hh, hd], [hw, -hh, hd], [hw, hh, hd], [-hw, hh, hd]],
        dtype=np.float64,
    )
    obj_corners = np.array(
        [
            [-hw, -hh, hd], [hw, -hh, hd], [hw, hh, hd], [-hw, hh, hd],
            [-hw, -hh, -hd], [hw, -hh, -hd], [hw, hh, -hd], [-hw, hh, -hd],
        ],
        dtype=np.float64,
    )

    fh, fw = vis.shape[:2]
    K = _camera_matrix_for_frame(fw, fh)
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj_front, img_corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj_front, img_corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        # Robust fallback: still draw a visible pseudo-3D cube from the 2D min-area box.
        front = img_corners
        rw, rh = rect[1]
        depth = max(8.0, 0.35 * min(float(rw), float(rh)))
        shift2d = np.array([-depth, -depth], dtype=np.float64)
        back = front + shift2d[None, :]
        front_i = np.round(front).astype(np.int32)
        back_i = np.round(back).astype(np.int32)
        _draw_cube_from_faces(vis, front_i, back_i, obj_id)
        return state

    if "rvec" in state and "tvec" in state and smooth_alpha > 0.0:
        rvec = smooth_alpha * state["rvec"] + (1.0 - smooth_alpha) * rvec
        tvec = smooth_alpha * state["tvec"] + (1.0 - smooth_alpha) * tvec
    state["rvec"] = rvec
    state["tvec"] = tvec

    proj, _ = cv2.projectPoints(obj_corners, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    front_i = np.round(proj[:4]).astype(np.int32)
    back_i = np.round(proj[4:]).astype(np.int32)
    _draw_cube_from_faces(vis, front_i, back_i, obj_id)
    return state

# ---------------------------------------------------------------------------
# ICP pose axes
# ---------------------------------------------------------------------------

def _resample_contour(mask_bool, num_pts):
    m = mask_bool.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    pts_all, w_all = [], []
    for c in cnts:
        if len(c) < 5:
            continue
        p = c.reshape(-1, 2).astype(np.float64)
        prev, nxt = np.roll(p, 1, 0), np.roll(p, -1, 0)
        v1, v2 = p - prev, nxt - p
        n1 = np.linalg.norm(v1, axis=1) + 1e-9
        n2 = np.linalg.norm(v2, axis=1) + 1e-9
        curv = np.arccos(np.clip(np.sum(v1 * v2, axis=1) / (n1 * n2), -1, 1))
        pts_all.append(p)
        w_all.append(1.0 + 3.0 * curv / np.pi)
    if not pts_all:
        return None
    pts = np.vstack(pts_all)
    w = np.concatenate(w_all)
    if len(pts) <= num_pts:
        return pts
    prob = w / (w.sum() + 1e-12)
    return pts[np.random.default_rng(0).choice(len(pts), num_pts, replace=False, p=prob)]


def _pca_axes(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) < 12:
        return None
    p = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    c = p.mean(0)
    cov = ((p - c).T @ (p - c)) / max(len(p), 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)
    return c, v[:, order[-1]] / (np.linalg.norm(v[:, order[-1]]) + 1e-12), \
              v[:, order[0]]  / (np.linalg.norm(v[:, order[0]])  + 1e-12)


def _kabsch(P, Q):
    mu_p, mu_q = P.mean(0), Q.mean(0)
    h = (P - mu_p).T @ (Q - mu_q)
    u, _, vt = np.linalg.svd(h)
    if np.linalg.det(u @ vt) < 0:
        vt = vt.copy(); vt[1] *= -1
    r = u @ vt
    return r.astype(np.float64), (mu_q - mu_p @ r.T).astype(np.float64)


def _icp(P, Q, prev_R=None, prev_t=None, max_iter=8, tol=1e-4):
    if P.shape[0] < 3 or Q.shape[0] < 3:
        return np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
    r = prev_R.copy() if prev_R is not None else np.eye(2, dtype=np.float64)
    t = prev_t.copy() if prev_t is not None else Q.mean(0) - (P @ r.T).mean(0)
    tree = cKDTree(Q) if cKDTree is not None else None
    prev_err = np.inf
    for _ in range(max_iter):
        p_t = P @ r.T + t
        if tree is not None:
            dist, idx = tree.query(p_t)
            err = float(np.mean(dist * dist))
        else:
            d2 = np.sum((p_t[:, None] - Q[None]) ** 2, axis=2)
            idx, err = np.argmin(d2, axis=1), float(np.mean(np.min(d2, axis=1)))
        ri, ti = _kabsch(p_t, Q[idx])
        r, t = ri @ r, t @ ri.T + ti
        if prev_err < np.inf and abs(prev_err - err) < tol:
            break
        prev_err = err
    return r, t


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
ONE_EURO_MIN_CUTOFF = 1.2
ONE_EURO_BETA = 0.02
ONE_EURO_D_CUTOFF = 1.0


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


def _one_euro_alpha(cutoff: float, dt: float) -> float:
    cutoff = max(float(cutoff), 1e-6)
    dt = max(float(dt), 1e-6)
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroScalar:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: float | None = None
        self.dx_prev: float = 0.0
        self.t_prev: float | None = None

    def filter(self, x: float, t_s: float) -> float:
        x = float(x)
        t_s = float(t_s)
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x
            self.t_prev = t_s
            self.dx_prev = 0.0
            return x
        dt = max(t_s - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = _one_euro_alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _one_euro_alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t_s
        return x_hat


def _apply_one_euro_pose_filter(
    state: dict,
    rv: np.ndarray,
    tv: np.ndarray,
    t_s: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    filters = state.get("one_euro_filters")
    if filters is None or len(filters) != 6:
        filters = [
            OneEuroScalar(min_cutoff, beta, d_cutoff),
            OneEuroScalar(min_cutoff, beta, d_cutoff),
            OneEuroScalar(min_cutoff, beta, d_cutoff),
            OneEuroScalar(min_cutoff, beta, d_cutoff),
            OneEuroScalar(min_cutoff, beta, d_cutoff),
            OneEuroScalar(min_cutoff, beta, d_cutoff),
        ]
        state["one_euro_filters"] = filters
    vec = np.concatenate([rv.reshape(3), tv.reshape(3)]).astype(np.float64)
    out = np.empty_like(vec)
    for i in range(6):
        out[i] = filters[i].filter(float(vec[i]), t_s)
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
    ) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """
        Many-point PnP on uniformly sampled mask/model rectangle perimeters.
        Seeds each cyclic alignment with IPPE on 4 corners, then refines with
        SOLVEPNP_ITERATIVE on all samples.
        """
        n = self._pnp_n
        img_n = _sample_quad_perimeter(img_corners4.astype(np.float64), n)
        model_n0 = _sample_quad_perimeter(model_corners4.astype(np.float64), n)
        step = n // 4
        best_rv, best_tv, best_err = None, None, np.inf
        for shift in range(4):
            mc4 = np.roll(model_corners4, shift, axis=0)
            ok, rv0, tv0 = cv2.solvePnP(
                mc4.astype(np.float64),
                img_corners4.astype(np.float64),
                K,
                dist,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if not ok:
                ok, rv0, tv0 = cv2.solvePnP(
                    mc4.astype(np.float64),
                    img_corners4.astype(np.float64),
                    K,
                    dist,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            if not ok:
                continue
            model_n = np.roll(model_n0, shift * step, axis=0)
            ok2, rv, tv = cv2.solvePnP(
                model_n,
                img_n,
                K,
                dist,
                rv0,
                tv0,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok2:
                rv, tv = rv0, tv0
            proj, _ = cv2.projectPoints(model_n, rv, tv, K, dist)
            err = float(
                np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_n, axis=1))
            )
            if err < best_err:
                best_err, best_rv, best_tv = err, rv, tv
        if best_rv is None:
            return self._pnp_best_pts(model_corners4, img_corners4, K, dist)
        return best_rv, best_tv, best_err

    # ------------------------------------------------------------------
    def estimate_pose(
        self,
        mask_bool: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        state: dict | None,
        timestamp_s: float,
        one_euro_min_cutoff: float = ONE_EURO_MIN_CUTOFF,
        one_euro_beta: float = ONE_EURO_BETA,
        one_euro_d_cutoff: float = ONE_EURO_D_CUTOFF,
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
            best_rv, best_tv, best_s = None, None, None
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
                rv, tv, err = self._pnp_best_dense(model_s, img_corners, K, dist)
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
            if best_rv is None or best_s is None:
                return state
            state["reg_sign"] = best_s
            rv, tv = best_rv, best_tv
        else:
            s = _reg_sign_from_state(state)
            model_s = self.model_pts * s
            rv, tv, _ = self._pnp_best_dense(model_s, img_corners, K, dist)
            if rv is None:
                return state

        rv, tv = _apply_one_euro_pose_filter(
            state,
            rv,
            tv,
            timestamp_s,
            one_euro_min_cutoff,
            one_euro_beta,
            one_euro_d_cutoff,
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
    """Project and draw X(red)/Y(green)/Z(blue) pose axes + euler-angle readout."""
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
    label_r = f"ID{obj_id} R({rx:.0f},{ry:.0f},{rz:.0f}) degree"
    label_t = f"ID{obj_id} T({tx:.1f},{ty:.1f},{tz:.1f})"
    (tw_r, th_r), _ = cv2.getTextSize(label_r, fnt, 0.42, 1)
    (tw_t, th_t), _ = cv2.getTextSize(label_t, fnt, 0.42, 1)
    tw = max(tw_r, tw_t)
    th = th_r + th_t + 6
    ox, oy = origin
    cv2.rectangle(vis, (ox, oy - th - 4), (ox + tw + 2, oy + 2), (0, 0, 0), -1)
    cv2.putText(vis, label_r, (ox + 1, oy - th_t - 4), fnt, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, label_t, (ox + 1, oy - 1), fnt, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


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
# SAM3D background worker — commented out (fal-ai dependency removed)
# ---------------------------------------------------------------------------
#
# class Sam3DWorker:
#     def __init__(self, output_dir, fal_key=None): ...
#     def trigger(self, frame, masks): ...
#     def get_status(self): ...
#     def pop_new_glbs(self): ...
#     def _run(self, frame, masks): ...  # uploads to fal-ai/sam-3/3d-objects


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
        print("`trimesh` not installed; pose axes unavailable (cube fallback).")
    elif mesh_estimators:
        loaded = ", ".join(f"ID{k}->object_{k-1}" for k in sorted(mesh_estimators))
        print(f"Pose estimators loaded for {loaded}")
    else:
        print("No mesh files found; using mask-only cube proportions.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize camera-side frame queue
    rotate_180 = detect_orbbec_camera(args.camera)
    print(f"Camera {args.camera}: {'Orbbec detected, rotating 180°' if rotate_180 else 'non-Orbbec, no rotation'}")

    # Estimate camera intrinsics from the native capture resolution.
    # No calibration file needed — adapts to Orbbec, iPhone, or any webcam.
    K_cam    = _estimate_intrinsics_from_cap(cap, TARGET_SIZE[0], TARGET_SIZE[1])
    dist_cam = np.zeros((4, 1), dtype=np.float64)
    print(
        f"Camera intrinsics (estimated): "
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

    com_trails: dict[int, list[tuple[int, int]]] = {}  # obj_id -> list of (x, y) COM positions
    pose_states: dict[int, dict] = {}          # per-obj state for MeshPoseEstimator
    cube_states: dict[int, dict | None] = {}   # fallback cube state (no mesh)
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
                                timestamp_s=time.perf_counter(),
                                one_euro_min_cutoff=args.one_euro_min_cutoff,
                                one_euro_beta=args.one_euro_beta,
                                one_euro_d_cutoff=args.one_euro_d_cutoff,
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
                            # Fallback disabled: do not draw per-object bounding-box cubes.
                            # cube_states[oid] = _draw_3d_cube_from_mask(
                            #     vis, binm, oid, cube_states.get(oid),
                            #     mesh_dims=None,
                            #     smooth_alpha=0.8,
                            # )
                            pass

                # Draw COM trails
                for oid, trail in com_trails.items():
                    col = point_color(oid)
                    for j in range(1, len(trail)):
                        cv2.line(vis, trail[j - 1], trail[j], col, 2, lineType=cv2.LINE_AA)
                    if trail:
                        cv2.circle(vis, trail[-1], 5, col, -1)
                        cv2.circle(vis, trail[-1], 7, (255, 255, 255), 1)

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
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time EdgeTAM tracking on Orbbec RGB stream.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--alpha", type=float, default=0.45,
                        help="Mask overlay alpha")
    parser.add_argument("--one-euro-min-cutoff", type=float, default=ONE_EURO_MIN_CUTOFF,
                        help="One Euro filter min cutoff (Hz-ish); higher = less smoothing")
    parser.add_argument("--one-euro-beta", type=float, default=ONE_EURO_BETA,
                        help="One Euro speed coefficient; higher = more responsive motion")
    parser.add_argument("--one-euro-d-cutoff", type=float, default=ONE_EURO_D_CUTOFF,
                        help="One Euro derivative cutoff (Hz-ish)")
    parser.add_argument("--half", action="store_true", default=True,
                        help="Use half-precision autocast (default: on)")
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--output", default="",
                        help="Optional path to save output video (e.g. out.mp4)")
    parser.add_argument(
        "--align-debug-out",
        default="",
        help="Optional path to save registration-debug image (SAM mask vs projected GLB) each frame.",
    )
    # --sam3d / --fal-key removed (fal-ai integration commented out)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
