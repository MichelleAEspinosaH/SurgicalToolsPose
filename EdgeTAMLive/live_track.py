#!/usr/bin/env python3
"""
Real-time surgical tool tracking with EdgeTAM.

Opens the Orbbec RGB camera (or any camera index), lets you click seed
points on the first frame, then streams live masks + oriented 3D cubes.

Usage:
    .venv/bin/python live_track.py                        # camera 0, default settings
    .venv/bin/python live_track.py --camera 1             # different camera index
    .venv/bin/python live_track.py --axis-smooth 0.9      # heavier axis smoothing
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
    # Approximate intrinsics for projection-only overlay.
    f = 1.2 * max(width, height)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
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


# ---------------------------------------------------------------------------
# Mesh-based 6DoF pose estimator
# ---------------------------------------------------------------------------

class MeshPoseEstimator:
    """
    6DoF pose estimator for an elongated surgical tool given its 3D mesh.

    Model-space convention (matches draw_pose_axes.py):
        X  (red)   — along tool primary axis (tip direction)
        Y  (green) — secondary in-plane axis (width)
        Z  (blue)  — tertiary out-of-plane axis (thickness)

    PnP uses the four midplane silhouette corners, which span the full
    length × width of the tool and lie in the Y=0 plane (coplanar → IPPE).
    """

    def __init__(self, mesh):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        cen = verts.mean(0)
        v = verts - cen
        cov = (v.T @ v) / len(v)
        eigvals, evecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]   # longest axis first
        aligned = v @ evecs[:, order]

        # Half-extents: 0=primary(length), 1=secondary(width), 2=tertiary(thickness)
        hP = (aligned[:, 0].max() - aligned[:, 0].min()) / 2
        hS = (aligned[:, 1].max() - aligned[:, 1].min()) / 2
        hT = (aligned[:, 2].max() - aligned[:, 2].min()) / 2
        self.hP, self.hS, self.hT = hP, hS, hT
        self.extents = np.array([2 * hP, 2 * hS, 2 * hT])

        # ── PnP model points ─────────────────────────────────────────────
        # Four midplane silhouette corners in model XZ plane (Y=0):
        #   X = secondary (width), Z = primary (along tool / length)
        # Ordered clockwise from top-left when the tool points upward:
        #   TL=(-hS, 0, hP)  TR=(hS, 0, hP)  BR=(hS, 0,-hP)  BL=(-hS, 0,-hP)
        self.model_pts = np.array(
            [[-hS, 0.0, hP], [hS, 0.0, hP], [hS, 0.0, -hP], [-hS, 0.0, -hP]],
            dtype=np.float64,
        )

        # ── Axis display points ───────────────────────────────────────────
        # In model space: X=along tool(Z-dir), Y=width(X-dir), Z=out-of-plane(Y-dir)
        ax = hP * 1.0                           # X axis: half tool length
        ay = hS * 1.5                           # Y axis: 1.5× half-width
        az = max(hT * 5.0, hP * 0.35)          # Z axis: boosted so it's visible
        self.axis_pts = np.array(
            [[0.0, 0.0, 0.0],   # origin
             [0.0, 0.0, ax],    # X tip (red)   = +Z model = along tool / tip
             [ay,  0.0, 0.0],   # Y tip (green) = +X model = in-plane width
             [0.0, az,  0.0]],  # Z tip (blue)  = +Y model = out-of-plane
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    def _pnp_best(
        self,
        img_corners: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """
        Try all 4 cyclic shifts of model_pts → img_corners correspondences
        with IPPE + ITERATIVE.  Return (rvec, tvec, reprojection_error).
        """
        best_rv, best_tv, best_err = None, None, np.inf
        for shift in range(4):
            pts = np.roll(self.model_pts, shift, axis=0)
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
    def estimate_pose(
        self,
        mask_bool: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        state: dict | None,
        smooth_alpha: float = 0.8,
    ) -> dict:
        """
        Estimate 6DoF pose from a binary mask.  Updates and returns the
        per-object state dict (keys: 'rvec', 'tvec').
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

        rv, tv, _ = self._pnp_best(img_corners, K, dist)
        if rv is None:
            return state

        if "rvec" in state and smooth_alpha > 0.0:
            rv = smooth_alpha * state["rvec"] + (1.0 - smooth_alpha) * rv
            tv = smooth_alpha * state["tvec"] + (1.0 - smooth_alpha) * tv
        state["rvec"] = rv
        state["tvec"] = tv
        return state


def _load_mesh_estimators() -> dict[int, MeshPoseEstimator]:
    """Load per-object GLB meshes and build MeshPoseEstimators (IDs 1..3)."""
    estimators: dict[int, MeshPoseEstimator] = {}
    if trimesh is None:
        return estimators
    base = Path(__file__).parent
    for obj_id in (1, 2, 3):
        stem = f"object_{obj_id - 1}"
        for ext in (".glb", ".ply"):
            p = base / (stem + ext)
            if not p.exists():
                continue
            try:
                mesh = trimesh.load(str(p), force="mesh")
                estimators[obj_id] = MeshPoseEstimator(mesh)
                break
            except Exception:
                continue
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

    proj, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, dist)
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
    label = f"ID{obj_id} ({rx:.0f},{ry:.0f},{rz:.0f})\u00b0"
    (tw, th), _ = cv2.getTextSize(label, fnt, 0.42, 1)
    ox, oy = origin
    cv2.rectangle(vis, (ox, oy - th - 4), (ox + tw + 2, oy + 2), (0, 0, 0), -1)
    cv2.putText(vis, label, (ox + 1, oy - 1), fnt, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


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

    # Approximate camera intrinsics (updated once we have the first frame)
    K_cam    = _camera_matrix_for_frame(TARGET_SIZE[0], TARGET_SIZE[1])
    dist_cam = np.zeros((4, 1), dtype=np.float64)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize camera-side frame queue
    rotate_180 = detect_orbbec_camera(args.camera)
    print(f"Camera {args.camera}: {'Orbbec detected, rotating 180°' if rotate_180 else 'non-Orbbec, no rotation'}")

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
                                binm, K_cam, dist_cam,
                                pose_states.get(oid),
                                smooth_alpha=args.axis_smooth,
                            )
                            _draw_pose_axes(
                                vis, pose_states[oid], K_cam, dist_cam,
                                est.axis_pts, oid,
                            )
                        else:
                            # Fallback: bounding-box cube (no mesh loaded)
                            cube_states[oid] = _draw_3d_cube_from_mask(
                                vis, binm, oid, cube_states.get(oid),
                                mesh_dims=None,
                                smooth_alpha=args.axis_smooth,
                            )

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
    parser.add_argument("--axis-smooth", type=float, default=0.8, metavar="ALPHA",
                        help="EMA smoothing for axes (0=none, 0.99=heavy, default 0.8)")
    parser.add_argument("--half", action="store_true", default=True,
                        help="Use half-precision autocast (default: on)")
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--output", default="",
                        help="Optional path to save output video (e.g. out.mp4)")
    # --sam3d / --fal-key removed (fal-ai integration commented out)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
