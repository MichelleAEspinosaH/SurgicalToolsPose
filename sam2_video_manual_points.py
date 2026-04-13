#!/usr/bin/env python3
"""
SAM2 video tracking from recorded RGB video (``--input``) and optional depth video
(``--depth-video``). Frames are rotated 180° first (see ``--no-rotate``), then
RGB is scaled by ``--rgb-div`` and depth frames are resized to match for display.

Record RGB + depth with ``record_rgb_depth.py`` (use ``--raw-depth-dir`` for metric .npy
depth used to draw three fixed-length PCA axes in 3D). Pass ``--depth-raw-dir`` here with
the same folder so axes use real mm values; otherwise axes fall back to jet colormap intensity.

The SAM2 *video* API loads a JPEG folder at ``init_state``; the point-picker uses frame 0.
"""
import argparse
import os
import tempfile
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch
from contextlib import nullcontext
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

def maybe_rotate_180(frame: np.ndarray, rotate: bool) -> np.ndarray:
    if not rotate or frame is None:
        return frame
    return cv2.rotate(frame, cv2.ROTATE_180)


def _scale_rgb_and_match_depth(
    bgr: np.ndarray, depth_bgr: np.ndarray, rgb_div: int
) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    div = max(1, int(rgb_div))
    tw, th = max(1, w // div), max(1, h // div)
    bgr_s = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
    d = depth_bgr
    if d.shape[0] != th or d.shape[1] != tw:
        d = cv2.resize(d, (tw, th), interpolation=cv2.INTER_NEAREST)
    return bgr_s, d


def _stream_depth_subdir(tmp_dir: str) -> str:
    """SAM2 only loads *.jpg whose stem is an integer; keep depth frames in a subfolder."""
    return os.path.join(tmp_dir, "depth")


def _raw_depth_subdir(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, "raw")


def build_jpeg_folder_from_rgb_depth_videos(
    rgb_path: str,
    depth_path: str,
    tmp_dir: str,
    rgb_div: int,
    max_frames: int,
    rotate_180: bool,
    depth_raw_dir: str = "",
) -> int:
    """
    Read paired frames from two video files, rotate 180°, scale RGB and resize depth
    to a common grid, write SAM2 RGB jpegs + depth/NNNNNN.jpg sidecar.
    Returns number of frame pairs written.
    """
    cap_r = cv2.VideoCapture(rgb_path)
    cap_d = cv2.VideoCapture(depth_path)
    if not cap_r.isOpened():
        raise RuntimeError(f"Could not open RGB video: {rgb_path}")
    if not cap_d.isOpened():
        cap_r.release()
        raise RuntimeError(f"Could not open depth video: {depth_path}")

    os.makedirs(tmp_dir, exist_ok=True)
    depth_dir = _stream_depth_subdir(tmp_dir)
    os.makedirs(depth_dir, exist_ok=True)
    raw_src = (depth_raw_dir or "").strip()
    raw_dst_dir = _raw_depth_subdir(tmp_dir)
    if raw_src:
        os.makedirs(raw_dst_dir, exist_ok=True)

    written = 0
    limit = max(1, int(max_frames))
    while written < limit:
        ok_r, fr = cap_r.read()
        ok_d, fd = cap_d.read()
        if not ok_r or fr is None or not ok_d or fd is None:
            break
        fr = maybe_rotate_180(fr, rotate_180)
        fd = maybe_rotate_180(fd, rotate_180)
        bgr_s, d_s = _scale_rgb_and_match_depth(fr, fd, rgb_div)
        cv2.imwrite(os.path.join(tmp_dir, f"{written:06d}.jpg"), bgr_s)
        cv2.imwrite(os.path.join(depth_dir, f"{written:06d}.jpg"), d_s)
        if raw_src:
            rp = os.path.join(raw_src, f"{written:06d}.npy")
            if os.path.isfile(rp):
                arr = np.load(rp).astype(np.float32)
                if rotate_180:
                    arr = cv2.rotate(arr, cv2.ROTATE_180)
                th, tw = bgr_s.shape[0], bgr_s.shape[1]
                if arr.shape[0] != th or arr.shape[1] != tw:
                    arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_NEAREST)
                np.save(os.path.join(raw_dst_dir, f"{written:06d}.npy"), arr)
        written += 1

    cap_r.release()
    cap_d.release()
    if written == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("No paired RGB/depth frames read from videos.")
    return written


def pick_points(first_frame: np.ndarray) -> list[tuple[int, float, float]]:
    win = "Select SAM2 points"
    points: list[tuple[int, float, float]] = []

    def draw() -> np.ndarray:
        vis = first_frame.copy()
        for obj_id, px_f, py_f in points:
            px, py = int(px_f), int(py_f)
            cv2.circle(vis, (px, py), 6, (0, 255, 255), -1)
            cv2.circle(vis, (px, py), 9, (255, 255, 255), 1)
            cv2.putText(
                vis,
                f"ID{obj_id}",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        cv2.putText(
            vis,
            "Left click: add NEW object point (new ID) | Backspace: undo | c: clear | Enter: start",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis,
            "Frozen frame 0 for point prompts; press Enter to run SAM2 on the clip.",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 220, 200),
            1,
        )
        return vis

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            obj_id = len(points) + 1
            points.append((obj_id, float(x), float(y)))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    cancelled = False
    while True:
        cv2.imshow(win, draw())
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):  # ENTER
            if points:
                break
        elif k in (8, 127):  # Backspace / Delete
            if points:
                points.pop()
        elif k == ord("c"):
            points.clear()
        elif k in (ord("q"), 27):
            cancelled = True
            break
    cv2.destroyWindow(win)
    return [] if cancelled else points


def point_color(obj_id: int) -> tuple[int, int, int]:
    hue = (obj_id * 47 + 20) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _mask_to_2d_bool(m: np.ndarray, fh: int, fw: int) -> np.ndarray:
    """SAM2 may return (H,W), (1,H,W), (1,1,H,W), etc.; normalize to (fh,fw) bool."""
    x = np.asarray(m, dtype=np.float32)
    x = np.squeeze(x)
    while x.ndim > 2:
        x = x[0]
    if x.ndim != 2:
        return np.zeros((fh, fw), dtype=bool)
    if x.shape[0] != fh or x.shape[1] != fw:
        x = cv2.resize(x, (fw, fh), interpolation=cv2.INTER_NEAREST)
    return x > 0.0


def overlay_masks_with_ids(
    frame: np.ndarray,
    obj_ids: list[int],
    masks: torch.Tensor,
    alpha: float = 0.45,
) -> np.ndarray:
    vis = frame.copy().astype(np.float32)
    fh, fw = frame.shape[:2]
    masks_np = masks.detach().cpu().numpy()
    n = min(len(obj_ids), masks_np.shape[0])
    for i in range(n):
        obj_id = obj_ids[i]
        binm = _mask_to_2d_bool(masks_np[i], fh, fw)
        if not np.any(binm):
            continue
        c = np.array(point_color(int(obj_id)), dtype=np.float32)
        vis[binm] = vis[binm] * (1.0 - alpha) + c * alpha
    return vis.astype(np.uint8)


def _resample_contour_xy(mask_bool: np.ndarray, num_pts: int) -> np.ndarray | None:
    """
    Curvature-aware contour samples for ICP.
    Uses all contours (outer + inner) so thin tools with gaps are represented better.
    """
    m = (mask_bool.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    pts_all = []
    w_all = []
    for c in cnts:
        if len(c) < 5:
            continue
        p = c.reshape(-1, 2).astype(np.float64)
        # Curvature proxy from local turning angle magnitude.
        prev = np.roll(p, 1, axis=0)
        nxt = np.roll(p, -1, axis=0)
        v1 = p - prev
        v2 = nxt - p
        n1 = np.linalg.norm(v1, axis=1) + 1e-9
        n2 = np.linalg.norm(v2, axis=1) + 1e-9
        cosang = np.sum(v1 * v2, axis=1) / (n1 * n2)
        cosang = np.clip(cosang, -1.0, 1.0)
        curv = np.arccos(cosang)  # [0, pi]
        # Preserve straight shafts but emphasize tips/hinges.
        w = 1.0 + 3.0 * (curv / np.pi)
        pts_all.append(p)
        w_all.append(w)
    if not pts_all:
        return None
    pts = np.vstack(pts_all)
    w = np.concatenate(w_all)
    if len(pts) <= num_pts:
        return pts
    prob = w / (np.sum(w) + 1e-12)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=num_pts, replace=False, p=prob)
    return pts[idx]


def _mask_pca_axes_2d(
    mask_bool: np.ndarray,
    max_samples: int = 8000,
    axis_len_scale: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    """Centroid + major/minor axis unit vectors + display half-lengths from pixel covariance."""
    ys, xs = np.where(mask_bool)
    n = len(xs)
    if n < 12:
        return None
    rng = np.random.default_rng(0)
    if n > max_samples:
        sel = rng.choice(n, size=max_samples, replace=False)
        xs, ys = xs[sel], ys[sel]
    P = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    c = P.mean(axis=0)
    Pc = P - c
    cov = (Pc.T @ Pc) / max(len(Pc), 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)
    v_minor = V[:, order[0]]
    v_major = V[:, order[-1]]
    # half-lengths in pixels (stabilized)
    lam_maj = float(max(w[order[-1]], 1e-8))
    lam_min = float(max(w[order[0]], 1e-8))
    half_major = axis_len_scale * np.sqrt(lam_maj)
    half_minor = axis_len_scale * np.sqrt(lam_min)
    half_major = float(np.clip(half_major, 20.0, 400.0))
    half_minor = float(np.clip(half_minor, 15.0, 300.0))
    return c, v_major / (np.linalg.norm(v_major) + 1e-12), v_minor / (np.linalg.norm(v_minor) + 1e-12), half_major, half_minor


def _kabsch_rigid_rows_2d(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Row-vector rigid transform: Q ≈ P @ R.T + t (least squares).
    P, Q: (n, 2)
    """
    mu_p = P.mean(axis=0)
    mu_q = Q.mean(axis=0)
    Pc = P - mu_p
    Qc = Q - mu_q
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[1, :] *= -1.0
        R = U @ Vt
    t = mu_q - mu_p @ R.T
    return R.astype(np.float64), t.astype(np.float64)


def draw_three_axes_from_depth(
    vis: np.ndarray,
    mask_bool: np.ndarray,
    depth_map: np.ndarray,
    obj_id: int,
    *,
    axis_half_len_px: float,
    metric_depth: bool,
) -> None:
    """
    3D PCA on masked pixels (u,v,depth): pinhole X,Y,Z when metric_depth else (u,v,Z_proxy).
    Draw three orthogonal axes as fixed-length segments in the image (2D projection).
    """
    fh, fw = vis.shape[:2]
    if depth_map.shape[0] != fh or depth_map.shape[1] != fw:
        depth_map = cv2.resize(
            depth_map.astype(np.float32), (fw, fh), interpolation=cv2.INTER_NEAREST
        )
    m = (mask_bool.astype(np.uint8) * 255)
    m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    valid = (m > 0) & (depth_map > 1e-3)
    ys, xs = np.where(valid)
    if len(xs) < 40:
        return

    rng = np.random.default_rng(0)
    if len(xs) > 8000:
        sel = rng.choice(len(xs), size=8000, replace=False)
        xs, ys = xs[sel], ys[sel]

    z = depth_map[ys, xs].astype(np.float64)
    uc = float(np.mean(xs))
    vc = float(np.mean(ys))
    fx = fy = max(0.5 * fw, 1e-3)
    cx_img = fw / 2.0
    cy_img = fh / 2.0

    if metric_depth:
        X = (xs.astype(np.float64) - cx_img) * z / fx
        Y = (ys.astype(np.float64) - cy_img) * z / fy
        P = np.stack([X, Y, z], axis=1)
    else:
        P = np.stack(
            [xs.astype(np.float64), ys.astype(np.float64), z],
            axis=1,
        )

    mu = P.mean(axis=0)
    Q = P - mu
    cov = (Q.T @ Q) / max(len(Q), 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    axes3 = [V[:, order[i]].astype(np.float64) for i in range(3)]
    for a in axes3:
        n = np.linalg.norm(a)
        if n > 1e-12:
            a /= n

    def image_dir(e3: np.ndarray) -> tuple[float, float]:
        ex, ey = float(e3[0]), float(e3[1])
        n2 = np.hypot(ex, ey)
        if n2 > 1e-4:
            return ex / n2, ey / n2
        view = np.array(
            [(uc - cx_img) / fx, (vc - cy_img) / fy, 1.0], dtype=np.float64
        )
        view /= np.linalg.norm(view) + 1e-9
        t = np.cross(e3, view)
        n2t = np.hypot(t[0], t[1])
        if n2t < 1e-4:
            return 1.0, 0.0
        return float(t[0] / n2t), float(t[1] / n2t)

    labels = ("X", "Y", "Z")
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    L = float(axis_half_len_px)
    oid = int(obj_id)

    for e3, col, lab in zip(axes3, colors, labels):
        ux, uy = image_dir(e3)
        p0 = (int(round(uc - ux * L)), int(round(vc - uy * L)))
        p1 = (int(round(uc + ux * L)), int(round(vc + uy * L)))
        p0 = (int(np.clip(p0[0], 0, fw - 1)), int(np.clip(p0[1], 0, fh - 1)))
        p1 = (int(np.clip(p1[0], 0, fw - 1)), int(np.clip(p1[1], 0, fh - 1)))
        cv2.line(vis, p0, p1, col, 2, lineType=cv2.LINE_AA)
        cv2.putText(
            vis,
            f"{lab}{oid}",
            (p1[0] + 3, p1[1] + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            col,
            1,
        )


def _icp_2d(
    P: np.ndarray,
    Q: np.ndarray,
    prev_R: np.ndarray | None = None,
    prev_t: np.ndarray | None = None,
    max_iter: int = 25,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterative Closest Point in 2D: align source P to target Q.
    P, Q: (n,2) and (m,2); returns R (2,2) such that PCA axes rotate as u' = R @ u.
    Uses row convention P_warped = P @ R.T + t each step; accumulates R_acc, t_acc with
    R_acc <- Ri @ R_acc, t_acc <- t_acc @ Ri.T + ti.
    """
    if P.shape[0] < 3 or Q.shape[0] < 3:
        return np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
    R_acc = prev_R.copy() if prev_R is not None else np.eye(2, dtype=np.float64)
    if prev_t is not None:
        t_acc = prev_t.copy()
    else:
        t_acc = Q.mean(axis=0) - (P @ R_acc.T).mean(axis=0)
    prev_err = np.inf
    tree = cKDTree(Q) if cKDTree is not None else None
    for _ in range(max_iter):
        P_t = (P @ R_acc.T) + t_acc
        if tree is not None:
            dist, idx = tree.query(P_t)
            Q_near = Q[idx]
            err = float(np.mean(dist * dist))
        else:
            diff = P_t[:, None, :] - Q[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            idx = np.argmin(d2, axis=1)
            Q_near = Q[idx]
            err = float(np.mean(np.min(d2, axis=1)))
        Ri, ti = _kabsch_rigid_rows_2d(P_t, Q_near)
        R_acc = Ri @ R_acc
        t_acc = t_acc @ Ri.T + ti
        if prev_err < np.inf and abs(prev_err - err) < tol:
            break
        prev_err = err
    return R_acc, t_acc


def draw_object_axes_icp(
    vis: np.ndarray,
    mask_bool: np.ndarray,
    obj_id: int,
    state: dict | None,
    contour_pts: int,
    icp_iter: int,
    axis_half_len_px: float = 40.0,
) -> dict | None:
    """
    Define PCA axes on first good mask; each frame align reference contour to the
    current contour via ICP (closest-point + Kabsch) and draw rotated axes at the
    current contour centroid.
    """
    fh, fw = vis.shape[:2]
    if not np.any(mask_bool):
        return state
    # Erode once to suppress noisy depth/silhouette boundaries before contour sampling.
    m = (mask_bool.astype(np.uint8) * 255)
    m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    clean_mask = m > 0
    ctr = _resample_contour_xy(clean_mask, contour_pts)
    if ctr is None:
        return state
    c_cur = ctr.mean(axis=0)
    X_cur = ctr - c_cur

    col = point_color(int(obj_id))
    L = float(axis_half_len_px)
    if state is None:
        pca = _mask_pca_axes_2d(clean_mask)
        if pca is None:
            return None
        _, u1, u2, _, _ = pca
        state = {
            "ref_contour_centered": ctr - c_cur,
            "u1": u1.astype(np.float64),
            "u2": u2.astype(np.float64),
            "prev_R": np.eye(2, dtype=np.float64),
            "prev_t": np.zeros(2, dtype=np.float64),
        }
        R = np.eye(2, dtype=np.float64)
    else:
        X_ref = state["ref_contour_centered"]
        if X_ref.shape[0] != X_cur.shape[0]:
            return state
        R, t = _icp_2d(
            X_ref,
            X_cur,
            prev_R=state.get("prev_R"),
            prev_t=state.get("prev_t"),
            max_iter=icp_iter,
            tol=1e-4,
        )
        state["prev_R"] = R
        state["prev_t"] = t

    u1t = R @ state["u1"]
    u2t = R @ state["u2"]
    u1t = u1t / (np.linalg.norm(u1t) + 1e-12)
    u2t = u2t / (np.linalg.norm(u2t) + 1e-12)

    cx, cy = float(c_cur[0]), float(c_cur[1])

    def _seg(p0, p1):
        p0 = (int(np.clip(p0[0], 0, fw - 1)), int(np.clip(p0[1], 0, fh - 1)))
        p1 = (int(np.clip(p1[0], 0, fw - 1)), int(np.clip(p1[1], 0, fh - 1)))
        cv2.arrowedLine(vis, p0, p1, col, 2, tipLength=0.2)

    c = np.array([cx, cy], dtype=np.float64)
    _seg(c - u1t * L, c + u1t * L)
    _seg(c - u2t * L, c + u2t * L)
    cv2.putText(
        vis,
        f"ID{obj_id} 2D",
        (int(cx) + 10, int(cy) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        col,
        1,
    )
    return state


def resolve_model_paths(repo_root: Path, model_size: str) -> tuple[str, str]:
    model_size = model_size.lower()
    if model_size == "tiny":
        return (
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            str(repo_root / "checkpoints" / "sam2.1_hiera_tiny.pt"),
        )
    if model_size == "small":
        return (
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            str(repo_root / "checkpoints" / "sam2.1_hiera_small.pt"),
        )
    if model_size in {"base", "base_plus", "b+"}:
        return (
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            str(repo_root / "checkpoints" / "sam2.1_hiera_base_plus.pt"),
        )
    if model_size == "large":
        return (
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            str(repo_root / "checkpoints" / "sam2.1_hiera_large.pt"),
        )
    raise ValueError(f"Unsupported model size: {model_size}")


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare_sam2_input_as_jpeg_folder(
    input_path: str, *, rotate_180: bool = True
) -> tuple[str, str]:
    """
    Build a temporary JPEG frame folder for SAM2 init_state().
    This avoids the optional decord dependency required for direct video-file loading.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = maybe_rotate_180(frame, rotate_180)
        out_path = os.path.join(tmp_dir, f"{frame_i:06d}.jpg")
        cv2.imwrite(out_path, frame)
        frame_i += 1
    cap.release()
    if frame_i == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Input video contained no readable frames.")
    return tmp_dir, tmp_dir


def read_first_scaled_rgb_depth_pair(
    rgb_path: str,
    depth_path: str,
    rgb_div: int,
    rotate_180: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    cap_r = cv2.VideoCapture(rgb_path)
    cap_d = cv2.VideoCapture(depth_path)
    if not cap_r.isOpened() or not cap_d.isOpened():
        if cap_r.isOpened():
            cap_r.release()
        if cap_d.isOpened():
            cap_d.release()
        return None
    ok_r, fr = cap_r.read()
    ok_d, fd = cap_d.read()
    cap_r.release()
    cap_d.release()
    if not ok_r or fr is None or not ok_d or fd is None:
        return None
    fr = maybe_rotate_180(fr, rotate_180)
    fd = maybe_rotate_180(fd, rotate_180)
    bgr_s, d_s = _scale_rgb_and_match_depth(fr, fd, rgb_div)
    return bgr_s, d_s


def main():
    parser = argparse.ArgumentParser(
        description="Apply official Meta SAM2 video tracking from manual point prompts."
    )
    parser.add_argument(
        "--input",
        default="movie.mp4.mov",
        help="RGB video path (OpenCV-readable). With --depth-video, paired frame-by-frame.",
    )
    parser.add_argument(
        "--depth-video",
        default="",
        help="Optional depth recording (e.g. from record_rgb_depth.py); shown alongside SAM2 output.",
    )
    parser.add_argument(
        "--rgb-div",
        type=int,
        default=3,
        help="Scale RGB to (W//N,H//N); with --depth-video, depth is resized to match (ignored if no depth).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frame pairs to decode (0 = until shorter video ends). Only used with --depth-video.",
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Do not rotate frames 180° (default is to rotate all inputs).",
    )
    parser.add_argument(
        "--sam2-repo",
        default="segment-anything-2",
        help="Path to cloned SAM2 repo",
    )
    parser.add_argument(
        "--model-size",
        default="base_plus",
        choices=["tiny", "small", "base_plus", "large"],
        help="Official SAM2.1 model size",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help="(Video predictor only) keep decoded frames in CPU RAM.",
    )
    parser.add_argument(
        "--no-offload-video",
        action="store_true",
        help="(Video predictor only) load every frame onto device at init.",
    )
    parser.add_argument(
        "--use-video-predictor",
        action="store_true",
        help="Use SAM2 video predictor (old path). Default uses SAM2ImagePredictor per-frame memory.",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay alpha")
    parser.add_argument("--output", default="", help="Optional output video path")
    parser.add_argument(
        "--no-post-playback",
        action="store_true",
        help="Disable real-time replay after SAM2 processing completes.",
    )
    parser.add_argument(
        "--contour-pts",
        type=int,
        default=96,
        help="Resampled contour points for ICP alignment",
    )
    parser.add_argument(
        "--icp-iter",
        type=int,
        default=8,
        help="Max ICP iterations per frame (closest point + Kabsch)",
    )
    parser.add_argument(
        "--no-axes",
        action="store_true",
        help="Disable axis overlay on masks",
    )
    parser.add_argument(
        "--depth-raw-dir",
        default="",
        help="Folder of 000000.npy … float32 depth in mm (from record_rgb_depth.py --raw-depth-dir); "
        "copied into the temp session for 3D axes. If omitted with --depth-video, axes use jet brightness as Z.",
    )
    parser.add_argument(
        "--axis-pixels",
        type=float,
        default=40.0,
        help="Half-length of each axis in the image (pixels from mask centroid to each tip).",
    )
    args = parser.parse_args()

    rotate_180 = not args.no_rotate
    depth_path = (args.depth_video or "").strip()
    has_depth_video = bool(depth_path)

    if not os.path.exists(args.input):
        print(f"Input video not found: {args.input}")
        return
    if has_depth_video and not os.path.isfile(depth_path):
        print(f"Depth video not found: {depth_path}")
        return
    raw_dir_arg = (args.depth_raw_dir or "").strip()
    if raw_dir_arg and not os.path.isdir(raw_dir_arg):
        print(f"--depth-raw-dir is not a directory: {raw_dir_arg}")
        return
    if has_depth_video and raw_dir_arg:
        print("3D axes: metric depth from .npy (pinhole back-projection + 3D PCA).")
    elif has_depth_video:
        print("3D axes: depth proxy from jet colormap grayscale (record with --raw-depth-dir for mm).")

    cap: cv2.VideoCapture | None = None
    use_jpeg_playback = False
    temp_frames_dir = ""
    sam2_video_path = ""

    if has_depth_video:
        pair0 = read_first_scaled_rgb_depth_pair(
            args.input,
            depth_path,
            max(1, args.rgb_div),
            rotate_180,
        )
        if pair0 is None:
            print("Could not read first RGB/depth frame pair from videos.")
            return
        first, _ = pair0
        print(
            f"RGB+depth mode: working size {first.shape[1]}x{first.shape[0]} "
            f"(RGB ÷ {max(1, args.rgb_div)}, depth resized to match)."
        )
    else:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Could not open input video: {args.input}")
            return
        ok, fr = cap.read()
        if not ok or fr is None:
            print("Could not read first frame.")
            cap.release()
            return
        first = maybe_rotate_180(fr, rotate_180)
        cap.release()
        cap = None

    if rotate_180:
        print("Applying 180° rotation to all RGB (and depth) frames.")

    points = pick_points(first)
    if not points:
        print("No points selected. Exiting.")
        if cap is not None:
            cap.release()
        return

    repo_root = Path(args.sam2_repo).expanduser().resolve()
    model_cfg, ckpt_path = resolve_model_paths(repo_root, args.model_size)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run: (cd segment-anything-2/checkpoints && ./download_ckpts.sh)")
        if cap is not None:
            cap.release()
        return

    device = choose_device(args.device)
    print(f"Using device={device}, model={args.model_size}")
    use_video_predictor = bool(args.use_video_predictor)
    if use_video_predictor:
        predictor = build_sam2_video_predictor(model_cfg, ckpt_path, device=device)
    else:
        model = build_sam2(model_cfg, ckpt_path, device=device)
        predictor = SAM2ImagePredictor(model)

    if has_depth_video:
        temp_frames_dir = tempfile.mkdtemp(prefix="sam2_rgb_depth_frames_")
        frame_limit = args.max_frames if args.max_frames > 0 else 10**9
        try:
            nbuf = build_jpeg_folder_from_rgb_depth_videos(
                args.input,
                depth_path,
                temp_frames_dir,
                max(1, args.rgb_div),
                frame_limit,
                rotate_180,
                depth_raw_dir=raw_dir_arg,
            )
        except RuntimeError as e:
            print(e)
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            if cap is not None:
                cap.release()
            return
        print(f"Decoded {nbuf} RGB+depth frame pair(s) into temp folder.")
        sam2_video_path = temp_frames_dir
        use_jpeg_playback = True
        if cap is not None:
            cap.release()
            cap = None
    else:
        sam2_video_path, temp_frames_dir = prepare_sam2_input_as_jpeg_folder(
            args.input, rotate_180=rotate_180
        )

    print(f"Prepared JPEG frame folder for SAM2: {sam2_video_path}")
    state = None
    if use_video_predictor:
        if args.offload_video_to_cpu:
            offload_video = True
        elif args.no_offload_video:
            offload_video = False
        elif device == "mps":
            offload_video = True
        else:
            offload_video = False
        if offload_video:
            print(
                "SAM2 offload_video_to_cpu=True: frames stay on CPU; each frame is moved to "
                f"{device} for the backbone (avoids MPS failures on long clips)."
            )
        state = predictor.init_state(
            video_path=sam2_video_path,
            offload_video_to_cpu=offload_video,
        )

        for obj_id, x, y in points:
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=int(obj_id),
                points=np.array([[x, y]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )

    fps_cap = cv2.VideoCapture(args.input)
    fps = fps_cap.get(cv2.CAP_PROP_FPS) if fps_cap.isOpened() else 0.0
    fps_cap.release()
    if not fps or fps <= 0:
        fps = 30.0
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    replay_path = args.output if args.output else os.path.join(
        tempfile.gettempdir(), f"sam2_replay_{os.getpid()}.mp4"
    )
    writer = cv2.VideoWriter(replay_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        print(f"Could not open output writer: {replay_path}")
        if temp_frames_dir and os.path.exists(temp_frames_dir):
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
        return

    if not use_jpeg_playback:
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print("Could not reopen input video for playback.")
            if writer is not None:
                writer.release()
            if temp_frames_dir and os.path.exists(temp_frames_dir):
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
            return
    cur_idx = -1
    axis_states: dict[int, dict | None] = {}
    # SAM2ImagePredictor memory (per-object logits from previous frame).
    prev_logits: dict[int, np.ndarray] = {}

    if use_video_predictor:
        pred_iter = predictor.propagate_in_video(state)
    else:
        pred_iter = None
    frame_idx = 0
    while True:
        if use_jpeg_playback:
            frame_path = os.path.join(sam2_video_path, f"{frame_idx:06d}.jpg")
            frame = cv2.imread(frame_path) if os.path.isfile(frame_path) else None
        else:
            ok, frame = cap.read()
            if ok and frame is not None:
                frame = maybe_rotate_180(frame, rotate_180)
            else:
                frame = None
        if frame is None:
            break

        if use_video_predictor:
            try:
                fidx, obj_ids, masks = next(pred_iter)
            except StopIteration:
                break
            if int(fidx) != int(frame_idx):
                frame_idx = int(fidx)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device == "cuda"
                else nullcontext()
            )
            obj_ids_list = []
            masks_list = []
            with torch.inference_mode():
                with amp_ctx:
                    predictor.set_image(frame_rgb)
                    for obj_id, x, y in points:
                        oid = int(obj_id)
                        if oid in prev_logits:
                            masks_np, scores, logits = predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                mask_input=prev_logits[oid],
                                multimask_output=False,
                            )
                        else:
                            masks_np, scores, logits = predictor.predict(
                                point_coords=np.array([[x, y]], dtype=np.float32),
                                point_labels=np.array([1], dtype=np.int32),
                                mask_input=None,
                                multimask_output=False,
                            )
                        prev_logits[oid] = logits
                        obj_ids_list.append(oid)
                        masks_list.append(masks_np[0].astype(np.float32))
            if not masks_list:
                frame_idx += 1
                continue
            obj_ids = obj_ids_list
            masks = torch.from_numpy(np.stack(masks_list, axis=0))

        if hasattr(obj_ids, "tolist"):
            obj_ids_list = [int(x) for x in obj_ids.tolist()]
        else:
            obj_ids_list = [int(i) for i in obj_ids]
        vis = overlay_masks_with_ids(frame, obj_ids_list, masks, alpha=args.alpha)

        dvis = None
        depth_mm_arr = None
        if has_depth_video:
            dj = os.path.join(_stream_depth_subdir(sam2_video_path), f"{frame_idx:06d}.jpg")
            if os.path.isfile(dj):
                dvis = cv2.imread(dj)
            rj = os.path.join(_raw_depth_subdir(sam2_video_path), f"{frame_idx:06d}.npy")
            if os.path.isfile(rj):
                depth_mm_arr = np.load(rj)

        if not args.no_axes:
            fh, fw = frame.shape[:2]
            masks_np = masks.detach().cpu().numpy()
            n_m = min(len(obj_ids_list), masks_np.shape[0])
            use_depth_axes = has_depth_video and (depth_mm_arr is not None or dvis is not None)
            for i in range(n_m):
                oid = int(obj_ids_list[i])
                binm = _mask_to_2d_bool(masks_np[i], fh, fw)
                if use_depth_axes:
                    if depth_mm_arr is not None:
                        draw_three_axes_from_depth(
                            vis,
                            binm,
                            depth_mm_arr,
                            oid,
                            axis_half_len_px=args.axis_pixels,
                            metric_depth=True,
                        )
                    else:
                        gray = cv2.cvtColor(dvis, cv2.COLOR_BGR2GRAY).astype(
                            np.float32
                        )
                        draw_three_axes_from_depth(
                            vis,
                            binm,
                            gray,
                            oid,
                            axis_half_len_px=args.axis_pixels,
                            metric_depth=False,
                        )
                else:
                    axis_states[oid] = draw_object_axes_icp(
                        vis,
                        binm,
                        oid,
                        axis_states.get(oid),
                        contour_pts=max(12, args.contour_pts),
                        icp_iter=max(3, args.icp_iter),
                        axis_half_len_px=args.axis_pixels,
                    )
        if frame_idx == 0:
            for obj_id, px_f, py_f in points:
                px, py = int(px_f), int(py_f)
                cv2.circle(vis, (px, py), 5, (0, 255, 255), -1)
                cv2.putText(
                    vis,
                    f"ID{obj_id}",
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("SAM2 manual points video", vis)
        if has_depth_video and dvis is not None:
            cv2.imshow("Depth (same grid as RGB)", dvis)
        writer.write(vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        frame_idx += 1

    if cap is not None:
        cap.release()
    writer.release()
    if not args.no_post_playback:
        print(f"Replaying result at source FPS ({fps:.2f})...")
        rp = cv2.VideoCapture(replay_path)
        if rp.isOpened():
            delay_ms = max(1, int(round(1000.0 / float(fps))))
            while True:
                ok, fr = rp.read()
                if not ok or fr is None:
                    break
                cv2.imshow("SAM2 manual points video (replay)", fr)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    break
            rp.release()
    if temp_frames_dir and os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
    if not args.output and os.path.isfile(replay_path):
        try:
            os.remove(replay_path)
        except OSError:
            pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
