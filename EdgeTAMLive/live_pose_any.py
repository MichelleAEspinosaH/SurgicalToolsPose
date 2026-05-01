#!/usr/bin/env python3
"""
Live 6DoF pose for arbitrary objects: EdgeTAM seeds → fal SAM3D GLBs → containment rigid
registration → native GLB axes overlay (60 px arrows) + COM trails.

Requires:
  - EdgeTAM checkpoint under EdgeTAM/checkpoints/edgetam.pt
  - scipy, trimesh, fal-client
  - export FAL_KEY="..."

Usage (from repo root or EdgeTAMLive):
  python live_pose_any.py
  python live_pose_any.py --camera 1 --glb-dir ./sam3d_live_objects

A second OpenCV window, ``3D mesh ↔ 2D mask (alignment)``, shows the projected
GLB silhouette vs the EdgeTAM mask (white=overlap, yellow=mask-only, magenta=mesh-only).
Both sides use ``cv2.RETR_EXTERNAL`` + solid fill for alignment (no holes in the 2D
solids); GLB meshes are passed through ``trimesh.repair.fill_holes`` when loading.
Pass ``--no-align-debug`` to hide the alignment window.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch

try:
    import trimesh  # type: ignore
except Exception:
    trimesh = None

try:
    from scipy.optimize import minimize  # type: ignore
except Exception:
    minimize = None

# Reuse EdgeTAM live pipeline pieces from sibling module.
from live_track_copy import (
    CHECKPOINT,
    LiveFrameProvider,
    TARGET_SIZE,
    _autocast_config,
    _bits_from_reg_sign,
    _draw_pose_hud,
    _draw_solid_alignment,
    _estimate_intrinsics_from_cap,
    _load_predictor,
    _mask_ellipse_params,
    _mask_image_plane_axis_unit_cam,
    _mask_iou_dice,
    _mask_to_2d_bool,
    _reg_sign_bits_from_index,
    _reg_sign_from_state,
    _rotmat_align_unit_vectors,
    _unit_pca_axis_signed,
    choose_device,
    detect_orbbec_camera,
    get_mask_contour,
    overlay_masks,
    pick_points_live,
    point_color,
    preprocess,
)

# Match live_track_copy registration hyperparameters.
_REG_NEIGHBORHOOD_W = 0.02
_REG_MAXITER_FIRST = 40
_REG_MAXITER_TRACK = 22

AXIS_LENGTH_PX = 60
# Match live_track.py default COM trail cap.
_MAX_COM_TRAIL = 60


def _euler_zyx_deg_from_rvec(rvec: np.ndarray) -> tuple[float, float, float]:
    """Euler ZYX degrees from Rodrigues vector (same convention as live_track / live_track_copy HUD)."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
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


def _draw_com_pose_readout(
    vis: np.ndarray,
    obj_id: int,
    cx: int,
    cy: int,
    pose_state: dict | None,
) -> None:
    """Near mask COM: pixel position + object R / T (same pose as mesh axes / HUD)."""
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    fh, fw = vis.shape[:2]
    x0 = int(np.clip(cx + 8, 0, fw - 2))
    y0 = int(np.clip(cy - 8, 28, fh - 8))

    if not pose_state or pose_state.get("rvec") is None:
        line = f"ID{obj_id} COM({cx},{cy})"
        cv2.putText(vis, line, (x0, y0), fnt, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, line, (x0, y0), fnt, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        return

    rx, ry, rz = _euler_zyx_deg_from_rvec(pose_state["rvec"])
    tvec = pose_state.get("tvec")
    if tvec is None:
        line = f"ID{obj_id} COM({cx},{cy})  R({rx:.0f},{ry:.0f},{rz:.0f})"
    else:
        tv = np.asarray(tvec, dtype=np.float64).reshape(3)
        line = (
            f"ID{obj_id} COM({cx},{cy})  "
            f"R({rx:.0f},{ry:.0f},{rz:.0f})  T({tv[0]:.2f},{tv[1]:.2f},{tv[2]:.2f})"
        )
    cv2.putText(vis, line, (x0, y0), fnt, 0.40, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, line, (x0, y0), fnt, 0.40, (0, 0, 0), 1, cv2.LINE_AA)


def _mask_outer_filled_bool(mask_bool: np.ndarray) -> np.ndarray:
    """
    Solidify mask using only cv2.RETR_EXTERNAL contours: each outer boundary is
    filled (cv2.FILLED), removing interior holes and ignoring nested contours.
    """
    if not np.any(mask_bool):
        return mask_bool
    m = (mask_bool.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask_bool
    out = np.zeros_like(m)
    for c in cnts:
        cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)
    return out > 0


# ---------------------------------------------------------------------------
# Native-coordinate mesh (no PCA — preserves GLB +X,+Y,+Z)
# ---------------------------------------------------------------------------


class NativeAxisMesh:
    """Vertices/faces in GLB native frame; origin at (0,0,0)."""

    def __init__(self, glb_path: Path | str):
        if trimesh is None:
            raise RuntimeError("trimesh is required to load GLB meshes.")
        path = Path(glb_path)
        loaded = trimesh.load(str(path), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not parts:
                raise ValueError(f"No mesh geometry in GLB: {path}")
            mesh = trimesh.util.concatenate(parts)
        elif not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"Unsupported mesh type in GLB: {type(loaded)}")
        else:
            mesh = loaded

        # Patch topological holes in the surface (does not change axes origin).
        try:
            from trimesh import repair as _trimesh_repair

            if hasattr(_trimesh_repair, "fill_holes"):
                _trimesh_repair.fill_holes(mesh)
        except Exception:
            pass

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        self.mesh_vertices = verts
        self.mesh_faces = faces
        # Cached vertex subsets for DT-based registration hot paths.
        self.sample_vertices_relock = _uniform_vertex_samples(verts, max_points=1400)
        self.sample_vertices_fast = _uniform_vertex_samples(verts, max_points=900)
        mn = verts.min(axis=0)
        mx = verts.max(axis=0)
        self.extents = (mx - mn).astype(np.float64)


def _mesh_silhouette_bool(
    verts: np.ndarray,
    faces: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    fh: int,
    fw: int,
) -> np.ndarray:
    pred_mask = np.zeros((fh, fw), dtype=np.uint8)
    proj_mesh, _ = cv2.projectPoints(verts.astype(np.float64), rvec, tvec, K, dist)
    pts2d = proj_mesh.reshape(-1, 2)
    for f_idx in faces:
        poly = np.round(pts2d[f_idx]).astype(np.int32)
        cv2.fillConvexPoly(pred_mask, poly, 255, lineType=cv2.LINE_AA)
    # Same as 2D mask: only outer silhouette, solid (no pinholes / nested gaps).
    return _mask_outer_filled_bool(pred_mask > 0)


def _mesh_silhouette_u8(
    verts: np.ndarray,
    faces: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    fh: int,
    fw: int,
) -> np.ndarray:
    return (_mesh_silhouette_bool(verts, faces, rvec, tvec, K, dist, fh, fw).astype(np.uint8) * 255)


def _layout_alignment_mosaic(panels: list[np.ndarray], per_row: int = 2) -> np.ndarray | None:
    """Horizontal chunks of ``per_row`` panels, then vertical stack; pad short last row."""
    if not panels:
        return None
    pr = max(1, int(per_row))
    rows: list[np.ndarray] = []
    for r0 in range(0, len(panels), pr):
        rows.append(np.hstack(panels[r0 : r0 + pr]))
    max_w = max(int(r.shape[1]) for r in rows)
    out_rows: list[np.ndarray] = []
    for r in rows:
        h, w = int(r.shape[0]), int(r.shape[1])
        if w >= max_w:
            out_rows.append(r)
            continue
        pad = np.zeros((h, max_w - w, r.shape[2]), dtype=r.dtype)
        out_rows.append(np.hstack([r, pad]))
    return np.vstack(out_rows) if len(out_rows) > 1 else out_rows[0]


def _alignment_debug_panel(
    frame_bgr: np.ndarray,
    est: NativeAxisMesh,
    pose_state: dict,
    mask_bool: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    obj_id: int,
    frame_idx: int,
) -> np.ndarray | None:
    """
    Troubleshooting view: projected 3D mesh fill vs EdgeTAM mask fill.
    Both use RETR_EXTERNAL-filled solids (no holes). White = overlap,
    yellow = mask only, magenta = mesh only (BGR).
    """
    rvec = pose_state.get("rvec")
    tvec = pose_state.get("tvec")
    if rvec is None or tvec is None or not np.any(mask_bool):
        return None
    fh, fw = mask_bool.shape[:2]
    s = _reg_sign_from_state(pose_state)
    verts_s = est.mesh_vertices * s.reshape(1, 3)
    mesh_u8 = _mesh_silhouette_u8(verts_s, est.mesh_faces, rvec, tvec, K, dist, fh, fw)
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    iou, dice = _mask_iou_dice(mask_u8, mesh_u8)
    panel = _draw_solid_alignment(frame_bgr, mask_u8, mesh_u8, obj_id, iou, dice)
    cont = pose_state.get("containment")
    if cont is not None:
        cv2.putText(
            panel,
            f"containment={float(cont):.3f}",
            (12, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        panel,
        f"frame {int(frame_idx)}",
        (12, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return panel


def _containment_metric(mask_bool: np.ndarray, sil_bool: np.ndarray) -> float:
    """Higher is better: mask covered by silhouette, penalize huge silhouette."""
    m = float(mask_bool.sum())
    if m < 1.0:
        return -1.0
    inter = float(np.logical_and(mask_bool, sil_bool).sum())
    sil_sum = float(sil_bool.sum())
    return (inter / m) - 0.05 * (sil_sum / m)


def _containment_from_pose(
    mask_bool: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    fh, fw = mask_bool.shape[:2]
    sil = _mesh_silhouette_bool(verts, faces, rvec, tvec, K, dist, fh, fw)
    return _containment_metric(mask_bool, sil)


def _uniform_vertex_samples(verts: np.ndarray, max_points: int = 1400) -> np.ndarray:
    """Uniformly subsample mesh vertices for fast projection-based scoring."""
    n = int(verts.shape[0])
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    k = int(max(32, min(max_points, n)))
    if k >= n:
        return np.asarray(verts, dtype=np.float64)
    idx = np.linspace(0, n - 1, num=k, dtype=np.int32)
    return np.asarray(verts[idx], dtype=np.float64)


def _prepare_mask_dt(mask_bool: np.ndarray) -> dict:
    """
    Build distance-transform maps once per frame/object.
    dt_outside is zero inside the mask and grows outside.
    """
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    inv = 255 - mask_u8
    dt_outside = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    ys, xs = np.where(mask_bool)
    if ys.size > 0:
        cx = float(xs.mean())
        cy = float(ys.mean())
    else:
        cx = cy = 0.0
    return {
        "dt_outside": dt_outside.astype(np.float32),
        "mask_area": float(mask_bool.sum()),
        "mask_cx": cx,
        "mask_cy": cy,
    }


def _dt_pose_score(
    verts_sample: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    dt_ctx: dict,
    fh: int,
    fw: int,
) -> float:
    """
    Fast score from projected sample points and mask distance transform.
    Higher is better.
    """
    if verts_sample.size == 0:
        return -1e9
    proj, _ = cv2.projectPoints(verts_sample, rvec, tvec, K, dist)
    pts = proj.reshape(-1, 2)
    xs = np.round(pts[:, 0]).astype(np.int32)
    ys = np.round(pts[:, 1]).astype(np.int32)
    inside = (xs >= 0) & (xs < fw) & (ys >= 0) & (ys < fh)
    if not np.any(inside):
        return -1e9
    xs_i = xs[inside]
    ys_i = ys[inside]
    dt_out = dt_ctx["dt_outside"][ys_i, xs_i]
    mean_out = float(np.mean(dt_out)) if dt_out.size > 0 else 1e6
    outside_ratio = float(np.mean(dt_out > 0.5)) if dt_out.size > 0 else 1.0

    # Keep projected sample spread roughly in the same scale as mask area.
    x0 = float(xs_i.min())
    x1 = float(xs_i.max())
    y0 = float(ys_i.min())
    y1 = float(ys_i.max())
    proj_area = max(1.0, (x1 - x0 + 1.0) * (y1 - y0 + 1.0))
    mask_area = max(1.0, float(dt_ctx["mask_area"]))
    area_pen = abs(float(np.log((proj_area + 1.0) / (mask_area + 1.0))))

    # Center consistency also helps avoid drift in low-texture motion.
    cx_p = float(xs_i.mean())
    cy_p = float(ys_i.mean())
    cdist = float(np.hypot(cx_p - float(dt_ctx["mask_cx"]), cy_p - float(dt_ctx["mask_cy"])))

    return float(-(0.90 * mean_out + 6.5 * outside_ratio + 0.35 * area_pen + 0.03 * cdist))


def _p0_native_extent_seed(
    est: NativeAxisMesh,
    mask_bool: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    s: np.ndarray,
    t_init: np.ndarray,
    z0: float,
) -> np.ndarray | None:
    ell = _mask_ellipse_params(mask_bool)
    if ell is None:
        return None

    maj_deg = float(ell.get("major_axis_deg", ell["angle_deg"]))
    target_maj = _mask_image_plane_axis_unit_cam(K, ell["center"], maj_deg, z0)

    ext = np.asarray(est.extents, dtype=np.float64)
    i0 = int(np.argmax(ext))

    tv = np.asarray(t_init, dtype=np.float64).reshape(3, 1)
    verts_s = est.mesh_vertices * s.reshape(1, 3)

    ax0 = np.zeros(3, dtype=np.float64)
    ax0[i0] = 1.0

    best_R: np.ndarray | None = None
    best_score = -1e18

    for flip_l in (1.0, -1.0):
        d0 = _unit_pca_axis_signed(i0, s, flip_l)
        r_align = _rotmat_align_unit_vectors(d0, target_maj)
        for k in range(4):
            psi = float(k) * (0.5 * np.pi)
            r_tw, _ = cv2.Rodrigues((psi * ax0).reshape(3, 1))
            r = (r_align @ r_tw).astype(np.float64)
            rvec, _ = cv2.Rodrigues(r)
            rv = rvec.astype(np.float64)
            sc = _containment_from_pose(mask_bool, verts_s, est.mesh_faces, rv, tv, K, dist)
            if sc > best_score:
                best_score = sc
                best_R = r.copy()

    if best_R is None:
        return None

    r_final, _ = cv2.Rodrigues(best_R.astype(np.float64))
    return np.concatenate([r_final.reshape(3), tv.reshape(3)]).astype(np.float64)


def _register_rigid_containment(
    est: NativeAxisMesh,
    mask_bool: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    state: dict | None,
) -> dict:
    """Rigid SE(3): maximize containment(mask ⊂ silhouette)."""
    if state is None:
        state = {}
    if minimize is None or not np.any(mask_bool):
        if minimize is None and not getattr(_register_rigid_containment, "_warned", False):
            print("scipy not installed; install scipy for rigid mesh–mask registration.")
            setattr(_register_rigid_containment, "_warned", True)
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
    fh, fw = mask_bool.shape[:2]
    dt_ctx = _prepare_mask_dt(mask_bool)

    best_score = -1e18
    best_rv = None
    best_tv = None
    best_s: np.ndarray | None = None

    for bits in bits_list:
        s = _reg_sign_bits_from_index(int(bits))
        verts_s = est.mesh_vertices * s.reshape(1, 3)
        verts_base = getattr(est, "sample_vertices_relock", est.mesh_vertices)
        verts_sample = np.asarray(verts_base, dtype=np.float64) * s.reshape(1, 3)
        p0 = p_base.astype(np.float64).copy()
        if prev_r is None:
            p0[3:] = t_init
            seed = _p0_native_extent_seed(est, mask_bool, K, dist, s, t_init, Z0)
            if seed is not None:
                p0 = seed.astype(np.float64)

        def _obj(p: np.ndarray, p0_ref=p0, vs=verts_s) -> float:
            rv = np.asarray(p[:3], dtype=np.float64).reshape(3, 1)
            tv = np.asarray(p[3:], dtype=np.float64).reshape(3, 1)
            sc = _dt_pose_score(verts_sample, rv, tv, K, dist, dt_ctx, fh, fw)
            reg = float(_REG_NEIGHBORHOOD_W) * float(np.sum((p - p0_ref) ** 2))
            return float(-sc + reg)

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
        sc = _dt_pose_score(verts_sample, rv, tv, K, dist, dt_ctx, fh, fw)
        if sc > best_score:
            best_score, best_rv, best_tv, best_s = sc, rv.copy(), tv.copy(), s.copy()

    if best_rv is None or best_s is None:
        return state

    state["reg_sign"] = best_s
    state["rvec"] = best_rv
    state["tvec"] = best_tv
    state["containment"] = float(best_score)
    state["score_type"] = "distance_transform"
    return state


def _pose_alignment_metrics(
    est: NativeAxisMesh,
    mask_bool: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    state: dict,
) -> dict:
    """Compute overlap and geometric quality metrics for current pose."""
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None or not np.any(mask_bool):
        return {"iou": 0.0, "dice": 0.0, "containment": -1.0, "centroid_px": 1e9}

    fh, fw = mask_bool.shape[:2]
    s = _reg_sign_from_state(state)
    verts_s = est.mesh_vertices * s.reshape(1, 3)
    mesh_u8 = _mesh_silhouette_u8(verts_s, est.mesh_faces, rvec, tvec, K, dist, fh, fw)
    mask_u8 = mask_bool.astype(np.uint8) * 255
    iou, dice = _mask_iou_dice(mask_u8, mesh_u8)
    containment = _containment_metric(mask_bool, mesh_u8 > 0)

    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        centroid_px = 1e9
    else:
        mx, my = float(xs.mean()), float(ys.mean())
        o_proj, _ = cv2.projectPoints(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            K,
            dist,
        )
        ox, oy = o_proj.reshape(-1, 2)[0]
        centroid_px = float(np.hypot(mx - float(ox), my - float(oy)))

    return {
        "iou": float(iou),
        "dice": float(dice),
        "containment": float(containment),
        "centroid_px": float(centroid_px),
    }


def _pose_confidence_from_metrics(metrics: dict) -> float:
    """Blend overlap and geometry checks into one confidence score in [0,1]."""
    iou = float(metrics.get("iou", 0.0))
    dice = float(metrics.get("dice", 0.0))
    containment = float(metrics.get("containment", -1.0))
    centroid_px = float(metrics.get("centroid_px", 1e9))
    containment_term = float(np.clip((containment + 0.15) / 0.85, 0.0, 1.0))
    centroid_term = float(np.exp(-centroid_px / 90.0))
    conf = 0.50 * iou + 0.25 * dice + 0.15 * containment_term + 0.10 * centroid_term
    return float(np.clip(conf, 0.0, 1.0))


def _smooth_pose_state(state: dict, alpha_t: float, alpha_r: float) -> None:
    """EMA on tvec/rvec; uses raw pose as input and writes filtered pose output."""
    rv_raw = state.get("rvec_raw")
    tv_raw = state.get("tvec_raw")
    if rv_raw is None or tv_raw is None:
        return
    rv_raw = np.asarray(rv_raw, dtype=np.float64).reshape(3, 1)
    tv_raw = np.asarray(tv_raw, dtype=np.float64).reshape(3, 1)
    rv_prev = state.get("rvec")
    tv_prev = state.get("tvec")
    if rv_prev is None or tv_prev is None:
        state["rvec"] = rv_raw
        state["tvec"] = tv_raw
        return
    rv_prev = np.asarray(rv_prev, dtype=np.float64).reshape(3, 1)
    tv_prev = np.asarray(tv_prev, dtype=np.float64).reshape(3, 1)
    a_t = float(np.clip(alpha_t, 0.05, 1.0))
    a_r = float(np.clip(alpha_r, 0.05, 1.0))
    state["tvec"] = (1.0 - a_t) * tv_prev + a_t * tv_raw
    state["rvec"] = (1.0 - a_r) * rv_prev + a_r * rv_raw


def _fast_pose_update_local(
    est: NativeAxisMesh,
    mask_bool: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    state: dict,
    rot_step_deg: float = 2.0,
    trans_step_xy: float = 0.003,
    trans_step_z: float = 0.008,
) -> dict:
    """
    Fast local search around previous pose.
    This is intentionally lightweight for per-frame updates.
    """
    if not np.any(mask_bool):
        return state
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None:
        return state

    rv0 = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tv0 = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    s = _reg_sign_from_state(state)
    verts_s = est.mesh_vertices * s.reshape(1, 3)
    verts_base = getattr(est, "sample_vertices_fast", est.mesh_vertices)
    verts_sample = np.asarray(verts_base, dtype=np.float64) * s.reshape(1, 3)
    fh, fw = mask_bool.shape[:2]
    dt_ctx = _prepare_mask_dt(mask_bool)

    ys, xs = np.where(mask_bool)
    if ys.size > 0:
        mx, my = float(xs.mean()), float(ys.mean())
        p0, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]], dtype=np.float64), rv0, tv0, K, dist)
        ox, oy = p0.reshape(-1, 2)[0]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        z = max(0.05, float(tv0[2, 0]))
        tv_base = tv0.copy()
        tv_base[0, 0] += (mx - float(ox)) / fx * z
        tv_base[1, 0] += (my - float(oy)) / fy * z
    else:
        tv_base = tv0.copy()

    ang = float(np.deg2rad(max(0.2, rot_step_deg)))
    rot_deltas = [
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([ang, 0.0, 0.0], dtype=np.float64),
        np.array([-ang, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, ang, 0.0], dtype=np.float64),
        np.array([0.0, -ang, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, ang], dtype=np.float64),
        np.array([0.0, 0.0, -ang], dtype=np.float64),
    ]
    trans_deltas = [
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([trans_step_xy, 0.0, 0.0], dtype=np.float64),
        np.array([-trans_step_xy, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, trans_step_xy, 0.0], dtype=np.float64),
        np.array([0.0, -trans_step_xy, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, trans_step_z], dtype=np.float64),
        np.array([0.0, 0.0, -trans_step_z], dtype=np.float64),
    ]

    best_score = float(_dt_pose_score(verts_sample, rv0, tv0, K, dist, dt_ctx, fh, fw))
    best_rv = rv0.copy()
    best_tv = tv0.copy()
    for dr in rot_deltas:
        rv = rv0 + dr.reshape(3, 1)
        for dt in trans_deltas:
            tv = tv_base + dt.reshape(3, 1)
            sc = float(_dt_pose_score(verts_sample, rv, tv, K, dist, dt_ctx, fh, fw))
            if sc > best_score:
                best_score = sc
                best_rv = rv.copy()
                best_tv = tv.copy()

    state["rvec_raw"] = best_rv
    state["tvec_raw"] = best_tv
    state["containment"] = float(best_score)
    if state.get("reg_sign") is None:
        state["reg_sign"] = s
    return state


def _native_axis_points_in_object_frame(s: np.ndarray) -> np.ndarray:
    """Origin + unit +X,+Y,+Z in signed object frame (matches verts_s = verts * s)."""
    s = np.asarray(s, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [s[0], 0.0, 0.0],
            [0.0, s[1], 0.0],
            [0.0, 0.0, s[2]],
        ],
        dtype=np.float64,
    )


def _draw_native_axes_fixed_pixel(
    vis: np.ndarray,
    state: dict,
    K: np.ndarray,
    dist: np.ndarray,
    length_px: float = AXIS_LENGTH_PX,
) -> None:
    """Project GLB native origin and axes; draw arrows with fixed image length."""
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None:
        return
    s = _reg_sign_from_state(state)
    pts_obj = _native_axis_points_in_object_frame(s)
    proj, _ = cv2.projectPoints(pts_obj.astype(np.float64), rvec, tvec, K, dist)
    pts2d = proj.reshape(-1, 2)

    fh, fw = vis.shape[:2]

    def clip_pt(p: np.ndarray) -> tuple[int, int]:
        return (int(np.clip(p[0], 0, fw - 1)), int(np.clip(p[1], 0, fh - 1)))

    origin = pts2d[0].ravel()
    colors = [(0, 0, 220), (0, 200, 0), (220, 80, 0)]  # X red, Y green, Z orange (BGR)
    labels = ("X", "Y", "Z")
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    o_int = clip_pt(origin)

    for k in range(3):
        tip_full = pts2d[k + 1].ravel()
        v = tip_full - origin
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            continue
        tip = origin + v * (length_px / n)
        tip_i = clip_pt(tip)
        cv2.arrowedLine(
            vis,
            o_int,
            tip_i,
            colors[k],
            2,
            tipLength=0.2,
            line_type=cv2.LINE_AA,
        )
        cv2.putText(vis, labels[k], (tip_i[0] + 4, tip_i[1] + 4), fnt, 0.50, colors[k], 1, cv2.LINE_AA)

    cv2.circle(vis, o_int, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(vis, o_int, 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)


def _fal_download_glb(
    fal_model: str,
    seed: int,
    image_url: str,
    mask_path: Path,
    glb_out: Path,
) -> tuple[bool, str]:
    try:
        import fal_client  # type: ignore
    except Exception as e:
        return False, f"fal_client import failed: {e}"

    try:
        mask_url = fal_client.upload_file(str(mask_path))
        result = fal_client.subscribe(
            fal_model,
            arguments={"image_url": image_url, "mask_urls": [mask_url], "seed": int(seed)},
            with_logs=False,
        )
    except Exception as e:
        return False, str(e)

    if not isinstance(result, dict):
        return False, "unexpected fal result type"
    model_glb = result.get("model_glb") or {}
    url = model_glb.get("url") if isinstance(model_glb, dict) else None
    if not isinstance(url, str) or not url:
        return False, "no model_glb.url in fal response"

    try:
        glb_out.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, str(glb_out))
    except Exception as e:
        return False, f"download failed: {e}"

    return glb_out.is_file(), "ok"


def _wait_fal_progress_ui(
    seed_frame: np.ndarray,
    seed_mask_bool: dict[int, np.ndarray],
    futures_map: dict,
    win_name: str,
) -> dict[int, tuple[bool, str]]:
    """Poll futures; show frozen frame + masks + per-ID status."""
    status: dict[int, str] = {oid: "queued…" for oid in futures_map.values()}
    results: dict[int, tuple[bool, str]] = {}

    def draw() -> np.ndarray:
        ids = sorted(seed_mask_bool.keys())
        if not ids:
            vis = seed_frame.copy()
        else:
            vis = seed_frame.copy().astype(np.float32)
            for oid in ids:
                binm = seed_mask_bool[oid]
                c = np.array(point_color(int(oid)), dtype=np.float32)
                vis[binm] = vis[binm] * 0.35 + c * 0.65
            vis = vis.astype(np.uint8)

        y = 26
        cv2.putText(
            vis,
            "Waiting for fal SAM3D (parallel)…",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 22
        for oid in ids:
            cv2.putText(
                vis,
                f"ID{oid}: {status.get(oid, '')}",
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (220, 220, 255),
                2,
                cv2.LINE_AA,
            )
            y += 20
        return vis

    cv2.namedWindow(win_name)
    pending = set(futures_map.keys())
    while pending:
        for fut in list(pending):
            if fut.done():
                oid = futures_map[fut]
                try:
                    ok, msg = fut.result()
                except Exception as e:
                    ok, msg = False, str(e)
                results[oid] = (ok, msg)
                status[oid] = "done" if ok else f"ERR: {msg[:40]}"
                pending.discard(fut)

        cv2.imshow(win_name, draw())
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break

    # Collect any stragglers if the window closed early
    for fut, oid in futures_map.items():
        if oid in results:
            continue
        if fut.done():
            try:
                results[oid] = fut.result()
            except Exception as e:
                results[oid] = (False, str(e))
        else:
            results[oid] = (False, "interrupted")

    cv2.destroyWindow(win_name)
    return results


def run(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    print(f"Device: {device}")

    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT}")
        return

    if trimesh is None:
        print("Install trimesh to load GLB meshes: pip install trimesh")
        return

    if not os.environ.get("FAL_KEY"):
        print("Set FAL_KEY environment variable for fal-ai.")
        return

    print("Loading EdgeTAM …")
    predictor = _load_predictor(device)
    image_size = predictor.image_size

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    rotate_180 = detect_orbbec_camera(args.camera)

    K_cam = _estimate_intrinsics_from_cap(cap, TARGET_SIZE[0], TARGET_SIZE[1])
    dist_cam = np.zeros((5, 1), dtype=np.float64)

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
    if not points or seed_frame is None:
        print("No seed selection; exiting.")
        stop_flag.set()
        cap.release()
        return

    fh, fw = seed_frame.shape[:2]

    tmp = tempfile.mkdtemp(prefix="edgetam_pose_any_")
    cv2.imwrite(os.path.join(tmp, "000000.jpg"), seed_frame)
    state = predictor.init_state(tmp, async_loading_frames=False)
    shutil.rmtree(tmp, ignore_errors=True)

    state["images"] = provider
    state["num_frames"] = 1_000_000

    for obj_id, x, y in points:
        predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=int(obj_id),
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

    ac_device, ac_dtype, ac_enabled = _autocast_config(device, args.half)

    seed_masks_raw: dict[int, np.ndarray] = {}
    ids_order: list[int] = []

    print("EdgeTAM: resolving seed masks (frame 0)…")
    t_masks0 = time.perf_counter()
    with torch.autocast(device_type=ac_device, dtype=ac_dtype, enabled=ac_enabled):
        gen0 = predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=0)
        _, obj_ids0, masks0 = next(gen0)
        if hasattr(obj_ids0, "tolist"):
            ids_order = [int(x) for x in obj_ids0.tolist()]
        else:
            ids_order = [int(x) for x in obj_ids0]
        masks_np = masks0.detach().cpu().numpy()
        for i in range(min(len(ids_order), masks_np.shape[0])):
            oid = ids_order[i]
            seed_masks_raw[oid] = _mask_to_2d_bool(masks_np[i], fh, fw)

    # Alignment / registration: hole-free solids (RETR_EXTERNAL fill only).
    seed_mask_bool = {oid: _mask_outer_filled_bool(mb) for oid, mb in seed_masks_raw.items()}
    t_masks_s = time.perf_counter() - t_masks0
    print(
        f"Timing: obtain_masks (EdgeTAM frame 0 + binarize + outer fill) = {t_masks_s:.3f} s"
    )

    t_3d0 = time.perf_counter()
    work_dir = Path(args.glb_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    seed_png = work_dir / "seed_frame.png"
    cv2.imwrite(str(seed_png), seed_frame)
    # fal SAM3D still receives the raw EdgeTAM mask PNGs.
    for oid, mb in seed_masks_raw.items():
        cv2.imwrite(str(work_dir / f"mask_{oid}.png"), (mb.astype(np.uint8) * 255))

    print("Uploading seed image to fal …")
    import fal_client  # type: ignore

    image_url = fal_client.upload_file(str(seed_png))

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max(1, len(seed_mask_bool))) as ex:
        for oid in sorted(seed_mask_bool.keys()):
            fut = ex.submit(
                _fal_download_glb,
                args.fal_model,
                args.seed,
                image_url,
                work_dir / f"mask_{oid}.png",
                work_dir / f"object_{oid}.glb",
            )
            futures_map[fut] = oid

        fal_results = _wait_fal_progress_ui(
            seed_frame,
            seed_mask_bool,
            futures_map,
            "fal SAM3D progress",
        )

    for oid, (ok, msg) in fal_results.items():
        print(f"  ID{oid} fal: {'OK' if ok else 'FAIL'} — {msg}")

    meshes: dict[int, NativeAxisMesh] = {}
    for oid in sorted(seed_mask_bool.keys()):
        glb_path = work_dir / f"object_{oid}.glb"
        if not glb_path.is_file():
            print(f"Missing GLB for ID{oid}, skipping pose.")
            continue
        try:
            meshes[oid] = NativeAxisMesh(glb_path)
        except Exception as e:
            print(f"Failed to load GLB for ID{oid}: {e}")

    t_3d_s = time.perf_counter() - t_3d0
    print(
        f"Timing: obtain_3d_objects (mask/seed PNG write + fal + download + load GLBs) = "
        f"{t_3d_s:.3f} s"
    )

    pose_states: dict[int, dict] = {}
    print("Initial containment registration on seed masks …")
    t_reg0 = time.perf_counter()
    for oid, mb in seed_mask_bool.items():
        if oid not in meshes:
            continue
        pose_states[oid] = _register_rigid_containment(
            meshes[oid], mb, K_cam, dist_cam, None
        )
        st = pose_states[oid]
        st["rvec_raw"] = st.get("rvec")
        st["tvec_raw"] = st.get("tvec")
        st["last_relock_frame"] = 0
        st["last_solver"] = "init"
        st["low_conf_frames"] = 0
        m = _pose_alignment_metrics(meshes[oid], mb, K_cam, dist_cam, st)
        st["iou"] = m["iou"]
        st["dice"] = m["dice"]
        st["centroid_px"] = m["centroid_px"]
        st["containment"] = m["containment"]
        st["confidence"] = _pose_confidence_from_metrics(m)
    t_reg_s = time.perf_counter() - t_reg0
    print(
        f"Timing: initial_registration (containment scipy, seed frame, all objects) = "
        f"{t_reg_s:.3f} s"
    )

    writer = None
    if args.output:
        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (fw, fh),
        )

    com_trails: dict[int, list[tuple[int, int]]] = {}

    align_win = "3D mesh ↔ 2D mask (alignment)"
    last_align_combo: np.ndarray | None = None
    print(
        "Speed: live_track.py is faster with stock settings because pose + SAM3D bootstrap "
        "are commented out there (mask overlay + COM trails only), so no registration at all "
        "and no fal work in the hot path. Here use --full-relock-every 8+, "
        "--align-refresh-every 3, --no-align-debug, and/or a smaller "
        "--max-trail to reduce CPU load."
    )
    print("Live tracking + pose. Press q / ESC to quit.")
    fps_t0 = time.perf_counter()
    fps_frames = 0
    stage_t = {"fast": 0.0, "full": 0.0}
    stage_n = {"fast": 0, "full": 0}
    try:
        with torch.autocast(device_type=ac_device, dtype=ac_dtype, enabled=ac_enabled):
            for fi, obj_ids, masks in predictor.propagate_in_video(
                state, start_frame_idx=1, max_frame_num_to_track=None
            ):
                frame = provider.get_raw(fi)
                if frame is None:
                    frame = seed_frame

                if hasattr(obj_ids, "tolist"):
                    ids = [int(x) for x in obj_ids.tolist()]
                else:
                    ids = [int(x) for x in obj_ids]

                vis = overlay_masks(frame, ids, masks, alpha=args.alpha)
                masks_np = masks.detach().cpu().numpy()
                reg_every = max(1, int(args.reg_every))
                fast_every = max(1, int(args.fast_update_every))
                full_relock_every = max(reg_every, int(args.full_relock_every))
                align_stride = max(1, int(args.align_refresh_every))
                build_align = (not args.no_align_debug) and (int(fi) % align_stride == 0)
                align_panels: list[np.ndarray] = []

                for i in range(min(len(ids), masks_np.shape[0])):
                    oid = ids[i]
                    binm_raw = _mask_to_2d_bool(masks_np[i], fh, fw)
                    binm = _mask_outer_filled_bool(binm_raw)

                    if np.any(binm):
                        cnt = get_mask_contour(binm)
                        if cnt is not None:
                            cv2.drawContours(vis, [cnt], -1, (255, 255, 255), 2, cv2.LINE_AA)

                        ys, xs = np.where(binm)
                        cx, cy = int(xs.mean()), int(ys.mean())
                        trail = com_trails.setdefault(oid, [])
                        trail.append((cx, cy))
                        mt = max(8, int(args.max_trail))
                        if len(trail) > mt:
                            com_trails[oid] = trail[-mt:]

                        est = meshes.get(oid)
                        if est is not None:
                            st0 = pose_states.get(oid, {})
                            conf0 = float(st0.get("confidence", 0.0))
                            low_frames = int(st0.get("low_conf_frames", 0))
                            last_relock = int(st0.get("last_relock_frame", 0))
                            run_full = (
                                st0.get("rvec") is None
                                or (int(fi) - last_relock >= full_relock_every)
                                or (conf0 < float(args.confidence_low) and low_frames >= int(args.lost_frames_trigger))
                            )
                            run_fast = (not run_full) and (int(fi) % fast_every == 0)

                            st = st0.copy()
                            if run_full:
                                t0 = time.perf_counter()
                                st = _register_rigid_containment(
                                    est,
                                    binm,
                                    K_cam,
                                    dist_cam,
                                    st,
                                )
                                stage_t["full"] += time.perf_counter() - t0
                                stage_n["full"] += 1
                                st["rvec_raw"] = st.get("rvec")
                                st["tvec_raw"] = st.get("tvec")
                                st["last_relock_frame"] = int(fi)
                                st["last_solver"] = "full"
                            elif run_fast:
                                t0 = time.perf_counter()
                                st = _fast_pose_update_local(
                                    est,
                                    binm,
                                    K_cam,
                                    dist_cam,
                                    st,
                                    rot_step_deg=float(args.fast_rot_step_deg),
                                    trans_step_xy=float(args.fast_trans_step_xy),
                                    trans_step_z=float(args.fast_trans_step_z),
                                )
                                stage_t["fast"] += time.perf_counter() - t0
                                stage_n["fast"] += 1
                                st["last_solver"] = "fast"
                            else:
                                st["last_solver"] = "hold"

                            m = _pose_alignment_metrics(est, binm, K_cam, dist_cam, st)
                            st["iou"] = m["iou"]
                            st["dice"] = m["dice"]
                            st["centroid_px"] = m["centroid_px"]
                            st["containment"] = m["containment"]
                            conf = _pose_confidence_from_metrics(m)
                            st["confidence"] = conf
                            if conf < float(args.confidence_low):
                                st["low_conf_frames"] = int(st.get("low_conf_frames", 0)) + 1
                            elif conf >= float(args.confidence_recover):
                                st["low_conf_frames"] = 0

                            # Increase smoothing when confidence is low to suppress visible jitter.
                            t_alpha = float(args.smooth_pos_alpha)
                            r_alpha = float(args.smooth_rot_alpha)
                            if conf < float(args.confidence_low):
                                t_alpha *= 0.65
                                r_alpha *= 0.65
                            _smooth_pose_state(st, t_alpha, r_alpha)
                            pose_states[oid] = st

                            _draw_native_axes_fixed_pixel(
                                vis, pose_states[oid], K_cam, dist_cam, AXIS_LENGTH_PX
                            )
                            if build_align:
                                panel = _alignment_debug_panel(
                                    frame,
                                    est,
                                    pose_states[oid],
                                    binm,
                                    K_cam,
                                    dist_cam,
                                    oid,
                                    fi,
                                )
                                if panel is not None:
                                    align_panels.append(panel)
                            _draw_com_pose_readout(
                                vis, oid, cx, cy, pose_states.get(oid)
                            )
                        else:
                            _draw_com_pose_readout(vis, oid, cx, cy, None)

                for oid, trail in com_trails.items():
                    col = point_color(oid)
                    for j in range(1, len(trail)):
                        cv2.line(vis, trail[j - 1], trail[j], col, 1, lineType=cv2.LINE_AA)
                    if trail:
                        cv2.circle(vis, trail[-1], 5, col, -1)
                        cv2.circle(vis, trail[-1], 7, (255, 255, 255), 1)

                _draw_pose_hud(vis, pose_states)

                cv2.imshow("EdgeTAM + 6DoF pose", vis)
                if not args.no_align_debug:
                    if align_panels:
                        mosaic = _layout_alignment_mosaic(align_panels, per_row=2)
                        if mosaic is not None:
                            last_align_combo = mosaic
                    if last_align_combo is not None:
                        cv2.imshow(align_win, last_align_combo)
                if writer is not None:
                    writer.write(vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if stop_flag.is_set():
                    break

                fps_frames += 1
                now = time.perf_counter()
                dt_fps = now - fps_t0
                if dt_fps >= 1.0:
                    fps_now = fps_frames / dt_fps
                    full_ms = (1000.0 * stage_t["full"] / max(1, stage_n["full"]))
                    fast_ms = (1000.0 * stage_t["fast"] / max(1, stage_n["fast"]))
                    print(
                        f"FPS (live loop): {fps_now:.2f} | "
                        f"fast={stage_n['fast']} ({fast_ms:.2f}ms avg) | "
                        f"full={stage_n['full']} ({full_ms:.2f}ms avg)"
                    )
                    fps_t0 = now
                    fps_frames = 0
                    stage_t = {"fast": 0.0, "full": 0.0}
                    stage_n = {"fast": 0, "full": 0}
    finally:
        stop_flag.set()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live 6DoF pose: EdgeTAM + fal SAM3D + containment registration.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--alpha", type=float, default=0.85, help="Mask overlay alpha.")
    parser.add_argument("--half", action="store_true", default=True)
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--output", default="", help="Optional mp4 output path.")
    parser.add_argument("--fal-model", default="fal-ai/sam-3/3d-objects", help="fal endpoint id.")
    parser.add_argument("--seed", type=int, default=42, help="SAM3D seed.")
    parser.add_argument(
        "--glb-dir",
        default=str(Path(__file__).resolve().parent / "sam3d_live_objects"),
        help="Directory for seed PNG, masks, downloaded GLBs.",
    )
    parser.add_argument(
        "--no-align-debug",
        action="store_true",
        help="Disable the troubleshooting window that shows projected 3D mesh vs 2D mask.",
    )
    parser.add_argument(
        "--reg-every",
        type=int,
        default=1,
        help="Legacy minimum period for full relock (kept for backward compatibility).",
    )
    parser.add_argument(
        "--fast-update-every",
        type=int,
        default=1,
        help="Run fast local pose update every N frames.",
    )
    parser.add_argument(
        "--full-relock-every",
        type=int,
        default=8,
        help="Run full scipy containment relock every N frames.",
    )
    parser.add_argument(
        "--confidence-low",
        type=float,
        default=0.35,
        help="Confidence threshold below which relock pressure increases.",
    )
    parser.add_argument(
        "--confidence-recover",
        type=float,
        default=0.50,
        help="Confidence threshold that resets low-confidence streak.",
    )
    parser.add_argument(
        "--lost-frames-trigger",
        type=int,
        default=6,
        help="Consecutive low-confidence frames before forced relock.",
    )
    parser.add_argument(
        "--smooth-pos-alpha",
        type=float,
        default=0.40,
        help="EMA alpha for translation smoothing (lower=more smoothing).",
    )
    parser.add_argument(
        "--smooth-rot-alpha",
        type=float,
        default=0.35,
        help="EMA alpha for rotation smoothing (lower=more smoothing).",
    )
    parser.add_argument(
        "--fast-rot-step-deg",
        type=float,
        default=2.2,
        help="Fast local search step for rotation (degrees).",
    )
    parser.add_argument(
        "--fast-trans-step-xy",
        type=float,
        default=0.003,
        help="Fast local search step for translation X/Y (camera units).",
    )
    parser.add_argument(
        "--fast-trans-step-z",
        type=float,
        default=0.008,
        help="Fast local search step for translation Z (camera units).",
    )
    parser.add_argument(
        "--align-refresh-every",
        type=int,
        default=1,
        help="Rebuild alignment debug mosaic every N frames (1=every frame).",
    )
    parser.add_argument(
        "--max-trail",
        type=int,
        default=_MAX_COM_TRAIL,
        help="Max COM trail points per object (same idea as live_track.py).",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()