#!/usr/bin/env python3
"""
Live 6DoF pose for arbitrary objects: EdgeTAM seeds → fal SAM3D GLBs →
dense contour PnP registration → axes overlay + COM trails.

Pose pipeline (ported from live_track.py "clean fast version", Apr 28 2026):
  1. EdgeTAM propagates instance masks from user seed points.
  2. fal SAM3D generates a GLB mesh per object.
  3. MeshPoseEstimator PCA-aligns the mesh, scales it to physical metres,
     builds a convex-hull model polygon, then each frame:
       - samples mask contour and model boundary uniformly (arc-length)
       - searches phase/reversal to find best 2D↔3D correspondence
       - solves PnP (ITERATIVE with warm-start, fallback EPNP)
       - smooths rvec/tvec with per-component Kalman filters
  4. Axes are drawn from the PCA-aligned object frame.

Position and orientation correctness:
  - Mesh vertices are rescaled to real metres via --object-scale-m so tvec Z
    is in metres, not arbitrary GLB units.
  - PCA aligns the longest axis to X so axes are stable across frames.
  - Eight axis-sign combinations are searched once (bootstrap) to resolve
    the mirror ambiguity inherent in monocular PnP.
  - Kalman filters suppress jitter without the phase/drift issues of EMA.

Requires:
  - EdgeTAM checkpoint: EdgeTAM/checkpoints/edgetam.pt
  - trimesh, fal-client
  - export FAL_KEY="..."

Usage:
  python live_pose_any.py
  python live_pose_any.py --camera 1 --object-scale-m 0.12
"""

from __future__ import annotations

import argparse
import os
import shutil
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

from live_track_copy import (
    CHECKPOINT,
    LiveFrameProvider,
    TARGET_SIZE,
    _autocast_config,
    _draw_pose_hud,
    _estimate_intrinsics_from_cap,
    _load_predictor,
    _mask_to_2d_bool,
    _reg_sign_from_state,
    choose_device,
    detect_orbbec_camera,
    overlay_masks,
    pick_points_live,
    point_color,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_COM_TRAIL = 60

# Kalman filter variances (rvec + tvec, one scalar filter per component).
KALMAN_PROCESS_VAR  = 2e-4
KALMAN_MEAS_VAR     = 4e-3

# PnP contour matching.
PNP_PER_EDGE        = 8     # points per model polygon edge
PNP_ROT_SMOOTH_W    = 0.03  # weight: rotation continuity vs previous frame
PNP_TRANS_SMOOTH_W  = 8.0   # weight: translation continuity vs previous frame
PNP_SHIFT_PENALTY   = 0.75  # extra cost for flipping contour direction

# SAM3D GLBs are unitless; this is the assumed longest-axis length in metres.
# Override with --object-scale-m for your specific tool.
_DEFAULT_OBJECT_SCALE_M = 0.15   # 15 cm

# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

class KalmanScalar:
    """1-D random-walk Kalman filter."""

    def __init__(self, process_var: float, meas_var: float):
        self.q = max(float(process_var), 1e-12)
        self.r = max(float(meas_var), 1e-12)
        self.x: float | None = None
        self.p: float = 1.0

    def filter(self, z: float) -> float:
        z = float(z)
        if self.x is None:
            self.x = z; self.p = 1.0; return z
        self.p += self.q
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
    """Six independent Kalman filters on rvec+tvec; filter state lives in ``state``."""
    filters = state.get("kalman_filters")
    if filters is None or len(filters) != 6:
        filters = [KalmanScalar(process_var, meas_var) for _ in range(6)]
        state["kalman_filters"] = filters
    vec = np.concatenate([rv.reshape(3), tv.reshape(3)]).astype(np.float64)
    out = np.array([filters[i].filter(float(vec[i])) for i in range(6)])
    return out[:3].reshape(3, 1), out[3:].reshape(3, 1)

# ---------------------------------------------------------------------------
# Contour / polygon sampling helpers
# ---------------------------------------------------------------------------

def _sample_poly_perimeter(poly: np.ndarray, n: int) -> np.ndarray:
    """Uniformly sample ``n`` points along a closed polygon (arc-length)."""
    D = int(poly.shape[1])
    if poly.shape[0] < 2:
        return np.repeat(poly[:1].astype(np.float64), n, axis=0)
    lens = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    L = float(lens.sum())
    if L < 1e-12:
        return np.repeat(poly[:1].astype(np.float64), n, axis=0)
    cum = np.zeros(len(poly) + 1, dtype=np.float64)
    for i in range(len(poly)):
        cum[i + 1] = cum[i] + lens[i]
    out = np.zeros((n, D), dtype=np.float64)
    for i in range(n):
        t = ((i + 0.5) / n) * L
        for k in range(len(poly)):
            if cum[k + 1] >= t - 1e-15:
                u = (t - cum[k]) / (lens[k] + 1e-12)
                out[i] = poly[k] * (1.0 - u) + poly[(k + 1) % len(poly)] * u
                break
    return out


def _sample_mask_contour(mask_u8: np.ndarray, n: int) -> np.ndarray | None:
    """Largest external contour of a binary mask, uniformly resampled to ``n`` points."""
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
    """Geodesic angle in degrees between two Rodrigues rotation vectors."""
    Ra, _ = cv2.Rodrigues(rvec_a.astype(np.float64))
    Rb, _ = cv2.Rodrigues(rvec_b.astype(np.float64))
    R = Ra @ Rb.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))))

# ---------------------------------------------------------------------------
# Mesh metric scaling
# ---------------------------------------------------------------------------

def _normalize_mesh_to_metric(verts: np.ndarray, scale_m: float) -> np.ndarray:
    """Rescale so the longest bounding-box axis equals ``scale_m`` metres.

    SAM3D outputs unitless meshes (~0–1). Without this, tvec has no physical
    meaning and PnP depth estimates are wrong.
    """
    longest = float((verts.max(0) - verts.min(0)).max())
    if longest < 1e-9:
        return verts
    s = scale_m / longest
    print(f"  [mesh scale] longest={longest:.4f} → ×{s:.5f}  ({scale_m*100:.1f} cm)")
    return verts * s

# ---------------------------------------------------------------------------
# MeshPoseEstimator
# ---------------------------------------------------------------------------

class MeshPoseEstimator:
    """
    Dense contour-PnP pose estimator for a single GLB mesh.

    Construction (once per object):
      - PCA-aligns vertices (longest variance axis → X).
      - Rescales to physical metres so PnP tvec is meaningful.
      - Builds a convex-hull model polygon in the XY plane.

    Per-frame ``estimate_pose``:
      - Searches 8 axis-sign combos on first call to resolve mirror ambiguity.
      - Matches mask contour to model polygon with phase/reversal search.
      - Calls cv2.solvePnP (ITERATIVE + warm-start; fallback EPNP).
      - Smooths result with 6 per-component Kalman filters.
    """

    def __init__(self, mesh, obj_id: int = 0, scale_m: float = _DEFAULT_OBJECT_SCALE_M):
        self.obj_id = obj_id

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces,    dtype=np.int32)

        # PCA alignment: longest variance axis → X
        cen    = verts.mean(0)
        v      = verts - cen
        cov    = (v.T @ v) / max(1, len(v))
        eigvals, evecs = np.linalg.eigh(cov)
        order   = np.argsort(eigvals)[::-1]   # descending eigenvalue
        aligned = v @ evecs[:, order]

        # Scale to physical metres
        aligned = _normalize_mesh_to_metric(aligned, scale_m)

        self.mesh_vertices = aligned.astype(np.float64)
        self.mesh_faces    = faces

        hP = (aligned[:, 0].max() - aligned[:, 0].min()) / 2.0
        hS = (aligned[:, 1].max() - aligned[:, 1].min()) / 2.0
        hT = (aligned[:, 2].max() - aligned[:, 2].min()) / 2.0
        self.hP, self.hS, self.hT = hP, hS, hT

        # Convex hull in PCA XY plane → dense PnP model polygon
        plane_xy = aligned[:, :2].astype(np.float32)
        hull = cv2.convexHull(plane_xy.reshape(-1, 1, 2)).reshape(-1, 2)
        if hull.shape[0] >= 3:
            peri   = float(cv2.arcLength(hull.reshape(-1, 1, 2), True))
            approx = cv2.approxPolyDP(hull.reshape(-1, 1, 2), 0.01 * peri, True).reshape(-1, 2)
            poly_xy = approx if approx.shape[0] >= 3 else hull
            self.model_pts = np.column_stack(
                [poly_xy[:, 0], poly_xy[:, 1], np.zeros(poly_xy.shape[0])]
            ).astype(np.float64)
        else:
            self.model_pts = np.array(
                [[-hP, -hS, 0.0], [hP, -hS, 0.0], [hP, hS, 0.0], [-hP, hS, 0.0]],
                dtype=np.float64,
            )

        # Axis display points (in object metres, used by _draw_pose_axes)
        ax = hP * 1.0
        ay = hS * 1.5
        az = max(hT * 5.0, hP * 0.35)
        self.axis_pts = np.array(
            [[0.0, 0.0, 0.0], [ax, 0.0, 0.0], [0.0, ay, 0.0], [0.0, 0.0, az]],
            dtype=np.float64,
        )

        # Semantic endpoints (larger cross-section → handle end)
        x    = aligned[:, 0]
        yz_r = np.linalg.norm(aligned[:, 1:3], axis=1)
        xmin, xmax = float(x.min()), float(x.max())
        band   = max(1e-6, 0.20 * (xmax - xmin))
        r_low  = float(np.mean(yz_r[x <= xmin + band])) if np.any(x <= xmin + band) else 0.0
        r_high = float(np.mean(yz_r[x >= xmax - band])) if np.any(x >= xmax - band) else 0.0
        if r_low >= r_high:
            self.handle_pt = np.array([xmin, 0.0, 0.0], np.float64)
            self.tip_pt    = np.array([xmax, 0.0, 0.0], np.float64)
        else:
            self.handle_pt = np.array([xmax, 0.0, 0.0], np.float64)
            self.tip_pt    = np.array([xmin, 0.0, 0.0], np.float64)

        self._pnp_n = max(
            4 * PNP_PER_EDGE,
            int(self.model_pts.shape[0]) * max(2, PNP_PER_EDGE // 2),
        )

    # ------------------------------------------------------------------
    def _orientation_penalty(
        self, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray,
        dist: np.ndarray, sign: np.ndarray | None = None,
    ) -> float:
        """Upright prior for obj_id == 1 (scissors): tip up, handle down."""
        if int(self.obj_id) != 1:
            return 0.0
        s   = np.ones(3, np.float64) if sign is None else np.asarray(sign, np.float64).reshape(3)
        pts = np.vstack([self.handle_pt, self.tip_pt]).astype(np.float64) * s.reshape(1, 3)
        proj, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
        p  = proj.reshape(-1, 2)
        v  = p[1] - p[0]
        n  = float(np.linalg.norm(v))
        if n < 1e-9:
            return 1e6
        ux, uy = float(v[0] / n), float(v[1] / n)
        hard   = 100.0 if float(p[1][1]) >= float(p[0][1]) else 0.0
        return hard + 4.0 * abs(ux) + 8.0 * max(0.0, uy)

    # ------------------------------------------------------------------
    def _pnp_best_contour(
        self,
        model_corners: np.ndarray,
        mask_bool: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        prev_rvec: np.ndarray | None = None,
        prev_tvec: np.ndarray | None = None,
        prev_phase: int | None = None,
        prev_reverse: bool | None = None,
    ) -> tuple:
        """Phase/reversal search over model↔mask contour; returns (rv, tv, err, phase, reverse)."""
        n      = int(self._pnp_n)
        img_n  = _sample_mask_contour(mask_bool.astype(np.uint8) * 255, n)
        if img_n is None:
            return None, None, np.inf, None, False
        model_n0 = _sample_poly_perimeter(model_corners.astype(np.float64), n)

        step = max(1, n // 8)
        # Always search all phases so the object can be tracked even after fast
        # motion or large rotation. The 0.01*dp phase-continuity penalty in the
        # score naturally prefers phases near the previous one when available,
        # without hard-limiting the search window (which caused frozen axes).
        candidates = [
            (int(ph), rev)
            for rev in (False, True)
            for ph in range(0, n, step)
        ]

        best_rv, best_tv, best_err = None, None, np.inf
        best_phase, best_reverse, best_score = None, False, np.inf

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
                ok2, rv2, tv2 = cv2.solvePnP(
                    model_n, img_n, K, dist, flags=cv2.SOLVEPNP_EPNP)
                if not ok2:
                    continue
                ok3, rv, tv = cv2.solvePnP(
                    model_n, img_n, K, dist, rv2, tv2,
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok3:
                    rv, tv = rv2, tv2

            proj, _ = cv2.projectPoints(model_n, rv, tv, K, dist)
            err   = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_n, axis=1)))
            score = err

            if prev_rvec is not None and prev_tvec is not None:
                d_rot = _rotation_delta_deg(rv, prev_rvec)
                z_ref = max(abs(float(prev_tvec.reshape(3)[2])), 1e-6)
                d_t   = float(np.linalg.norm(tv.reshape(3) - prev_tvec.reshape(3)) / z_ref)
                score += PNP_ROT_SMOOTH_W * d_rot + PNP_TRANS_SMOOTH_W * d_t
            if prev_phase is not None:
                dp = min(abs(int(phase) - int(prev_phase)),
                         n - abs(int(phase) - int(prev_phase)))
                score += 0.01 * float(dp)
            if prev_reverse is not None and bool(rev) != bool(prev_reverse):
                score += PNP_SHIFT_PENALTY

            if score < best_score:
                best_score = score
                best_err, best_rv, best_tv = err, rv, tv
                best_phase, best_reverse   = int(phase), bool(rev)

        return best_rv, best_tv, best_err, best_phase, best_reverse

    # ------------------------------------------------------------------
    def estimate_pose(
        self,
        mask_bool: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        state: dict,
        kalman_process_var: float = KALMAN_PROCESS_VAR,
        kalman_meas_var:    float = KALMAN_MEAS_VAR,
    ) -> dict:
        """Run one frame of contour-PnP pose estimation; updates and returns ``state``."""
        if state is None:
            state = {}
        if not np.any(mask_bool):
            return state

        # Erode to remove thin mask edges that confuse contour sampling.
        clean = cv2.erode(mask_bool.astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
        if int(np.count_nonzero(clean)) < 20:
            return state

        prev_rv = state.get("rvec_raw") or state.get("rvec")
        prev_tv = state.get("tvec_raw") or state.get("tvec")

        if "reg_sign" not in state:
            # Bootstrap: exhaustive search over all 8 axis-sign combinations
            # to resolve the mirror ambiguity in monocular PnP.
            best: tuple = (np.inf, None, None, None, None, None)
            for bits in range(8):
                s = np.array(
                    [-1.0 if (bits >> i) & 1 else 1.0 for i in range(3)], np.float64)
                rv_i, tv_i, err_i, ph_i, rev_i = self._pnp_best_contour(
                    self.model_pts * s, clean, K, dist,
                    prev_rvec=prev_rv, prev_tvec=prev_tv,
                    prev_phase=state.get("contour_phase"),
                    prev_reverse=state.get("contour_reverse"),
                )
                if rv_i is None:
                    continue
                score = float(err_i) + self._orientation_penalty(rv_i, tv_i, K, dist, s)
                if score < float(best[0]):
                    best = (score, rv_i, tv_i, ph_i, rev_i, s)
            if best[1] is None:
                return state
            _, rv, tv, phase, reverse, s_use = best
            state["reg_sign"] = np.asarray(s_use, np.float64)
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
# Axis visualization
# ---------------------------------------------------------------------------

def _draw_pose_axes(
    vis: np.ndarray,
    state: dict,
    K: np.ndarray,
    dist: np.ndarray,
    axis_pts: np.ndarray,
    obj_id: int,
    max_arrow_px: float = 60.0,
) -> None:
    """Project object-local X/Y/Z axes and draw capped arrows on ``vis``."""
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    if rvec is None or tvec is None:
        return
    s     = _reg_sign_from_state(state)
    proj, _ = cv2.projectPoints(
        (axis_pts * s).astype(np.float64), rvec, tvec, K, dist)
    pts2d = proj.reshape(-1, 2)
    fh, fw = vis.shape[:2]

    def clip_pt(p: np.ndarray) -> tuple[int, int]:
        return (int(np.clip(p[0], 0, fw - 1)), int(np.clip(p[1], 0, fh - 1)))

    def cap(o: np.ndarray, t: np.ndarray) -> np.ndarray:
        v = t - o; n = float(np.linalg.norm(v))
        return o if n < 1e-9 else (o + v * (max_arrow_px / n) if n > max_arrow_px else t)

    o      = pts2d[0].astype(np.float64)
    origin = clip_pt(o)
    colors = [(0, 0, 220), (0, 200, 0), (220, 80, 0)]   # X red, Y green, Z orange
    labels = ("X", "Y", "Z")
    fnt    = cv2.FONT_HERSHEY_SIMPLEX
    for k in range(3):
        tip = clip_pt(cap(o, pts2d[k + 1].astype(np.float64)))
        cv2.arrowedLine(vis, origin, tip, colors[k], 2, tipLength=0.2, line_type=cv2.LINE_AA)
        cv2.putText(vis, labels[k], (tip[0] + 4, tip[1] + 4), fnt, 0.50,
                    colors[k], 1, cv2.LINE_AA)
    cv2.circle(vis, origin, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(vis, origin, 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)

# ---------------------------------------------------------------------------
# COM + pose readout near mask centroid
# ---------------------------------------------------------------------------

def _draw_com_pose_readout(
    vis: np.ndarray,
    obj_id: int,
    cx: int,
    cy: int,
    pose_state: dict | None,
) -> None:
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    fh, fw = vis.shape[:2]
    x0 = int(np.clip(cx + 8, 0, fw - 2))
    y0 = int(np.clip(cy - 8, 28, fh - 8))
    if not pose_state or pose_state.get("rvec") is None:
        line = f"ID{obj_id} COM({cx},{cy})"
        cv2.putText(vis, line, (x0, y0), fnt, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, line, (x0, y0), fnt, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        return
    rvec = pose_state["rvec"]
    tvec = pose_state.get("tvec")
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    sy   = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        rx = float(np.degrees(np.arctan2(float(R[2, 1]), float(R[2, 2]))))
        ry = float(np.degrees(np.arctan2(-float(R[2, 0]), sy)))
        rz = float(np.degrees(np.arctan2(float(R[1, 0]), float(R[0, 0]))))
    else:
        rx = float(np.degrees(np.arctan2(-float(R[1, 2]), float(R[1, 1]))))
        ry = float(np.degrees(np.arctan2(-float(R[2, 0]), sy)))
        rz = 0.0
    if tvec is not None:
        tv   = np.asarray(tvec, np.float64).reshape(3)
        line = (f"ID{obj_id} COM({cx},{cy})  R({rx:.0f},{ry:.0f},{rz:.0f})"
                f"  T({tv[0]:.2f},{tv[1]:.2f},{tv[2]:.2f}m)")
    else:
        line = f"ID{obj_id} COM({cx},{cy})  R({rx:.0f},{ry:.0f},{rz:.0f})"
    cv2.putText(vis, line, (x0, y0), fnt, 0.40, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, line, (x0, y0), fnt, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# fal SAM3D download
# ---------------------------------------------------------------------------

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
        result   = fal_client.subscribe(
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
    status: dict[int, str] = {oid: "queued…" for oid in futures_map.values()}
    results: dict[int, tuple[bool, str]] = {}

    def draw() -> np.ndarray:
        ids = sorted(seed_mask_bool.keys())
        vis = seed_frame.copy().astype(np.float32) if ids else seed_frame.copy()
        for oid in ids:
            binm = seed_mask_bool[oid]
            c    = np.array(point_color(int(oid)), dtype=np.float32)
            vis[binm] = vis[binm] * 0.35 + c * 0.65
        vis = vis.astype(np.uint8) if ids else vis
        y   = 26
        cv2.putText(vis, "Waiting for fal SAM3D (parallel)…",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22
        for oid in sorted(seed_mask_bool.keys()):
            cv2.putText(vis, f"ID{oid}: {status.get(oid, '')}",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 255), 2, cv2.LINE_AA)
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
                status[oid]  = "done" if ok else f"ERR: {msg[:40]}"
                pending.discard(fut)
        cv2.imshow(win_name, draw())
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
    for fut, oid in futures_map.items():
        if oid in results:
            continue
        results[oid] = fut.result() if fut.done() else (False, "interrupted")
    cv2.destroyWindow(win_name)
    return results


def _confirm_seed_masks_ui(
    seed_frame: np.ndarray,
    seed_mask_bool: dict[int, np.ndarray],
    win_name: str = "Confirm seed masks",
) -> bool:
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
            cv2.destroyWindow(win_name)
            return True
        if k in (ord("r"), ord("R"), ord("n"), ord("N"), ord("q"), 27):
            cv2.destroyWindow(win_name)
            return False

# ---------------------------------------------------------------------------
# GLB loading + MeshPoseEstimator construction (parallel-safe)
# ---------------------------------------------------------------------------

def _load_mesh_and_register(
    oid: int,
    glb_path: Path,
    mb: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    scale_m: float,
) -> tuple[int, MeshPoseEstimator | None, dict]:
    """Load GLB, repair, build MeshPoseEstimator, run seed-frame pose estimation."""
    if trimesh is None:
        print("trimesh required — pip install trimesh")
        return oid, None, {}

    # Repaired-GLB cache: skip fill_holes if a pre-repaired copy is newer than the source.
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

    est = MeshPoseEstimator(mesh, obj_id=oid, scale_m=scale_m)
    st: dict = {}
    if np.any(mb):
        st = est.estimate_pose(mb, K, dist, st)
    print(f"[ID{oid}] estimator ready  scale={scale_m*100:.1f}cm  "
          f"model_pts={est.model_pts.shape[0]}  pnp_n={est._pnp_n}"
          + ("  (repaired cache)" if use_cache else ""))
    return oid, est, st

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    print(f"Device: {device}")

    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT}"); return
    if trimesh is None:
        print("Install trimesh: pip install trimesh"); return
    if not os.environ.get("FAL_KEY"):
        print("Set FAL_KEY environment variable."); return

    print("Loading EdgeTAM …")
    predictor  = _load_predictor(device)
    image_size = predictor.image_size

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}"); return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    rotate_180 = detect_orbbec_camera(args.camera)

    K_cam    = _estimate_intrinsics_from_cap(cap, TARGET_SIZE[0], TARGET_SIZE[1])
    dist_cam = np.zeros((5, 1), dtype=np.float64)

    provider = LiveFrameProvider(cap, image_size, rotate_180)
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

    # Pre-upload seed frame to fal while EdgeTAM computes + user confirms masks.
    work_dir = Path(args.glb_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    seed_png = work_dir / "seed_frame.png"
    cv2.imwrite(str(seed_png), seed_frame)
    _seed_upload_ex: ThreadPoolExecutor | None = None
    seed_url_future = None
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

    seed_masks_raw: dict[int, np.ndarray] = {}
    ids_order: list[int] = []

    print("EdgeTAM: resolving seed masks (frame 0)…")
    t0 = time.perf_counter()
    with torch.autocast(device_type=ac_device, dtype=ac_dtype, enabled=ac_enabled):
        gen0 = predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=0)
        _, obj_ids0, masks0 = next(gen0)
        ids_order = [int(x) for x in (obj_ids0.tolist() if hasattr(obj_ids0, "tolist") else obj_ids0)]
        masks_np  = masks0.detach().cpu().numpy()
        for i in range(min(len(ids_order), masks_np.shape[0])):
            seed_masks_raw[ids_order[i]] = _mask_to_2d_bool(masks_np[i], fh, fw)
    print(f"Timing: EdgeTAM seed masks = {time.perf_counter()-t0:.3f}s")

    seed_mask_bool = dict(seed_masks_raw)

    print("Please confirm seed masks before sending to SAM3D …")
    if not _confirm_seed_masks_ui(seed_frame, seed_mask_bool):
        print("Seed mask confirmation declined; aborting.")
        stop_flag.set(); cap.release(); cv2.destroyAllWindows(); return

    # Write mask PNGs (seed frame was already written above)
    t_3d0 = time.perf_counter()
    for oid, mb in seed_masks_raw.items():
        cv2.imwrite(str(work_dir / f"mask_{oid}.png"), (mb.astype(np.uint8) * 255))

    import fal_client  # type: ignore
    if seed_url_future is not None:
        print("fal: collecting background seed upload …")
        image_url = seed_url_future.result()
        if _seed_upload_ex is not None:
            _seed_upload_ex.shutdown(wait=False)
    else:
        print("Uploading seed image to fal …")
        image_url = fal_client.upload_file(str(seed_png))

    futures_map: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, len(seed_mask_bool))) as ex:
        for oid in sorted(seed_mask_bool.keys()):
            fut = ex.submit(
                _fal_download_glb, args.fal_model, args.seed, image_url,
                work_dir / f"mask_{oid}.png", work_dir / f"object_{oid}.glb",
            )
            futures_map[fut] = oid
        fal_results = _wait_fal_progress_ui(
            seed_frame, seed_mask_bool, futures_map, "fal SAM3D progress")

    for oid, (ok, msg) in fal_results.items():
        print(f"  ID{oid} fal: {'OK' if ok else 'FAIL'} — {msg}")
    print(f"Timing: fal SAM3D = {time.perf_counter()-t_3d0:.3f}s")

    # Load GLBs + run seed pose in parallel across objects.
    estimators:  dict[int, MeshPoseEstimator] = {}
    pose_states: dict[int, dict]              = {}
    print("Loading GLBs + initial pose (parallel) …")
    t_reg0 = time.perf_counter()
    n_glb  = sum(1 for oid in seed_mask_bool if (work_dir / f"object_{oid}.glb").is_file())
    glb_futs: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, n_glb)) as reg_ex:
        for oid in sorted(seed_mask_bool.keys()):
            glb_path = work_dir / f"object_{oid}.glb"
            if not glb_path.is_file():
                print(f"Missing GLB for ID{oid}, skipping."); continue
            mb  = seed_mask_bool.get(oid, np.zeros((fh, fw), dtype=bool))
            fut = reg_ex.submit(
                _load_mesh_and_register, oid, glb_path, mb, K_cam, dist_cam,
                args.object_scale_m,
            )
            glb_futs[fut] = oid
        for fut in as_completed(glb_futs):
            oid, est, st = fut.result()
            if est is not None:
                estimators[oid]  = est
                pose_states[oid] = st
    print(f"Timing: load + seed pose = {time.perf_counter()-t_reg0:.3f}s")

    writer = None
    if args.output:
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (fw, fh))

    com_trails: dict[int, list[tuple[int, int]]] = {}

    print("Live tracking + pose. Press q / ESC to quit.")
    fps_t0, fps_frames = time.perf_counter(), 0
    try:
        with torch.autocast(device_type=ac_device, dtype=ac_dtype, enabled=ac_enabled):
            for fi, obj_ids, masks in predictor.propagate_in_video(
                state, start_frame_idx=1, max_frame_num_to_track=None
            ):
                frame = provider.get_raw(fi)
                if frame is None:
                    frame = seed_frame

                ids      = [int(x) for x in (obj_ids.tolist() if hasattr(obj_ids, "tolist") else obj_ids)]
                vis      = overlay_masks(frame, ids, masks, alpha=args.alpha)
                masks_np = masks.detach().cpu().numpy()

                for i in range(min(len(ids), masks_np.shape[0])):
                    oid  = ids[i]
                    binm = _mask_to_2d_bool(masks_np[i], fh, fw)
                    if not np.any(binm):
                        continue

                    # COM trail
                    ys, xs = np.where(binm)
                    cx, cy = int(xs.mean()), int(ys.mean())
                    trail  = com_trails.setdefault(oid, [])
                    trail.append((cx, cy))
                    mt = max(8, int(args.max_trail))
                    if len(trail) > mt:
                        com_trails[oid] = trail[-mt:]

                    est = estimators.get(oid)
                    if est is not None:
                        pose_states[oid] = est.estimate_pose(
                            binm, K_cam, dist_cam,
                            pose_states.get(oid, {}),
                            kalman_process_var=args.kalman_process_var,
                            kalman_meas_var=args.kalman_meas_var,
                        )
                        _draw_pose_axes(
                            vis, pose_states[oid], K_cam, dist_cam, est.axis_pts, oid)
                        _draw_com_pose_readout(vis, oid, cx, cy, pose_states.get(oid))
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
                if writer is not None:
                    writer.write(vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("Exiting: q / ESC pressed."); break
                if stop_flag.is_set():
                    print("Exiting: camera stalled."); break

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
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live 6DoF pose: EdgeTAM + fal SAM3D + dense contour PnP.")
    parser.add_argument("--camera",    type=int,   default=0)
    parser.add_argument("--device",    default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--alpha",     type=float, default=0.85, help="Mask overlay alpha.")
    parser.add_argument("--half",      action="store_true", default=True)
    parser.add_argument("--no-half",   dest="half", action="store_false")
    parser.add_argument("--output",    default="", help="Optional mp4 output path.")
    parser.add_argument("--fal-model", default="fal-ai/sam-3/3d-objects")
    parser.add_argument("--seed",      type=int, default=42, help="SAM3D seed.")
    parser.add_argument("--glb-dir",
        default=str(Path(__file__).resolve().parent / "sam3d_live_objects"),
        help="Directory for seed PNG, masks, and downloaded GLBs.")
    parser.add_argument("--object-scale-m", type=float, default=_DEFAULT_OBJECT_SCALE_M,
        help="Physical length of the longest object axis in metres (default 0.15 = 15 cm). "
             "Scales SAM3D mesh so PnP tvec is in real-world metres. "
             "e.g. 0.1175 for scissors, 0.16 for scalpel.")
    parser.add_argument("--kalman-process-var", type=float, default=KALMAN_PROCESS_VAR,
        help="Kalman process variance for rvec/tvec smoothing.")
    parser.add_argument("--kalman-meas-var",    type=float, default=KALMAN_MEAS_VAR,
        help="Kalman measurement variance for rvec/tvec smoothing.")
    parser.add_argument("--max-trail", type=int, default=_MAX_COM_TRAIL,
        help="Max COM trail points per object.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
