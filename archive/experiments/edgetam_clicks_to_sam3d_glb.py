#!/usr/bin/env python3
"""
Click objects on one image, segment each object with EdgeTAM, then run fal SAM3D
for each object mask and export GLB files preserving object IDs.

Usage:
  cd /Users/michelleespinosa/Desktop/SurgicalToolsPose
  source .venv/bin/activate
  export FAL_KEY="..."
  python tests/edgetam_clicks_to_sam3d_glb.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch

try:
    import trimesh  # type: ignore
except Exception:
    trimesh = None


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_runtime_device_info(selected_device: str, pnp_only: bool) -> None:
    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(
        f"Runtime devices: cuda_available={cuda_ok}, mps_available={mps_ok}, selected={selected_device}"
    )
    if pnp_only:
        print(
            "Note: --pnp-only uses OpenCV/trimesh CPU ops. GPU acceleration applies to EdgeTAM segmentation stage."
        )


def pick_points(image_bgr: np.ndarray) -> list[tuple[int, int, int]]:
    win = "Click objects (ENTER=run, Backspace=undo, c=clear, q=quit)"
    points: list[tuple[int, int, int]] = []

    def draw() -> np.ndarray:
        vis = image_bgr.copy()
        for obj_id, x, y in points:
            cv2.circle(vis, (x, y), 6, (0, 255, 255), -1)
            cv2.circle(vis, (x, y), 9, (255, 255, 255), 1)
            cv2.putText(
                vis,
                f"ID{obj_id}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        cv2.putText(
            vis,
            "Left click = new object ID | Enter = run",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return vis

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((len(points) + 1, int(x), int(y)))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    cancelled = False
    while True:
        cv2.imshow(win, draw())
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            if points:
                break
        elif key in (8, 127) and points:
            points.pop()
        elif key == ord("c"):
            points.clear()
        elif key in (ord("q"), 27):
            cancelled = True
            break
    cv2.destroyWindow(win)
    return [] if cancelled else points


def build_predictor(edge_tam_repo: Path, model_cfg: str, checkpoint: Path, device: str):
    repo = edge_tam_repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from sam2.build_sam import build_sam2  # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

    sam_model = build_sam2(model_cfg, str(checkpoint), device=device)
    return SAM2ImagePredictor(sam_model)


def estimate_intrinsics_from_image_size(
    width: int, height: int, hfov_deg: float = 79.0, vfov_deg: float = 62.0
) -> np.ndarray:
    hfov = np.radians(float(hfov_deg))
    vfov = np.radians(float(vfov_deg))
    fx = width / (2.0 * np.tan(hfov / 2.0))
    fy = height / (2.0 * np.tan(vfov / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def order_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    center = corners.mean(axis=0)
    ang = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    ordered = corners[np.argsort(ang)]
    idx0 = int(np.argmin(ordered[:, 1] * 10000.0 + ordered[:, 0]))
    return np.roll(ordered, -idx0, axis=0)


def sample_quad_perimeter(corners4: np.ndarray, n: int) -> np.ndarray:
    corners4 = np.asarray(corners4, dtype=np.float64)
    lens = np.linalg.norm(np.roll(corners4, -1, axis=0) - corners4, axis=1)
    L = float(lens.sum())
    out = np.zeros((n, 2), dtype=np.float64)
    if L < 1e-12:
        return np.repeat(corners4[:1], n, axis=0)
    cum = np.zeros(5, dtype=np.float64)
    for i in range(4):
        cum[i + 1] = cum[i] + lens[i]
    for i in range(n):
        t = ((i + 0.5) / n) * L
        k = int(np.searchsorted(cum, t, side="right") - 1)
        k = int(np.clip(k, 0, 3))
        u = (t - cum[k]) / (lens[k] + 1e-12)
        out[i] = corners4[k] * (1.0 - u) + corners4[(k + 1) % 4] * u
    return out


def sample_mask_contour(mask_u8: np.ndarray, n: int) -> np.ndarray | None:
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
        u = (t - cum[k]) / (d[k] + 1e-12)
        out[i] = poly[k] * (1.0 - u) + poly[(k + 1) % len(poly)] * u
    return out


def mesh_projection_iou(
    mask_u8: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
) -> float:
    fh, fw = mask_u8.shape[:2]
    pred = np.zeros((fh, fw), dtype=np.uint8)
    dist = np.zeros((4, 1), dtype=np.float64)
    proj, _ = cv2.projectPoints(verts.astype(np.float64), rvec, tvec, K, dist)
    pts2d = proj.reshape(-1, 2)
    for f in faces:
        poly = np.round(pts2d[f]).astype(np.int32)
        cv2.fillConvexPoly(pred, poly, 255, lineType=cv2.LINE_AA)
    a = pred > 0
    b = mask_u8 > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def build_model_quad_from_mesh(
    glb_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if trimesh is None:
        return None
    mesh = trimesh.load(str(glb_path), force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.shape[0] < 8:
        return None
    cen = verts.mean(axis=0)
    v = verts - cen
    cov = (v.T @ v) / max(len(v), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    aligned = v @ eigvecs[:, order]
    hP = (aligned[:, 0].max() - aligned[:, 0].min()) / 2.0
    hS = (aligned[:, 1].max() - aligned[:, 1].min()) / 2.0
    model_pts = np.array(
        [[-hP, -hS, 0.0], [hP, -hS, 0.0], [hP, hS, 0.0], [-hP, hS, 0.0]],
        dtype=np.float64,
    )
    axis_len = max(hP, hS) * 0.8 + 1e-6
    axis_pts = np.array(
        [[0.0, 0.0, 0.0], [axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, axis_len]],
        dtype=np.float64,
    )
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return model_pts, axis_pts, aligned.astype(np.float64), faces


def estimate_pose_from_mask_and_glb(
    mask_u8: np.ndarray,
    glb_path: Path,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    quad = build_model_quad_from_mesh(glb_path)
    if quad is None:
        return None
    model_pts, axis_pts, mesh_verts, mesh_faces = quad
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 20:
        return None
    pts2d = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    rect = cv2.minAreaRect(pts2d.reshape(-1, 1, 2))
    img_corners = order_corners_clockwise(cv2.boxPoints(rect).astype(np.float64))
    contour_dense = sample_mask_contour(mask_u8, n=32)
    if contour_dense is None:
        contour_dense = sample_quad_perimeter(img_corners, n=32)
    model_dense0 = sample_quad_perimeter(model_pts[:, :2], n=32)
    dist = np.zeros((4, 1), dtype=np.float64)
    best = None
    best_score = np.inf
    for order_mode in ("forward", "reverse"):
        contour_use = contour_dense if order_mode == "forward" else contour_dense[::-1].copy()
        for shift in range(4):
            model_dense_xy = np.roll(model_dense0, shift * (len(model_dense0) // 4), axis=0)
            model_dense = np.c_[model_dense_xy[:, 0], model_dense_xy[:, 1], np.zeros((len(model_dense_xy),))]
            # Dense correspondences first; fallback to corner PnP if needed.
            ok, rvec, tvec = cv2.solvePnP(
                model_dense, contour_use, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                obj = np.roll(model_pts, shift, axis=0)
                ok, rvec, tvec = cv2.solvePnP(obj, img_corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
                if not ok:
                    continue
            proj_dense, _ = cv2.projectPoints(model_dense, rvec, tvec, K, dist)
            reproj = float(np.mean(np.linalg.norm(proj_dense.reshape(-1, 2) - contour_use, axis=1)))
            iou = mesh_projection_iou(mask_u8, mesh_verts, mesh_faces, rvec, tvec, K)
            # Blend dense reprojection with silhouette overlap.
            score = reproj + 50.0 * (1.0 - iou)
            if score < best_score:
                best_score = score
                best = (rvec, tvec, axis_pts, mesh_verts, mesh_faces)
                if iou >= 0.75:
                    return best
    return best


def draw_pose_axes(
    image_bgr: np.ndarray,
    obj_id: int,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_pts: np.ndarray,
    K: np.ndarray,
) -> None:
    dist = np.zeros((4, 1), dtype=np.float64)
    proj, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, dist)
    p = proj.reshape(-1, 2)
    o = (int(p[0, 0]), int(p[0, 1]))
    x = (int(p[1, 0]), int(p[1, 1]))
    y = (int(p[2, 0]), int(p[2, 1]))
    z = (int(p[3, 0]), int(p[3, 1]))
    cv2.arrowedLine(image_bgr, o, x, (0, 0, 220), 2, tipLength=0.2, line_type=cv2.LINE_AA)
    cv2.arrowedLine(image_bgr, o, y, (0, 200, 0), 2, tipLength=0.2, line_type=cv2.LINE_AA)
    cv2.arrowedLine(image_bgr, o, z, (220, 80, 0), 2, tipLength=0.2, line_type=cv2.LINE_AA)
    cv2.putText(
        image_bgr,
        f"ID{obj_id}",
        (o[0] + 6, o[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def draw_projected_mesh(
    image_bgr: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
) -> None:
    dist = np.zeros((4, 1), dtype=np.float64)
    proj, _ = cv2.projectPoints(mesh_verts.astype(np.float64), rvec, tvec, K, dist)
    pts2d = proj.reshape(-1, 2)
    overlay = image_bgr.copy()
    # Draw a light face fill + wireframe to visualize mesh alignment.
    for f in mesh_faces:
        poly = np.round(pts2d[f]).astype(np.int32)
        cv2.fillConvexPoly(overlay, poly, (180, 130, 255), lineType=cv2.LINE_AA)
        cv2.polylines(image_bgr, [poly], True, (255, 0, 255), 1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.20, image_bgr, 0.80, 0.0, dst=image_bgr)


def segment_points(
    predictor,
    image_bgr: np.ndarray,
    points: list[tuple[int, int, int]],
) -> dict[int, np.ndarray]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks_by_id: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        predictor.set_image(image_rgb)
        for obj_id, x, y in points:
            coords = np.array([[x, y]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=True,
            )
            if masks.ndim != 3 or len(masks) == 0:
                continue
            # Prefer candidate masks that actually contain the clicked point.
            valid = []
            for i in range(len(masks)):
                m = masks[i] > 0
                if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and bool(m[y, x]):
                    valid.append(i)
            if valid:
                best = max(valid, key=lambda i: float(scores[i]))
            else:
                best = int(np.argmax(scores))
            masks_by_id[obj_id] = (masks[best] > 0).astype(np.uint8)
    return masks_by_id


def save_debug_overlay(
    image_bgr: np.ndarray, masks_by_id: dict[int, np.ndarray], out_path: Path
) -> None:
    vis = image_bgr.copy().astype(np.float32)
    for obj_id, mask in masks_by_id.items():
        hue = (obj_id * 47 + 20) % 180
        hsv = np.uint8([[[hue, 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0].astype(np.float32)
        vis[mask.astype(bool)] = vis[mask.astype(bool)] * 0.55 + bgr * 0.45
    cv2.imwrite(str(out_path), vis.astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Click points -> EdgeTAM masks -> fal SAM3D -> export object GLBs."
    )
    parser.add_argument("--image", default="EdgeTAMLive/test.jpg", help="Input image path.")
    parser.add_argument(
        "--edge-tam-repo", default="EdgeTAMLive/EdgeTAM", help="Path to EdgeTAM repo root."
    )
    parser.add_argument(
        "--model-cfg",
        default="configs/edgetam.yaml",
        help="EdgeTAM model config path relative to --edge-tam-repo.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/edgetam.pt",
        help="EdgeTAM checkpoint path relative to --edge-tam-repo.",
    )
    parser.add_argument(
        "--output-dir",
        default="EdgeTAMLive/sam3d_click_exports",
        help="Directory to write masks, metadata, and GLBs.",
    )
    parser.add_argument("--fal-model", default="fal-ai/sam-3/3d-objects", help="fal model endpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for SAM3D.")
    parser.add_argument(
        "--pnp-only",
        action="store_true",
        help="Skip EdgeTAM/fal and run only PnP from existing object_<id>_mask.png + object_<id>.glb in --output-dir.",
    )
    parser.add_argument("--hfov-deg", type=float, default=79.0, help="Horizontal FOV for estimated intrinsics.")
    parser.add_argument("--vfov-deg", type=float, default=62.0, help="Vertical FOV for estimated intrinsics.")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Inference device for EdgeTAM."
    )
    args = parser.parse_args()
    selected_device = choose_device(args.device)
    print_runtime_device_info(selected_device, args.pnp_only)

    image_path = Path(args.image).expanduser().resolve()
    edge_tam_repo = Path(args.edge_tam_repo).expanduser().resolve()
    model_cfg = args.model_cfg
    ckpt_path = (edge_tam_repo / args.checkpoint).resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {image_path}")

    h, w = image_bgr.shape[:2]
    K = estimate_intrinsics_from_image_size(w, h, args.hfov_deg, args.vfov_deg)
    intrinsics_path = out_dir / "estimated_intrinsics.json"
    intrinsics_path.write_text(
        json.dumps({"hfov_deg": args.hfov_deg, "vfov_deg": args.vfov_deg, "K": K.tolist()}, indent=2),
        encoding="utf-8",
    )
    print(
        f"Estimated intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}"
    )
    print(f"Saved estimated intrinsics: {intrinsics_path}")

    if args.pnp_only:
        pose_overlay = image_bgr.copy()
        mesh_overlay = image_bgr.copy()
        mapping: dict[int, dict] = {}
        mask_files = sorted(out_dir.glob("object_*_mask.png"))
        if not mask_files:
            raise SystemExit(f"No mask files found in {out_dir} (expected object_<id>_mask.png).")
        for mask_path in mask_files:
            parts = mask_path.stem.split("_")
            if len(parts) < 3:
                continue
            try:
                obj_id = int(parts[1])
            except ValueError:
                continue
            glb_path = out_dir / f"object_{obj_id}.glb"
            if not glb_path.is_file():
                print(f"[ID {obj_id}] Missing GLB: {glb_path}")
                mapping[obj_id] = {"mask_path": str(mask_path), "glb_path": None, "pose_pnp": None}
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[ID {obj_id}] Could not read mask: {mask_path}")
                continue
            pose = None
            pose_out = estimate_pose_from_mask_and_glb(mask, glb_path, K)
            if pose_out is not None:
                rvec, tvec, axis_pts, mesh_verts, mesh_faces = pose_out
                pose = {"rvec": rvec.reshape(3).tolist(), "tvec": tvec.reshape(3).tolist()}
                draw_pose_axes(pose_overlay, obj_id, rvec, tvec, axis_pts, K)
                draw_projected_mesh(mesh_overlay, mesh_verts, mesh_faces, rvec, tvec, K)
                draw_pose_axes(mesh_overlay, obj_id, rvec, tvec, axis_pts, K)
                print(f"[ID {obj_id}] PnP ok")
            else:
                print(f"[ID {obj_id}] PnP failed")
            mapping[obj_id] = {"mask_path": str(mask_path), "glb_path": str(glb_path), "pose_pnp": pose}

        mapping_path = out_dir / "id_mapping_pnp_only.json"
        mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        pose_overlay_path = out_dir / "pnp_pose_overlay.png"
        cv2.imwrite(str(pose_overlay_path), pose_overlay)
        mesh_overlay_path = out_dir / "pnp_mesh_overlay.png"
        cv2.imwrite(str(mesh_overlay_path), mesh_overlay)
        print(f"Saved PnP mapping: {mapping_path}")
        print(f"Saved pose overlay: {pose_overlay_path}")
        print(f"Saved mesh overlay: {mesh_overlay_path}")
        print(f"Done. Export directory: {out_dir}")
        return

    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    if not os.environ.get("FAL_KEY"):
        raise SystemExit("Set FAL_KEY first: export FAL_KEY='your_fal_key'")

    points = pick_points(image_bgr)
    if not points:
        raise SystemExit("No points selected.")

    device = selected_device
    if device == "cpu":
        raise SystemExit("GPU is required for full run. Use --device mps or --device cuda.")
    print(f"Using device: {device}")
    predictor = build_predictor(edge_tam_repo, model_cfg, ckpt_path, device=device)
    masks_by_id = segment_points(predictor, image_bgr, points)
    if not masks_by_id:
        raise SystemExit("EdgeTAM did not return any masks.")

    overlay_path = out_dir / "edgetam_overlay.png"
    save_debug_overlay(image_bgr, masks_by_id, overlay_path)
    print(f"Saved overlay: {overlay_path}")

    import fal_client

    image_upload_url = fal_client.upload_file(str(image_path))
    print("Uploaded source image to fal storage.")

    mapping: dict[int, dict] = {}
    pose_overlay = image_bgr.copy()
    mesh_overlay = image_bgr.copy()
    for obj_id in sorted(masks_by_id):
        mask = (masks_by_id[obj_id] * 255).astype(np.uint8)
        mask_path = out_dir / f"object_{obj_id}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        mask_url = fal_client.upload_file(str(mask_path))
        print(f"[ID {obj_id}] Uploaded mask, running {args.fal_model} ...")
        result = fal_client.subscribe(
            args.fal_model, arguments={"image_url": image_upload_url, "mask_urls": [mask_url], "seed": args.seed}, with_logs=True
        )

        model_glb = result.get("model_glb", {}) if isinstance(result, dict) else {}
        glb_url = model_glb.get("url")
        glb_path = None
        if isinstance(glb_url, str) and glb_url:
            glb_path = out_dir / f"object_{obj_id}.glb"
            urlretrieve(glb_url, glb_path)
            print(f"[ID {obj_id}] Saved GLB: {glb_path}")
        else:
            print(f"[ID {obj_id}] No model_glb returned.")

        pose = None
        if glb_path is not None:
            pose_out = estimate_pose_from_mask_and_glb(mask, glb_path, K)
            if pose_out is not None:
                rvec, tvec, axis_pts, mesh_verts, mesh_faces = pose_out
                pose = {"rvec": rvec.reshape(3).tolist(), "tvec": tvec.reshape(3).tolist()}
                draw_pose_axes(pose_overlay, obj_id, rvec, tvec, axis_pts, K)
                draw_projected_mesh(mesh_overlay, mesh_verts, mesh_faces, rvec, tvec, K)
                draw_pose_axes(mesh_overlay, obj_id, rvec, tvec, axis_pts, K)
                print(f"[ID {obj_id}] PnP ok")
            else:
                print(f"[ID {obj_id}] PnP failed")

        mapping[obj_id] = {
            "point_xy": [int(p[1]), int(p[2])] if (p := next(x for x in points if x[0] == obj_id)) else None,
            "mask_path": str(mask_path),
            "glb_path": str(glb_path) if glb_path else None,
            "pose_pnp": pose,
            "fal_result": result,
        }

    mapping_path = out_dir / "id_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    pose_overlay_path = out_dir / "pnp_pose_overlay.png"
    cv2.imwrite(str(pose_overlay_path), pose_overlay)
    mesh_overlay_path = out_dir / "pnp_mesh_overlay.png"
    cv2.imwrite(str(mesh_overlay_path), mesh_overlay)
    print(f"Saved ID mapping: {mapping_path}")
    print(f"Saved pose overlay: {pose_overlay_path}")
    print(f"Saved mesh overlay: {mesh_overlay_path}")
    print(f"Done. Export directory: {out_dir}")


if __name__ == "__main__":
    main()

