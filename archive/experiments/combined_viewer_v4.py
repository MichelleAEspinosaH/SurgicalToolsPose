# ******************************************************************************
#  combined_viewer_v4.py
#
#  Extends v3 with a segmentation layer between YOLO and SIFT.
#
#  Segmentation pipeline (NEW in v4)
#  ----------------------------------
#  After YOLO provides a bounding box, the code builds a tight, tool-shaped
#  mask in three stages:
#
#    Stage 1 — YOLO segmentation mask
#      If the loaded model is a segmentation model (e.g. yolo26n-seg.pt),
#      it already outputs a per-pixel mask.  That mask is used directly.
#      This is the most accurate path and requires no extra compute.
#
#    Stage 2 — GrabCut (fallback for detection-only models)
#      If the model is detection-only, GrabCut is initialised with the
#      bounding box and run for a few iterations.  It separates foreground
#      (tool) from background using colour statistics.
#
#    Stage 3 — Canny edge + contour refinement
#      Applied on top of Stage 1 or 2.  Canny edges inside the bbox are
#      computed, the largest closed contour is found, and its filled polygon
#      replaces the coarser mask.  This sharpens the boundary to follow the
#      exact tool outline rather than a rectangular or colour blob.
#
#  Result: SIFT only finds keypoints on the tool surface (not on the hand,
#  arm, or background even when those overlap the bbox).  Optical flow points
#  that drift outside the refined contour mask are dropped immediately.
#
#  A semi-transparent magenta overlay of the segmentation mask is drawn on
#  both the "Reference Features" and "Tracking" windows so you can verify
#  the segmentation at a glance.
#
#  All v3 features are preserved:
#    v — 20-second 3-D recording pass with scatter-plot viewer
#    r — single-frame reference (YOLO auto-detect)
#    t — manual ROI
#    last-known COM ghost indicator and COM-region reinit
#    q / ESC — quit
# ******************************************************************************

import os
import time
import threading
from collections import deque
from multiprocessing import Process

import cv2
import numpy as np
from ultralytics import YOLO
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

# --- Constants ----------------------------------------------------------------
ESC_KEY           = 27
PRINT_INTERVAL    = 1
MIN_DEPTH         = 20
MAX_DEPTH         = 10000
TRAIL_LENGTH      = 50
REINIT_RATIO      = 0.2
MAX_FEATURES      = 30
COM_SMOOTH        = 0.2
POSE_SMOOTH       = 0.15
RECORD_SECONDS    = 20
RECORD_MAX_KF     = 40
COM_SEARCH_RADIUS = 120

# Segmentation tuning
GRABCUT_ITERS     = 5      # GrabCut iteration count (more = slower but better)
CANNY_LOW         = 40     # Canny lower threshold
CANNY_HIGH        = 120    # Canny upper threshold
SEG_OVERLAY_ALPHA = 0.35   # transparency of segmentation overlay (0=invisible, 1=opaque)

YOLO_MODEL_PATH   = "runs/detect/surgical_tools/weights/best.pt"
EXCLUDED_CLASSES  = {"human", "person", "hand", "arm"}

LK_PARAMS = dict(
    winSize  = (21, 21),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
# ------------------------------------------------------------------------------


# ==============================================================================
# Depth pipeline helpers  (unchanged from v3)
# ==============================================================================

class TemporalFilter:
    def __init__(self, alpha):
        self.alpha          = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(
                frame, self.alpha, self.previous_frame, 1 - self.alpha, 0
            )
        self.previous_frame = result
        return result


class DepthReader:
    """
    Background-thread Orbbec depth reader.
    get_latest()     → colourised BGR display image
    get_latest_raw() → float32 depth array in sensor units (mm)
    """
    def __init__(self, pipeline):
        self.pipeline        = pipeline
        self.latest_image    = None
        self.latest_raw      = None
        self.latest_distance = None
        self.lock            = threading.Lock()
        self.running         = True
        self.temporal_filter = TemporalFilter(alpha=0.5)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        last_print = time.time()
        while self.running:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None or depth_frame.get_format() != OBFormat.Y16:
                continue

            w, h  = depth_frame.get_width(), depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            data  = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
            data  = data.astype(np.float32) * scale
            data  = np.where((data > MIN_DEPTH) & (data < MAX_DEPTH), data, 0).astype(np.uint16)
            data  = self.temporal_filter.process(data)

            dist = float(data[h // 2, w // 2])
            now  = time.time()
            if now - last_print >= PRINT_INTERVAL:
                print("Center distance:", dist)
                last_print = now

            img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

            with self.lock:
                self.latest_image    = img
                self.latest_raw      = data.astype(np.float32)
                self.latest_distance = dist

    def get_latest(self):
        with self.lock:
            return self.latest_image

    def get_latest_raw(self):
        with self.lock:
            return self.latest_raw.copy() if self.latest_raw is not None else None

    def stop(self):
        self.running = False
        self._thread.join()


# ==============================================================================
# Segmentation helpers  (NEW in v4)
# ==============================================================================

def _yolo_seg_mask(results, best_idx, h, w):
    """
    Extracts the per-pixel mask from a YOLO segmentation result for detection
    index best_idx.  Returns a uint8 mask (255 = foreground) at full (h, w)
    resolution, or None if the model produced no mask data.
    """
    try:
        masks = results[0].masks
        if masks is None:
            return None
        # masks.data is (N, mh, mw) float32 in [0, 1]
        m = masks.data[best_idx].cpu().numpy()
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return (m > 0.5).astype(np.uint8) * 255
    except Exception:
        return None


def _grabcut_mask(color_image, bbox):
    """
    Runs GrabCut inside bbox to separate tool from background.
    Returns a uint8 mask (255 = probable foreground).
    """
    h, w = color_image.shape[:2]
    x1, y1, x2, y2 = bbox
    # GrabCut needs a rectangle with positive width/height
    rect = (
        max(0, x1 + 2),
        max(0, y1 + 2),
        max(1, x2 - x1 - 4),
        max(1, y2 - y1 - 4),
    )
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    gc_mask   = np.zeros((h, w), np.uint8)

    try:
        cv2.grabCut(color_image, gc_mask, rect,
                    bgd_model, fgd_model, GRABCUT_ITERS,
                    cv2.GC_INIT_WITH_RECT)
        fg = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        return fg
    except Exception:
        # Fall back to plain bbox mask
        m = np.zeros((h, w), dtype=np.uint8)
        m[y1:y2, x1:x2] = 255
        return m


def _canny_contour_refine(color_image, coarse_mask, bbox):
    """
    Runs Canny edge detection inside the bounding box, then finds the largest
    contour that overlaps with coarse_mask and fills it to produce a tighter
    tool-shaped mask.

    Falls back to coarse_mask if no suitable contour is found.
    """
    x1, y1, x2, y2 = bbox
    h, w = color_image.shape[:2]

    # Work only inside the bbox to keep it fast
    roi_color = color_image[y1:y2, x1:x2]
    roi_mask  = coarse_mask[y1:y2, x1:x2]

    # Blur slightly to suppress texture noise before Canny
    blurred = cv2.GaussianBlur(roi_color, (5, 5), 0)
    gray_roi = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges    = cv2.Canny(gray_roi, CANNY_LOW, CANNY_HIGH)

    # Dilate edges slightly so thin tool edges close into contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges  = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return coarse_mask

    # Keep contours that have meaningful overlap with the coarse foreground mask
    best_cnt   = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:          # ignore tiny fragments
            continue
        tmp = np.zeros_like(roi_mask)
        cv2.drawContours(tmp, [cnt], -1, 255, -1)
        overlap = int(cv2.bitwise_and(tmp, roi_mask).sum())
        if overlap > best_score:
            best_score = overlap
            best_cnt   = cnt

    if best_cnt is None or best_score < 500:
        return coarse_mask

    # Fill the best contour back into full-frame mask coordinates
    refined = np.zeros((h, w), dtype=np.uint8)
    shifted = best_cnt + np.array([x1, y1])
    cv2.drawContours(refined, [shifted], -1, 255, -1)

    # Morphological close to fill small holes inside the contour
    refined = cv2.morphologyEx(
        refined, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    return refined


def get_object_mask(color_image, yolo_model):
    """
    Full segmentation pipeline:
      1. Run YOLO to get bbox + class.
      2. Try YOLO segmentation mask.
      3. Fall back to GrabCut if no seg mask.
      4. Refine with Canny contour.

    Returns (mask, bbox, seg_contour, cls_name, conf)
      mask        — tight uint8 tool mask (255 = object)
      bbox        — (x1, y1, x2, y2)
      seg_contour — largest contour as (N,1,2) int32 for drawing, or None
      cls_name    — detected class name
      conf        — detection confidence

    Returns (None, None, None, None, None) if nothing valid detected.
    """
    h, w    = color_image.shape[:2]
    results = yolo_model(color_image, verbose=False)

    if not results or len(results[0].boxes) == 0:
        return None, None, None, None, None

    boxes = results[0].boxes
    valid = [
        i for i in range(len(boxes))
        if yolo_model.names.get(int(boxes.cls[i]), "").lower()
           not in EXCLUDED_CLASSES
    ]
    if not valid:
        return None, None, None, None, None

    best            = max(valid, key=lambda i: float(boxes.conf[i]))
    x1, y1, x2, y2 = map(int, boxes.xyxy[best])
    cls_id          = int(boxes.cls[best])
    conf            = float(boxes.conf[best])
    cls_name        = yolo_model.names.get(cls_id, str(cls_id))
    bbox            = (x1, y1, x2, y2)

    # --- Stage 1: YOLO segmentation mask -------------------------------------
    seg_mask = _yolo_seg_mask(results, best, h, w)
    if seg_mask is not None:
        # Restrict to bbox region just in case mask bleeds outside
        bbox_mask         = np.zeros((h, w), dtype=np.uint8)
        bbox_mask[y1:y2, x1:x2] = 255
        coarse_mask       = cv2.bitwise_and(seg_mask, bbox_mask)
        source            = "seg"
    else:
        # --- Stage 2: GrabCut ------------------------------------------------
        coarse_mask = _grabcut_mask(color_image, bbox)
        source      = "grabcut"

    # --- Stage 3: Canny contour refinement -----------------------------------
    mask = _canny_contour_refine(color_image, coarse_mask, bbox)

    # Extract the contour from the final mask for drawing
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_contour = max(cnts, key=cv2.contourArea) if cnts else None

    # Safety: if the refined mask is nearly empty, fall back to coarse
    if mask.sum() < 1000:
        mask = coarse_mask
        seg_contour = None

    print(f"  Segmentation: {source}  class={cls_name}  conf={conf:.2f}  "
          f"mask_px={int(mask.sum() // 255)}")

    return mask, bbox, seg_contour, cls_name, conf


def draw_seg_overlay(image, mask, contour, alpha=SEG_OVERLAY_ALPHA):
    """
    Returns a copy of image with:
      • a semi-transparent magenta fill over the segmented region
      • the contour outline drawn in bright magenta
    """
    out = image.copy()

    # Fill overlay
    colour_layer       = out.copy()
    colour_layer[mask > 0] = (200, 0, 200)   # magenta fill
    out = cv2.addWeighted(colour_layer, alpha, out, 1 - alpha, 0)

    # Contour outline
    if contour is not None:
        cv2.drawContours(out, [contour], -1, (255, 0, 255), 2)

    return out


# ==============================================================================
# SIFT helpers  (updated to use segmentation mask)
# ==============================================================================

def _bbox_mask(shape, bbox):
    """Returns a uint8 mask with 255 inside bbox, 0 outside."""
    m = np.zeros(shape[:2], dtype=np.uint8)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        m[y1:y2, x1:x2] = 255
    return m


def compute_reference_features(ref_gray, ref_color, ref_mask, bbox,
                                seg_contour, max_features):
    """
    Detects SIFT keypoints within the segmentation mask, clipped to bbox.
    If ref_mask is None but bbox is given, only the bbox region is searched.
    Draws the bbox (blue), segmentation overlay (magenta), and keypoints (green).
    Returns (kp, des, annotated_image).
    """
    # Guarantee SIFT never goes outside the bounding box
    if bbox is not None:
        bm = _bbox_mask(ref_gray.shape, bbox)
        detect_mask = cv2.bitwise_and(ref_mask, bm) if ref_mask is not None else bm
    else:
        detect_mask = ref_mask

    sift    = cv2.SIFT_create(nfeatures=max_features)
    kp, des = sift.detectAndCompute(ref_gray, detect_mask)

    viz = ref_color.copy()
    # Draw segmentation overlay first (behind bbox and keypoints)
    if ref_mask is not None:
        viz = draw_seg_overlay(viz, ref_mask, seg_contour)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(viz, "reference bbox", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    viz = cv2.drawKeypoints(
        viz, kp, None, color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return kp, des, viz


def get_sift_points_in_frame(ref_kp, ref_des, query_gray, query_mask, max_features,
                              bbox=None):
    """
    SIFT matching restricted to the object bounding box.
    If bbox is given, query_mask is further clipped to bbox so keypoints are
    only detected inside the box — nothing in the background is matched.
    """
    if bbox is not None:
        bm = _bbox_mask(query_gray.shape, bbox)
        query_mask = cv2.bitwise_and(query_mask, bm) if query_mask is not None else bm

    sift      = cv2.SIFT_create()
    kp2, des2 = sift.detectAndCompute(query_gray, query_mask)
    if ref_des is None or des2 is None or len(ref_des) < 2 or len(des2) < 2:
        return None, None

    flann   = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(ref_des, des2, k=2)
    good    = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if not good:
        return None, None

    good       = sorted(good, key=lambda m: m.distance)[:max_features]
    live_pts   = np.array([[kp2[m.trainIdx].pt] for m in good], dtype=np.float32)
    ref_2d     = np.array([ref_kp[m.queryIdx].pt for m in good], dtype=np.float32)
    ref_2d    -= ref_2d.mean(axis=0)
    obj_pts_3d = np.column_stack([ref_2d, np.zeros(len(good), dtype=np.float32)])
    return live_pts, obj_pts_3d


def point_color(idx):
    hue = (idx * 37) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ==============================================================================
# 3-D reconstruction helpers  (unchanged from v3)
# ==============================================================================

def _depth_at_rgb_pixel(u, v, raw_depth, rgb_w, rgb_h):
    dh, dw = raw_depth.shape[:2]
    u_d    = int(round(u * dw / rgb_w))
    v_d    = int(round(v * dh / rgb_h))
    return float(raw_depth[np.clip(v_d, 0, dh - 1), np.clip(u_d, 0, dw - 1)])


def _unproject(u, v, z, cam_matrix):
    if z <= 0:
        return None
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float32)


def record_and_reconstruct_3d(cap, depth_reader, yolo_model, cam_matrix, max_features):
    """20-second multi-view 3-D reconstruction — unchanged from v3."""
    print(f"\nRecording 3-D reference for {RECORD_SECONDS} s.")
    print("Move the tool slowly so the camera sees all sides.")

    recorded = []
    t0       = time.time()
    while time.time() - t0 < RECORD_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue
        raw = depth_reader.get_latest_raw()

        remaining = RECORD_SECONDS - (time.time() - t0)
        disp      = frame.copy()
        cv2.putText(disp,
                    f"Recording  {remaining:.1f} s  — move tool slowly",
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 210), 2)
        cv2.circle(disp, (disp.shape[1] - 34, 32), 14, (0, 0, 210), -1)
        cv2.imshow("Recording", disp)
        cv2.waitKey(1)
        if raw is not None:
            recorded.append((frame.copy(), raw))

    cv2.destroyWindow("Recording")
    if not recorded:
        print("  No frames captured.")
        return (None,) * 8

    print(f"  {len(recorded)} frames — extracting 3-D keypoints…")

    sift  = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    step  = max(1, len(recorded) // RECORD_MAX_KF)
    kf_data = []

    for i in range(0, len(recorded), step):
        rgb, raw_d   = recorded[i]
        rgb_h, rgb_w = rgb.shape[:2]

        # Use the segmentation mask clipped to bbox for keyframe extraction
        mask, bbox, _, _, _ = get_object_mask(rgb, yolo_model)
        if bbox is None:
            continue

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # Clip mask to bbox so SIFT only fires inside the detected object
        detect_mask = cv2.bitwise_and(mask, _bbox_mask(gray.shape, bbox)) \
                      if mask is not None else _bbox_mask(gray.shape, bbox)
        kp, des = sift.detectAndCompute(gray, detect_mask)
        if not kp or des is None:
            continue

        pts3d_list, colors_list, kp_valid, des_valid = [], [], [], []
        for j, k in enumerate(kp):
            u, v = k.pt
            z    = _depth_at_rgb_pixel(u, v, raw_d, rgb_w, rgb_h)
            pt3  = _unproject(u, v, z, cam_matrix)
            if pt3 is None:
                continue
            pts3d_list.append(pt3)
            ui, vi = np.clip(int(u), 0, rgb_w - 1), np.clip(int(v), 0, rgb_h - 1)
            colors_list.append(rgb[vi, ui])
            kp_valid.append(k)
            des_valid.append(des[j])

        if len(kp_valid) < 4:
            continue

        kf_data.append(dict(
            frame  = rgb,  gray = gray,
            kp     = kp_valid,
            des    = np.array(des_valid, dtype=np.float32),
            pts3d  = np.array(pts3d_list, dtype=np.float32),
            colors = np.array(colors_list, dtype=np.uint8),
            bbox   = bbox,  mask = mask,
        ))

    if not kf_data:
        print("  No keyframes with valid depth — check YOLO and depth feed.")
        return (None,) * 8

    print(f"  {len(kf_data)} keyframes. Registering into common frame…")

    ref_kf    = kf_data[0]
    all_pts3d = list(ref_kf["pts3d"])
    all_colors = list(ref_kf["colors"])

    for kf in kf_data[1:]:
        try:
            matches = flann.knnMatch(ref_kf["des"], kf["des"], k=2)
        except Exception:
            continue
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        if len(good) < 4:
            continue
        src = np.array([kf["pts3d"][m.trainIdx]     for m in good], dtype=np.float32)
        dst = np.array([ref_kf["pts3d"][m.queryIdx] for m in good], dtype=np.float32)
        try:
            ok, T34, _ = cv2.estimateAffine3D(src, dst, confidence=0.99)
        except Exception:
            continue
        if not ok or T34 is None:
            continue
        T44    = np.vstack([T34, [0, 0, 0, 1]])
        h_pts  = np.column_stack([kf["pts3d"], np.ones(len(kf["pts3d"]))])
        warped = (T44 @ h_pts.T).T[:, :3]
        all_pts3d.extend(warped.tolist())
        all_colors.extend(kf["colors"].tolist())

    all_pts3d  = np.array(all_pts3d,  dtype=np.float32)
    all_colors = np.array(all_colors, dtype=np.uint8)
    point_cloud = np.column_stack([all_pts3d, all_colors])

    responses  = [k.response for k in ref_kf["kp"]]
    idx_sorted = np.argsort(responses)[::-1][:max_features]
    ref_kp     = [ref_kf["kp"][i] for i in idx_sorted]
    ref_des    = ref_kf["des"][idx_sorted]
    ref_obj    = ref_kf["pts3d"][idx_sorted].copy()
    ref_obj   -= ref_obj.mean(axis=0)

    ref_bbox   = ref_kf["bbox"]
    ref_mask   = ref_kf["mask"]
    ref_frame  = ref_kf["frame"]

    # Annotated reference viz with segmentation overlay
    ref_viz = ref_frame.copy()
    ref_viz = draw_seg_overlay(ref_viz, ref_mask, None)
    x1, y1, x2, y2 = ref_bbox
    cv2.rectangle(ref_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(ref_viz, "3-D reference bbox", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    ref_viz = cv2.drawKeypoints(
        ref_viz, ref_kp, None, color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    print(f"  3-D model: {len(all_pts3d)} pts  |  tracking seed: {len(ref_kp)} kp")
    return ref_frame, ref_kp, ref_des, ref_obj, point_cloud, ref_bbox, ref_mask, ref_viz


# ==============================================================================
# 3-D viewer subprocess  (unchanged from v3)
# ==============================================================================

def _plot_3d_worker(pts, colors):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure("3-D Tool Model", figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    r   = colors[:, 2] / 255.0
    g   = colors[:, 1] / 255.0
    b   = colors[:, 0] / 255.0
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=np.column_stack([r, g, b]), s=8, alpha=0.9, depthshade=True)
    ax.set_xlabel("X (mm)");  ax.set_ylabel("Y (mm)");  ax.set_zlabel("Z (mm)")
    ax.set_title("3-D Tool Model — drag to rotate")
    ranges   = np.array([pts[:, i].max() - pts[:, i].min() for i in range(3)])
    max_r    = ranges.max() / 2.0
    mid      = np.array([(pts[:, i].max() + pts[:, i].min()) / 2.0 for i in range(3)])
    ax.set_xlim(mid[0] - max_r, mid[0] + max_r)
    ax.set_ylim(mid[1] - max_r, mid[1] + max_r)
    ax.set_zlim(mid[2] - max_r, mid[2] + max_r)
    plt.tight_layout();  plt.show()


def show_3d_model(point_cloud):
    pts    = point_cloud[:, :3]
    colors = point_cloud[:, 3:].astype(np.uint8)
    Process(target=_plot_3d_worker, args=(pts, colors), daemon=True).start()


# ==============================================================================
# Tracker  (updated: uses live seg mask to cull drifting points)
# ==============================================================================

class COMTracker:
    """
    Lucas-Kanade optical-flow tracker.
    New in v4: update() accepts live_seg_mask.  Any tracked point that lands
    on a background pixel (mask == 0) is dropped immediately, tightening
    containment beyond the rectangular bbox used in v2/v3.
    """

    def __init__(self):
        self.prev_gray      = None
        self.points         = None
        self.obj_pts_3d     = None
        self.original_count = 0
        self.trails         = {}
        self.init_positions = {}
        self.com_trail      = deque(maxlen=TRAIL_LENGTH)
        self.smooth_com     = None
        self.last_known_com = None
        self.smooth_rvec    = None
        self.smooth_tvec    = None
        self.axis_length    = 50.0
        self.ref_bbox       = None

    def initialise(self, gray, live_pts, obj_pts_3d, ref_bbox=None,
                   preserve_com=False):
        saved_com  = self.smooth_com
        saved_last = self.last_known_com

        self.prev_gray      = gray.copy()
        self.points         = live_pts.copy()
        self.obj_pts_3d     = obj_pts_3d.copy()
        self.original_count = len(live_pts)
        self.trails         = {i: deque(maxlen=TRAIL_LENGTH) for i in range(len(live_pts))}
        self.init_positions = {}
        for i, p in enumerate(live_pts):
            x, y = float(p[0][0]), float(p[0][1])
            self.trails[i].append((x, y))
            self.init_positions[i] = (x, y)
        self.com_trail   = deque(maxlen=TRAIL_LENGTH)
        self.smooth_rvec = None
        self.smooth_tvec = None

        if preserve_com:
            self.smooth_com     = saved_com
            self.last_known_com = saved_last
            if saved_com is not None:
                self.com_trail.append(saved_com)
        else:
            self.smooth_com     = None
            self.last_known_com = None

        if ref_bbox is not None:
            self.ref_bbox = ref_bbox
        spread           = float(np.std(obj_pts_3d[:, :2]))
        self.axis_length = max(30.0, spread * 1.5)

    def update(self, gray, cam_matrix, dist_coeffs,
               live_bbox=None, live_seg_mask=None):
        """
        live_seg_mask — if provided, points outside the tool segmentation
        mask are dropped in addition to the bbox cull.
        """
        if self.points is None or len(self.points) == 0 or self.prev_gray is None:
            return

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None, **LK_PARAMS
        )
        surviving = [i for i, st in enumerate(status) if st[0] == 1]

        # Bbox cull (coarse)
        if live_bbox is not None:
            bx1, by1, bx2, by2 = live_bbox
            surviving = [
                i for i in surviving
                if bx1 <= new_pts[i][0][0] <= bx2
                and by1 <= new_pts[i][0][1] <= by2
            ]

        # Segmentation mask cull (fine) — drop points on background pixels
        if live_seg_mask is not None:
            mh, mw = live_seg_mask.shape[:2]
            def _in_mask(pt):
                px = np.clip(int(pt[0][0]), 0, mw - 1)
                py = np.clip(int(pt[0][1]), 0, mh - 1)
                return live_seg_mask[py, px] > 0

            surviving = [i for i in surviving if _in_mask(new_pts[i])]

        good_live, good_obj = [], []
        for i in surviving:
            x, y = float(new_pts[i][0][0]), float(new_pts[i][0][1])
            good_live.append(new_pts[i])
            good_obj.append(self.obj_pts_3d[i])
            self.trails[i].append((x, y))

        self.points    = (
            np.array(good_live, dtype=np.float32)
            if good_live else np.empty((0, 1, 2), dtype=np.float32)
        )
        self.prev_gray = gray.copy()

        tips = [self.trails[i][-1] for i in surviving if self.trails[i]]
        if tips:
            raw_x = float(np.mean([p[0] for p in tips]))
            raw_y = float(np.mean([p[1] for p in tips]))
            self.smooth_com = (
                (raw_x, raw_y) if self.smooth_com is None else (
                    COM_SMOOTH * raw_x + (1 - COM_SMOOTH) * self.smooth_com[0],
                    COM_SMOOTH * raw_y + (1 - COM_SMOOTH) * self.smooth_com[1],
                )
            )
            self.com_trail.append(self.smooth_com)
            self.last_known_com = self.smooth_com

        if len(surviving) >= 4 and cam_matrix is not None:
            obj_pts = np.array(good_obj, dtype=np.float32)
            img_pts = np.array(
                [[self.trails[i][-1][0], self.trails[i][-1][1]] for i in surviving],
                dtype=np.float32,
            )
            use_guess = self.smooth_rvec is not None
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, cam_matrix, dist_coeffs,
                    rvec=self.smooth_rvec.copy() if use_guess else None,
                    tvec=self.smooth_tvec.copy() if use_guess else None,
                    useExtrinsicGuess=use_guess,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    if self.smooth_rvec is None:
                        self.smooth_rvec, self.smooth_tvec = rvec.copy(), tvec.copy()
                    else:
                        self.smooth_rvec = POSE_SMOOTH * rvec + (1 - POSE_SMOOTH) * self.smooth_rvec
                        self.smooth_tvec = POSE_SMOOTH * tvec + (1 - POSE_SMOOTH) * self.smooth_tvec
            except Exception:
                pass

    def add_points(self, gray, new_pts, new_obj3d):
        """Merge newly SIFT-matched points into the existing tracked set."""
        if self.points is None or len(self.points) == 0:
            self.initialise(gray, new_pts, new_obj3d,
                            ref_bbox=self.ref_bbox, preserve_com=True)
            return
        start_idx = (max(self.trails.keys()) + 1) if self.trails else 0
        for j, (pt, obj) in enumerate(zip(new_pts, new_obj3d)):
            idx = start_idx + j
            self.trails[idx] = deque(maxlen=TRAIL_LENGTH)
            x, y = float(pt[0][0]), float(pt[0][1])
            self.trails[idx].append((x, y))
            self.init_positions[idx] = (x, y)
        self.points     = np.concatenate([self.points,     new_pts],    axis=0)
        self.obj_pts_3d = np.concatenate([self.obj_pts_3d, new_obj3d],  axis=0)
        if len(self.points) > self.original_count:
            self.original_count = len(self.points)
        self.prev_gray = gray.copy()

    def draw(self, image, cam_matrix, dist_coeffs,
             live_bbox=None, live_seg_mask=None, live_contour=None):
        out = image.copy()

        # Segmentation overlay on tracking frame (semi-transparent magenta)
        if live_seg_mask is not None:
            out = draw_seg_overlay(out, live_seg_mask, live_contour)

        # Live YOLO bbox (green rectangle)
        if live_bbox is not None:
            x1, y1, x2, y2 = live_bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, "live bbox", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Per-point trails
        for idx, trail in self.trails.items():
            pts = list(trail)
            if not pts:
                continue
            color = point_color(idx)
            n     = len(pts)
            for i in range(1, n):
                brightness = i / n
                seg_c = tuple(int(c * brightness) for c in color)
                p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p2 = (int(pts[i][0]),     int(pts[i][1]))
                cv2.line(out, p1, p2, seg_c, max(1, int(brightness * 3)))
            tip = (int(pts[-1][0]), int(pts[-1][1]))
            cv2.circle(out, tip, 5, color, -1)
            cv2.circle(out, tip, 5, (255, 255, 255), 1)

        # COM trail
        com_pts = list(self.com_trail)
        n = len(com_pts)
        for i in range(1, n):
            brightness = i / n
            c  = (0, int(255 * brightness), int(255 * brightness))
            p1 = (int(com_pts[i - 1][0]), int(com_pts[i - 1][1]))
            p2 = (int(com_pts[i][0]),     int(com_pts[i][1]))
            cv2.line(out, p1, p2, c, 2)

        # COM crosshair + pose axes
        if com_pts:
            cx, cy = int(com_pts[-1][0]), int(com_pts[-1][1])
            arm = 14
            cv2.line(out,   (cx - arm, cy), (cx + arm, cy), (0, 255, 255), 2)
            cv2.line(out,   (cx, cy - arm), (cx, cy + arm), (0, 255, 255), 2)
            cv2.circle(out, (cx, cy), 10, (0, 255, 255), 2)
            cv2.putText(out, f"COM ({cx}, {cy})", (cx + 14, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if self.smooth_rvec is not None and cam_matrix is not None:
                L       = self.axis_length
                axes_3d = np.array([[0,0,0],[L,0,0],[0,L,0],[0,0,-L]], dtype=np.float32)
                try:
                    proj, _ = cv2.projectPoints(
                        axes_3d, self.smooth_rvec, self.smooth_tvec,
                        cam_matrix, dist_coeffs,
                    )
                    proj   = proj.reshape(-1, 2)
                    shift  = np.array([cx, cy], dtype=np.float64) - proj[0]
                    origin = (cx, cy)
                    x_tip  = tuple((proj[1] + shift).astype(int))
                    y_tip  = tuple((proj[2] + shift).astype(int))
                    z_tip  = tuple((proj[3] + shift).astype(int))
                    cv2.arrowedLine(out, origin, x_tip, (0,   0, 255), 2, tipLength=0.25)
                    cv2.arrowedLine(out, origin, y_tip, (0, 255,   0), 2, tipLength=0.25)
                    cv2.arrowedLine(out, origin, z_tip, (255,  0,   0), 2, tipLength=0.25)
                    for tip, label, col in [
                        (x_tip, "X", (0,0,255)), (y_tip, "Y", (0,255,0)),
                        (z_tip, "Z", (255,0,0))
                    ]:
                        cv2.putText(out, label, (tip[0]+4, tip[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                except Exception:
                    pass

        # Warped reference bbox via homography
        if self.ref_bbox is not None and len(self.init_positions) >= 4:
            common = [i for i in self.init_positions if i in self.trails and self.trails[i]]
            if len(common) >= 4:
                src = np.array([self.init_positions[i] for i in common], dtype=np.float32)
                dst = np.array([self.trails[i][-1]     for i in common], dtype=np.float32)
                H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if H is not None:
                    x1, y1, x2, y2 = self.ref_bbox
                    corners = np.array(
                        [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32
                    ).reshape(-1, 1, 2)
                    warped = cv2.perspectiveTransform(corners, H)
                    cv2.polylines(out, [warped.astype(np.int32)],
                                  isClosed=True, color=(255, 0, 0), thickness=2)

        # Ghost COM when all points lost
        if self.active_count == 0 and self.last_known_com is not None:
            gx, gy      = int(self.last_known_com[0]), int(self.last_known_com[1])
            ghost_color = (80, 200, 200)
            arm         = 14
            cv2.line(out, (gx - arm, gy), (gx + arm, gy), ghost_color, 1)
            cv2.line(out, (gx, gy - arm), (gx, gy + arm), ghost_color, 1)
            cv2.circle(out, (gx, gy), COM_SEARCH_RADIUS, ghost_color, 1)
            cv2.putText(out, "searching…", (gx + 16, gy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ghost_color, 1)

        cv2.putText(out, f"Tracking: {self.active_count} pts",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    @property
    def active_count(self):
        return len(self.points) if self.points is not None else 0

    @property
    def needs_reinit(self):
        if self.original_count == 0:
            return False
        return self.active_count < max(1, int(self.original_count * REINIT_RATIO))

    def reset(self):
        self.prev_gray      = None
        self.points         = None
        self.obj_pts_3d     = None
        self.original_count = 0
        self.trails         = {}
        self.com_trail      = deque(maxlen=TRAIL_LENGTH)
        self.smooth_com     = None
        self.last_known_com = None
        self.smooth_rvec    = None
        self.smooth_tvec    = None


# ==============================================================================
# Main
# ==============================================================================

def main():
    # --- Load YOLO model (fine-tuned with fallback) ---------------------------
    yolo_model = None

    if os.path.exists(YOLO_MODEL_PATH):
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print("Fine-tuned YOLO model loaded:", YOLO_MODEL_PATH)
            print("Classes:", yolo_model.names)
        except Exception as e:
            print(f"Could not load fine-tuned model: {e}")

    if yolo_model is None:
        FALLBACK = "yolo26n-seg.pt"
        try:
            yolo_model = YOLO(FALLBACK)
            print(f"Using fallback model: {FALLBACK}")
            print("Classes:", yolo_model.names)
        except Exception as e:
            print(f"Could not load fallback ({FALLBACK}): {e}")
            yolo_model = None

    # --- RGB camera -----------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open RGB camera.")
        return
    print("RGB camera opened.")

    # --- Depth sensor ---------------------------------------------------------
    config   = Config()
    pipeline = Pipeline()
    try:
        profile_list  = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None
        depth_profile = profile_list.get_default_video_stream_profile()
        assert depth_profile is not None
        print("Depth profile:", depth_profile)
        config.enable_stream(depth_profile)
    except Exception as e:
        print("Could not configure depth stream:", e)
        cap.release()
        return

    pipeline.start(config)
    depth_reader = DepthReader(pipeline)

    # --- Camera matrix --------------------------------------------------------
    ret0, frame0 = cap.read()
    if ret0:
        fh, fw     = frame0.shape[:2]
        focal      = fw
        cam_matrix = np.array(
            [[focal, 0, fw/2], [0, focal, fh/2], [0, 0, 1]], dtype=np.float32
        )
    else:
        cam_matrix = None
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    print("\nControls:")
    print("  v       — 20-s 3-D recording reference")
    print("  r       — single-frame reference (YOLO + segmentation auto-detect)")
    print("  t       — freeze frame and draw your own ROI")
    print("  q / ESC — quit\n")
    print("Segmentation overlay (magenta) shows the tool mask used for SIFT.")

    reference_gray  = None
    reference_kp    = None
    reference_des   = None
    ref_mask_saved  = None
    ref_bbox        = None
    ref_contour     = None

    tracker = COMTracker()

    max_features_holder = [MAX_FEATURES]
    cv2.namedWindow("Tracking")
    cv2.createTrackbar(
        "Max points", "Tracking", MAX_FEATURES, 200,
        lambda v: max_features_holder.__setitem__(0, max(5, v)),
    )

    while True:
        try:
            ret, color_image = cap.read()
            if not ret:
                print("Frame grab failed.")
                break
            color_image = cv2.rotate(color_image, cv2.ROTATE_180)
            cv2.imshow("RGB Camera", color_image)

            depth_image = depth_reader.get_latest()
            depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
            if depth_image is not None:
                cv2.imshow("Depth Viewer", depth_image)

            # ------------------------------------------------------------------
            # Tracking loop
            # ------------------------------------------------------------------
            if reference_gray is not None:
                query_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                # Run segmentation every frame for live mask + bbox
                if yolo_model:
                    live_mask, live_bbox, live_contour, _, _ = get_object_mask(
                        color_image, yolo_model
                    )
                else:
                    live_mask    = ref_mask_saved
                    live_bbox    = ref_bbox
                    live_contour = ref_contour

                # Always update existing points via optical flow.
                # The seg mask is NOT used for culling here — it is imperfect
                # and would drop valid points on the tool if even slightly off.
                # Bbox cull alone is sufficient to stop background drift.
                tracker.update(
                    query_gray, cam_matrix, dist_coeffs,
                    live_bbox=live_bbox,
                )

                # If below 20 % threshold, try to seed additional SIFT points
                # and MERGE them with the surviving tracked set.
                if tracker.needs_reinit:
                    pts, obj3d = None, None
                    last_com   = tracker.last_known_com

                    # Search around last known COM first
                    if last_com is not None:
                        h_img, w_img = color_image.shape[:2]
                        cx_s, cy_s   = int(last_com[0]), int(last_com[1])
                        com_mask     = np.zeros((h_img, w_img), dtype=np.uint8)
                        cv2.circle(com_mask, (cx_s, cy_s), COM_SEARCH_RADIUS, 255, -1)
                        if live_mask is not None:
                            com_mask = cv2.bitwise_and(com_mask, live_mask)
                        pts, obj3d = get_sift_points_in_frame(
                            reference_kp, reference_des, query_gray, com_mask,
                            max_features_holder[0], bbox=live_bbox,
                        )

                    # Fall back to full live seg mask
                    if (pts is None or len(pts) == 0) and live_mask is not None:
                        pts, obj3d = get_sift_points_in_frame(
                            reference_kp, reference_des, query_gray, live_mask,
                            max_features_holder[0], bbox=live_bbox,
                        )

                    if pts is not None and len(pts) > 0:
                        if tracker.active_count > 0:
                            # Keep surviving points and add fresh ones on top
                            tracker.add_points(query_gray, pts, obj3d)
                            print(f"Added {len(pts)} pts  (total {tracker.active_count}).")
                        else:
                            tracker.initialise(
                                query_gray, pts, obj3d,
                                ref_bbox=tracker.ref_bbox,
                                preserve_com=True,
                            )
                            print(f"Re-initialised with {len(pts)} pts.")

                tracking_image = tracker.draw(
                    color_image, cam_matrix, dist_coeffs,
                    live_bbox=live_bbox,
                    live_seg_mask=live_mask,
                    live_contour=live_contour,
                )
                cv2.imshow("Tracking", tracking_image)

            # ------------------------------------------------------------------
            # Key handling
            # ------------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ESC_KEY:
                break

            # ------ v : 3-D recording -----------------------------------------
            elif key == ord("v"):
                result = record_and_reconstruct_3d(
                    cap, depth_reader, yolo_model, cam_matrix,
                    max_features_holder[0],
                )
                (ref_frame, ref_kp_3d, ref_des_3d, ref_obj3d,
                 point_cloud, ref_bbox, ref_mask_saved, ref_viz) = result

                if ref_frame is None:
                    print("3-D recording failed — try 'r' or 't'.")
                    continue

                show_3d_model(point_cloud)
                cv2.imshow("Reference Features", ref_viz)

                reference_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
                reference_kp   = ref_kp_3d
                reference_des  = ref_des_3d

                init_pts = np.array([[k.pt] for k in ref_kp_3d], dtype=np.float32)
                tracker.reset()
                if len(init_pts) > 0:
                    tracker.initialise(
                        reference_gray, init_pts, ref_obj3d, ref_bbox=ref_bbox
                    )
                    print(f"3-D reference ready — tracking {len(init_pts)} points.")
                else:
                    print("3-D reference built but no keypoints found.")

            # ------ r : single-frame reference --------------------------------
            elif key == ord("r"):
                ref_color      = color_image.copy()
                reference_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)

                if yolo_model:
                    ref_mask_saved, ref_bbox, ref_contour, ref_cls, ref_conf = \
                        get_object_mask(ref_color, yolo_model)
                    if ref_bbox is not None:
                        print(f"  YOLO+seg: {ref_cls}  conf={ref_conf:.2f}  bbox={ref_bbox}")
                else:
                    ref_mask_saved, ref_bbox, ref_contour = None, None, None

                reference_kp, reference_des, ref_viz = compute_reference_features(
                    reference_gray, ref_color, ref_mask_saved, ref_bbox,
                    ref_contour, max_features_holder[0],
                )
                cv2.imshow("Reference Features", ref_viz)

                init_pts   = np.array([[kp.pt] for kp in reference_kp], dtype=np.float32)
                ref_2d     = np.array([kp.pt   for kp in reference_kp], dtype=np.float32)
                ref_2d    -= ref_2d.mean(axis=0)
                init_obj3d = np.column_stack(
                    [ref_2d, np.zeros(len(reference_kp), dtype=np.float32)]
                )
                tracker.reset()
                if len(init_pts) > 0:
                    tracker.initialise(
                        reference_gray, init_pts, init_obj3d, ref_bbox=ref_bbox
                    )
                    print(f"Reference captured — tracking {len(init_pts)} points.")
                else:
                    print("Reference captured but no SIFT points found.")

                if ref_mask_saved is None:
                    print("(No detection — SIFT uses full image.)")

            # ------ t : manual ROI -------------------------------------------
            elif key == ord("t"):
                roi_frame = color_image.copy()
                print("Draw ROI: drag a rectangle around the tool.")
                print("  SPACE or ENTER to confirm   ESC to cancel")
                rx, ry, rw, rh = cv2.selectROI(
                    "Select ROI", roi_frame, fromCenter=False, showCrosshair=True
                )
                cv2.destroyWindow("Select ROI")

                if rw > 0 and rh > 0:
                    ref_color      = roi_frame
                    reference_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
                    ref_bbox       = (rx, ry, rx + rw, ry + rh)

                    # Run segmentation inside the manually drawn ROI
                    bbox_mask = np.zeros(reference_gray.shape, dtype=np.uint8)
                    bbox_mask[ry:ry + rh, rx:rx + rw] = 255
                    refined   = _canny_contour_refine(ref_color, bbox_mask, ref_bbox)
                    cnts, _   = cv2.findContours(refined, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
                    ref_contour    = max(cnts, key=cv2.contourArea) if cnts else None
                    ref_mask_saved = refined if refined.sum() > 1000 else bbox_mask

                    reference_kp, reference_des, ref_viz = compute_reference_features(
                        reference_gray, ref_color, ref_mask_saved, ref_bbox,
                        ref_contour, max_features_holder[0],
                    )
                    cv2.imshow("Reference Features", ref_viz)

                    init_pts   = np.array([[kp.pt] for kp in reference_kp], dtype=np.float32)
                    ref_2d     = np.array([kp.pt   for kp in reference_kp], dtype=np.float32)
                    ref_2d    -= ref_2d.mean(axis=0)
                    init_obj3d = np.column_stack(
                        [ref_2d, np.zeros(len(reference_kp), dtype=np.float32)]
                    )
                    tracker.reset()
                    if len(init_pts) > 0:
                        tracker.initialise(
                            reference_gray, init_pts, init_obj3d, ref_bbox=ref_bbox
                        )
                        print(f"Manual ROI (segmented) — tracking {len(init_pts)} points.")
                    else:
                        print("ROI selected but no SIFT points found inside it.")
                else:
                    print("ROI selection cancelled.")

        except KeyboardInterrupt:
            break

    depth_reader.stop()
    cap.release()
    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    main()
