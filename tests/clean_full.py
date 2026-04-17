# ******************************************************************************
#  clean_full.py
#
#  Pipeline from combined_viewer_v3.py: depth, SIFT + optical-flow tracking,
#  COM / pose, 3-D recording ('v'), ROI ('t'), single-frame ref ('r').
#
#  Starts from former clean_full settings: externally fine-tuned YOLO
#  (surgical_tools_ft), conf=0.4, RGB rotated 180°. One YOLO pass per frame
#  feeds both live bbox/mask and (when idle) Detection+SIFT preview.
#
#  Originally extended combined_viewer_v2 with a 3-D reference model built from a
#  short recording rather than a single snapshot.
#
#  Workflow
#  --------
#  1. Press 'v' — a RECORD_SECONDS countdown starts.
#     Move the surgical tool slowly to show different sides to the camera.
#  2. Every valid frame where YOLO detects the tool contributes SIFT keypoints
#     whose 3-D positions come from the live Orbbec depth map — real metric
#     coordinates, not a flat z = 0 plane.
#  3. All per-frame point clouds are registered into a single coordinate frame
#     using 3-D affine transforms estimated from SIFT matches between frames.
#  4. A 3-D scatter plot of the reconstructed model opens in a separate window
#     (drag to rotate). Tracking starts immediately in the background.
#  5. solvePnP now operates on real-depth obj_pts_3d, giving more accurate
#     and stable pose-axis estimates.
#
#  Fallback controls (identical to v2)
#  ------------------------------------
#  r       — single-frame reference with YOLO auto-detect
#  t       — freeze frame, draw your own ROI with the mouse
#  v       — record & reconstruct 3-D reference  ← NEW
#  q / ESC — quit
#
#  Display windows
#  ---------------
#  RGB Camera         : live colour feed (rotated 180°)
#  Depth Viewer       : depth colormap (rotated 180° to match RGB)
#  Detection + SIFT   : YOLO plot + full-frame SIFT (only before tracking)
#  Reference Features : annotated snapshot used as tracking seed
#  Tracking           : per object ID — own SIFT ref + optical-flow / COM / pose
#  3-D Tool Model     : interactive scatter plot (separate process)
#
#  SAM 2 video (optional): export CLEAN_FULL_SAM2=1 to segment/track inside
#  YOLO boxes each frame (same YOLO pass). Or run: python sam2_yolo_video.py
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
ESC_KEY          = 27
PRINT_INTERVAL   = 1        # seconds between depth console prints
MIN_DEPTH        = 20       # depth units (mm on most Orbbec sensors)
MAX_DEPTH        = 10000
TRAIL_LENGTH     = 50       # frames of history kept per point / COM
REINIT_RATIO     = 0.3      # re-initialise when < 30 % of points survive
MAX_FEATURES     = 30       # default max SIFT keypoints (trackbar-tunable)
COM_SMOOTH       = 0.2      # EMA alpha for COM  (lower = smoother)
POSE_SMOOTH      = 0.15     # EMA alpha for rvec/tvec
RECORD_SECONDS   = 20       # length of the 3-D recording pass
RECORD_MAX_KF    = 40       # max keyframes sampled from the recording
COM_SEARCH_RADIUS = 120     # px radius around last COM to search for features on reinit

YOLO_MODEL_PATH  = os.environ.get(
    "YOLO_MODEL_PATH",
    "runs/detect/surgical_tools_ft/run1/weights/best.pt",
)
YOLO_CONF        = 0.4
ENABLE_SAM2_VIDEO = os.environ.get("CLEAN_FULL_SAM2", "").lower() in (
    "1",
    "true",
    "yes",
)
EXCLUDED_CLASSES = {"human", "person", "hand", "arm"}

# Drop queued RGB frames before each grab (reduces latency when the pipeline is slower than the camera).
CAM_BUFFER_FLUSH_GRABS = int(os.environ.get("CLEAN_FULL_CAM_FLUSH", "4"))
# Minimum IoU to keep the same track_id across frames when matching boxes.
TRACK_IOU_MATCH_THRESH = float(os.environ.get("CLEAN_FULL_TRACK_IOU", "0.25"))

LK_PARAMS = dict(
    winSize  = (21, 21),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
# ------------------------------------------------------------------------------


# ==============================================================================
# Depth pipeline helpers
# ==============================================================================

class TemporalFilter:
    """
    EMA blend to reduce per-frame depth noise.
        result = alpha * new + (1 - alpha) * previous
    """
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
    Reads Orbbec depth frames in a background thread.

    get_latest()     → colourised BGR image  (for display)
    get_latest_raw() → float32 depth array   (for 3-D reconstruction)
                       values are in the same units returned by the sensor
                       after applying the depth scale (typically mm).
    """

    def __init__(self, pipeline):
        self.pipeline        = pipeline
        self.latest_image    = None
        self.latest_raw      = None   # ← NEW: raw float32 depth for 3-D work
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

            data = np.frombuffer(
                depth_frame.get_data(), dtype=np.uint16
            ).reshape((h, w))
            data = data.astype(np.float32) * scale
            data = np.where(
                (data > MIN_DEPTH) & (data < MAX_DEPTH), data, 0
            ).astype(np.uint16)
            data = self.temporal_filter.process(data)   # returns float32

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
        """Return a copy of the most recent raw depth frame (float32, in mm)."""
        with self.lock:
            return self.latest_raw.copy() if self.latest_raw is not None else None

    def stop(self):
        self.running = False
        self._thread.join()


# ==============================================================================
# YOLO + SIFT helpers  (same as v2)
# ==============================================================================

def get_object_mask_from_results(r, hw, yolo_model):
    """
    Build bbox mask from an existing ultralytics Results object (no extra inference).
    hw = (h, w) image size.
    """
    h, w = hw[0], hw[1]
    if r.boxes is None or len(r.boxes) == 0:
        return None, None, None, None

    boxes = r.boxes
    valid = [
        i for i in range(len(boxes))
        if str(yolo_model.names.get(int(boxes.cls[i]), "")).lower()
        not in EXCLUDED_CLASSES
    ]
    if not valid:
        return None, None, None, None

    best = max(valid, key=lambda i: float(boxes.conf[i]))
    x1, y1, x2, y2 = map(int, boxes.xyxy[best])
    cls_id = int(boxes.cls[best])
    conf = float(boxes.conf[best])
    cls_name = yolo_model.names.get(cls_id, str(cls_id))

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask, (x1, y1, x2, y2), cls_name, conf


def bbox_iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


def enumerate_valid_detections(r, hw, yolo_model):
    """
    All non-excluded YOLO boxes as dicts: bbox, mask, cls_name, conf, cls_id, yolo_idx.
    Sorted by confidence (highest first).
    """
    h, w = hw[0], hw[1]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    boxes = r.boxes
    valid = [
        i
        for i in range(len(boxes))
        if str(yolo_model.names.get(int(boxes.cls[i]), "")).lower()
        not in EXCLUDED_CLASSES
    ]
    if not valid:
        return []

    valid.sort(key=lambda i: float(boxes.conf[i]), reverse=True)
    out = []
    for i in valid:
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        cls_name = yolo_model.names.get(cls_id, str(cls_id))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        out.append(
            {
                "yolo_idx": i,
                "bbox": (x1, y1, x2, y2),
                "mask": mask,
                "cls_name": cls_name,
                "conf": conf,
                "cls_id": cls_id,
            }
        )
    return out


def assign_track_ids_to_detections(
    detections,
    id_to_bbox: dict,
    next_id_start: int,
    iou_thresh: float = TRACK_IOU_MATCH_THRESH,
):
    """
    Greedy IoU matching then new IDs. Sets det['track_id'] on each detection.
    Returns (next_id_after_max_assigned, new_id_to_bbox for this frame).
    """
    if not detections:
        # Keep last boxes so IDs can match again after a brief missed detection.
        return next_id_start, dict(id_to_bbox)

    pairs = []
    for di, det in enumerate(detections):
        for tid, pb in id_to_bbox.items():
            iou = bbox_iou_xyxy(det["bbox"], pb)
            if iou >= iou_thresh:
                pairs.append((iou, di, int(tid)))
    pairs.sort(key=lambda x: -x[0])

    det_to_tid = {}
    used_di = set()
    used_tid = set()
    for iou, di, tid in pairs:
        if di in used_di or tid in used_tid:
            continue
        det_to_tid[di] = tid
        used_di.add(di)
        used_tid.add(tid)

    nid = next_id_start
    for di in range(len(detections)):
        if di not in det_to_tid:
            det_to_tid[di] = nid
            nid += 1

    new_map = {}
    for di, det in enumerate(detections):
        tid = det_to_tid[di]
        det["track_id"] = tid
        new_map[tid] = det["bbox"]
    return nid, new_map


def get_object_mask(color_image, yolo_model):
    """
    Runs YOLO and returns (mask, bbox, cls_name, conf) for the highest-
    confidence detection that is not in EXCLUDED_CLASSES.
    """
    h, w = color_image.shape[:2]
    results = yolo_model(color_image, conf=YOLO_CONF, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None, None, None, None
    return get_object_mask_from_results(results[0], (h, w), yolo_model)


def compute_reference_features(ref_gray, ref_color, ref_mask, bbox, max_features):
    """
    Detects up to max_features SIFT keypoints inside the YOLO bbox.
    Returns (kp, des, annotated_image).
    """
    sift    = cv2.SIFT_create(nfeatures=max_features)
    kp, des = sift.detectAndCompute(ref_gray, ref_mask)

    viz = ref_color.copy()
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


def get_sift_points_in_frame(ref_kp, ref_des, query_gray, query_mask, max_features):
    """
    Matches reference descriptors to the current frame within query_mask.
    Returns (live_pts, obj_pts_3d) or (None, None).
    """
    sift      = cv2.SIFT_create()
    kp2, des2 = sift.detectAndCompute(query_gray, query_mask)
    if ref_des is None or des2 is None or len(ref_des) < 2 or len(des2) < 2:
        return None, None

    flann   = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(ref_des, des2, k=2)
    good    = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if not good:
        return None, None

    good     = sorted(good, key=lambda m: m.distance)[:max_features]
    live_pts = np.array([[kp2[m.trainIdx].pt] for m in good], dtype=np.float32)
    ref_2d   = np.array([ref_kp[m.queryIdx].pt for m in good], dtype=np.float32)
    ref_2d  -= ref_2d.mean(axis=0)
    obj_pts_3d = np.column_stack(
        [ref_2d, np.zeros(len(good), dtype=np.float32)]
    )
    return live_pts, obj_pts_3d


def point_color(idx, hue_base: int = 0):
    """Unique BGR colour per point index; hue_base separates different tracked objects."""
    hue = (hue_base + idx * 37) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def overlay_binary_mask(image, mask, color, alpha: float = 0.3):
    """Blend a binary mask onto image with given BGR color."""
    if mask is None:
        return image
    binm = mask > 0
    if not np.any(binm):
        return image
    out = image.copy().astype(np.float32)
    col = np.array(color, dtype=np.float32)
    out[binm] = out[binm] * (1.0 - alpha) + col * alpha
    return out.astype(np.uint8)


def select_sift_prompt_points_in_detections(gray, dets):
    """
    YOLO -> SIFT-in-box stage: pick one strong SIFT keypoint per detection mask.
    Returns dict[yolo_idx] = (x, y).
    """
    sift = cv2.SIFT_create()
    out = {}
    for det in dets:
        kp, des = sift.detectAndCompute(gray, det["mask"])
        if not kp or des is None:
            continue
        # Highest-response keypoint is typically stable as a point prompt.
        best = max(kp, key=lambda k: float(k.response))
        out[int(det["yolo_idx"])] = (float(best.pt[0]), float(best.pt[1]))
    return out


# ==============================================================================
# 3-D reconstruction helpers  (NEW in v3)
# ==============================================================================

def _depth_at_rgb_pixel(u, v, raw_depth, rgb_w, rgb_h):
    """
    Look up the depth at RGB pixel (u, v), scaling to the depth frame's
    (potentially different) resolution.  Returns 0.0 for invalid readings.
    """
    dh, dw = raw_depth.shape[:2]
    u_d    = int(round(u * dw / rgb_w))
    v_d    = int(round(v * dh / rgb_h))
    u_d    = np.clip(u_d, 0, dw - 1)
    v_d    = np.clip(v_d, 0, dh - 1)
    return float(raw_depth[v_d, u_d])


def _unproject(u, v, z, cam_matrix):
    """
    Back-project a 2-D image point + depth value to a 3-D camera-space
    point using the pinhole camera model.  Returns None if z is invalid.
    """
    if z <= 0:
        return None
    fx = cam_matrix[0, 0]
    fy = cam_matrix[1, 1]
    cx = cam_matrix[0, 2]
    cy = cam_matrix[1, 2]
    return np.array(
        [(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float32
    )


def record_and_reconstruct_3d(cap, depth_reader, yolo_model, cam_matrix, max_features):
    """
    Records RECORD_SECONDS of video while the user moves the tool, then
    builds a sparse 3-D model by:
      1. Extracting SIFT keypoints with real Z values from the depth map
         in each keyframe where YOLO detects the tool.
      2. Matching keyframes to the first good frame with FLANN and
         estimating a 3-D affine transform (cv2.estimateAffine3D).
      3. Transforming all subsequent-frame point clouds into the first
         frame's coordinate system to produce a unified model.

    Returns
    -------
    ref_frame   : BGR image of the first good keyframe
    ref_kp      : list[cv2.KeyPoint] — keypoints on ref_frame
    ref_des     : (N, 128) float32 — SIFT descriptors
    obj_pts_3d  : (N, 3) float32 — 3-D positions centred at the object origin
    point_cloud : (M, 6) float32 — X Y Z B G R for the 3-D viewer
    ref_bbox    : (x1, y1, x2, y2)
    ref_mask    : binary mask (same size as ref_frame)
    ref_viz     : annotated reference image
    All values are None on failure.
    """
    print(f"\nRecording 3-D reference for {RECORD_SECONDS} s.")
    print("Move the tool slowly so the camera sees all sides.")

    recorded = []   # list of (rgb_frame, raw_depth_array)
    t0       = time.time()

    while time.time() - t0 < RECORD_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        raw = depth_reader.get_latest_raw()

        remaining = RECORD_SECONDS - (time.time() - t0)
        disp = frame.copy()
        cv2.putText(
            disp,
            f"Recording  {remaining:.1f} s  — move the tool slowly",
            (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 210), 2,
        )
        # Red recording dot
        cv2.circle(disp, (disp.shape[1] - 34, 32), 14, (0, 0, 210), -1)
        cv2.imshow("Recording", disp)
        cv2.waitKey(1)

        if raw is not None:
            recorded.append((frame.copy(), raw))

    cv2.destroyWindow("Recording")

    if not recorded:
        print("  No frames captured.")
        return (None,) * 8

    print(f"  {len(recorded)} frames captured — extracting 3-D keypoints…")

    # ------------------------------------------------------------------
    # Extract keypoints + real 3-D positions from evenly-spaced keyframes
    # ------------------------------------------------------------------
    sift  = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    step    = max(1, len(recorded) // RECORD_MAX_KF)
    kf_data = []   # list of dicts

    for i in range(0, len(recorded), step):
        rgb, raw_d  = recorded[i]
        rgb_h, rgb_w = rgb.shape[:2]

        mask, bbox, _, _ = get_object_mask(rgb, yolo_model)
        if bbox is None:
            continue

        gray    = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, mask)
        if not kp or des is None:
            continue

        pts3d_list  = []
        colors_list = []
        kp_valid    = []
        des_valid   = []

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
            frame  = rgb,
            gray   = gray,
            kp     = kp_valid,
            des    = np.array(des_valid, dtype=np.float32),
            pts3d  = np.array(pts3d_list, dtype=np.float32),
            colors = np.array(colors_list, dtype=np.uint8),
            bbox   = bbox,
            mask   = mask,
        ))

    if not kf_data:
        print("  No keyframes had valid depth data — check YOLO detection and depth feed.")
        return (None,) * 8

    print(f"  {len(kf_data)} keyframes with depth. Registering into common frame…")

    # ------------------------------------------------------------------
    # Register all keyframes into the first keyframe's coordinate frame
    # ------------------------------------------------------------------
    ref_kf     = kf_data[0]
    all_pts3d  = list(ref_kf["pts3d"])
    all_colors = list(ref_kf["colors"])

    for kf in kf_data[1:]:
        if ref_kf["des"] is None or kf["des"] is None:
            continue
        try:
            matches = flann.knnMatch(ref_kf["des"], kf["des"], k=2)
        except Exception:
            continue

        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        if len(good) < 4:
            continue

        src_pts = np.array([kf["pts3d"][m.trainIdx]     for m in good], dtype=np.float32)
        dst_pts = np.array([ref_kf["pts3d"][m.queryIdx] for m in good], dtype=np.float32)

        try:
            ok, T34, _ = cv2.estimateAffine3D(src_pts, dst_pts, confidence=0.99)
        except Exception:
            continue

        if not ok or T34 is None:
            continue

        # Transform this keyframe's 3-D points into the reference frame
        T44   = np.vstack([T34, [0, 0, 0, 1]])
        h_pts = np.column_stack([kf["pts3d"], np.ones(len(kf["pts3d"]))])
        warped = (T44 @ h_pts.T).T[:, :3]

        all_pts3d.extend(warped.tolist())
        all_colors.extend(kf["colors"].tolist())

    all_pts3d  = np.array(all_pts3d,  dtype=np.float32)
    all_colors = np.array(all_colors, dtype=np.uint8)
    point_cloud = np.column_stack([all_pts3d, all_colors])   # Nx6: X Y Z B G R

    # ------------------------------------------------------------------
    # Build tracking reference from the first keyframe
    # (best MAX_FEATURES points by SIFT response, centred in 3-D)
    # ------------------------------------------------------------------
    responses  = [k.response for k in ref_kf["kp"]]
    idx_sorted = np.argsort(responses)[::-1][:max_features]

    ref_kp  = [ref_kf["kp"][i]  for i in idx_sorted]
    ref_des =  ref_kf["des"][idx_sorted]
    ref_obj =  ref_kf["pts3d"][idx_sorted].copy()

    # Centre so solvePnP works in an object-relative coordinate frame
    centroid = ref_obj.mean(axis=0)
    ref_obj -= centroid

    ref_bbox  = ref_kf["bbox"]
    ref_mask  = ref_kf["mask"]
    ref_frame = ref_kf["frame"]

    # Annotated reference image
    ref_viz = ref_frame.copy()
    x1, y1, x2, y2 = ref_bbox
    cv2.rectangle(ref_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(ref_viz, "3-D reference bbox", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    ref_viz = cv2.drawKeypoints(
        ref_viz, ref_kp, None, color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    print(
        f"  3-D model: {len(all_pts3d)} points  |  "
        f"tracking seed: {len(ref_kp)} keypoints"
    )
    return ref_frame, ref_kp, ref_des, ref_obj, point_cloud, ref_bbox, ref_mask, ref_viz


# ------------------------------------------------------------------------------
# 3-D viewer — runs in a separate process so it never blocks OpenCV
# ------------------------------------------------------------------------------

def _plot_3d_worker(pts, colors):
    """
    Entry point for the 3-D viewer subprocess.
    pts    : (N, 3) float32 — X Y Z
    colors : (N, 3) uint8   — B G R
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    fig = plt.figure("3-D Tool Model", figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")

    r = colors[:, 2] / 255.0   # OpenCV stores BGR
    g = colors[:, 1] / 255.0
    b = colors[:, 0] / 255.0
    rgb_norm = np.column_stack([r, g, b])

    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=rgb_norm, s=8, alpha=0.9, depthshade=True,
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z — depth (mm)")
    ax.set_title("Reconstructed 3-D Tool Model — drag to rotate")

    # Force equal axis scaling so Z depth variation isn't squished into a line.
    # Matplotlib 3D doesn't support set_aspect('equal') natively, so we manually
    # compute the largest range and apply it to all three axes symmetrically.
    ranges = np.array([
        pts[:, 0].max() - pts[:, 0].min(),
        pts[:, 1].max() - pts[:, 1].min(),
        pts[:, 2].max() - pts[:, 2].min(),
    ])
    max_range = ranges.max() / 2.0
    mid = np.array([
        (pts[:, 0].max() + pts[:, 0].min()) / 2.0,
        (pts[:, 1].max() + pts[:, 1].min()) / 2.0,
        (pts[:, 2].max() + pts[:, 2].min()) / 2.0,
    ])
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.show()   # blocks until the window is closed, but runs in its own process


def show_3d_model(point_cloud):
    """
    Opens the interactive 3-D scatter plot in a separate process.
    The tracking loop continues running uninterrupted.
    """
    pts    = point_cloud[:, :3]
    colors = point_cloud[:, 3:].astype(np.uint8)
    p = Process(target=_plot_3d_worker, args=(pts, colors), daemon=True)
    p.start()


# ==============================================================================
# Tracker  (identical to v2)
# ==============================================================================

class COMTracker:
    """
    Continuous Lucas-Kanade optical-flow tracker with:
      • per-point colour trails
      • EMA-smoothed Centre of Mass + COM trail
      • EMA-smoothed solvePnP pose axes (X / Y / Z arrows at COM)
      • warped reference bounding box via RANSAC homography
    """

    def __init__(self):
        self.prev_gray       = None
        self.points          = None
        self.obj_pts_3d      = None
        self.original_count  = 0
        self.trails          = {}
        self.init_positions  = {}
        self.com_trail       = deque(maxlen=TRAIL_LENGTH)
        self.smooth_com      = None
        self.last_known_com  = None   # survives reinit — used as search anchor
        self.smooth_rvec     = None
        self.smooth_tvec     = None
        self.axis_length     = 50.0
        self.ref_bbox        = None

    def initialise(self, gray, live_pts, obj_pts_3d, ref_bbox=None,
                   preserve_com=False):
        """
        preserve_com=True keeps smooth_com / last_known_com alive across a
        reinit so the COM indicator doesn't snap to (0,0) when the hand
        briefly blocks the tool.
        """
        saved_com     = self.smooth_com
        saved_last    = self.last_known_com

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
        self.com_trail  = deque(maxlen=TRAIL_LENGTH)
        self.smooth_rvec = None
        self.smooth_tvec = None

        if preserve_com:
            self.smooth_com    = saved_com
            self.last_known_com = saved_last
            if saved_com is not None:
                self.com_trail.append(saved_com)
        else:
            self.smooth_com    = None
            self.last_known_com = None

        if ref_bbox is not None:
            self.ref_bbox = ref_bbox
        spread           = float(np.std(obj_pts_3d[:, :2]))
        self.axis_length = max(30.0, spread * 1.5)

    def update(self, gray, cam_matrix, dist_coeffs, live_bbox=None):
        """
        live_bbox — when provided, points that drift outside it are dropped.
        """
        if self.points is None or len(self.points) == 0 or self.prev_gray is None:
            return

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None, **LK_PARAMS
        )
        surviving = [i for i, st in enumerate(status) if st[0] == 1]

        if live_bbox is not None:
            bx1, by1, bx2, by2 = live_bbox
            surviving = [
                i for i in surviving
                if bx1 <= new_pts[i][0][0] <= bx2
                and by1 <= new_pts[i][0][1] <= by2
            ]

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
            if self.smooth_com is None:
                self.smooth_com = (raw_x, raw_y)
            else:
                self.smooth_com = (
                    COM_SMOOTH * raw_x + (1 - COM_SMOOTH) * self.smooth_com[0],
                    COM_SMOOTH * raw_y + (1 - COM_SMOOTH) * self.smooth_com[1],
                )
            self.com_trail.append(self.smooth_com)
            self.last_known_com = self.smooth_com   # always up-to-date anchor

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
                        self.smooth_rvec = rvec.copy()
                        self.smooth_tvec = tvec.copy()
                    else:
                        self.smooth_rvec = (
                            POSE_SMOOTH * rvec + (1 - POSE_SMOOTH) * self.smooth_rvec
                        )
                        self.smooth_tvec = (
                            POSE_SMOOTH * tvec + (1 - POSE_SMOOTH) * self.smooth_tvec
                        )
            except Exception:
                pass

    def draw(self, image, cam_matrix, dist_coeffs, live_bbox=None,
             hue_shift: int = 0, status_y: int = 24, track_label: str = ""):
        out = image.copy()
        self.draw_on(
            out, cam_matrix, dist_coeffs, live_bbox=live_bbox,
            hue_shift=hue_shift, status_y=status_y, track_label=track_label,
        )
        return out

    def draw_on(
        self,
        out,
        cam_matrix,
        dist_coeffs,
        live_bbox=None,
        hue_shift: int = 0,
        status_y: int = 24,
        track_label: str = "",
    ):
        """Draw trails / COM / pose on an existing BGR image (for multi-object)."""
        # Live YOLO bbox (green)
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
            color = point_color(idx, hue_base=hue_shift)
            n     = len(pts)
            for i in range(1, n):
                brightness = i / n
                seg = tuple(int(c * brightness) for c in color)
                p1  = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p2  = (int(pts[i][0]),     int(pts[i][1]))
                cv2.line(out, p1, p2, seg, max(1, int(brightness * 3)))
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
            cv2.putText(out, f"COM ({cx}, {cy})",
                        (cx + 14, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if self.smooth_rvec is not None and cam_matrix is not None:
                L       = self.axis_length
                axes_3d = np.array(
                    [[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, -L]],
                    dtype=np.float32,
                )
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
                    cv2.putText(out, "X", (x_tip[0] + 4, x_tip[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,   0, 255), 1)
                    cv2.putText(out, "Y", (y_tip[0] + 4, y_tip[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,   0), 1)
                    cv2.putText(out, "Z", (z_tip[0] + 4, z_tip[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,  0,   0), 1)
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
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        dtype=np.float32,
                    ).reshape(-1, 1, 2)
                    warped = cv2.perspectiveTransform(corners, H)
                    cv2.polylines(
                        out, [warped.astype(np.int32)],
                        isClosed=True, color=(255, 0, 0), thickness=2,
                    )

        # Ghost COM — drawn when active tracking is lost but we still have a
        # last known position.  Shows where the tracker expects the tool to be.
        if self.active_count == 0 and self.last_known_com is not None:
            gx, gy = int(self.last_known_com[0]), int(self.last_known_com[1])
            arm = 14
            ghost_color = (80, 200, 200)   # dim cyan
            cv2.line(out, (gx - arm, gy), (gx + arm, gy), ghost_color, 1)
            cv2.line(out, (gx, gy - arm), (gx, gy + arm), ghost_color, 1)
            cv2.circle(out, (gx, gy), COM_SEARCH_RADIUS, ghost_color, 1)
            cv2.putText(out, "searching…", (gx + 16, gy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ghost_color, 1)

        label = track_label.strip() or "Track"
        cv2.putText(
            out,
            f"{label}: {self.active_count} pts",
            (10, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    @property
    def active_count(self):
        return len(self.points) if self.points is not None else 0

    @property
    def needs_reinit(self):
        if self.original_count == 0:
            return False
        return self.active_count < max(1, int(self.original_count * REINIT_RATIO))

    def reset(self):
        self.prev_gray       = None
        self.points          = None
        self.obj_pts_3d      = None
        self.original_count  = 0
        self.trails          = {}
        self.com_trail       = deque(maxlen=TRAIL_LENGTH)
        self.smooth_com      = None
        self.last_known_com  = None
        self.smooth_rvec     = None
        self.smooth_tvec     = None


# ==============================================================================
# Main
# ==============================================================================

def main():
    # --- Load YOLO (externally trained weights, same as testyolo / clean_full) --
    yolo_model = None
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO loaded:", YOLO_MODEL_PATH, "| conf=", YOLO_CONF)
        print("Classes:", yolo_model.names)
    except Exception as e:
        print("Could not load YOLO weights:", e)
        return

    sam2_tracker = None
    if ENABLE_SAM2_VIDEO:
        try:
            from sam2_yolo_track import SamYoloVideoTracker, overlay_sam_on_bgr

            sam2_tracker = SamYoloVideoTracker(
                yolo_model=yolo_model,
                yolo_weights="",
                yolo_conf=YOLO_CONF,
            )
            print("SAM2 video overlay ON (CLEAN_FULL_SAM2). Window: SAM2 track")
        except Exception as e:
            print("SAM2 video disabled:", e)

    # --- RGB camera -----------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open RGB camera.")
        return
    print("RGB camera opened.")
    rgb_fps = cap.get(cv2.CAP_PROP_FPS)
    rgb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rgb_fps_note = (
        ""
        if (rgb_fps and rgb_fps > 0)
        else " — if 0, driver did not report FPS (actual rate is device-dependent)"
    )
    print(f"RGB stream: {rgb_w}x{rgb_h}, CAP_PROP_FPS={rgb_fps}{rgb_fps_note}")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if CAM_BUFFER_FLUSH_GRABS > 0:
        print(
            f"RGB buffer: flush {CAM_BUFFER_FLUSH_GRABS} grab(s) per frame "
            f"(CLEAN_FULL_CAM_FLUSH to change)",
        )

    # --- Depth sensor (Orbbec SDK) -------------------------------------------
    config   = Config()
    pipeline = Pipeline()
    try:
        profile_list  = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None
        depth_profile = profile_list.get_default_video_stream_profile()
        assert depth_profile is not None
        print("Depth profile:", depth_profile)
        try:
            dw = depth_profile.get_width()
            dh = depth_profile.get_height()
            dfps = depth_profile.get_fps()
            print(f"Depth stream (profile): {dw}x{dh} @ {dfps} FPS")
        except Exception as e:
            print("Depth stream: could not read width/height/FPS from profile:", e)
        config.enable_stream(depth_profile)
    except Exception as e:
        print("Could not configure depth stream:", e)
        cap.release()
        return

    pipeline.start(config)
    depth_reader = DepthReader(pipeline)

    # --- Approximate camera matrix from first RGB frame (after 180° rotation) -
    ret0, frame0 = cap.read()
    if ret0:
        frame0 = cv2.rotate(frame0, cv2.ROTATE_180)
        fh, fw = frame0.shape[:2]
        focal = fw
        cam_matrix = np.array(
            [[focal, 0, fw / 2], [0, focal, fh / 2], [0, 0, 1]],
            dtype=np.float32,
        )
    else:
        cam_matrix = None
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    sift_preview = cv2.SIFT_create(nfeatures=500)
    det_window_open = True

    print("\nControls:")
    print("  v       — record 3-D reference (NEW: move tool to show all sides)")
    print("  r       — single-frame reference: SIFT + track each YOLO object (IDs on screen)")
    print("  t       — freeze frame and draw your own ROI")
    print("  q / ESC — quit\n")

    active_tracks: dict = {}  # track_id -> {tracker, ref_kp, ref_des, ref_bbox, cls}
    next_track_id = [1]
    track_bbox_memory: dict = {}

    max_features_holder = [MAX_FEATURES]
    cv2.namedWindow("Tracking")
    cv2.createTrackbar(
        "Max points", "Tracking",
        MAX_FEATURES, 200,
        lambda v: max_features_holder.__setitem__(0, max(5, v)),
    )

    while True:
        try:
            for _ in range(max(0, CAM_BUFFER_FLUSH_GRABS)):
                cap.grab()
            ret, color_image = cap.read()
            if not ret:
                print("Frame grab failed.")
                break
            color_image = cv2.rotate(color_image, cv2.ROTATE_180)
            cv2.imshow("RGB Camera", color_image)

            depth_image = depth_reader.get_latest()
            if depth_image is not None:
                depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
                cv2.imshow("Depth Viewer", depth_image)

            results = yolo_model(color_image, conf=YOLO_CONF, verbose=False)[0]
            h, w = color_image.shape[:2]
            dets = enumerate_valid_detections(results, (h, w), yolo_model)
            next_track_id[0], track_bbox_memory = assign_track_ids_to_detections(
                dets, track_bbox_memory, next_track_id[0]
            )

            if sam2_tracker is not None:
                gray_now = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                # Strict pipeline: YOLO detections -> SIFT inside each bbox mask -> one point/object for SAM2.
                sam_prompt_points = select_sift_prompt_points_in_detections(gray_now, dets)
                sam2_out = sam2_tracker.step_with_yolo_result(
                    color_image,
                    results,
                    sift_points_by_yolo_idx=sam_prompt_points,
                )
                rgb_vis = color_image.copy()
                if sam2_out is not None:
                    svis = overlay_sam_on_bgr(color_image, sam2_out, alpha=0.45)
                    # Visualize the exact positive point prompts sent to SAM2.
                    for det in dets:
                        yidx = int(det["yolo_idx"])
                        if yidx not in sam_prompt_points:
                            continue
                        px, py = sam_prompt_points[yidx]
                        tid = int(det["track_id"])
                        c = point_color(tid, hue_base=(tid * 29) % 180)
                        p = (int(px), int(py))
                        cv2.circle(svis, p, 5, c, -1)
                        cv2.circle(svis, p, 8, (255, 255, 255), 1)
                        cv2.putText(
                            svis,
                            f"ID{tid} SIFT->SAM2",
                            (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            c,
                            1,
                        )
                        cv2.circle(rgb_vis, p, 5, c, -1)
                        cv2.circle(rgb_vis, p, 8, (255, 255, 255), 1)
                        cv2.putText(
                            rgb_vis,
                            f"ID{tid} SIFT",
                            (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            c,
                            1,
                        )
                    cv2.imshow("SAM2 track", svis)
                cv2.imshow("RGB Camera", rgb_vis)
            else:
                sam2_out = None

            # ------------------------------------------------------------------
            # Idle: YOLO plot + full-frame SIFT + persistent track IDs
            # ------------------------------------------------------------------
            if not active_tracks:
                det_vis = results.plot()
                for det in dets:
                    x1, y1, x2, y2 = det["bbox"]
                    tid = det["track_id"]
                    det_vis = overlay_binary_mask(
                        det_vis,
                        det["mask"],
                        point_color(tid, hue_base=(tid * 29) % 180),
                        alpha=0.22,
                    )
                    lab = f"ID{tid} {det['cls_name']}"
                    cv2.putText(
                        det_vis,
                        lab,
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
                g = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                kp_prev, _ = sift_preview.detectAndCompute(g, None)
                det_vis = cv2.drawKeypoints(
                    det_vis,
                    kp_prev,
                    None,
                    color=(0, 255, 0),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                )
                cv2.imshow("Detection + SIFT", det_vis)
                det_window_open = True
            else:
                if det_window_open:
                    try:
                        cv2.destroyWindow("Detection + SIFT")
                    except cv2.error:
                        pass
                    det_window_open = False

            # ------------------------------------------------------------------
            # Tracking loop (per track_id: own SIFT reference + COMTracker)
            # ------------------------------------------------------------------
            if active_tracks:
                query_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                sorted_tids = sorted(active_tracks.keys())

                for tid in sorted_tids:
                    st = active_tracks[tid]
                    tr = st["tracker"]
                    ref_kp = st["ref_kp"]
                    ref_des = st["ref_des"]
                    ref_bbox = st["ref_bbox"]
                    live_det = next(
                        (d for d in dets if d["track_id"] == tid),
                        None,
                    )
                    live_bbox = live_det["bbox"] if live_det else None
                    live_mask = live_det["mask"] if live_det else None

                    if tr.needs_reinit:
                        pts, obj3d = None, None
                        last_com = tr.last_known_com

                        if last_com is not None:
                            h_img, w_img = color_image.shape[:2]
                            cx_s, cy_s = int(last_com[0]), int(last_com[1])
                            com_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                            cv2.circle(
                                com_mask, (cx_s, cy_s), COM_SEARCH_RADIUS, 255, -1
                            )
                            pts, obj3d = get_sift_points_in_frame(
                                ref_kp,
                                ref_des,
                                query_gray,
                                com_mask,
                                max_features_holder[0],
                            )

                        if (pts is None or len(pts) == 0) and live_mask is not None:
                            pts, obj3d = get_sift_points_in_frame(
                                ref_kp,
                                ref_des,
                                query_gray,
                                live_mask,
                                max_features_holder[0],
                            )

                        if pts is not None and len(pts) > 0:
                            tr.initialise(
                                query_gray,
                                pts,
                                obj3d,
                                ref_bbox=tr.ref_bbox,
                                preserve_com=True,
                            )
                            print(
                                f"ID{tid}: re-init {len(pts)} pts "
                                f"({'COM' if last_com else 'YOLO mask'}).",
                            )
                    else:
                        tr.update(
                            query_gray,
                            cam_matrix,
                            dist_coeffs,
                            live_bbox=live_bbox,
                        )

                tracking_image = color_image.copy()
                if sam2_out is not None:
                    tracking_image = overlay_sam_on_bgr(tracking_image, sam2_out, alpha=0.25)
                else:
                    for det in dets:
                        tid = det["track_id"]
                        tracking_image = overlay_binary_mask(
                            tracking_image,
                            det["mask"],
                            point_color(tid, hue_base=(tid * 29) % 180),
                            alpha=0.20,
                        )
                for row, tid in enumerate(sorted_tids):
                    st = active_tracks[tid]
                    tr = st["tracker"]
                    live_det = next(
                        (d for d in dets if d["track_id"] == tid),
                        None,
                    )
                    live_bbox = live_det["bbox"] if live_det else None
                    lbl = f"ID{tid} {st.get('cls', '')}"
                    tr.draw_on(
                        tracking_image,
                        cam_matrix,
                        dist_coeffs,
                        live_bbox=live_bbox,
                        hue_shift=(tid * 41) % 180,
                        status_y=22 + row * 26,
                        track_label=lbl,
                    )
                cv2.imshow("Tracking", tracking_image)

            # ------------------------------------------------------------------
            # Key handling
            # ------------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ESC_KEY:
                break

            # ------ v : 3-D recording reference (NEW) -------------------------
            elif key == ord("v"):
                result = record_and_reconstruct_3d(
                    cap, depth_reader, yolo_model, cam_matrix,
                    max_features_holder[0],
                )
                (ref_frame, ref_kp_3d, ref_des_3d, ref_obj3d,
                 point_cloud, ref_bbox, ref_mask_saved, ref_viz) = result

                if ref_frame is None:
                    print("3-D recording failed — try again or use 'r' / 't'.")
                    continue

                # Show the 3-D model in a separate window
                show_3d_model(point_cloud)

                # Show the annotated reference image
                cv2.imshow("Reference Features", ref_viz)

                ref_gray_3d = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
                init_pts = np.array([[k.pt] for k in ref_kp_3d], dtype=np.float32)

                tid = next_track_id[0]
                next_track_id[0] += 1
                active_tracks.clear()
                tr = COMTracker()
                if len(init_pts) > 0:
                    tr.initialise(ref_gray_3d, init_pts, ref_obj3d, ref_bbox=ref_bbox)
                    active_tracks[tid] = {
                        "tracker": tr,
                        "ref_kp": ref_kp_3d,
                        "ref_des": ref_des_3d,
                        "ref_bbox": ref_bbox,
                        "cls": "3D",
                    }
                    if ref_bbox is not None:
                        track_bbox_memory.clear()
                        track_bbox_memory[tid] = ref_bbox
                    print(f"3-D reference ready — ID{tid}, {len(init_pts)} points.")
                else:
                    print("3-D reference built but no keypoints found.")

            # ------ r : single-frame reference --------------------------------
            elif key == ord("r"):
                ref_color = color_image.copy()
                ref_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
                if not dets:
                    print("No valid YOLO detections on this frame — use 't' or try again.")
                    continue

                active_tracks.clear()
                last_viz = None
                for det in dets:
                    tid = det["track_id"]
                    kp, des, ref_viz = compute_reference_features(
                        ref_gray,
                        ref_color,
                        det["mask"],
                        det["bbox"],
                        max_features_holder[0],
                    )
                    last_viz = ref_viz
                    if des is None or not kp:
                        print(f"ID{tid} {det['cls_name']}: no SIFT points in box.")
                        continue
                    init_pts = np.array([[k.pt] for k in kp], dtype=np.float32)
                    ref_2d = np.array([k.pt for k in kp], dtype=np.float32)
                    ref_2d -= ref_2d.mean(axis=0)
                    init_obj3d = np.column_stack(
                        [ref_2d, np.zeros(len(kp), dtype=np.float32)]
                    )
                    tr = COMTracker()
                    tr.initialise(ref_gray, init_pts, init_obj3d, ref_bbox=det["bbox"])
                    active_tracks[tid] = {
                        "tracker": tr,
                        "ref_kp": kp,
                        "ref_des": des,
                        "ref_bbox": det["bbox"],
                        "cls": det["cls_name"],
                    }
                    print(
                        f"ID{tid} {det['cls_name']}: {len(kp)} SIFT pts "
                        f"(conf={det['conf']:.2f}).",
                    )
                if last_viz is not None:
                    cv2.imshow("Reference Features", last_viz)
                if not active_tracks:
                    print("No tracks started — no SIFT features in any detection box.")

            # ------ t : manual ROI --------------------------------------------
            elif key == ord("t"):
                roi_frame = color_image.copy()
                print("Draw ROI: drag a rectangle around the tool.")
                print("  SPACE or ENTER — confirm    ESC — cancel")
                rx, ry, rw, rh = cv2.selectROI(
                    "Select ROI", roi_frame,
                    fromCenter=False, showCrosshair=True,
                )
                cv2.destroyWindow("Select ROI")

                if rw > 0 and rh > 0:
                    ref_color = roi_frame
                    ref_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
                    ref_bbox = (rx, ry, rx + rw, ry + rh)

                    ref_mask_saved = np.zeros(ref_gray.shape, dtype=np.uint8)
                    ref_mask_saved[ry : ry + rh, rx : rx + rw] = 255

                    ref_kp, ref_des, ref_viz = compute_reference_features(
                        ref_gray, ref_color, ref_mask_saved, ref_bbox,
                        max_features_holder[0],
                    )
                    cv2.imshow("Reference Features", ref_viz)

                    init_pts = np.array([[kp.pt] for kp in ref_kp], dtype=np.float32)
                    ref_2d = np.array([kp.pt for kp in ref_kp], dtype=np.float32)
                    ref_2d -= ref_2d.mean(axis=0)
                    init_obj3d = np.column_stack(
                        [ref_2d, np.zeros(len(ref_kp), dtype=np.float32)]
                    )

                    tid = next_track_id[0]
                    next_track_id[0] += 1
                    active_tracks.clear()
                    track_bbox_memory.clear()
                    track_bbox_memory[tid] = ref_bbox
                    tr = COMTracker()
                    if len(init_pts) > 0:
                        tr.initialise(ref_gray, init_pts, init_obj3d, ref_bbox=ref_bbox)
                        active_tracks[tid] = {
                            "tracker": tr,
                            "ref_kp": ref_kp,
                            "ref_des": ref_des,
                            "ref_bbox": ref_bbox,
                            "cls": "ROI",
                        }
                        print(f"Manual ROI — ID{tid}, tracking {len(init_pts)} points.")
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
    import argparse

    _ap = argparse.ArgumentParser(description="clean_full.py — v3 pipeline + optional SAM2")
    _ap.add_argument(
        "--sam2-only",
        action="store_true",
        help="Run standalone YOLO + SAM2 webcam demo (no depth / SIFT pipeline)",
    )
    _args = _ap.parse_args()
    if _args.sam2_only:
        from sam2_yolo_video import main as _sam2_main

        _sam2_main()
    else:
        main()
