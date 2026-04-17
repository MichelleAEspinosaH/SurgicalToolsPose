# ******************************************************************************
#  combined_viewer_v2.py
#
#  Identical to combined_viewer.py, but uses a YOLOv8 detection model that has
#  been fine-tuned on the Roboflow surgical-tools dataset via train_yolo.py.
#
#  Run train_yolo.py first to produce the weights, then launch this file:
#      python3 train_yolo.py        # one-time training step
#      python3 combined_viewer_v2.py
#
#  Display windows:
#    - "RGB Camera"        : live color video (read with OpenCV)
#    - "Depth Viewer"      : live depth map (read with Orbbec SDK), shown as a
#                            rainbow color gradient (blue = far, red = close)
#    - "Reference Features": static snapshot shown after pressing 'r'; shows
#                            the YOLO bounding box and all SIFT keypoints on
#                            the detected object
#    - "Tracking"          : live RGB feed overlaid with:
#                              • each feature point's continuous trail
#                              • the Center of Mass (COM) with its own trail
#                              • X / Y / Z pose axes drawn at the COM
#                              • the reference bounding box warped to follow
#                                the object as it moves
#
#  How tracking works:
#    1. Press 'r' — YOLO detects the surgical tool using the fine-tuned model,
#       SIFT finds keypoints strictly inside the detected bounding box, and
#       those positions become the starting points for optical flow.
#    2. Every frame — Lucas-Kanade Optical Flow propagates each point from
#       the previous frame to the current one. No re-detection; paths are
#       continuous and unbroken.
#    3. If fewer than REINIT_RATIO of the original points survive, SIFT
#       matching is run again automatically to recover.
#    4. COM is the mean position of all active points, smoothed with EMA.
#    5. solvePnP estimates the 3D orientation of the object each frame.
#       The rotation and translation are also smoothed with EMA so the
#       axes never jump. useExtrinsicGuess keeps each solution anchored
#       to the previous one.
#
#  Keyboard controls:
#    r       — capture reference and start tracking (YOLO auto-detects the bbox)
#    t       — freeze the current frame and draw your own ROI with the mouse;
#              use this if YOLO did not draw a bounding box on the reference frame
#    q / ESC — quit
#
#  Depth runs in a background thread so it never blocks the RGB feed.
# ******************************************************************************

import time
import threading
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

# --- Constants ----------------------------------------------------------------
ESC_KEY         = 27
PRINT_INTERVAL  = 1       # seconds between depth prints
MIN_DEPTH       = 20      # cm
MAX_DEPTH       = 10000   # cm
TRAIL_LENGTH    = 50      # frames of history per point / COM
REINIT_RATIO    = 0.3     # re-init when fewer than 30 % of points remain
MAX_FEATURES    = 30      # default max SIFT keypoints to track (tunable via trackbar)
COM_SMOOTH      = 0.2     # EMA alpha for COM position  (lower = smoother)
POSE_SMOOTH     = 0.15    # EMA alpha for rvec / tvec   (lower = smoother)

# Path to the fine-tuned model produced by train_yolo.py.
# If training placed the file elsewhere, update this path to match.
YOLO_MODEL_PATH = "runs/detect/surgical_tools/weights/best.pt"

# YOLO class names that should never be selected as the tracked object.
# Detections whose class matches any entry here are silently ignored so that
# hands, arms, or full bodies never become the "object" the tracker follows.
EXCLUDED_CLASSES = {"human", "person", "hand", "arm"}

# Lucas-Kanade optical flow settings
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
    Blends each depth frame with the previous result to reduce sensor noise.
        result = alpha * new_frame + (1 - alpha) * previous_frame
    """
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


class DepthReader:
    """
    Reads Orbbec depth frames in a background thread so the RGB loop is
    never blocked or delayed.  Call get_latest() to get the most recent
    colourised depth image.
    """
    def __init__(self, pipeline):
        self.pipeline        = pipeline
        self.latest_image    = None
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

            data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
            data = data.astype(np.float32) * scale
            data = np.where((data > MIN_DEPTH) & (data < MAX_DEPTH), data, 0).astype(np.uint16)
            data = self.temporal_filter.process(data)

            dist = data[h // 2, w // 2]
            now  = time.time()
            if now - last_print >= PRINT_INTERVAL:
                print("Center distance:", dist, "cm")
                last_print = now

            img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

            with self.lock:
                self.latest_image    = img
                self.latest_distance = dist

    def get_latest(self):
        with self.lock:
            return self.latest_image

    def stop(self):
        self.running = False
        self._thread.join()


# ==============================================================================
# YOLO + SIFT helpers
# ==============================================================================

def get_object_mask(color_image, yolo_model):
    """
    Runs the fine-tuned YOLO detection model and returns:
      mask     — binary mask WHITE (255) inside the detected bounding box
      bbox     — (x1, y1, x2, y2) of the highest-confidence detection
      cls_name — name of the detected class (e.g. "scissors")
      conf     — detection confidence (0–1)

    The fine-tuned model is a detection model (not segmentation), so we
    always use the bounding box to build the mask.  This keeps SIFT features
    strictly inside the detected tool region.

    Returns (None, None, None, None) if nothing is detected.
    """
    h, w    = color_image.shape[:2]
    results = yolo_model(color_image, verbose=False)

    if not results or len(results[0].boxes) == 0:
        return None, None, None, None

    boxes = results[0].boxes

    # Collect indices of detections whose class is NOT in EXCLUDED_CLASSES,
    # sorted by descending confidence so we always pick the best valid one.
    valid = [
        i for i in range(len(boxes))
        if yolo_model.names.get(int(boxes.cls[i]), "").lower()
           not in EXCLUDED_CLASSES
    ]
    if not valid:
        return None, None, None, None

    best     = max(valid, key=lambda i: float(boxes.conf[i]))
    x1, y1, x2, y2 = map(int, boxes.xyxy[best])
    cls_id   = int(boxes.cls[best])
    conf     = float(boxes.conf[best])
    cls_name = yolo_model.names.get(cls_id, str(cls_id))

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    return mask, (x1, y1, x2, y2), cls_name, conf


def compute_reference_features(ref_gray, ref_color, ref_mask, bbox, max_features):
    """
    Detects up to max_features SIFT keypoints inside the YOLO bounding box on
    the reference frame.  Returns (kp, des, annotated_image) with bounding box
    (blue) and keypoints (green) drawn.
    """
    sift    = cv2.SIFT_create(nfeatures=max_features)
    kp, des = sift.detectAndCompute(ref_gray, ref_mask)

    viz = ref_color.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(viz, "reference bbox", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    viz = cv2.drawKeypoints(viz, kp, None, color=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des, viz


def get_sift_points_in_frame(ref_kp, ref_des, query_gray, query_mask, max_features):
    """
    Runs SIFT matching between the reference descriptors and the current frame,
    restricting detection to the query_mask region (the YOLO bounding box).

    Returns:
      live_pts   — (N, 1, 2) float32 pixel positions in the current frame
      obj_pts_3d — (N, 3) float32 object-space coordinates (centred, z = 0)
    Returns (None, None) if matching fails.
    """
    sift      = cv2.SIFT_create()
    kp2, des2 = sift.detectAndCompute(query_gray, query_mask)

    if ref_des is None or des2 is None or len(ref_des) < 2 or len(des2) < 2:
        return None, None

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50)
    )
    matches = flann.knnMatch(ref_des, des2, k=2)
    good    = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if not good:
        return None, None

    good = sorted(good, key=lambda m: m.distance)[:max_features]

    live_pts = np.array([[kp2[m.trainIdx].pt] for m in good], dtype=np.float32)
    ref_2d   = np.array([ref_kp[m.queryIdx].pt for m in good], dtype=np.float32)
    ref_2d  -= ref_2d.mean(axis=0)
    obj_pts_3d = np.column_stack(
        [ref_2d, np.zeros(len(good), dtype=np.float32)]
    )
    return live_pts, obj_pts_3d


def point_color(idx):
    """Unique BGR colour per point index, spread around the HSV wheel."""
    hue = (idx * 37) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ==============================================================================
# Tracker
# ==============================================================================

class COMTracker:
    """
    Tracks feature points continuously with Lucas-Kanade Optical Flow,
    computes the Center of Mass (COM) of all active points, and estimates
    the object's 3-D pose with solvePnP so that X/Y/Z axes can be drawn.

    Smoothing strategy
    ------------------
    • COM position : EMA with alpha = COM_SMOOTH
    • rvec / tvec  : EMA with alpha = POSE_SMOOTH
    • solvePnP uses useExtrinsicGuess = True after the first frame so each
      solution stays anchored to the previous one — no sudden flips.
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
        self.smooth_rvec    = None
        self.smooth_tvec    = None
        self.axis_length    = 50.0
        self.ref_bbox       = None

    def initialise(self, gray, live_pts, obj_pts_3d, ref_bbox=None):
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
        self.smooth_com  = None
        self.smooth_rvec = None
        self.smooth_tvec = None
        if ref_bbox is not None:
            self.ref_bbox = ref_bbox
        spread = float(np.std(obj_pts_3d[:, :2]))
        self.axis_length = max(30.0, spread * 1.5)

    def update(self, gray, cam_matrix, dist_coeffs, live_bbox=None):
        """
        live_bbox — (x1, y1, x2, y2) of the current YOLO detection.
        When provided, any point that has drifted outside that rectangle is
        immediately discarded so features never accumulate on the background
        or on the operator's hand.
        """
        if self.points is None or len(self.points) == 0 or self.prev_gray is None:
            return

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None, **LK_PARAMS
        )
        surviving = [i for i, st in enumerate(status) if st[0] == 1]

        # If a live bounding box is known, drop points that have wandered out.
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

        self.points    = np.array(good_live, dtype=np.float32) if good_live \
                         else np.empty((0, 1, 2), dtype=np.float32)
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
                        self.smooth_rvec = POSE_SMOOTH * rvec + (1 - POSE_SMOOTH) * self.smooth_rvec
                        self.smooth_tvec = POSE_SMOOTH * tvec + (1 - POSE_SMOOTH) * self.smooth_tvec
            except Exception:
                pass

    def draw(self, image, cam_matrix, dist_coeffs, live_bbox=None):
        out = image.copy()

        # --- Live YOLO bounding box (green) -----------------------------------
        # Drawn from a fresh YOLO detection this frame so it always reflects
        # where the model currently sees the tool.
        if live_bbox is not None:
            x1, y1, x2, y2 = live_bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, "live bbox", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Feature-point trails
        for idx, trail in self.trails.items():
            pts = list(trail)
            if not pts:
                continue
            color = point_color(idx)
            n = len(pts)
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

        # COM crosshair and pose axes
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
                L = self.axis_length
                axes_3d = np.array([
                    [0, 0,  0],
                    [L, 0,  0],
                    [0, L,  0],
                    [0, 0, -L],
                ], dtype=np.float32)
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

        # Reference bounding box warped to current frame via homography
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
                    cv2.polylines(out, [warped.astype(np.int32)],
                                  isClosed=True, color=(255, 0, 0), thickness=2)

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
        self.smooth_rvec    = None
        self.smooth_tvec    = None


# ==============================================================================
# Main
# ==============================================================================

def main():
    # --- Load fine-tuned YOLO model (with fallback) ---------------------------
    import os
    yolo_model = None

    if os.path.exists(YOLO_MODEL_PATH):
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print("Fine-tuned YOLO model loaded:", YOLO_MODEL_PATH)
            print("Classes:", yolo_model.names)
        except Exception as e:
            print(f"Could not load fine-tuned model ({YOLO_MODEL_PATH}): {e}")

    if yolo_model is None:
        # Fine-tuned weights don't exist yet (run train_yolo.py to create them).
        # Fall back to the general surgical model so bounding boxes still appear.
        FALLBACK_MODEL = "yolo26n-seg.pt"
        try:
            yolo_model = YOLO(FALLBACK_MODEL)
            print(f"Fine-tuned model not found at '{YOLO_MODEL_PATH}'.")
            print(f"Using fallback model: {FALLBACK_MODEL}")
            print("Run train_yolo.py once to enable the surgical-tools fine-tuned model.")
            print("Classes:", yolo_model.names)
        except Exception as e:
            print(f"Could not load fallback model ({FALLBACK_MODEL}): {e}")
            print("Object detection will be disabled.")
            yolo_model = None

    # --- Open RGB camera with OpenCV ------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open RGB camera.")
        return
    print("RGB camera opened.")

    # --- Open depth sensor with the Orbbec SDK --------------------------------
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

    # --- Camera matrix (approximate, from frame size) -------------------------
    ret0, frame0 = cap.read()
    if ret0:
        fh, fw = frame0.shape[:2]
        focal  = fw
        cam_matrix = np.array([
            [focal, 0,     fw / 2],
            [0,     focal, fh / 2],
            [0,     0,     1     ],
        ], dtype=np.float32)
    else:
        cam_matrix = None
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    print("Press 'r' to capture the reference image and start tracking (YOLO auto-detects).")
    print("Press 't' to freeze the frame and draw your own ROI if YOLO misses the object.")
    print("Press 'q' or ESC to quit.")

    reference_gray = None
    reference_kp   = None
    reference_des  = None
    ref_mask_saved = None
    ref_bbox       = None

    tracker = COMTracker()

    max_features_holder = [MAX_FEATURES]
    cv2.namedWindow("Tracking")
    cv2.createTrackbar(
        "Max points", "Tracking",
        MAX_FEATURES, 200,
        lambda v: max_features_holder.__setitem__(0, max(5, v)),
    )

    while True:
        try:
            # RGB frame
            ret, color_image = cap.read()
            if not ret:
                print("Frame grab failed on RGB camera.")
                break
            color_image = cv2.rotate(color_image, cv2.ROTATE_180)
            cv2.imshow("RGB Camera", color_image)

            # Depth frame
            depth_image = depth_reader.get_latest()
            if depth_image is not None:
                cv2.imshow("Depth Viewer", depth_image)

            # Tracking
            if reference_gray is not None:
                query_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                # Run YOLO every frame for a fresh live bounding box
                if yolo_model:
                    live_mask, live_bbox, _, _ = get_object_mask(color_image, yolo_model)
                else:
                    live_mask, live_bbox = ref_mask_saved, ref_bbox

                if tracker.needs_reinit:
                    print("Reinitialising tracking...")
                    pts, obj3d = get_sift_points_in_frame(
                        reference_kp, reference_des, query_gray, live_mask,
                        max_features_holder[0],
                    )
                    if pts is not None and len(pts) > 0:
                        tracker.initialise(query_gray, pts, obj3d)
                    else:
                        print("Could not find enough points to reinitialise.")
                else:
                    tracker.update(query_gray, cam_matrix, dist_coeffs,
                                   live_bbox=live_bbox)

                tracking_image = tracker.draw(color_image, cam_matrix, dist_coeffs,
                                              live_bbox=live_bbox)
                cv2.imshow("Tracking", tracking_image)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ESC_KEY:
                break

            elif key == ord('r'):
                ref_color      = color_image.copy()
                reference_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)

                if yolo_model:
                    ref_mask_saved, ref_bbox, ref_cls, ref_conf = get_object_mask(ref_color, yolo_model)
                    if ref_bbox is not None:
                        print(f"  YOLO detected: {ref_cls}  conf={ref_conf:.2f}  bbox={ref_bbox}")
                else:
                    ref_mask_saved, ref_bbox = None, None

                reference_kp, reference_des, ref_viz = compute_reference_features(
                    reference_gray, ref_color, ref_mask_saved, ref_bbox,
                    max_features_holder[0],
                )
                cv2.imshow("Reference Features", ref_viz)

                init_pts = np.array([[kp.pt] for kp in reference_kp], dtype=np.float32)
                ref_2d   = np.array([kp.pt for kp in reference_kp], dtype=np.float32)
                ref_2d  -= ref_2d.mean(axis=0)
                init_obj3d = np.column_stack(
                    [ref_2d, np.zeros(len(reference_kp), dtype=np.float32)]
                )

                tracker.reset()
                if len(init_pts) > 0:
                    tracker.initialise(reference_gray, init_pts, init_obj3d, ref_bbox=ref_bbox)
                    print(f"Reference captured — tracking {len(init_pts)} points.")
                else:
                    print("Reference captured but no points found.")

                if ref_mask_saved is None:
                    print("(YOLO found no object — using full image for SIFT.)")

            elif key == ord('t'):
                # ---- Manual ROI selection ----------------------------------------
                # Freeze the current frame and let the user drag a rectangle.
                # Useful when YOLO doesn't fire or detects the wrong object.
                roi_frame = color_image.copy()
                print("Draw ROI: drag a rectangle around the tool.")
                print("  SPACE or ENTER — confirm    ESC — cancel")
                rx, ry, rw, rh = cv2.selectROI(
                    "Select ROI",
                    roi_frame,
                    fromCenter=False,
                    showCrosshair=True,
                )
                cv2.destroyWindow("Select ROI")

                if rw > 0 and rh > 0:
                    ref_color      = roi_frame
                    reference_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
                    ref_bbox       = (rx, ry, rx + rw, ry + rh)

                    # Build a mask that covers only the selected rectangle
                    ref_mask_saved = np.zeros(reference_gray.shape, dtype=np.uint8)
                    ref_mask_saved[ry:ry + rh, rx:rx + rw] = 255

                    reference_kp, reference_des, ref_viz = compute_reference_features(
                        reference_gray, ref_color, ref_mask_saved, ref_bbox,
                        max_features_holder[0],
                    )
                    cv2.imshow("Reference Features", ref_viz)

                    init_pts = np.array(
                        [[kp.pt] for kp in reference_kp], dtype=np.float32
                    )
                    ref_2d = np.array(
                        [kp.pt for kp in reference_kp], dtype=np.float32
                    )
                    ref_2d -= ref_2d.mean(axis=0)
                    init_obj3d = np.column_stack(
                        [ref_2d, np.zeros(len(reference_kp), dtype=np.float32)]
                    )

                    tracker.reset()
                    if len(init_pts) > 0:
                        tracker.initialise(
                            reference_gray, init_pts, init_obj3d, ref_bbox=ref_bbox
                        )
                        print(f"Manual ROI captured — tracking {len(init_pts)} points.")
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
