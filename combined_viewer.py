# ******************************************************************************
#  combined_viewer.py
#
#  Combines depth_frame.py, rgb_test.py, and the SIFT query logic from
#  pipeline.py into a single live viewer.
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
#
#  How tracking works:
#    1. Press 'r' — YOLO detects the object, SIFT finds keypoints on it,
#       and SIFT matching locates those points in the live frame right now.
#       Those pixel positions become the starting points for optical flow.
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
#    r       — capture reference and start tracking
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
YOLO_MODEL_PATH = "yolo26n-seg.pt"
TRAIL_LENGTH    = 50      # frames of history per point / COM
REINIT_RATIO    = 0.3     # re-init when fewer than 30 % of points remain
MAX_FEATURES    = 30      # default max SIFT keypoints to track (tunable via trackbar)
COM_SMOOTH      = 0.2     # EMA alpha for COM position  (lower = smoother)
POSE_SMOOTH     = 0.15    # EMA alpha for rvec / tvec   (lower = smoother)

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

            w, h   = depth_frame.get_width(), depth_frame.get_height()
            scale  = depth_frame.get_depth_scale()

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
    Returns (mask, bbox) where mask is WHITE over the detected object and
    bbox is (x1, y1, x2, y2).  Uses pixel-level segmentation when available,
    falls back to bounding box.  Returns (None, None) if nothing detected.
    """
    h, w    = color_image.shape[:2]
    results = yolo_model(color_image, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None, None

    boxes    = results[0].boxes
    best     = int(boxes.conf.argmax())
    x1, y1, x2, y2 = map(int, boxes.xyxy[best])
    mask = np.zeros((h, w), dtype=np.uint8)

    if results[0].masks is not None:
        seg = results[0].masks.data[best].cpu().numpy()
        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        mask[seg > 0.5] = 255
    else:
        mask[y1:y2, x1:x2] = 255

    return mask, (x1, y1, x2, y2)


def compute_reference_features(ref_gray, ref_color, ref_mask, bbox, max_features):
    """
    Detects up to max_features SIFT keypoints inside the YOLO mask on the
    reference frame.  Returns (kp, des, annotated_image) with bounding box
    (blue) and keypoints (green) drawn.
    """
    sift   = cv2.SIFT_create(nfeatures=max_features)
    kp, des = sift.detectAndCompute(ref_gray, ref_mask)

    viz = ref_color.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 0, 0), 2)
    viz = cv2.drawKeypoints(viz, kp, None, color=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des, viz


def get_sift_points_in_frame(ref_kp, ref_des, query_gray, query_mask, max_features):
    """
    Runs SIFT matching between the reference descriptors and the current frame.

    Returns:
      live_pts   — (N, 1, 2) float32 pixel positions in the current frame
      obj_pts_3d — (N, 3) float32 object-space coordinates derived from the
                   reference keypoint positions (centred, z = 0).
                   These are the "known 3-D points" for solvePnP.
    Returns (None, None) if matching fails.

    Only the top max_features matches (ranked by match quality / distance)
    are kept, so the number of tracked points stays under control.
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

    # Keep only the best max_features matches, ranked by match distance
    # (lower distance = more confident / closer descriptor match)
    good = sorted(good, key=lambda m: m.distance)[:max_features]

    live_pts = np.array([[kp2[m.trainIdx].pt] for m in good], dtype=np.float32)

    # Build 3-D object points from the reference image layout (z = 0 plane).
    # Centering them makes the origin coincide with the object's geometric
    # centre, which is where we want to draw the axes.
    ref_2d    = np.array([ref_kp[m.queryIdx].pt for m in good], dtype=np.float32)
    ref_2d   -= ref_2d.mean(axis=0)
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

    Why optical flow instead of per-frame SIFT
    ------------------------------------------
    SIFT detects points independently each frame; there is no memory of
    where a point was the frame before, so trails are broken and jumpy.
    Optical flow propagates the exact same pixel patches frame-to-frame,
    giving continuous, unbroken paths.
    """

    def __init__(self):
        self.prev_gray      = None
        self.points         = None        # (N, 1, 2) float32 — live positions
        self.obj_pts_3d     = None        # (N, 3) float32 — object-space coords
        self.original_count = 0
        self.trails         = {}          # idx -> deque[(x, y)]
        self.init_positions = {}          # idx -> (x, y) at initialise time
        self.com_trail      = deque(maxlen=TRAIL_LENGTH)
        self.smooth_com     = None        # EMA-smoothed (x, y)
        self.smooth_rvec    = None        # EMA-smoothed rotation vector
        self.smooth_tvec    = None        # EMA-smoothed translation vector
        self.axis_length    = 50.0        # object-space units for drawn axes
        self.ref_bbox       = None        # (x1,y1,x2,y2) from YOLO at capture time

    # ------------------------------------------------------------------
    def initialise(self, gray, live_pts, obj_pts_3d, ref_bbox=None):
        """
        Set starting positions (from SIFT matching) and reset all state.
        gray       — current grayscale frame
        live_pts   — (N, 1, 2) float32 starting pixel positions
        obj_pts_3d — (N, 3) float32 corresponding object-space points
        ref_bbox   — optional (x1,y1,x2,y2) YOLO bounding box to carry forward
        """
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
            self.ref_bbox = ref_bbox   # keep previous bbox if none supplied

        # Scale axis length to roughly match the spread of the object points
        spread = float(np.std(obj_pts_3d[:, :2]))
        self.axis_length = max(30.0, spread * 1.5)

    # ------------------------------------------------------------------
    def update(self, gray, cam_matrix, dist_coeffs):
        """
        Advance tracking by one frame.
          1. Optical flow: move each point from prev frame to current frame.
          2. Drop points optical flow could not track.
          3. Update COM with EMA smoothing.
          4. Run solvePnP on surviving points; smooth rvec/tvec with EMA.
        """
        if self.points is None or len(self.points) == 0 or self.prev_gray is None:
            return

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None, **LK_PARAMS
        )
        surviving = [i for i, st in enumerate(status) if st[0] == 1]

        good_live = []
        good_obj  = []
        for i in surviving:
            x, y = float(new_pts[i][0][0]), float(new_pts[i][0][1])
            good_live.append(new_pts[i])
            good_obj.append(self.obj_pts_3d[i])
            self.trails[i].append((x, y))

        self.points     = np.array(good_live, dtype=np.float32) if good_live \
                          else np.empty((0, 1, 2), dtype=np.float32)
        self.prev_gray  = gray.copy()

        # --- COM with EMA smoothing ----------------------------------------
        tips = [self.trails[i][-1] for i in surviving if self.trails[i]]
        if tips:
            raw_x = float(np.mean([p[0] for p in tips]))
            raw_y = float(np.mean([p[1] for p in tips]))
            if self.smooth_com is None:
                self.smooth_com = (raw_x, raw_y)
            else:
                sx = COM_SMOOTH * raw_x + (1 - COM_SMOOTH) * self.smooth_com[0]
                sy = COM_SMOOTH * raw_y + (1 - COM_SMOOTH) * self.smooth_com[1]
                self.smooth_com = (sx, sy)
            self.com_trail.append(self.smooth_com)

        # --- Pose estimation (solvePnP) with EMA smoothing -----------------
        # Need at least 4 points; camera matrix must be available.
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
                        # EMA on rotation and translation vectors
                        self.smooth_rvec = POSE_SMOOTH * rvec + (1 - POSE_SMOOTH) * self.smooth_rvec
                        self.smooth_tvec = POSE_SMOOTH * tvec + (1 - POSE_SMOOTH) * self.smooth_tvec
            except Exception:
                pass  # solvePnP can fail on degenerate configs; silently skip

    # ------------------------------------------------------------------
    def draw(self, image, cam_matrix, dist_coeffs):
        """
        Render onto a copy of image:
          • Coloured trails for each feature point (fading older segments)
          • COM trail in cyan, COM crosshair, COM coordinates
          • X (red) / Y (green) / Z (blue) pose axes anchored at the COM
        """
        out = image.copy()

        # --- Feature-point trails ------------------------------------------
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

        # --- COM trail -------------------------------------------------------
        com_pts = list(self.com_trail)
        n = len(com_pts)
        for i in range(1, n):
            brightness = i / n
            c   = (0, int(255 * brightness), int(255 * brightness))
            p1  = (int(com_pts[i - 1][0]), int(com_pts[i - 1][1]))
            p2  = (int(com_pts[i][0]),     int(com_pts[i][1]))
            cv2.line(out, p1, p2, c, 2)

        # --- COM crosshair and axes ----------------------------------------
        if com_pts:
            cx, cy = int(com_pts[-1][0]), int(com_pts[-1][1])

            # Crosshair
            arm = 14
            cv2.line(out,   (cx - arm, cy), (cx + arm, cy), (0, 255, 255), 2)
            cv2.line(out,   (cx, cy - arm), (cx, cy + arm), (0, 255, 255), 2)
            cv2.circle(out, (cx, cy), 10, (0, 255, 255), 2)
            cv2.putText(out, f"COM ({cx}, {cy})",
                        (cx + 14, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Pose axes — only when solvePnP has a solution and camera matrix exists
            if self.smooth_rvec is not None and cam_matrix is not None:
                L = self.axis_length
                # 3-D axis endpoints in object space
                axes_3d = np.array([
                    [0, 0,  0],   # origin
                    [L, 0,  0],   # +X
                    [0, L,  0],   # +Y
                    [0, 0, -L],   # +Z (toward camera in OpenCV convention)
                ], dtype=np.float32)

                try:
                    proj, _ = cv2.projectPoints(
                        axes_3d, self.smooth_rvec, self.smooth_tvec,
                        cam_matrix, dist_coeffs,
                    )
                    proj = proj.reshape(-1, 2)

                    # The solvePnP origin projected to screen; shift so axes
                    # originate exactly at the COM crosshair.
                    shift  = np.array([cx, cy], dtype=np.float64) - proj[0]
                    origin = (cx, cy)
                    x_tip  = tuple((proj[1] + shift).astype(int))
                    y_tip  = tuple((proj[2] + shift).astype(int))
                    z_tip  = tuple((proj[3] + shift).astype(int))

                    cv2.arrowedLine(out, origin, x_tip, (0,   0, 255), 2, tipLength=0.25)
                    cv2.arrowedLine(out, origin, y_tip, (0, 255,   0), 2, tipLength=0.25)
                    cv2.arrowedLine(out, origin, z_tip, (255, 0,   0), 2, tipLength=0.25)
                    cv2.putText(out, "X", (x_tip[0] + 4, x_tip[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,   0, 255), 1)
                    cv2.putText(out, "Y", (y_tip[0] + 4, y_tip[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,   0), 1)
                    cv2.putText(out, "Z", (z_tip[0] + 4, z_tip[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,  0,   0), 1)
                except Exception:
                    pass

        # --- Reference bounding box warped to current frame ------------------
        # Build a homography from the initial keypoint positions (at capture
        # time) to their current optical-flow positions.  That homography is
        # then applied to the four corners of the YOLO reference bbox so the
        # rectangle follows the object as it moves, scales, and rotates.
        if self.ref_bbox is not None and len(self.init_positions) >= 4:
            common = [i for i in self.init_positions if i in self.trails and self.trails[i]]
            if len(common) >= 4:
                src = np.array([self.init_positions[i] for i in common], dtype=np.float32)
                dst = np.array([self.trails[i][-1]     for i in common], dtype=np.float32)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if H is not None:
                    x1, y1, x2, y2 = self.ref_bbox
                    corners = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        dtype=np.float32,
                    ).reshape(-1, 1, 2)
                    warped  = cv2.perspectiveTransform(corners, H)
                    cv2.polylines(out, [warped.astype(np.int32)],
                                  isClosed=True, color=(255, 0, 0), thickness=2)

        # Active point count
        cv2.putText(out, f"Tracking: {self.active_count} pts",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    # ------------------------------------------------------------------
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
    # --- Load YOLO model ------------------------------------------------------
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded:", YOLO_MODEL_PATH)
    except Exception as e:
        print("Warning: Could not load YOLO model —", e)
        yolo_model = None

    # --- Open RGB camera with OpenCV ------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open RGB camera. Try changing the index in cv2.VideoCapture(0).")
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

    # --- Camera matrix --------------------------------------------------------
    # Approximate intrinsics from the frame size (no calibration file needed).
    # A proper calibration would give better axis accuracy.
    ret0, frame0 = cap.read()
    if ret0:
        fh, fw    = frame0.shape[:2]
        focal     = fw                        # rough estimate: focal ≈ image width
        cam_matrix = np.array([
            [focal, 0,     fw / 2],
            [0,     focal, fh / 2],
            [0,     0,     1     ],
        ], dtype=np.float32)
    else:
        cam_matrix = None
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    print("Press 'r' to capture the reference image and start tracking.")
    print("Press 'q' or ESC to quit.")

    reference_gray = None
    reference_kp   = None
    reference_des  = None
    ref_mask_saved = None

    tracker = COMTracker()

    # Trackbar for live point-count control.
    # A mutable list lets the callback update the value in place.
    # Range: 5 – 200 points.  Changing it takes effect on the next 'r' press
    # or automatic reinitialisation.
    max_features_holder = [MAX_FEATURES]
    cv2.namedWindow("Tracking")
    cv2.createTrackbar(
        "Max points", "Tracking",
        MAX_FEATURES, 200,
        lambda v: max_features_holder.__setitem__(0, max(5, v)),
    )

    while True:
        try:
            # ------------------------------------------------------------------
            # RGB frame
            # ------------------------------------------------------------------
            ret, color_image = cap.read()
            if not ret:
                print("Frame grab failed on RGB camera.")
                break
            cv2.imshow("RGB Camera", color_image)

            # ------------------------------------------------------------------
            # Depth frame
            # ------------------------------------------------------------------
            depth_image = depth_reader.get_latest()
            if depth_image is not None:
                cv2.imshow("Depth Viewer", depth_image)

            # ------------------------------------------------------------------
            # Continuous optical flow tracking + pose estimation
            # ------------------------------------------------------------------
            if reference_gray is not None:
                query_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                if tracker.needs_reinit:
                    print("Reinitialising tracking (too many points lost)...")
                    pts, obj3d = get_sift_points_in_frame(
                        reference_kp, reference_des, query_gray, ref_mask_saved,
                        max_features_holder[0],
                    )
                    if pts is not None and len(pts) > 0:
                        tracker.initialise(query_gray, pts, obj3d)
                    else:
                        print("Could not find enough points to reinitialise.")
                else:
                    tracker.update(query_gray, cam_matrix, dist_coeffs)

                tracking_image = tracker.draw(color_image, cam_matrix, dist_coeffs)
                cv2.imshow("Tracking", tracking_image)

            # ------------------------------------------------------------------
            # Key handling
            # ------------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ESC_KEY:
                break

            elif key == ord('r'):
                ref_color      = color_image.copy()
                reference_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)

                if yolo_model:
                    ref_mask_saved, ref_bbox = get_object_mask(ref_color, yolo_model)
                else:
                    ref_mask_saved, ref_bbox = None, None

                reference_kp, reference_des, ref_viz = compute_reference_features(
                    reference_gray, ref_color, ref_mask_saved, ref_bbox,
                    max_features_holder[0],
                )
                cv2.imshow("Reference Features", ref_viz)

                # The reference IS the current frame, so the keypoint positions
                # from SIFT are already the correct starting points for optical
                # flow — no second matching pass needed.
                init_pts = np.array([[kp.pt] for kp in reference_kp], dtype=np.float32)

                # Build object-space 3-D points from reference keypoint layout
                # (centred so the origin sits at the object's geometric centre).
                ref_2d = np.array([kp.pt for kp in reference_kp], dtype=np.float32)
                ref_2d -= ref_2d.mean(axis=0)
                init_obj3d = np.column_stack(
                    [ref_2d, np.zeros(len(reference_kp), dtype=np.float32)]
                )

                tracker.reset()
                if len(init_pts) > 0:
                    tracker.initialise(reference_gray, init_pts, init_obj3d, ref_bbox=ref_bbox)
                    print(f"Reference captured — tracking {len(init_pts)} points.")
                else:
                    print("Reference captured but no matching points found yet.")

                if ref_mask_saved is None:
                    print("(YOLO found no object — using full image for SIFT.)")

        except KeyboardInterrupt:
            break

    depth_reader.stop()
    cap.release()
    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    main()
