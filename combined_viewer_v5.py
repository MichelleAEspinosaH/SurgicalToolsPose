# ******************************************************************************
#  combined_viewer_v5.py
#
#  Ultralytics YOLO instance segmentation + multi-object tracking.
#  Significantly simpler than v4: no SIFT, no optical flow, no manual
#  reference — YOLO handles detection, segmentation, and tracking entirely.
#
#  What's new compared to previous versions
#  -----------------------------------------
#  • RGB and depth frames are rotated 180° on read (camera mounted upside-down)
#  • Instance segmentation: each detected tool gets a per-pixel mask, drawn as
#    a semi-transparent coloured overlay with a crisp contour outline
#  • Ultralytics ByteTrack assigns a consistent numerical ID to each tool
#    across frames — ID persists through occlusion and brief disappearance
#  • Depth value (in mm) at each instance's centre is overlaid on the frame
#  • Fine-tuned model (train_yolo_seg.py) is loaded automatically if present;
#    falls back to the bundled yolo26n-seg.pt for immediate use without training
#
#  Training (one-time, optional but recommended)
#  -----------------------------------------------
#      python3 train_yolo_seg.py
#  This fine-tunes a YOLOv8-nano segmentation model on the Roboflow
#  surgical-tools dataset and saves it to:
#      runs/segment/surgical_tools_seg/weights/best.pt
#  v5 loads that path automatically on the next launch.
#
#  Display windows
#  ---------------
#  Tracking   : rotated RGB with segmentation masks, tracking IDs, depth
#  Depth      : rotated rainbow depth map
#
#  Controls
#  --------
#  q / ESC — quit
# ******************************************************************************

import os
import time
import threading

import cv2
import numpy as np
from ultralytics import YOLO
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

# --- Constants ----------------------------------------------------------------
ESC_KEY        = 27
PRINT_INTERVAL = 1        # seconds between depth console prints
MIN_DEPTH      = 20       # mm
MAX_DEPTH      = 10000    # mm

CONF_THRESHOLD = 0.35     # minimum detection confidence shown
IOU_THRESHOLD  = 0.45     # NMS overlap threshold
TRACKER        = "bytetrack.yaml"   # built into ultralytics; swap for botrack.yaml

MASK_ALPHA     = 0.40     # opacity of the segmentation fill (0 = invisible)

# Path to the fine-tuned segmentation model (produced by train_yolo_seg.py).
YOLO_SEG_MODEL_PATH = "runs/segment/surgical_tools_seg/weights/best.pt"
FALLBACK_SEG_MODEL  = "yolo26n-seg.pt"

# Classes to ignore even when detected — keeps hands/arms out of the overlay.
EXCLUDED_CLASSES = {"human", "person", "hand", "arm"}
# ------------------------------------------------------------------------------


# ==============================================================================
# Depth pipeline
# ==============================================================================

class TemporalFilter:
    """EMA blend to reduce depth sensor noise."""
    def __init__(self, alpha=0.5):
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
    get_latest()     → colourised BGR image (for display, already rotated 180°)
    get_latest_raw() → float32 depth array in mm (already rotated 180°)
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

            data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
            data = data.astype(np.float32) * scale
            data = np.where((data > MIN_DEPTH) & (data < MAX_DEPTH), data, 0).astype(np.uint16)
            data = self.temporal_filter.process(data)   # float32

            dist = float(data[h // 2, w // 2])
            now  = time.time()
            if now - last_print >= PRINT_INTERVAL:
                print(f"Center depth: {dist:.0f} mm")
                last_print = now

            img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

            # Rotate 180° — camera is mounted upside-down
            img_rot  = cv2.rotate(img,                  cv2.ROTATE_180)
            data_rot = cv2.rotate(data.astype(np.float32), cv2.ROTATE_180)

            with self.lock:
                self.latest_image    = img_rot
                self.latest_raw      = data_rot
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
# Drawing helpers
# ==============================================================================

def instance_color(track_id):
    """Consistent, visually distinct BGR colour for a given tracking ID."""
    hue = (int(track_id) * 67) % 180
    hsv = np.uint8([[[hue, 230, 220]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def depth_at_point(cx, cy, raw_depth, frame_w, frame_h):
    """
    Look up depth (mm) at image pixel (cx, cy), scaling to the depth frame
    resolution.  Returns 0 if invalid.
    """
    if raw_depth is None:
        return 0
    dh, dw = raw_depth.shape[:2]
    dx = int(round(cx * dw / frame_w))
    dy = int(round(cy * dh / frame_h))
    dx = np.clip(dx, 0, dw - 1)
    dy = np.clip(dy, 0, dh - 1)
    return float(raw_depth[dy, dx])


def draw_instances(frame, results, raw_depth):
    """
    Draws instance segmentation results onto frame:
      • semi-transparent coloured mask fill per instance
      • contour outline
      • bounding box
      • label: tracking ID, class name, confidence
      • depth value at the instance centre (if depth is available)

    Instances whose class name is in EXCLUDED_CLASSES are skipped.
    """
    out   = frame.copy()
    fh, fw = frame.shape[:2]

    if results is None or len(results) == 0:
        return out

    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return out

    boxes   = r.boxes
    n       = len(boxes)
    has_ids = boxes.id is not None

    # Resize masks to frame dimensions if present
    masks_np = None
    if r.masks is not None:
        # masks.data shape: (N, mh, mw) float32 in [0, 1]
        raw_masks = r.masks.data.cpu().numpy()
        masks_np  = np.stack([
            cv2.resize(m, (fw, fh), interpolation=cv2.INTER_LINEAR)
            for m in raw_masks
        ])   # (N, fh, fw)

    for i in range(n):
        cls_id   = int(boxes.cls[i])
        cls_name = r.names.get(cls_id, str(cls_id))
        conf     = float(boxes.conf[i])

        # Skip excluded classes and low-confidence detections
        if cls_name.lower() in EXCLUDED_CLASSES:
            continue
        if conf < CONF_THRESHOLD:
            continue

        track_id = int(boxes.id[i]) if has_ids and boxes.id[i] is not None else i
        color    = instance_color(track_id)

        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        cx, cy          = (x1 + x2) // 2, (y1 + y2) // 2

        # --- Segmentation mask fill ------------------------------------------
        if masks_np is not None:
            mask_bin = (masks_np[i] > 0.5).astype(np.uint8)

            # Semi-transparent colour fill
            colour_layer        = out.copy()
            colour_layer[mask_bin == 1] = color
            out = cv2.addWeighted(colour_layer, MASK_ALPHA, out, 1 - MASK_ALPHA, 0)

            # Contour outline
            cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color, 2)

        # --- Bounding box ----------------------------------------------------
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # --- Label -----------------------------------------------------------
        label = f"ID {track_id}  {cls_name}  {conf:.2f}"
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_y = max(y1 - 6, lh + 4)
        cv2.rectangle(out,
                      (x1, label_y - lh - 4),
                      (x1 + lw + 4, label_y + baseline),
                      color, -1)
        cv2.putText(out, label, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                    lineType=cv2.LINE_AA)

        # --- Depth at instance centre ----------------------------------------
        d = depth_at_point(cx, cy, raw_depth, fw, fh)
        if d > 0:
            depth_label = f"{d:.0f} mm"
            cv2.putText(out, depth_label, (cx - 28, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                        lineType=cv2.LINE_AA)
            cv2.putText(out, depth_label, (cx - 28, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        lineType=cv2.LINE_AA)

    # Instance count HUD
    visible = sum(
        1 for i in range(n)
        if r.names.get(int(boxes.cls[i]), "").lower() not in EXCLUDED_CLASSES
        and float(boxes.conf[i]) >= CONF_THRESHOLD
    )
    cv2.putText(out, f"Instances: {visible}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                lineType=cv2.LINE_AA)

    return out


# ==============================================================================
# Main
# ==============================================================================

def main():
    # --- Load segmentation model (fine-tuned → fallback) ---------------------
    yolo_model = None

    if os.path.exists(YOLO_SEG_MODEL_PATH):
        try:
            yolo_model = YOLO(YOLO_SEG_MODEL_PATH)
            print(f"Fine-tuned seg model loaded: {YOLO_SEG_MODEL_PATH}")
            print(f"Classes: {yolo_model.names}")
        except Exception as e:
            print(f"Could not load fine-tuned model: {e}")

    if yolo_model is None:
        try:
            yolo_model = YOLO(FALLBACK_SEG_MODEL)
            print(f"Fine-tuned model not found — using fallback: {FALLBACK_SEG_MODEL}")
            print("Run train_yolo_seg.py to enable the surgical-tools model.")
            print(f"Classes: {yolo_model.names}")
        except Exception as e:
            print(f"Could not load fallback model: {e}")
            return

    # --- RGB camera -----------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open RGB camera.")
        return
    print("RGB camera opened.")

    # --- Depth sensor (Orbbec SDK) -------------------------------------------
    config   = Config()
    pipeline = Pipeline()
    try:
        profile_list  = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None
        depth_profile = profile_list.get_default_video_stream_profile()
        assert depth_profile is not None
        print(f"Depth profile: {depth_profile}")
        config.enable_stream(depth_profile)
    except Exception as e:
        print(f"Could not configure depth stream: {e}")
        cap.release()
        return

    pipeline.start(config)
    depth_reader = DepthReader(pipeline)

    print("\nRunning — press q or ESC to quit.")
    print(f"Tracker : {TRACKER}")
    print(f"Model   : {YOLO_SEG_MODEL_PATH if os.path.exists(YOLO_SEG_MODEL_PATH) else FALLBACK_SEG_MODEL}")

    while True:
        try:
            # --- Read RGB frame and rotate 180° -------------------------------
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed.")
                break
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # --- Read depth ---------------------------------------------------
            depth_display = depth_reader.get_latest()
            raw_depth     = depth_reader.get_latest_raw()

            if depth_display is not None:
                cv2.imshow("Depth", depth_display)

            # --- YOLO: track + segment ----------------------------------------
            # persist=True tells ByteTrack to maintain ID state between calls.
            results = yolo_model.track(
                source  = frame,
                persist = True,
                tracker = TRACKER,
                conf    = CONF_THRESHOLD,
                iou     = IOU_THRESHOLD,
                verbose = False,
            )

            # --- Draw and show ------------------------------------------------
            tracking_frame = draw_instances(frame, results, raw_depth)
            cv2.imshow("Tracking", tracking_frame)

            # --- Quit ---------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ESC_KEY:
                break

        except KeyboardInterrupt:
            break

    depth_reader.stop()
    cap.release()
    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    main()
