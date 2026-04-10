# ******************************************************************************
#  combined_viewer_v6.py
#
#  Live RGB (OpenCV) + Orbbec depth; both previews rotated 180°.
#  Object detection with externally fine-tuned YOLO (same style as testyolo.py:
#  model(..., conf=0.5), results.plot()).
#
#  Default weights: runs/detect/surgical_tools_ft/run1/weights/best.pt
#  Override: YOLO_MODEL_PATH env or --weights path/to/best.pt
#
#  Depth runs in a background thread so it never blocks the RGB feed.
#
#  Keys: q / ESC — quit
# ******************************************************************************

import argparse
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

# --- Constants ----------------------------------------------------------------
ESC_KEY         = 27
PRINT_INTERVAL  = 1
MIN_DEPTH       = 20
MAX_DEPTH       = 10000
# Match testyolo.py
YOLO_CONF       = 0.5


def default_yolo_model_path():
    for candidate in (
        Path("runs/detect/surgical_tools_ft/run1/weights/best.pt"),
        Path("runs/detect/surgical_tools_v6/weights/best.pt"),
        Path("runs/segment/surgical_tools_v6/weights/best.pt"),
    ):
        if candidate.is_file():
            return str(candidate)
    return "runs/detect/surgical_tools_ft/run1/weights/best.pt"


YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH") or default_yolo_model_path()
# ------------------------------------------------------------------------------


class TemporalFilter:
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
    """Background Orbbec depth reader; get_latest() returns a colourised BGR image."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.latest_image = None
        self.latest_distance = None
        self.lock = threading.Lock()
        self.running = True
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

            w, h = depth_frame.get_width(), depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
            data = data.astype(np.float32) * scale
            data = np.where((data > MIN_DEPTH) & (data < MAX_DEPTH), data, 0).astype(np.uint16)
            data = self.temporal_filter.process(data)

            dist = data[h // 2, w // 2]
            now = time.time()
            if now - last_print >= PRINT_INTERVAL:
                print("Center distance:", dist, "cm")
                last_print = now

            img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

            with self.lock:
                self.latest_image = img
                self.latest_distance = dist

    def get_latest(self):
        with self.lock:
            return self.latest_image

    def stop(self):
        self.running = False
        self._thread.join()


def draw_yolo_detections(bgr_image, yolo_model, conf=YOLO_CONF):
    """Same pattern as testyolo.py: infer with conf, return Ultralytics plot() overlay."""
    if yolo_model is None:
        return bgr_image
    r = yolo_model(bgr_image, conf=conf, verbose=False)[0]
    plotted = r.plot()
    return plotted


def run_live_viewer(yolo_weights=None, yolo_conf=YOLO_CONF):
    weights_path = yolo_weights or YOLO_MODEL_PATH
    try:
        yolo_model = YOLO(weights_path)
        print("YOLO model loaded:", weights_path)
    except Exception as e:
        print("Warning: Could not load YOLO model —", e)
        yolo_model = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open RGB camera. Try changing the index in cv2.VideoCapture(0).")
        return
    print("RGB camera opened.")

    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
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

    print("Press 'q' or ESC to quit.")

    while True:
        try:
            ret, color_image = cap.read()
            if not ret:
                print("Frame grab failed on RGB camera.")
                break
            color_image = cv2.rotate(color_image, cv2.ROTATE_180)
            display = draw_yolo_detections(color_image, yolo_model, conf=yolo_conf)
            cv2.imshow("RGB Camera", display)

            depth_image = depth_reader.get_latest()
            if depth_image is not None:
                depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
                cv2.imshow("Depth Viewer", depth_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ESC_KEY:
                break
        except KeyboardInterrupt:
            break

    depth_reader.stop()
    cap.release()
    cv2.destroyAllWindows()
    pipeline.stop()


def main():
    parser = argparse.ArgumentParser(
        description="combined_viewer_v6 — RGB + depth (180°) with fine-tuned YOLO detection (testyolo.py style)."
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="YOLO weights (default: surgical_tools_ft best.pt or YOLO_MODEL_PATH).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=YOLO_CONF,
        help=f"Detection confidence threshold (default {YOLO_CONF}, same as testyolo.py).",
    )
    args = parser.parse_args()

    run_live_viewer(yolo_weights=args.weights, yolo_conf=args.conf)


if __name__ == "__main__":
    main()
