#!/usr/bin/env python3
# ******************************************************************************
#  sam2_yolo_video.py
#
#  Webcam demo: fine-tuned YOLO proposes boxes each frame; SAM 2.1 tracks
#  segmentations inside those boxes across time (Ultralytics
#  SAM2DynamicInteractivePredictor).
#
#  python sam2_yolo_video.py
#  python sam2_yolo_video.py --weights path/to/best.pt --sam sam2.1_s.pt
#
#  q / ESC — quit
# ******************************************************************************

import argparse
import os

import cv2

from sam2_yolo_track import SamYoloVideoTracker, overlay_sam_on_bgr


def main():
    parser = argparse.ArgumentParser(description="YOLO boxes + SAM2 video segmentation/tracking")
    parser.add_argument(
        "--weights",
        default=os.environ.get(
            "YOLO_MODEL_PATH",
            "runs/detect/surgical_tools_ft/run1/weights/best.pt",
        ),
    )
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--sam", default="sam2.1_t.pt", help="SAM2.1 checkpoint (t/s/b/l)")
    parser.add_argument("--imgsz", type=int, default=512, help="SAM inference size (lower = faster)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-rotate", action="store_true", help="disable 180° rotation")
    parser.add_argument(
        "--refresh",
        type=int,
        default=20,
        help="re-seed SAM from YOLO boxes every N frames (0 = only on count change)",
    )
    parser.add_argument("--max-objects", type=int, default=10)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open camera", args.camera)
        return

    tracker = SamYoloVideoTracker(
        yolo_weights=args.weights,
        yolo_conf=args.conf,
        sam_model=args.sam,
        sam_imgsz=args.imgsz,
        max_obj_num=args.max_objects,
        refresh_every=args.refresh,
    )

    print("YOLO + SAM2 running. q or ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not args.no_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        yolo_vis, sam_res = tracker.step(frame)

        if sam_res is not None:
            blended = overlay_sam_on_bgr(frame, sam_res, alpha=0.5)
            cv2.imshow("SAM2 track (masks)", blended)

        cv2.imshow("YOLO detections", yolo_vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
