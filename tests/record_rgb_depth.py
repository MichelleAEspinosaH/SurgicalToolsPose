#!/usr/bin/env python3
"""
Record OpenCV RGB and Orbbec depth (jet colormap) to two MP4 files, one frame pair
per iteration (approximate sync). Use the same camera index as in rgb_test.py.

Example:
  python3 record_rgb_depth.py --rgb-out clip_rgb.mp4 --depth-out clip_depth.mp4

Then (with metric depth for 3D axes):
  python3 record_rgb_depth.py --rgb-out clip_rgb.mp4 --depth-out clip_depth.mp4 --raw-depth-dir clip_depth_npy
  python3 sam2_video_manual_points.py --input clip_rgb.mp4 --depth-video clip_depth.mp4 --depth-raw-dir clip_depth_npy
"""
import argparse
import os

import cv2
import numpy as np

try:
    from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline
except ImportError as e:
    raise SystemExit("Install pyorbbecsdk for depth recording: pip install pyorbbecsdk") from e

MIN_DEPTH_MM = 20
MAX_DEPTH_MM = 10000


class _TemporalFilterDepth:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(
                frame, self.alpha, self.previous_frame, 1 - self.alpha, 0
            )
        self.previous_frame = result
        return result


class DepthJetRecorder:
    def __init__(self) -> None:
        self._pipeline = None
        self._temporal = _TemporalFilterDepth(alpha=0.5)

    def start(self) -> bool:
        pipeline = Pipeline()
        config = Config()
        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if profile_list is None:
                return False
            depth_profile = profile_list.get_default_video_stream_profile()
            if depth_profile is None:
                return False
            config.enable_stream(depth_profile)
            pipeline.start(config)
        except Exception as e:
            print(f"Orbbec depth failed: {e}")
            return False
        self._pipeline = pipeline
        return True

    def read_jet_and_depth_mm(self) -> tuple[np.ndarray, np.ndarray] | None:
        """One depth frame → (jet BGR, float32 depth in mm, same H×W)."""
        if self._pipeline is None:
            return None
        try:
            frames = self._pipeline.wait_for_frames(1000)
            if frames is None:
                return None
            depth_frame = frames.get_depth_frame()
            if depth_frame is None or depth_frame.get_format() != OBFormat.Y16:
                return None
            w, h = depth_frame.get_width(), depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            depth_data = np.frombuffer(
                depth_frame.get_data(), dtype=np.uint16
            ).reshape((h, w))
            depth_data = depth_data.astype(np.float32) * scale
            depth_data = np.where(
                (depth_data > MIN_DEPTH_MM) & (depth_data < MAX_DEPTH_MM),
                depth_data,
                0,
            ).astype(np.uint16)
            depth_data = self._temporal.process(depth_data)
            mm = depth_data.astype(np.float32)
            depth_u8 = cv2.normalize(
                mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            jet = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
            return jet, mm
        except Exception:
            return None

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Record RGB (OpenCV) + depth (Orbbec) to two MP4s.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV RGB camera index")
    parser.add_argument("--rgb-out", default="recording_rgb.mp4", help="Output RGB video path")
    parser.add_argument("--depth-out", default="recording_depth.mp4", help="Output depth colormap video")
    parser.add_argument(
        "--raw-depth-dir",
        default="",
        help="If set, save float32 depth (mm) per frame as NNNNNN.npy for SAM2 3D axes.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    raw_dir = (args.raw_depth_dir or "").strip()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return

    depth = DepthJetRecorder()
    if not depth.start():
        cap.release()
        return

    ok, sample_rgb = cap.read()
    if not ok:
        print("Could not read RGB.")
        depth.stop()
        cap.release()
        return
    pair0 = depth.read_jet_and_depth_mm()
    if pair0 is None:
        print("Could not read depth.")
        depth.stop()
        cap.release()
        return
    d0, mm0 = pair0

    rw, rh = sample_rgb.shape[1], sample_rgb.shape[0]
    dw, dh = d0.shape[1], d0.shape[0]
    print(f"RGB {rw}x{rh}  →  {args.rgb_out}")
    print(f"Depth {dw}x{dh} (jet)  →  {args.depth_out}")
    if raw_dir:
        os.makedirs(raw_dir, exist_ok=True)
        print(f"Raw depth (mm) .npy  →  {raw_dir}/")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_rgb = cv2.VideoWriter(args.rgb_out, fourcc, float(args.fps), (rw, rh))
    w_dep = cv2.VideoWriter(args.depth_out, fourcc, float(args.fps), (dw, dh))
    w_rgb.write(sample_rgb)
    w_dep.write(d0)
    if raw_dir:
        np.save(os.path.join(raw_dir, "000000.npy"), mm0.astype(np.float32))

    n = 1
    while True:
        ok, rgb = cap.read()
        pair = depth.read_jet_and_depth_mm()
        if not ok or rgb is None or pair is None:
            break
        djet, mm = pair
        w_rgb.write(rgb)
        w_dep.write(djet)
        if raw_dir:
            np.save(os.path.join(raw_dir, f"{n:06d}.npy"), mm.astype(np.float32))
        n += 1
        cv2.imshow("RGB (q to stop)", rgb)
        cv2.imshow("Depth", djet)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    w_rgb.release()
    w_dep.release()
    cap.release()
    depth.stop()
    cv2.destroyAllWindows()
    print(f"Saved {n} paired frame(s).")


if __name__ == "__main__":
    main()
