#!/usr/bin/env python3
"""Read the RGB camera stream and record it to a file. Press q or Esc to stop."""
import argparse
import cv2

def main():
    parser = argparse.ArgumentParser(description="Record RGB camera stream to file.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", default="recording.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    print(f"Stream: {w}x{h} @ {fps:.2f} fps  →  {args.output}")

    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    # Warm up the sensor
    for _ in range(8):
        cap.grab()

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_count += 1
        cv2.imshow("Recording (q to stop)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved {frame_count} frames to {args.output}")

if __name__ == "__main__":
    main()
