#!/usr/bin/env python3
import argparse
import os
import tempfile
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor


def pick_points(first_frame: np.ndarray) -> list[tuple[int, float, float]]:
    win = "Select SAM2 points"
    points: list[tuple[int, float, float]] = []

    def draw() -> np.ndarray:
        vis = first_frame.copy()
        for obj_id, px_f, py_f in points:
            px, py = int(px_f), int(py_f)
            cv2.circle(vis, (px, py), 6, (0, 255, 255), -1)
            cv2.circle(vis, (px, py), 9, (255, 255, 255), 1)
            cv2.putText(
                vis,
                f"ID{obj_id}",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        cv2.putText(
            vis,
            "Left click: add NEW object point (new ID) | Backspace: undo | c: clear | Enter: start",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return vis

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            obj_id = len(points) + 1
            points.append((obj_id, float(x), float(y)))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    cancelled = False
    while True:
        cv2.imshow(win, draw())
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):  # ENTER
            if points:
                break
        elif k in (8, 127):  # Backspace / Delete
            if points:
                points.pop()
        elif k == ord("c"):
            points.clear()
        elif k in (ord("q"), 27):
            cancelled = True
            break
    cv2.destroyWindow(win)
    return [] if cancelled else points


def point_color(obj_id: int) -> tuple[int, int, int]:
    hue = (obj_id * 47 + 20) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _mask_to_2d_bool(m: np.ndarray, fh: int, fw: int) -> np.ndarray:
    """SAM2 may return (H,W), (1,H,W), (1,1,H,W), etc.; normalize to (fh,fw) bool."""
    x = np.asarray(m, dtype=np.float32)
    x = np.squeeze(x)
    while x.ndim > 2:
        x = x[0]
    if x.ndim != 2:
        return np.zeros((fh, fw), dtype=bool)
    if x.shape[0] != fh or x.shape[1] != fw:
        x = cv2.resize(x, (fw, fh), interpolation=cv2.INTER_NEAREST)
    return x > 0.0


def overlay_masks_with_ids(
    frame: np.ndarray,
    obj_ids: list[int],
    masks: torch.Tensor,
    alpha: float = 0.45,
) -> np.ndarray:
    vis = frame.copy().astype(np.float32)
    fh, fw = frame.shape[:2]
    masks_np = masks.detach().cpu().numpy()
    n = min(len(obj_ids), masks_np.shape[0])
    for i in range(n):
        obj_id = obj_ids[i]
        binm = _mask_to_2d_bool(masks_np[i], fh, fw)
        if not np.any(binm):
            continue
        c = np.array(point_color(int(obj_id)), dtype=np.float32)
        vis[binm] = vis[binm] * (1.0 - alpha) + c * alpha
    return vis.astype(np.uint8)


def resolve_model_paths(repo_root: Path, model_size: str) -> tuple[str, str]:
    model_size = model_size.lower()
    if model_size == "tiny":
        return (
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            str(repo_root / "checkpoints" / "sam2.1_hiera_tiny.pt"),
        )
    if model_size == "small":
        return (
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            str(repo_root / "checkpoints" / "sam2.1_hiera_small.pt"),
        )
    if model_size in {"base", "base_plus", "b+"}:
        return (
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            str(repo_root / "checkpoints" / "sam2.1_hiera_base_plus.pt"),
        )
    if model_size == "large":
        return (
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            str(repo_root / "checkpoints" / "sam2.1_hiera_large.pt"),
        )
    raise ValueError(f"Unsupported model size: {model_size}")


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare_sam2_input_as_jpeg_folder(input_path: str) -> tuple[str, str]:
    """
    Build a temporary JPEG frame folder for SAM2 init_state().
    This avoids the optional decord dependency required for direct video-file loading.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_path = os.path.join(tmp_dir, f"{frame_i:06d}.jpg")
        cv2.imwrite(out_path, frame)
        frame_i += 1
    cap.release()
    if frame_i == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Input video contained no readable frames.")
    return tmp_dir, tmp_dir


def main():
    parser = argparse.ArgumentParser(
        description="Apply official Meta SAM2 video tracking from manual point prompts."
    )
    parser.add_argument("--input", default="movie.mp4.mov", help="Input video path")
    parser.add_argument(
        "--sam2-repo",
        default="segment-anything-2",
        help="Path to cloned SAM2 repo",
    )
    parser.add_argument(
        "--model-size",
        default="base_plus",
        choices=["tiny", "small", "base_plus", "large"],
        help="Official SAM2.1 model size",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay alpha")
    parser.add_argument("--output", default="", help="Optional output video path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input video not found: {args.input}")
        print("Pass a valid path with --input")
        return

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Could not open input video: {args.input}")
        return

    ok, first = cap.read()
    if not ok:
        print("Could not read first frame.")
        cap.release()
        return

    points = pick_points(first)
    if not points:
        print("No points selected. Exiting.")
        cap.release()
        return

    repo_root = Path(args.sam2_repo).expanduser().resolve()
    model_cfg, ckpt_path = resolve_model_paths(repo_root, args.model_size)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run: (cd segment-anything-2/checkpoints && ./download_ckpts.sh)")
        cap.release()
        return

    device = choose_device(args.device)
    print(f"Using device={device}, model={args.model_size}")
    predictor = build_sam2_video_predictor(model_cfg, ckpt_path, device=device)
    sam2_video_path, temp_frames_dir = prepare_sam2_input_as_jpeg_folder(args.input)
    print(f"Prepared temporary JPEG frame folder for SAM2: {sam2_video_path}")
    state = predictor.init_state(video_path=sam2_video_path)

    for obj_id, x, y in points:
        predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=int(obj_id),
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    h, w = first.shape[:2]
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, float(fps), (w, h))

    # Re-open to read frames in sync with predictor output indices.
    cap.release()
    cap = cv2.VideoCapture(args.input)
    cur_idx = -1

    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        while cur_idx < frame_idx:
            ok, frame = cap.read()
            if not ok:
                frame = None
                break
            cur_idx += 1
        if frame is None:
            break

        if hasattr(obj_ids, "tolist"):
            obj_ids_list = [int(x) for x in obj_ids.tolist()]
        else:
            obj_ids_list = [int(i) for i in obj_ids]
        vis = overlay_masks_with_ids(frame, obj_ids_list, masks, alpha=args.alpha)
        if frame_idx == 0:
            for obj_id, px_f, py_f in points:
                px, py = int(px_f), int(py_f)
                cv2.circle(vis, (px, py), 5, (0, 255, 255), -1)
                cv2.putText(
                    vis,
                    f"ID{obj_id}",
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("SAM2 manual points video", vis)
        if writer is not None:
            writer.write(vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    if writer is not None:
        writer.release()
    if temp_frames_dir and os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
