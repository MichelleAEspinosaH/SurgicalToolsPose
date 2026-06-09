#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


TARGET_SIZE = (640, 360)


def _ensure_sam3_importable(repo_hint: str) -> None:
    candidates = [
        Path(repo_hint).expanduser().resolve(),
        Path("sam3").resolve(),
    ]
    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))


def _build_sam3_video_predictor(repo_hint: str, checkpoint_path: str = ""):
    _ensure_sam3_importable(repo_hint)
    build_sam3_video_predictor = None
    import_err = None
    for mod_path in ("sam3.model_builder", "sam3.sam3.model_builder"):
        try:
            module = __import__(mod_path, fromlist=["build_sam3_video_predictor"])
            build_sam3_video_predictor = getattr(module, "build_sam3_video_predictor")
            break
        except Exception as e:
            import_err = e
    if build_sam3_video_predictor is None:
        raise RuntimeError(
            "Could not import SAM3.\n"
            "Install it first:\n"
            "  git clone https://github.com/facebookresearch/sam3.git\n"
            "  .venv/bin/python -m pip install -e sam3\n"
            "If running from inside the sam3 repo, this script also supports\n"
            "the nested module path (sam3.sam3.model_builder).\n"
        ) from import_err

    if checkpoint_path:
        for kwargs in (
            {"checkpoint": checkpoint_path},
            {"ckpt_path": checkpoint_path},
            {"checkpoint_path": checkpoint_path},
        ):
            try:
                return build_sam3_video_predictor(**kwargs)
            except TypeError:
                continue
    return build_sam3_video_predictor()


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    return cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def point_color(obj_id: int) -> tuple[int, int, int]:
    hue = (obj_id * 47 + 20) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def pick_points(first_frame: np.ndarray) -> list[tuple[int, float, float]]:
    win = "Select SAM3 points"
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
            "Left click: add NEW object | Backspace: undo | c: clear | Enter: start",
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
        if k in (13, 10):
            if points:
                break
        elif k in (8, 127):
            if points:
                points.pop()
        elif k == ord("c"):
            points.clear()
        elif k in (ord("q"), 27):
            cancelled = True
            break
    cv2.destroyWindow(win)
    return [] if cancelled else points


def _mask_to_2d_bool(mask_i: Any, fh: int, fw: int) -> np.ndarray:
    if isinstance(mask_i, torch.Tensor):
        x = mask_i.detach().cpu().numpy()
    else:
        x = np.asarray(mask_i)
    x = np.squeeze(x)
    while x.ndim > 2:
        x = x[0]
    if x.ndim != 2:
        return np.zeros((fh, fw), dtype=bool)
    if x.shape[:2] != (fh, fw):
        x = cv2.resize(x.astype(np.float32), (fw, fh), interpolation=cv2.INTER_NEAREST)
    return x > 0.0


def _extract_obj_masks(outputs: Any, fh: int, fw: int) -> list[tuple[int, np.ndarray]]:
    pairs: list[tuple[int, np.ndarray]] = []
    if outputs is None:
        return pairs

    if isinstance(outputs, dict):
        obj_ids = outputs.get("obj_ids", outputs.get("out_obj_ids"))
        mask_logits = outputs.get("mask_logits", outputs.get("out_mask_logits"))
        if obj_ids is not None and mask_logits is not None:
            if hasattr(obj_ids, "tolist"):
                obj_ids = obj_ids.tolist()
            if isinstance(mask_logits, torch.Tensor):
                mask_logits = mask_logits.detach().cpu().numpy()
            for i, obj_id in enumerate(obj_ids):
                pairs.append((int(obj_id), _mask_to_2d_bool(mask_logits[i], fh, fw)))
            return pairs
        for k, v in outputs.items():
            if isinstance(k, (int, np.integer, str)) and str(k).isdigit():
                pairs.append((int(k), _mask_to_2d_bool(v, fh, fw)))
        if pairs:
            return pairs

    if isinstance(outputs, list):
        for i, entry in enumerate(outputs):
            if not isinstance(entry, dict):
                continue
            oid = int(entry.get("obj_id", i + 1))
            m = entry.get("mask_logits", entry.get("mask"))
            if m is not None:
                pairs.append((oid, _mask_to_2d_bool(m, fh, fw)))
        return pairs

    return pairs


def run_video(args) -> None:
    if not os.path.exists(args.input):
        print(f"Input video not found: {args.input}")
        return

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Could not open input video: {args.input}")
        return
    ok, first = cap.read()
    cap.release()
    if not ok:
        print("Could not read first frame.")
        return

    first = preprocess_frame(first)
    fh, fw = first.shape[:2]
    points = pick_points(first)
    if not points:
        print("No points selected. Exiting.")
        return

    predictor = _build_sam3_video_predictor(args.sam3_repo, args.checkpoint)
    print("Starting SAM3 session...")
    try:
        response = predictor.handle_request(
            request=dict(type="start_session", resource_path=args.input)
        )
    except Exception as e:
        raise RuntimeError(
            "SAM3 failed to start session.\n"
            "If you use gated checkpoints from Hugging Face, run `hf auth login` first.\n"
            "You can also pass an explicit local checkpoint with --checkpoint."
        ) from e
    session_id = response["session_id"]

    for obj_id, x, y in points:
        points_tensor = torch.tensor([[x / float(fw), y / float(fh)]], dtype=torch.float32)
        labels_tensor = torch.tensor([1], dtype=torch.int32)
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                points=points_tensor,
                point_labels=labels_tensor,
                obj_id=int(obj_id),
            )
        )

    cap = cv2.VideoCapture(args.input)
    cur_idx = -1
    frame = None
    print("SAM3 tracking started. Press 'q' or ESC to stop.")

    for response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        frame_idx = int(response["frame_index"])
        outputs = response.get("outputs")
        while cur_idx < frame_idx:
            ok, frame = cap.read()
            if not ok:
                frame = None
                break
            frame = preprocess_frame(frame)
            cur_idx += 1
        if frame is None:
            break

        vis = frame.copy().astype(np.float32)
        for obj_id, binm in _extract_obj_masks(outputs, frame.shape[0], frame.shape[1]):
            if not np.any(binm):
                continue
            color = np.array(point_color(obj_id), dtype=np.float32)
            vis[binm] = vis[binm] * (1.0 - args.alpha) + color * args.alpha

        vis = vis.astype(np.uint8)
        cv2.putText(
            vis,
            f"frame={frame_idx}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("SAM3", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SAM3 video tracking from manual point prompts.")
    parser.add_argument(
        "--input",
        default="/Users/michelleespinosa/Desktop/SurgicalToolsPose/recording_rgb.mp4",
        help="Input video path",
    )
    parser.add_argument("--sam3-repo", default="sam3", help="Path to SAM3 repo")
    parser.add_argument("--checkpoint", default="", help="Optional SAM3 checkpoint path override")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay alpha")
    args = parser.parse_args()
    run_video(args)


if __name__ == "__main__":
    main()
