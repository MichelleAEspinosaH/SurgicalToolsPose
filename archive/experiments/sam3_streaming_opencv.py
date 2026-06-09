#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def _ensure_sam3_importable(repo_hint: str) -> None:
    candidates = [
        Path(repo_hint).expanduser().resolve(),
        Path("sam3").resolve(),
    ]
    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))


def _build_sam3_image_stack(repo_hint: str):
    _ensure_sam3_importable(repo_hint)
    try:
        from sam3.model_builder import build_sam3_image_model  # type: ignore
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import SAM3 image APIs.\n"
            "Install SAM3 in this env:\n"
            "  git clone https://github.com/facebookresearch/sam3.git\n"
            "  .venv/bin/python -m pip install -e sam3\n"
        ) from e
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    return model, processor


def _choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mask_to_bool(mask: Any, h: int, w: int) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.asarray(mask)
    m = np.squeeze(m)
    while m.ndim > 2:
        m = m[0]
    if m.ndim != 2:
        return np.zeros((h, w), dtype=bool)
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return m > 0.0


def _draw_streaming_outputs(frame_bgr: np.ndarray, outputs: dict[str, Any], alpha: float = 0.45) -> np.ndarray:
    vis = frame_bgr.copy().astype(np.float32)
    h, w = frame_bgr.shape[:2]
    masks = outputs.get("masks")
    boxes = outputs.get("boxes")
    object_ids = outputs.get("object_ids", [])

    if masks is not None:
        if isinstance(masks, torch.Tensor):
            masks_np = masks.detach().cpu().numpy()
        else:
            masks_np = np.asarray(masks)
        for i in range(int(masks_np.shape[0]) if masks_np.ndim >= 3 else 0):
            obj_id = int(object_ids[i]) if i < len(object_ids) else i + 1
            color = np.array(
                [
                    int((obj_id * 53 + 30) % 255),
                    int((obj_id * 97 + 50) % 255),
                    int((obj_id * 151 + 70) % 255),
                ],
                dtype=np.float32,
            )
            binm = _mask_to_bool(masks_np[i], h, w)
            if np.any(binm):
                vis[binm] = vis[binm] * (1.0 - alpha) + color * alpha

    if boxes is not None:
        b = boxes.detach().cpu().numpy() if isinstance(boxes, torch.Tensor) else np.asarray(boxes)
        if b.ndim == 2 and b.shape[1] >= 4:
            for i in range(b.shape[0]):
                obj_id = int(object_ids[i]) if i < len(object_ids) else i + 1
                x1, y1, x2, y2 = [int(v) for v in b[i, :4]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(
                    vis,
                    f"ID{obj_id}",
                    (x1 + 4, max(18, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

    return vis.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3 streaming video inference with OpenCV RGB input")
    parser.add_argument(
        "--input",
        default="/Users/michelleespinosa/Desktop/SurgicalToolsPose/recording_rgb.mp4",
        help="Video path or camera index (e.g., 0)",
    )
    parser.add_argument("--sam3-repo", default="sam3", help="Path to local sam3 repo")
    parser.add_argument("--text", default="person", help="Text prompt for SAM3")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay alpha")
    args = parser.parse_args()

    device = _choose_device(args.device)
    model, processor = _build_sam3_image_stack(args.sam3_repo)
    print(f"Using device={device}")

    stream_session = processor.init_video_session(
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    stream_session = processor.add_text_prompt(
        inference_session=stream_session,
        text=args.text,
    )

    source: Any = int(args.input) if str(args.input).isdigit() else args.input
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open input: {args.input}")
        return

    outputs_per_frame: dict[int, dict[str, Any]] = {}
    frame_idx = 0
    print("Streaming inference started. Press 'q' or ESC to stop.")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = processor(images=frame_rgb, device=device, return_tensors="pt")
        model_outputs = model(
            inference_session=stream_session,
            frame=inputs.pixel_values[0],
            reverse=False,
        )
        processed_outputs = processor.postprocess_outputs(
            stream_session,
            model_outputs,
            original_sizes=inputs.original_sizes,
        )
        outputs_per_frame[frame_idx] = processed_outputs

        vis = _draw_streaming_outputs(frame_bgr, processed_outputs, alpha=args.alpha)
        cv2.putText(
            vis,
            f"frame={frame_idx}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("SAM3 Streaming", vis)

        if (frame_idx + 1) % 10 == 0:
            print(f"Processed {frame_idx + 1} frames...")

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        frame_idx += 1
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"✓ Streaming inference complete! Processed {len(outputs_per_frame)} frames")
    if outputs_per_frame:
        first = outputs_per_frame[min(outputs_per_frame.keys())]
        n_obj = len(first.get("object_ids", []))
        print(f"Detected {n_obj} objects in first frame")
        boxes = first.get("boxes")
        masks = first.get("masks")
        if boxes is not None:
            shape = tuple(boxes.shape) if hasattr(boxes, "shape") else np.asarray(boxes).shape
            print(f"Boxes are in XYXY format (absolute pixel coordinates): {shape}")
        if masks is not None:
            shape = tuple(masks.shape) if hasattr(masks, "shape") else np.asarray(masks).shape
            print(f"Masks are at original video resolution: {shape}")


if __name__ == "__main__":
    main()
