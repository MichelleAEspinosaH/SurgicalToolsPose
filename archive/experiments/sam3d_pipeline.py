#!/usr/bin/env python3
"""
SAM3 tracking pipeline with mask/keyframe export for SAM-3D experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    try:
        from sam3.model_builder import build_sam3_video_predictor  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import SAM3.\n"
            "Install it first, for example:\n"
            "  git clone https://github.com/facebookresearch/sam3.git\n"
            "  .venv/bin/pip install -e sam3\n"
        ) from e

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


def pick_points(first_frame: np.ndarray) -> list[tuple[int, float, float]]:
    win = "Select SAM3 points"
    points: list[tuple[int, float, float]] = []

    def draw() -> np.ndarray:
        vis = first_frame.copy()
        for obj_id, px_f, py_f in points:
            px, py = int(px_f), int(py_f)
            cv2.circle(vis, (px, py), 6, (0, 255, 255), -1)
            cv2.circle(vis, (px, py), 9, (255, 255, 255), 1)
            cv2.putText(vis, f"ID{obj_id}", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(
            vis,
            "Left click: add object | Backspace: undo | c: clear | Enter: start",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return vis

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((len(points) + 1, float(x), float(y)))

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


def _extract_masks_from_outputs(outputs: Any, fh: int, fw: int) -> list[tuple[int, np.ndarray]]:
    pairs: list[tuple[int, np.ndarray]] = []
    if outputs is None:
        return pairs

    # SAM2-like dict shape (compat fallback).
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

        # Dict as {obj_id: mask}
        for k, v in outputs.items():
            if isinstance(k, (int, np.integer, str)) and str(k).isdigit():
                pairs.append((int(k), _mask_to_2d_bool(v, fh, fw)))
        if pairs:
            return pairs

    # List of per-object dict entries.
    if isinstance(outputs, list):
        for idx, entry in enumerate(outputs):
            if not isinstance(entry, dict):
                continue
            obj_id = int(entry.get("obj_id", idx + 1))
            m = entry.get("mask_logits", entry.get("mask"))
            if m is None:
                continue
            pairs.append((obj_id, _mask_to_2d_bool(m, fh, fw)))
        return pairs

    return pairs


def _attempt_sam3d_stub(rgb_path: str, mask_path: str, output_dir: str) -> bool:
    try:
        from inference import Inference, load_image, load_masks  # type: ignore
    except Exception:
        return False
    cfg = os.environ.get("SAM3D_CONFIG_PATH", "")
    if not cfg:
        return False
    try:
        infer = Inference(cfg, compile=False)  # type: ignore[arg-type]
        image = load_image(rgb_path)
        masks = load_masks(str(Path(mask_path).parent), extension=".png")
        if not masks:
            return False
        _ = infer(image, masks[0], seed=42)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def run(args) -> None:
    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}")
        return

    cap = cv2.VideoCapture(args.input)
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

    predictor = _build_sam3_video_predictor(args.sam3_repo, checkpoint_path=args.sam3_checkpoint)
    print("Starting SAM3 session...")
    response = predictor.handle_request(request=dict(type="start_session", resource_path=args.input))
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

    out_root = Path(args.output_dir).resolve()
    rgb_dir = out_root / "rgb_frames"
    masks_dir = out_root / "masks"
    keyframes_dir = out_root / "sam3d_keyframes"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    cur_idx = -1
    frame = None
    keyframe_records: dict[int, dict[str, int]] = {}
    frame_count = 0

    print("Tracking + export running... press q to stop early.")
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

        fh2, fw2 = frame.shape[:2]
        rgb_path = rgb_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(rgb_path), frame)

        obj_masks = _extract_masks_from_outputs(outputs, fh2, fw2)
        for obj_id, binm in obj_masks:
            if not np.any(binm):
                continue
            obj_dir = masks_dir / f"obj_{obj_id:03d}"
            obj_dir.mkdir(parents=True, exist_ok=True)
            mask_path = obj_dir / f"{frame_idx:06d}.png"
            cv2.imwrite(str(mask_path), (binm.astype(np.uint8) * 255))

            area = int(np.count_nonzero(binm))
            rec = keyframe_records.get(obj_id)
            if rec is None or area > rec["area"]:
                keyframe_records[obj_id] = {"frame_idx": frame_idx, "area": area}

        frame_count += 1
        vis = frame.copy()
        cv2.putText(vis, f"frame={frame_idx} exported={frame_count}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("SAM3 Export Pipeline", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    keyframes_json: list[dict[str, object]] = []
    for obj_id, rec in sorted(keyframe_records.items()):
        fi = int(rec["frame_idx"])
        rgb_path = rgb_dir / f"{fi:06d}.png"
        mask_path = masks_dir / f"obj_{obj_id:03d}" / f"{fi:06d}.png"
        if not rgb_path.exists() or not mask_path.exists():
            continue
        keyframes_json.append(
            {"obj_id": obj_id, "frame_idx": fi, "rgb_path": str(rgb_path), "mask_path": str(mask_path), "area_px": int(rec["area"])}
        )
        shutil.copy2(rgb_path, keyframes_dir / f"obj_{obj_id:03d}_rgb.png")
        shutil.copy2(mask_path, keyframes_dir / f"obj_{obj_id:03d}_mask.png")

    meta = {
        "input_video": args.input,
        "pipeline": "sam3_video_predictor",
        "output_dir": str(out_root),
        "num_exported_frames": frame_count,
        "selected_points": [{"obj_id": int(o), "x": float(x), "y": float(y)} for o, x, y in points],
        "keyframes": keyframes_json,
    }
    with open(out_root / "sam3d_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.try_sam3d and keyframes_json:
        print("Attempting SAM-3D test call on first keyframe...")
        k0 = keyframes_json[0]
        ok = _attempt_sam3d_stub(
            rgb_path=str(k0["rgb_path"]),
            mask_path=str(k0["mask_path"]),
            output_dir=str(out_root / "sam3d_outputs"),
        )
        if ok:
            print("SAM-3D call succeeded (see sam3d_outputs).")
        else:
            print("SAM-3D call skipped/failed. Set SAM3D_CONFIG_PATH and install sam-3d deps.")

    print(f"Done. Exports written to: {out_root}")
    print(f"Manifest: {out_root / 'sam3d_manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3-to-SAM3D export pipeline")
    parser.add_argument("--input", default="/Users/michelleespinosa/Desktop/SurgicalToolsPose/recording_rgb.mp4", help="Input video path")
    parser.add_argument("--output-dir", default="sam3d_exports", help="Output folder")
    parser.add_argument("--sam3-repo", default="sam3", help="Path to local SAM3 repo")
    parser.add_argument("--sam3-checkpoint", default="", help="Optional SAM3 checkpoint override")
    parser.add_argument("--try-sam3d", action="store_true", help="Attempt a minimal SAM-3D call if dependencies are available")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
