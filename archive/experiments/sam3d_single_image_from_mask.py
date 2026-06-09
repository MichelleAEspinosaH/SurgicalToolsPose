#!/usr/bin/env python3
"""
Build a mask from local inputs and run a single-image SAM-3D inference test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


def build_mask(mask_png_path: Path, image_path: Path, output_mask_path: Path) -> Path:
    mask_gray = cv2.imread(str(mask_png_path), cv2.IMREAD_GRAYSCALE)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if mask_gray is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_png_path}")
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {image_path}")

    rotated_mask = cv2.rotate(mask_gray, cv2.ROTATE_180)
    if rotated_mask.shape[:2] != image_bgr.shape[:2]:
        rotated_mask = cv2.resize(
            rotated_mask,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    binary_mask = (rotated_mask > 0).astype("uint8")
    masked_image = image_bgr * binary_mask[:, :, None]

    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_mask_path), masked_image):
        raise RuntimeError(f"Failed to write mask image to: {output_mask_path}")
    return output_mask_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mask_png_path = repo_root / "EdgeTAMLive" / "segmented_scissors_mask.png"
    image_path = repo_root / "EdgeTAMLive" / "testing.jpg"

    generated_dir = repo_root / "EdgeTAMLive" / "sam3d_generated_mask"
    output_mask_path = generated_dir / "0.png"
    splat_output_path = repo_root / "EdgeTAMLive" / "splat.ply"

    build_mask(mask_png_path=mask_png_path, image_path=image_path, output_mask_path=output_mask_path)

    # Import inference code exactly like the notebook example.
    sys.path.append("notebook")
    from inference import Inference, load_image, load_single_mask  # type: ignore

    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    image = load_image(str(image_path))
    mask = load_single_mask(str(generated_dir), index=0)

    output = inference(image, mask, seed=42)
    output["gs"].save_ply(str(splat_output_path))

    print(f"Generated mask: {output_mask_path}")
    print(f"Saved gaussian splat: {splat_output_path}")


if __name__ == "__main__":
    main()
