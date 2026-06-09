#!/usr/bin/env python3
"""
Smoke test for fal SAM 3D Objects API.

  pip install fal-client
  export FAL_KEY="..."

  # Blocking wait (simplest — same as submit + get internally):
  python tests/fal_sam3d_smoke_test.py

  # Explicit queue API (docs style): submit → request_id → wait with .get()
  python tests/fal_sam3d_smoke_test.py --submit

  # submit + webhook (your server receives the result; this script exits after enqueue):
  python tests/fal_sam3d_smoke_test.py --submit --webhook https://your.server/hook

  # Your RGB image + single mask PNG (same resolution):
  python tests/fal_sam3d_smoke_test.py path/to/frame.png path/to/mask.png
  python tests/fal_sam3d_smoke_test.py --submit path/to/frame.png path/to/mask.png

Or set SAM3D_IMAGE_URL to a public image URL and omit image argv.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve


def _build_arguments(args: argparse.Namespace) -> dict:
    import fal_client

    image_url = os.environ.get(
        "SAM3D_IMAGE_URL",
        "https://v3b.fal.media/files/b/0a8439e5/TyAmfW5w_sqRXRzWVBGsW_car.jpeg",
    )
    mask_urls: list[str] = []

    pos = args.image_paths
    if len(pos) >= 2:
        img_path, mask_path = pos[0], pos[1]
        print("Uploading:", img_path, mask_path)
        image_url = fal_client.upload_file(img_path)
        mask_urls = [fal_client.upload_file(mask_path)]
    elif len(pos) == 1:
        print("Uploading:", pos[0])
        image_url = fal_client.upload_file(pos[0])

    arguments: dict = {"image_url": image_url, "seed": 42}
    if mask_urls:
        arguments["mask_urls"] = mask_urls
    return arguments


def main() -> None:
    if not os.environ.get("FAL_KEY"):
        print("Set FAL_KEY first, e.g. export FAL_KEY='your_key'", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="fal SAM 3D Objects smoke test")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Use fal_client.submit() + request_id + wait (no webhook unless --webhook)",
    )
    parser.add_argument(
        "--webhook",
        default="",
        help="With --submit: fal_webhook URL (optional). If set, script does not wait for result.",
    )
    parser.add_argument(
        "image_paths",
        nargs="*",
        help="Optional: image.png [mask.png] (uploaded via fal_client.upload_file)",
    )
    parser.add_argument(
        "--download-dir",
        default="",
        help="Optional output directory to auto-download model_glb/gaussian_splat/artifacts_zip.",
    )
    cli = parser.parse_args()

    import fal_client

    arguments = _build_arguments(cli)

    if cli.submit:
        # Same pattern as fal docs; webhook_url is optional.
        submit_kw: dict = {}
        if cli.webhook.strip():
            submit_kw["webhook_url"] = cli.webhook.strip()

        handler = fal_client.submit(
            "fal-ai/sam-3/3d-objects",
            arguments=arguments,
            **submit_kw,
        )
        request_id = handler.request_id
        print("Submitted. request_id:", request_id)

        if submit_kw.get("webhook_url"):
            print(
                "Webhook set — fal will POST the result there. "
                "Optional: fal_client.status('fal-ai/sam-3/3d-objects', request_id, with_logs=True)"
            )
            return

        # No webhook: block until done (equivalent to polling status until Completed).
        print("Waiting for result (handler.get()) …")
        result = handler.get()
    else:
        print("Calling fal-ai/sam-3/3d-objects (subscribe — blocks until done) …")
        result = fal_client.subscribe(
            "fal-ai/sam-3/3d-objects",
            arguments=arguments,
            with_logs=True,
        )

    print("Done. Keys:", list(result.keys()) if isinstance(result, dict) else type(result))
    if isinstance(result, dict):
        for key in ("model_glb", "gaussian_splat", "metadata", "individual_glbs"):
            if key in result:
                print(f"  {key}:", result[key])
        if cli.download_dir:
            out_dir = Path(cli.download_dir).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            for key in ("model_glb", "gaussian_splat", "artifacts_zip"):
                val = result.get(key)
                if not isinstance(val, dict):
                    continue
                url = val.get("url")
                file_name = val.get("file_name") or f"{key}.bin"
                if not isinstance(url, str) or not url:
                    continue
                out_path = out_dir / str(file_name)
                print(f"Downloading {key} -> {out_path}")
                urlretrieve(url, out_path)
            print(f"Downloaded outputs to: {out_dir}")


if __name__ == "__main__":
    main()
