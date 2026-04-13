# SurgicalToolsPose

Surgical tool tracking and visualization experiments using OpenCV RGB video, Orbbec depth, YOLO, and SAM2.

The repo currently has two main paths:
- **SAM2 manual-point tracking** (`sam2_video_manual_points.py`) for recorded videos or live camera input.
- **YOLO/SIFT + depth viewers** (`combined_viewer*.py`) for real-time prototyping with Orbbec depth.

## What Is In This Repo

### Main scripts

| File | Purpose |
|---|---|
| `sam2_video_manual_points.py` | Main SAM2 workflow: manual point prompts, mask propagation, ICP-based axes overlay, 180-degree rotation, and 640x360 preprocessing. Supports video file input or live camera index input. |
| `record_stream.py` | Record RGB camera stream to MP4. |
| `record_rgb_depth.py` | Record synchronized RGB (OpenCV) + depth colormap (Orbbec) MP4s; optionally saves raw depth `.npy` frames. |
| `sam2_yolo_video.py` | Live webcam demo: YOLO detections + Ultralytics SAM2 dynamic tracking. |
| `sam2_yolo_track.py` | Reusable tracker class used by `sam2_yolo_video.py`. |

### Legacy / experimental scripts

| File | Purpose |
|---|---|
| `combined_viewer.py`, `combined_viewer_v2.py` ... `combined_viewer_v6.py` | Iterations of YOLO + SIFT + optical-flow + depth visualization pipelines. |
| `clean_full.py`, `buildingup.py`, `yolo_sift_sam2_ids.py`, `pipeline.py` | Experimental integrated pipelines and utilities. |
| `depth_frame.py`, `rgb_test.py` | Basic depth and RGB test scripts. |
| `train_yolo.py`, `train_yolo_seg.py`, `testyolo.py` | YOLO training/testing utilities. |

## Setup

### 1) Python dependencies

Create a virtual environment and install the core packages used by current scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python numpy torch ultralytics scipy pyorbbecsdk
```

Notes:
- `pyorbbecsdk` is only required for depth-camera scripts.
- `sam2_video_manual_points.py` uses the official SAM2 repo code from `segment-anything-2/`.

### 2) SAM2 repo and checkpoints

`sam2_video_manual_points.py` expects a local clone at `segment-anything-2` by default.

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git segment-anything-2
cd segment-anything-2/checkpoints
./download_ckpts.sh
cd ../..
```

## Primary Workflows

### A) Run SAM2 manual-point tracking on a video

```bash
python3 sam2_video_manual_points.py --input "movie.mp4.mov"
```

Useful options:
- `--sam2-repo segment-anything-2`
- `--model-size {tiny,small,base_plus,large}`
- `--device {auto,cuda,mps,cpu}`
- `--output out.mp4`
- `--compile` (tries `torch.compile`, first run is slower)

Behavior:
- Frames are rotated 180 degrees.
- Frames are resized to 640x360 before processing.
- ICP-based axes are overlaid with fixed axis length.

### B) Run SAM2 manual-point tracking on a live camera

Pass camera index as `--input`:

```bash
python3 sam2_video_manual_points.py --input 0
```

### C) Record RGB stream only

```bash
python3 record_stream.py --camera 0 --output recording.mp4
```

### D) Record RGB + depth stream

```bash
python3 record_rgb_depth.py --camera 0 --rgb-out clip_rgb.mp4 --depth-out clip_depth.mp4
```

To also save raw depth arrays:

```bash
python3 record_rgb_depth.py --camera 0 --rgb-out clip_rgb.mp4 --depth-out clip_depth.mp4 --raw-depth-dir clip_depth_npy
```

### E) YOLO + SAM2 webcam demo

```bash
python3 sam2_yolo_video.py --camera 0
```

## Training Notes

- `train_yolo.py` and `train_yolo_seg.py` are utility scripts for dataset/model experiments and may need local edits (dataset paths, API key, model choices) before use.
- YOLO run outputs are written under `runs/` (ignored by git).

## Keyboard Controls (Common)

- `q` or `Esc`: quit
- In `sam2_video_manual_points.py` point-selection window:
  - Left click: add a new object point
  - Backspace/Delete: remove last point
  - `c`: clear points
  - Enter: start tracking

## Repo Notes

- Large files are intentionally git-ignored (`*.pt`, datasets, `runs/`, caches).
- Several scripts are iterative prototypes; prefer `sam2_video_manual_points.py` for current SAM2 tracking work.
