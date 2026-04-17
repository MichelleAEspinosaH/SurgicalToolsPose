# SurgicalToolsPose

Surgical tool tracking and pose-estimation experiments built around EdgeTAM/SAM2 and OpenCV.

## Current Primary Workflow

The active script is:

- `EdgeTAMLive/live_track.py`

It performs:

- live camera capture with optional Orbbec-specific 180-degree rotation,
- first-click frame freeze for multi-object seed selection,
- EdgeTAM mask propagation from the selected seed frame,
- per-object mesh pose estimation from GLB files (`object_0.glb`, `object_1.glb`, `object_2.glb`),
- projected XYZ axis overlay per object,
- on-screen pose readout for each tracked object:
  - `R(rx, ry, rz)` in degrees,
  - `T(tx, ty, tz)` translation values,
- optional output video writing.

## Repository Layout

- `EdgeTAMLive/` - main live tracking pipeline and mesh/registration assets.
- `tests/` - legacy/experimental scripts and utility runners moved from repo root.

Examples now in `tests/` include:

- `tests/sam2_video_manual_points.py`
- `tests/sam2_yolo_video.py`
- `tests/combined_viewer*.py`
- `tests/train_yolo.py`, `tests/train_yolo_seg.py`, `tests/testyolo.py`
- `tests/depth_frame.py`, `tests/rgb_test.py`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python numpy torch ultralytics scipy pyorbbecsdk trimesh
```

If using official SAM2 tooling, keep a local clone (used by scripts under `tests/`):

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git segment-anything-2
cd segment-anything-2/checkpoints
./download_ckpts.sh
cd ../..
```

## Run Live Tracking

From repo root:

```bash
python3 EdgeTAMLive/live_track.py --camera 0
```

Useful options:

- `--camera <index>`
- `--device {auto,cuda,mps,cpu}`
- `--alpha <mask_alpha>`
- `--axis-smooth <0..1>`
- `--output out.mp4`

Controls:

- left click: add object seed point (first click freezes frame),
- backspace/delete: remove last point,
- `c`: clear and unfreeze,
- enter: start tracking,
- `q`/`Esc`: quit.

## Notes

- `live_track.py` now prioritizes GLB-based pose overlays; bbox-cube fallback drawing is disabled.
- Keep large binary assets and ad-hoc outputs out of commits unless intentionally versioned.
