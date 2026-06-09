# SurgicalToolsPose

Live **6DoF pose estimation** for surgical tools from a single RGB camera.

The pipeline combines [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) instance segmentation, [fal SAM3D](https://fal.ai/models/fal-ai/sam-3/3d-objects) mesh generation, and dense PnP pose tracking with Kalman smoothing.

## Quick start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd SurgicalToolsPose

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r EdgeTAMLive/requirements.txt

# 3. EdgeTAM checkpoint (one-time)
cd EdgeTAMLive/EdgeTAM/checkpoints
./download_ckpts.sh    # downloads edgetam.pt
cd ../../..

# 4. fal API key (for 3D mesh generation)
export FAL_KEY="your-fal-api-key"

# 5. Camera calibration (recommended, one-time)
cd EdgeTAMLive
python live_pose_any.py --calibrate-only --camera 0

# 6. Live tracking
python live_pose_any.py --camera 0
```

## What it does

```
Live camera  →  click seed points  →  EdgeTAM masks
       →  confirm masks  →  fal SAM3D GLB per tool
       →  dense PnP 6DoF pose each frame  →  XYZ axes + CSV log
```

| Stage | Technology | Purpose |
|-------|------------|---------|
| Segmentation | EdgeTAM | Track tool masks frame-to-frame |
| 3D model | fal SAM3D | Build a GLB mesh from one RGB frame + mask |
| Pose | OpenCV PnP + Kalman | 6DoF pose from mask + mesh |
| Calibration | Checkerboard (OpenCV) | Camera intrinsics for metric pose |

## Repository layout

```
SurgicalToolsPose/
├── README.md
├── EdgeTAMLive/                # Active pipeline (use this)
│   ├── live_pose_any.py
│   ├── requirements.txt
│   ├── camera_calibration.npz  # Created by you (gitignored)
│   ├── sam3d_live_objects/     # Runtime masks & GLBs (gitignored)
│   └── EdgeTAM/
│       └── checkpoints/edgetam.pt
└── archive/                    # Older scripts & experiments (reference only)
    ├── legacy_live/            # live_track.py, live_track_copy.py, …
    ├── experiments/            # SAM2/YOLO prototypes, one-off tests
    ├── samples/sam3d_bootstrap/
    └── weights/
```

## Camera calibration

Use the [MRPT 9×7 checkerboard](https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf) (20 mm squares, print at **1:1 scale** on A4).

```bash
cd EdgeTAMLive
python live_pose_any.py --calibrate-only --camera 0
```

Capture **15+ views** (SPACE), then press **C** or **Enter**. Saves `camera_calibration.npz`.

Subsequent runs auto-load this file. Pose HUD shows rotation in **degrees** and translation in **mm**.

For better metric depth, optionally pass a known working distance or tool width:

```bash
python live_pose_any.py --camera 0 --surface-distance-cm 40
python live_pose_any.py --camera 0 --tool-width-mm 12
```

## Live tracking controls

| Step | Action |
|------|--------|
| Seed selection | Left-click on each tool → **Enter** |
| Confirm masks | **Y** / **Enter** to continue, **N** / **Esc** to abort |
| During tracking | **Q** / **Esc** to quit |

Seed-selection shortcuts: **Backspace** undo last point, **C** clear all.

## CLI reference

```bash
python live_pose_any.py --camera 0              # Live tracking
python live_pose_any.py --calibrate-only        # Checkerboard calibration only
python live_pose_any.py --output out.mp4        # Save annotated video
python live_pose_any.py --device mps            # Apple Silicon GPU
python live_pose_any.py --no-half               # Disable FP16 inference
```

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | OpenCV camera index |
| `--device` | `auto` | `cuda`, `mps`, or `cpu` |
| `--alpha` | `0.45` | Mask overlay opacity |
| `--surface-distance-cm` | `0` | Known camera-to-surface distance |
| `--tool-width-mm` | `0` | Known tool width for metric translation |
| `--kalman-process-var` | `5e-3` | Pose smoothing (lower = smoother) |
| `--fal-model` | `fal-ai/sam-3/3d-objects` | fal SAM3D model ID |

## Output

- **On-screen HUD**: `R(rx,ry,rz)deg  T(tx,ty,tz)mm` per object
- **CSV log**: `posesN.csv` in the working directory (`tx_mm`, `ty_mm`, `tz_mm`, Euler angles)
- **Meshes**: `sam3d_live_objects/object_<id>.glb`

## Requirements

- Python 3.10+
- Webcam or phone camera (Continuity Camera, USB, etc.)
- macOS, Linux, or Windows with OpenCV camera support
- [fal.ai](https://fal.ai) API key (`FAL_KEY`)
- GPU recommended (CUDA or Apple MPS); CPU works but is slower

Optional: `pyorbbecsdk` for Orbbec factory intrinsics (otherwise checkerboard calibration is used).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Checkpoint not found` | Run `EdgeTAM/checkpoints/download_ckpts.sh` |
| `Set FAL_KEY` | `export FAL_KEY=...` before running |
| Board not detected | Print PDF at 1:1 scale; improve lighting |
| Wrong camera | Try `--camera 1` or `--camera 2` |
| Pose jitter | Lower `--kalman-process-var` (e.g. `2e-4`) |

## License

EdgeTAM is subject to its own [license](EdgeTAMLive/EdgeTAM/LICENSE). Application code in this repository is provided for research use.
