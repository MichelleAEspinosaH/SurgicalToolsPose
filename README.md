# SurgicalToolsPose

Live **6DoF pose estimation** for surgical tools from a single RGB camera (webcam or phone).

The pipeline combines:

- **[EdgeTAM](https://github.com/facebookresearch/EdgeTAM)** — instance segmentation & mask tracking
- **[fal SAM3D](https://fal.ai/models/fal-ai/sam-3/3d-objects)** — one-shot 3D mesh (GLB) from RGB + mask
- **OpenCV PnP + Kalman filtering** — dense 6DoF pose each frame

**Main script:** `EdgeTAMLive/live_pose_any.py`

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Download models & API keys](#download-models--api-keys)
4. [Camera calibration (required first)](#camera-calibration-required-first)
5. [Run live tracking](#run-live-tracking)
6. [Controls & workflow](#controls--workflow)
7. [Outputs](#outputs)
8. [CLI reference](#cli-reference)
9. [Repository layout](#repository-layout)
10. [Troubleshooting](#troubleshooting)
11. [Links & references](#links--references)

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | EdgeTAM requires 3.10 or newer |
| **Camera** | Webcam, phone (Continuity Camera / USB), or Orbbec |
| **GPU (recommended)** | NVIDIA CUDA or Apple Silicon (MPS); CPU works but is slow |
| **fal.ai account** | For SAM3D mesh generation ([sign up](https://fal.ai)) |
| **Checkerboard** | [MRPT 9×7 PDF](https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf) — print at **1:1 scale** on A4 |
| **Internet** | Needed once per session for fal SAM3D mesh download |

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd SurgicalToolsPose
```

### 2. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install --upgrade pip
```

### 3. Install PyTorch

Install the build that matches your machine: **[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)**

Example (Apple Silicon):

```bash
pip install torch torchvision
```

Example (CUDA):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install pipeline dependencies

```bash
pip install -r EdgeTAMLive/requirements.txt
```

### 5. Install EdgeTAM (required)

EdgeTAM is vendored at `EdgeTAMLive/EdgeTAM/`. Install it in editable mode so `sam2` imports work:

```bash
cd EdgeTAMLive/EdgeTAM
pip install -e .
cd ../..
```

> If you see a warning about failing to build the SAM 2 CUDA extension, you can usually ignore it — the live pipeline still works on CPU/MPS.

---

## Download models & API keys

### EdgeTAM checkpoint (`edgetam.pt`)

The live script expects this file:

```
EdgeTAMLive/EdgeTAM/checkpoints/edgetam.pt
```

**Download (choose one):**

```bash
cd EdgeTAMLive/EdgeTAM/checkpoints
curl -L -O https://github.com/facebookresearch/EdgeTAM/releases/download/v1.0/edgetam.pt
cd ../../..
```

| Resource | Link |
|----------|------|
| Direct download | https://github.com/facebookresearch/EdgeTAM/releases/download/v1.0/edgetam.pt |
| EdgeTAM GitHub | https://github.com/facebookresearch/EdgeTAM |
| Hugging Face mirror | https://huggingface.co/facebook/EdgeTAM |
| EdgeTAM paper | https://arxiv.org/abs/2501.07256 |

Verify the file exists (~56 MB):

```bash
ls -lh EdgeTAMLive/EdgeTAM/checkpoints/edgetam.pt
```

### fal.ai API key (SAM3D meshes)

1. Create an account at **[fal.ai](https://fal.ai)**
2. Open **[fal.ai/dashboard/keys](https://fal.ai/dashboard/keys)** and create an API key
3. Export it before every run (or add to your shell profile):

```bash
export FAL_KEY="your-fal-api-key-here"
```

| Resource | Link |
|----------|------|
| SAM3D model used by this repo | https://fal.ai/models/fal-ai/sam-3/3d-objects |
| API docs | https://fal.ai/docs |
| Python client | https://fal.ai/docs/clients/python |
| Pricing (~$0.02 / object) | https://fal.ai/pricing |

The default model ID in code is `fal-ai/sam-3/3d-objects` (override with `--fal-model`).

---

## Camera calibration (required first)

Calibration estimates camera **intrinsics** (focal length, distortion). Without it, pose translation in **cm** will be inaccurate.

The pipeline uses a **centimetre** coordinate system end to end:

1. Checkerboard object points are in **cm** (`--checkerboard-square-cm 2` = 2 cm squares on the MRPT print).
2. SAM3D GLBs (glTF metres) are scaled **metres → cm** before PnP.
3. HUD and CSV report translation in **cm** (e.g. `tz ≈ 40` at ~40 cm distance).

### Squares vs inner corners (important)

OpenCV detects **inner corners** (where four squares meet), not square count.

| What you count on the print | MRPT PDF board | Different board (example) |
|-----------------------------|----------------|---------------------------|
| **Squares** (black/white cells) | **9 × 7** | 8 × 10 |
| **Inner corners** (`--checkerboard-cols/rows`) | **8 × 6** | 7 × 9 |

Rule: **inner corners = squares − 1** in each direction.

The code defaults (`--checkerboard-cols 8 --checkerboard-rows 6`) match the **MRPT 9×7** PDF. If your physical board differs, pass the correct inner-corner counts.

### Step 1 — Print the checkerboard

Download and print at **100% / actual size** (1:1 scale) on A4 — not “fit to page”:

**[MRPT 9×7 checkerboard PDF](https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf)**

| Property | Value |
|----------|-------|
| Squares on print | 9 wide × 7 tall |
| Square size | **2 cm** (20 mm when printed 1:1) |
| CLI inner corners | `--checkerboard-cols 8 --checkerboard-rows 6` |
| Square size flag | `--checkerboard-square-cm 2` |

**Verify with a ruler** after printing: one square must measure 2 cm. Wrong print scale breaks metric depth.

Mount flat on stiff backing (clipboard, foam board). Use matte paper; avoid glare.

### Step 2 — Run calibration

Use the **same camera, index, zoom, and focus** you will use for tracking:

```bash
cd EdgeTAMLive
python live_pose_any.py --calibrate-only --camera 0
```

If your phone is not camera 0, try `--camera 1` or `--camera 2`.

Lock phone settings when possible (disable autofocus / digital zoom).

### Step 3 — Capture 15+ diverse views

Default minimum is **15** samples; aim for **20–30** for best results.

| Key | Action |
|-----|--------|
| **SPACE** | Capture view (only when green corners are detected) |
| **U** / Backspace | Undo last capture |
| **C** or **Enter** | Finish calibration (needs ≥ 15 samples) |
| **Q** or **ESC** | Cancel |

**Vary each capture** — do not take 15 nearly identical frames:

- Board in **center, left, right, top, bottom** of the image
- **Near and far** from the camera
- **Tilted / rotated** (~30–45°) for distortion estimation
- Sharp frames only (no motion blur)

More samples with more variety:

```bash
python live_pose_any.py --calibrate-only --camera 0 --calibration-min-samples 25
```

### Step 4 — Check quality (RMS reprojection error)

When calibration finishes you will see a summary including **RMS reprojection error** in pixels.

| RMS error | Quality |
|-----------|---------|
| **< 0.5 px** | Good |
| 0.5 – 1.0 px | Acceptable |
| **> 1.0 px** | Re-calibrate — more views, better lighting, flatter board |

**Inspect a saved file anytime:**

```bash
python live_pose_any.py --show-calibration
```

Or read the `.npz` directly:

```bash
python3 -c "import numpy as np; d=np.load('camera_calibration.npz'); print(f'RMS: {float(d[\"reprojection_error\"]):.4f} px')"
```

On every tracking run, the startup log prints the same summary when `camera_calibration.npz` is auto-loaded.

### What is saved in `camera_calibration.npz`

| Field | Meaning |
|-------|---------|
| `K`, `dist` | Camera intrinsics and distortion |
| `reprojection_error` | RMS quality metric (px) |
| `checkerboard_cols/rows` | Inner corners used |
| `square_cm` | Physical square size in cm |
| `num_samples` | Number of captured views |
| `width`, `height` | Resolution used during calibration |

The file is saved locally and **auto-loaded** on future runs (gitignored — not committed).

### Metric translation (cm)

With checkerboard calibration loaded, **no extra flags are needed** for cm poses. Optional overrides if SAM3D mesh scale is still slightly off:

```bash
python live_pose_any.py --camera 0 --surface-distance-cm 40   # known camera-to-object depth (cm)
python live_pose_any.py --camera 0 --tool-width-cm 1.2        # known tool width (cm)
```

If depth looks wrong after switching from an older mm-based build, delete cached `sam3d_live_objects/*_repaired.glb` files and re-run tracking.

---

## Run live tracking

**Prerequisites checklist:**

- [ ] `edgetam.pt` downloaded
- [ ] `pip install -e EdgeTAMLive/EdgeTAM` done
- [ ] `FAL_KEY` exported
- [ ] `camera_calibration.npz` exists (from calibration step)

```bash
cd EdgeTAMLive
export FAL_KEY="your-fal-api-key"    # if not already in your shell
python live_pose_any.py --camera 0
```

You should see:

```
Using checkerboard calibration:
Calibration file: .../camera_calibration.npz
  RMS reprojection error: 0.xx px (good)
  ...
Pose display: rotation (deg), translation (cm) — checkerboard cal (2 cm squares); SAM3D mesh scaled to cm for PnP
```

### Useful run options

```bash
python live_pose_any.py --camera 0 --output tracking.mp4    # save video
python live_pose_any.py --camera 0 --device mps             # Apple Silicon GPU
python live_pose_any.py --camera 0 --no-half                # disable FP16
python live_pose_any.py --calibrate-checkerboard --camera 0 # re-calibrate, then track
```

---

## Controls & workflow

### Pipeline steps

```
1. Live video opens
2. Click one seed point per tool → Enter
3. EdgeTAM computes masks on the frozen seed frame
4. Confirm masks → Y / Enter  (or N / Esc to abort)
5. fal SAM3D builds a GLB mesh per object (progress window)
6. Live tracking with pose axes + HUD
```

### Seed selection window

| Input | Action |
|-------|--------|
| **Left click** | Add seed point (first click freezes the frame) |
| **Backspace** | Remove last point |
| **C** | Clear all points and unfreeze |
| **Enter** | Start tracking (needs ≥ 1 point) |
| **Q** / **ESC** | Cancel |

### During tracking

| Input | Action |
|-------|--------|
| **Q** / **ESC** | Quit |

---

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Pose HUD | On-screen | `R(rx,ry,rz)deg  T(tx,ty,tz)cm` per object |
| CSV log | `posesN.csv` (cwd) | Per-frame Euler angles + `tx_cm`, `ty_cm`, `tz_cm` |
| Seed frame | `sam3d_live_objects/seed_frame.png` | Frozen RGB used for SAM3D |
| Masks | `sam3d_live_objects/mask_<id>.png` | Per-object masks sent to fal |
| GLB meshes | `sam3d_live_objects/object_<id>.glb` | 3D models from fal SAM3D |
| Video (optional) | `--output path.mp4` | Annotated recording |

---

## CLI reference

```bash
python live_pose_any.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | OpenCV camera index |
| `--device` | `auto` | `cuda`, `mps`, or `cpu` |
| `--calibrate-only` | off | Run checkerboard calibration and exit |
| `--show-calibration` | off | Print RMS / metadata from saved `.npz` and exit |
| `--calibrate-checkerboard` | off | Calibrate first, then track |
| `--calibration-out` | `camera_calibration.npz` | Calibration save/load path |
| `--checkerboard-cols` | `8` | Inner corners width (MRPT 9×7 board) |
| `--checkerboard-rows` | `6` | Inner corners height |
| `--checkerboard-square-cm` | `2` | Physical square size in cm (MRPT PDF → 2 cm / 20 mm) |
| `--calibration-min-samples` | `15` | Min views before calibrating |
| `--intrinsics-file` | auto | Override calibration `.npz` path |
| `--surface-distance-cm` | `0` | Known camera-to-surface distance |
| `--tool-width-cm` | `0` | Known tool width for metric scale (cm) |
| `--alpha` | `0.45` | Mask overlay opacity |
| `--kalman-process-var` | `5e-3` | Pose smoothing (lower = smoother) |
| `--kalman-meas-var` | `1e-3` | Kalman measurement noise |
| `--fal-model` | `fal-ai/sam-3/3d-objects` | fal model ID |
| `--seed` | `42` | SAM3D random seed |
| `--glb-dir` | `sam3d_live_objects/` | Masks & GLB output directory |
| `--output` | `""` | Save annotated MP4 |
| `--align-debug-out` | `""` | Per-frame registration debug PNG |
| `--no-orbbec-intrinsics` | off | Skip Orbbec SDK intrinsics lookup |
| `--no-half` | off | Disable FP16 inference |

---

## Repository layout

```
SurgicalToolsPose/
├── README.md
├── EdgeTAMLive/                     ← run everything from here
│   ├── live_pose_any.py             ← main application
│   ├── requirements.txt
│   ├── camera_calibration.npz       ← you create this (gitignored)
│   ├── sam3d_live_objects/          ← runtime masks & GLBs (gitignored)
│   └── EdgeTAM/                     ← segmentation backend
│       ├── checkpoints/
│       │   └── edgetam.pt           ← download separately (~56 MB)
│       └── configs/edgetam.yaml
└── archive/                         ← older experiments (reference only)
    ├── legacy_live/
    ├── experiments/
    ├── samples/sam3d_bootstrap/
    └── weights/
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Checkpoint not found: .../edgetam.pt` | Download from [releases](https://github.com/facebookresearch/EdgeTAM/releases/download/v1.0/edgetam.pt) into `EdgeTAMLive/EdgeTAM/checkpoints/` |
| `No module named 'sam2'` | Run `pip install -e EdgeTAMLive/EdgeTAM` |
| `Set FAL_KEY environment variable` | `export FAL_KEY=...` from [dashboard/keys](https://fal.ai/dashboard/keys) |
| `Could not open camera 0` | Try `--camera 1`; close other apps using the camera |
| Checkerboard not detected | Print PDF at **1:1** scale; improve lighting; hold board flat |
| Confused by board size | MRPT PDF = **9×7 squares** = **8×6 inner corners** (not 8×10 squares) |
| Wrong board size flags | MRPT 9×7 → `--checkerboard-cols 8 --checkerboard-rows 6 --checkerboard-square-cm 2`; other prints → inner corners = squares − 1 |
| RMS > 1.0 px | Capture 20+ varied views; flatten board; fix glare; verify 2 cm squares with ruler |
| `fal_client import failed` | `pip install fal-client` |
| `Install trimesh` | `pip install trimesh` |
| Pose jitter | `--kalman-process-var 2e-4` |
| Translation in cm looks wrong | Re-calibrate; confirm log shows `mesh units: metres→cm`; delete old `*_repaired.glb` caches if needed; optional `--surface-distance-cm` / `--tool-width-cm` |
| Orbbec upside-down image | Auto-detected; same rotation used in calibration and tracking |

---

## Links & references

### This pipeline

| Item | URL |
|------|-----|
| Checkerboard (MRPT 9×7, 2 cm squares) | https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf |
| fal SAM3D model | https://fal.ai/models/fal-ai/sam-3/3d-objects |
| fal API keys | https://fal.ai/dashboard/keys |
| fal documentation | https://fal.ai/docs |

### EdgeTAM

| Item | URL |
|------|-----|
| GitHub | https://github.com/facebookresearch/EdgeTAM |
| Checkpoint `edgetam.pt` | https://github.com/facebookresearch/EdgeTAM/releases/download/v1.0/edgetam.pt |
| Hugging Face | https://huggingface.co/facebook/EdgeTAM |
| Demo (Hugging Face Spaces) | https://huggingface.co/spaces/facebook/EdgeTAM |
| Paper | https://arxiv.org/abs/2501.07256 |

### Dependencies

| Item | URL |
|------|-----|
| PyTorch install | https://pytorch.org/get-started/locally/ |
| OpenCV | https://opencv.org/ |
| trimesh | https://trimesh.org/ |
| fal Python client | https://pypi.org/project/fal-client/ |

### Optional

| Item | URL |
|------|-----|
| Orbbec SDK (`pyorbbecsdk`) | Install from [Orbbec SDK](https://www.orbbec.com/developers/) for factory camera intrinsics |

---

## License

EdgeTAM is subject to its own [license](EdgeTAMLive/EdgeTAM/LICENSE). Application code in this repository is provided for research use.
