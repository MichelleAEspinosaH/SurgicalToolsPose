# SurgicalToolsPose

Live **6DoF pose estimation** for surgical tools from a single RGB camera (webcam or phone).

| Component | Role |
|-----------|------|
| **[EdgeTAM](https://github.com/facebookresearch/EdgeTAM)** | Instance segmentation & mask tracking |
| **[fal SAM3D](https://fal.ai/models/fal-ai/sam-3/3d-objects)** | One-shot 3D mesh (GLB) from RGB + mask |
| **OpenCV PnP + Kalman** | Dense 6DoF pose each frame |

**Main script:** [`EdgeTAMLive/live_pose_any.py`](EdgeTAMLive/live_pose_any.py) — run from `EdgeTAMLive/`.  
**Short guide:** [`EdgeTAMLive/README.md`](EdgeTAMLive/README.md)

---

## Quick start

```bash
git clone https://github.com/MichelleAEspinosaH/SurgicalToolsPose.git
cd SurgicalToolsPose

python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision
pip install -r EdgeTAMLive/requirements.txt
pip install -e EdgeTAMLive/EdgeTAM

# EdgeTAM checkpoint (~56 MB)
curl -L -o EdgeTAMLive/EdgeTAM/checkpoints/edgetam.pt \
  https://github.com/facebookresearch/EdgeTAM/releases/download/v1.0/edgetam.pt

# Checkerboard calibration (once)
cd EdgeTAMLive
python live_pose_any.py --calibrate-only --camera 0

# Live tracking
export FAL_KEY="your-fal-api-key"
python live_pose_any.py --camera 0
```

Pose translation is reported in **centimetres** when checkerboard calibration is loaded (PnP internally uses millimetres; the pipeline converts for display).

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Download models & API keys](#download-models--api-keys)
4. [Camera calibration](#camera-calibration)
5. [Run live tracking](#run-live-tracking)
6. [Controls & workflow](#controls--workflow)
7. [Outputs](#outputs)
8. [CLI reference](#cli-reference)
9. [Repository layout](#repository-layout)
10. [Local files (not in git)](#local-files-not-in-git)
11. [Troubleshooting](#troubleshooting)
12. [Links & references](#links--references)

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.9+** | 3.10+ recommended; EdgeTAM may warn on older builds |
| **Camera** | Webcam, phone (Continuity Camera / USB), or Orbbec |
| **GPU (recommended)** | NVIDIA CUDA or Apple Silicon (MPS); CPU works but is slow |
| **fal.ai account** | For SAM3D mesh generation ([sign up](https://fal.ai)) |
| **Checkerboard** | [MRPT 9×7 PDF](https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf) — print at **1:1 scale** on A4 |
| **Internet** | Needed once per session for fal SAM3D mesh download |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/MichelleAEspinosaH/SurgicalToolsPose.git
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

The live script expects:

```
EdgeTAMLive/EdgeTAM/checkpoints/edgetam.pt
```

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

Verify (~56 MB):

```bash
ls -lh EdgeTAMLive/EdgeTAM/checkpoints/edgetam.pt
```

> `EdgeTAM/checkpoints/download_ckpts.sh` downloads **SAM 2.1** weights, not `edgetam.pt`. Use the curl command above.

### fal.ai API key (SAM3D meshes)

1. Create an account at **[fal.ai](https://fal.ai)**
2. Open **[fal.ai/dashboard/keys](https://fal.ai/dashboard/keys)** and create an API key
3. Export before every run (or add to your shell profile):

```bash
export FAL_KEY="your-fal-api-key-here"
```

| Resource | Link |
|----------|------|
| SAM3D model | https://fal.ai/models/fal-ai/sam-3/3d-objects |
| API docs | https://fal.ai/docs |
| Python client | https://fal.ai/docs/clients/python |
| Pricing (~$0.02 / object) | https://fal.ai/pricing |

Default model ID: `fal-ai/sam-3/3d-objects` (override with `--fal-model`).

---

## Camera calibration

Calibration estimates camera **intrinsics** (focal length, distortion). Without it, pose translation in **cm** will be inaccurate.

### Coordinate system & units

The pipeline mixes **cm** (calibration, display) and **mm** (SAM3D mesh PnP). Keep this straight when reading logs or CSV.

| Stage | Units | What happens |
|-------|-------|----------------|
| Checkerboard object points | **cm** | `--checkerboard-square-cm 2`; stored as `square_cm` in `.npz` |
| Camera intrinsics `K`, `dist` | **pixels** | From checkerboard calibration |
| SAM3D GLB mesh | **m → cm** | fal normalizes longest axis to **1 m** (not real tool size); pipeline ×100 to cm for PnP |
| PnP translation (`tvec`) | **mm** | OpenCV `solvePnP` on mesh vertices |
| HUD & CSV (`tx_cm`, `ty_cm`, `tz_cm`) | **cm** | Raw mm × **0.1** before display |
| Rotation (HUD / CSV) | **degrees** | Euler ZYX |

**Example:** seed log `raw 400.0 mm` → HUD `tz ≈ 40.0 cm` at ~40 cm distance.

Optional CLI overrides compare **cm** against **mm** depth directly (no extra ×0.1):

```bash
python live_pose_any.py --camera 0 --surface-distance-cm 40
python live_pose_any.py --camera 0 --tool-width-cm 1.2
```

If depth still looks wrong after calibration, delete stale repaired meshes and re-run:

```bash
rm sam3d_live_objects/*_repaired.glb
```

Check startup logs for `[IDn] mesh units:` and `[IDn] SAM3D depth scale:`.

### Squares vs inner corners

OpenCV detects **inner corners** (where four squares meet), not square count.

| What you count on the print | MRPT PDF board | Different board (example) |
|-----------------------------|----------------|---------------------------|
| **Squares** (black/white cells) | **9 × 7** | 8 × 10 |
| **Inner corners** (`--checkerboard-cols/rows`) | **8 × 6** | 7 × 9 |

Rule: **inner corners = squares − 1** in each direction.

### Print the board

**[MRPT 9×7 checkerboard PDF](https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf)** — print at **100% / actual size** (1:1), not “fit to page”.

| Property | Value |
|----------|-------|
| Squares on print | 9 wide × 7 tall |
| Square size | **2 cm** (20 mm at 1:1) |
| CLI | `--checkerboard-cols 8 --checkerboard-rows 6 --checkerboard-square-cm 2` |

Verify one square measures **2 cm** with a ruler. Mount flat on stiff backing; avoid glare.

### Run calibration

Use the **same camera, index, zoom, and focus** as tracking:

```bash
cd EdgeTAMLive
python live_pose_any.py --calibrate-only --camera 0
```

| Key | Action |
|-----|--------|
| **SPACE** | Capture (when green corners detected) |
| **U** / Backspace | Undo last capture |
| **C** / **Enter** | Finish (≥ 15 samples) |
| **Q** / **ESC** | Cancel |

Capture **20–30 varied views**: center/corners of frame, near/far, tilted angles, sharp frames only.

```bash
python live_pose_any.py --calibrate-only --camera 0 --calibration-min-samples 25
```

### Check quality (RMS)

| RMS reprojection error | Quality |
|------------------------|---------|
| **< 0.5 px** | Good |
| 0.5 – 1.0 px | Acceptable |
| **> 1.0 px** | Re-calibrate |

```bash
python live_pose_any.py --show-calibration
```

The same summary prints when calibration is auto-loaded at tracking startup.

### Saved file: `camera_calibration.npz`

| Field | Meaning |
|-------|---------|
| `K`, `dist` | Intrinsics & distortion |
| `reprojection_error` | RMS quality (px) |
| `checkerboard_cols/rows` | Inner corners |
| `square_cm` | Physical square size (cm) |
| `num_samples` | Captured views |
| `width`, `height` | Calibration resolution |

Legacy files with `square_mm` still load (auto-converted to cm).

### Optional metric overrides

See [Coordinate system & units](#coordinate-system--units) above. `--surface-distance-cm` and `--tool-width-cm` bypass the default mm→cm mesh scale when you know the true depth or tool width.

---

## Run live tracking

**Checklist:**

- [ ] `edgetam.pt` downloaded
- [ ] `pip install -e EdgeTAMLive/EdgeTAM` done
- [ ] `FAL_KEY` exported
- [ ] `camera_calibration.npz` exists

```bash
cd EdgeTAMLive
export FAL_KEY="your-fal-api-key"
python live_pose_any.py --camera 0
```

Expected startup log:

```
Using checkerboard calibration:
Calibration file: .../camera_calibration.npz
  RMS reprojection error: 0.xx px (good)
  ...
Pose display: rotation (deg), translation (cm) — checkerboard cal (2 cm squares); SAM3D mesh scaled to cm for PnP
```

After mask confirmation, fal SAM3D runs **in parallel** per object (upload → infer → download). The seed-frame window shows status per ID; console logs timing, e.g. `ID1 fal: OK — upload 0.3s, infer 13.2s, download 0.5s`.

```bash
python live_pose_any.py --camera 0 --output tracking.mp4
python live_pose_any.py --camera 0 --device mps
python live_pose_any.py --calibrate-checkerboard --camera 0
```

---

## Controls & workflow

```
1. Live video opens
2. Click seed points per tool → Enter
3. EdgeTAM masks on frozen seed frame
4. Confirm masks → Y / Enter
5. fal SAM3D builds GLB per object (parallel; status overlay per ID)
6. Mesh registration + pinhole depth fit on seed mask
7. Live tracking with pose axes + HUD (translation in cm)
```

| Seed window | Action |
|-------------|--------|
| **Left click** | Add seed (first click freezes frame) |
| **Backspace** | Remove last point |
| **C** | Clear all / unfreeze |
| **Enter** | Start tracking |
| **Q** / **ESC** | Cancel / quit |

---

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Pose HUD | On-screen | `R(rx,ry,rz)deg  T(tx,ty,tz)cm` |
| CSV log | `posesN.csv` (cwd) | `tx_cm`, `ty_cm`, `tz_cm` + Euler angles |
| Seed frame | `sam3d_live_objects/seed_frame.png` | RGB sent to SAM3D |
| Masks | `sam3d_live_objects/mask_<id>.png` | Per-object masks |
| GLB meshes | `sam3d_live_objects/object_<id>.glb` | fal SAM3D models (1 m normalized axis) |
| Repaired GLB cache | `sam3d_live_objects/object_<id>_repaired.glb` | Local mesh cache (delete if scale wrong) |
| Video | `--output path.mp4` | Annotated recording |

---

## CLI reference

All commands run from `EdgeTAMLive/`:

```bash
python live_pose_any.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | OpenCV camera index |
| `--device` | `auto` | `cuda`, `mps`, or `cpu` |
| `--calibrate-only` | off | Calibrate and exit |
| `--show-calibration` | off | Print RMS / metadata from `.npz` and exit |
| `--calibrate-checkerboard` | off | Calibrate, then track |
| `--calibration-out` | `camera_calibration.npz` | Calib save/load path |
| `--checkerboard-cols` | `8` | Inner corners width (MRPT 9×7) |
| `--checkerboard-rows` | `6` | Inner corners height |
| `--checkerboard-square-cm` | `2` | Square size in cm |
| `--calibration-min-samples` | `15` | Min views before calibrating |
| `--intrinsics-file` | auto | Override calibration `.npz` |
| `--surface-distance-cm` | `0` | Known depth override (cm) |
| `--tool-width-cm` | `0` | Known tool width (cm); overrides auto scale |
| `--max-trail` | `60` | COM trail length (frames) |
| `--alpha` | `0.45` | Mask overlay opacity |
| `--kalman-process-var` | `5e-3` | Pose smoothing |
| `--kalman-meas-var` | `1e-3` | Kalman measurement noise |
| `--fal-model` | `fal-ai/sam-3/3d-objects` | fal model ID |
| `--seed` | `42` | SAM3D random seed |
| `--glb-dir` | `sam3d_live_objects/` | Masks & GLB directory |
| `--output` | `""` | Save annotated MP4 |
| `--align-debug-out` | `""` | Registration debug PNG |
| `--no-orbbec-intrinsics` | off | Skip Orbbec SDK intrinsics |
| `--no-half` | off | Disable FP16 inference |

---

## Repository layout

```
SurgicalToolsPose/
├── README.md                        ← you are here
├── .gitignore
├── EdgeTAMLive/                     ← run everything from here
│   ├── README.md                    ← quick reference
│   ├── live_pose_any.py             ← main application
│   ├── requirements.txt
│   ├── camera_calibration.npz       ← local (gitignored)
│   ├── sam3d_live_objects/          ← runtime artifacts (gitignored)
│   └── EdgeTAM/                     ← segmentation backend
│       ├── checkpoints/edgetam.pt   ← download separately
│       └── configs/edgetam.yaml
└── archive/                         ← legacy experiments (reference only)
    ├── README.md
    ├── legacy_live/
    ├── experiments/
    ├── samples/sam3d_bootstrap/
    └── weights/
```

---

## Local files (not in git)

These may exist on your machine but are **not committed**:

| Path | Purpose |
|------|---------|
| `.venv/` | Python virtual environment |
| `camera_calibration.npz` | Your checkerboard intrinsics |
| `sam3d_live_objects/` | Session masks, GLBs, repaired meshes |
| `poses*.csv` | Pose logs from tracking runs |
| `*.pt` / `*.mp4` | Downloaded weights & recordings |
| `pyorbbecsdk/`, `sam3/`, `segment-anything-2/` | Optional local SDK clones |
| `tests/` | **Obsolete** — duplicate of `archive/experiments/`; safe to delete |

To clean a local workspace:

```bash
rm -rf tests/ __pycache__ poses*.csv
rm -rf sam3d_live_objects/*_repaired.glb   # if mesh scale looks wrong
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `TypeError: unsupported operand type(s) for \|` on import | Use **Python 3.9+**; script uses postponed type annotations |
| `Checkpoint not found: .../edgetam.pt` | [Download edgetam.pt](https://github.com/facebookresearch/EdgeTAM/releases/download/v1.0/edgetam.pt) into `EdgeTAMLive/EdgeTAM/checkpoints/` |
| `No module named 'sam2'` | `pip install -e EdgeTAMLive/EdgeTAM` |
| `Set FAL_KEY environment variable` | `export FAL_KEY=...` from [dashboard/keys](https://fal.ai/dashboard/keys) |
| `Could not open camera 0` | Try `--camera 1`; close other camera apps |
| Checkerboard not detected | Print at **1:1**; improve lighting; flat board |
| Confused by board size | MRPT = **9×7 squares** = **8×6 inner corners** |
| Wrong board flags | Inner corners = squares − 1; MRPT → `--checkerboard-cols 8 --checkerboard-rows 6` |
| Calibrated with wrong board (e.g. 7×9) | Re-calibrate with correct MRPT flags |
| RMS > 1.0 px | 20+ varied views; verify 2 cm squares with ruler |
| `fal_client import failed` | `pip install fal-client` |
| `Install trimesh` | `pip install trimesh` |
| fal SAM3D slow (60–120 s per ID) | Normal for remote infer; check console `infer` time; WiFi rarely the bottleneck |
| Pose jitter | `--kalman-process-var 2e-4` |
| Translation in cm wrong (~10× off) | PnP is mm internally; HUD should show cm (×0.1). If still wrong: re-calibrate, check `[IDn] mesh units:` log, delete `*_repaired.glb` |
| Orbbec upside-down | Auto-detected; same rotation in cal & tracking |

---

## Links & references

### This pipeline

| Item | URL |
|------|-----|
| Checkerboard (MRPT 9×7, 2 cm) | https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf |
| fal SAM3D | https://fal.ai/models/fal-ai/sam-3/3d-objects |
| fal API keys | https://fal.ai/dashboard/keys |

### EdgeTAM

| Item | URL |
|------|-----|
| GitHub | https://github.com/facebookresearch/EdgeTAM |
| Checkpoint | https://github.com/facebookresearch/EdgeTAM/releases/download/v1.0/edgetam.pt |
| Paper | https://arxiv.org/abs/2501.07256 |

### Dependencies

| Item | URL |
|------|-----|
| PyTorch | https://pytorch.org/get-started/locally/ |
| OpenCV | https://opencv.org/ |
| trimesh | https://trimesh.org/ |
| fal-client | https://pypi.org/project/fal-client/ |

---

## License

EdgeTAM is subject to its own [license](EdgeTAMLive/EdgeTAM/LICENSE). Application code in this repository is provided for research use.
