# EdgeTAMLive

Run the live surgical-tool pose pipeline from this directory.

**Full documentation:** [../README.md](../README.md)

## Quick start

```bash
# One-time setup (from repo root)
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision
pip install -r EdgeTAMLive/requirements.txt
pip install -e EdgeTAMLive/EdgeTAM
curl -L -o EdgeTAMLive/EdgeTAM/checkpoints/edgetam.pt \
  https://github.com/facebookresearch/EdgeTAM/releases/download/v1.0/edgetam.pt

# Calibrate camera (MRPT 9×7 board, 2 cm squares)
cd EdgeTAMLive
python live_pose_any.py --calibrate-only --camera 0

# Inspect calibration quality
python live_pose_any.py --show-calibration

# Live tracking
export FAL_KEY="your-key"
python live_pose_any.py --camera 0
```

## Main script

| File | Purpose |
|------|---------|
| `live_pose_any.py` | Calibration, EdgeTAM tracking, fal SAM3D meshes, 6DoF PnP pose |
| `requirements.txt` | Python dependencies (excluding EdgeTAM itself) |
| `camera_calibration.npz` | Your checkerboard intrinsics (created locally, gitignored) |
| `sam3d_live_objects/` | Runtime masks & GLBs per session (gitignored) |
| `EdgeTAM/` | Segmentation backend — install with `pip install -e EdgeTAM` |

## Units

- Checkerboard calibration: **cm** (`--checkerboard-square-cm 2`)
- Pose HUD & CSV: translation in **cm**, rotation in **degrees**
- MRPT board: **9×7 squares** = **8×6 inner corners** (OpenCV defaults)

## Common commands

```bash
python live_pose_any.py --calibrate-only --camera 0
python live_pose_any.py --show-calibration
python live_pose_any.py --camera 0 --device mps
python live_pose_any.py --camera 0 --output tracking.mp4
python live_pose_any.py --calibrate-checkerboard --camera 0
```

See [../README.md](../README.md) for CLI flags, troubleshooting, and calibration best practices.
