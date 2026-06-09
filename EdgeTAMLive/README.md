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

## Units (cm vs mm)

| Stage | Units | Notes |
|-------|-------|-------|
| Checkerboard calibration | **cm** | `--checkerboard-square-cm 2`; saved as `square_cm` in `.npz` |
| SAM3D mesh vertices | **cm** | fal GLB longest axis is 1 m → pipeline ×100 to cm |
| PnP `tvec` (internal) | **mm** | OpenCV solvePnP on the mesh |
| HUD & CSV translation | **cm** | Raw mm × `0.1` before display (`tx_cm`, `ty_cm`, `tz_cm`) |
| Rotation | **degrees** | Euler ZYX in HUD and CSV |

Legacy calibration files with `square_mm` still load (auto-converted to cm).

## Checkerboard (MRPT)

OpenCV counts **inner corners**, not squares:

| Print | MRPT PDF | CLI defaults |
|-------|----------|--------------|
| **9 × 7 squares** | yes | `--checkerboard-cols 8 --checkerboard-rows 6` |
| **8 × 6 inner corners** | yes | (same as defaults) |

Print the [MRPT PDF](https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf) at **1:1** (100%, not fit-to-page). Verify one square = **2 cm** with a ruler.

## fal SAM3D workflow

After you confirm EdgeTAM masks, fal runs **in parallel** (one job per object ID):

1. Upload mask → **uploading mask…**
2. Remote infer → **SAM3D running…** (often the slow step; can be 10–120 s per ID)
3. Download GLB → **downloading GLB…**

Console logs per-ID timing, e.g. `ID1 fal: OK — upload 0.3s, infer 13.2s, download 0.5s`.

Meshes are saved to `sam3d_live_objects/object_<id>.glb`. On registration, logs show mesh unit conversion and seed depth, e.g.:

```
[ID1] mesh units: SAM3D 1 m axis → cm (raw extent 1 m, now 100.0 cm)
[ID1] SAM3D depth scale: 1.023  (seed tz=40.0 cm, raw 400.0 mm → ~40.9 cm after pinhole fit)
```

## Common commands

```bash
python live_pose_any.py --calibrate-only --camera 0
python live_pose_any.py --show-calibration
python live_pose_any.py --camera 0 --device mps
python live_pose_any.py --camera 0 --output tracking.mp4
python live_pose_any.py --calibrate-checkerboard --camera 0
python live_pose_any.py --camera 0 --surface-distance-cm 40   # optional depth override
python live_pose_any.py --camera 0 --tool-width-cm 1.2        # optional width override
```

## Quick troubleshooting

| Issue | Fix |
|-------|-----|
| Depth ~10× too large in HUD | Should not happen after mm→cm fix; re-run with fresh GLBs |
| Wrong board flags | MRPT = 8×6 inner corners, not 7×9 |
| `TypeError: ... \| ...` on import | Use Python 3.9+; script uses postponed annotations |
| `No module named 'sam2'` | `pip install -e EdgeTAM` from this folder |
| Stale/wrong mesh scale | `rm sam3d_live_objects/*_repaired.glb` and re-track |
| fal very slow on one ID | Normal for complex masks; check console infer time |

See [../README.md](../README.md) for full CLI, calibration best practices, and troubleshooting.
