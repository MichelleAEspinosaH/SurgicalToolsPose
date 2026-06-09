# Archive

Older scripts, experiments, and sample assets from earlier development.
**Not required** to run the current pipeline — see the root [README](../README.md) and [EdgeTAMLive/README.md](../EdgeTAMLive/README.md).

The active entry point is `EdgeTAMLive/live_pose_any.py` (checkerboard calibration in **cm**, metric HUD/CSV in **cm**; SAM3D PnP internally in **mm**).

## Layout

| Folder | Contents |
|--------|----------|
| `legacy_live/` | Previous live tracking scripts (`live_track.py`, `live_track_copy.py`, `live_track_pose.py`) |
| `experiments/` | One-off tests, YOLO trainers, SAM2/SAM3 prototypes, fal smoke tests |
| `samples/sam3d_bootstrap/` | Example GLB meshes from early SAM3D bootstrap runs |
| `weights/` | `yolo26n-seg.pt` and other legacy model weights |

## Note

These files are kept for reference. Paths in docstrings may still say `tests/` — run from `archive/experiments/` instead, e.g.:

```bash
python archive/experiments/fal_sam3d_smoke_test.py --submit
```

Update imports if you reuse a script.
