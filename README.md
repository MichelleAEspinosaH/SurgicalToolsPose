# SurgicalToolsPose

Real-time surgical tool pose estimation combining an **Orbbec depth camera** and an **OpenCV RGB feed**.  
Features include YOLO object detection, SIFT keypoint matching, Lucas-Kanade optical flow tracking, Center-of-Mass (COM) estimation, and solvePnP 3-D pose axes.

---

## Repository Files

| File | Description |
|---|---|
| `combined_viewer.py` | Main tracker — YOLO + SIFT + Lucas-Kanade optical flow + solvePnP pose axes |
| `combined_viewer_v2.py` | V2 variant — fine-tuned YOLO detection, manual ROI (`t` key), excluded-class filtering |
| `train_yolo.py` | Fine-tuning script for the Roboflow surgical-tools dataset |
| `depth_frame.py` | Original Orbbec depth reference script |
| `rgb_test.py` | Original OpenCV RGB capture reference script |
| `pipeline.py` | Original SIFT matching pipeline reference script |
| `yolo26n-seg.pt` | YOLO segmentation model weights |
| `.gitignore` | Excludes `pyorbbecsdk/`, `runs/`, `Log/`, caches |

---

## How to Run

```bash
# (Optional) Fine-tune YOLO on the surgical-tools dataset first
python3 train_yolo.py

# Launch the tracker
python3 combined_viewer_v2.py
```

## Keyboard Controls

| Key | Action |
|---|---|
| `r` | Capture reference frame — YOLO auto-detects the bounding box |
| `t` | Freeze frame and draw a manual ROI if YOLO misses the object |
| `q` / `ESC` | Quit |
