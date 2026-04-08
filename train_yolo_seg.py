# ******************************************************************************
#  train_yolo_seg.py
#
#  Fine-tunes a YOLOv8 SEGMENTATION model on the Roboflow surgical-tools
#  dataset so combined_viewer_v5.py can load it for instance segmentation
#  and tracking.
#
#  Run this ONCE before launching combined_viewer_v5.py:
#      python3 train_yolo_seg.py
#
#  The trained model is saved to:
#      runs/segment/surgical_tools_seg/weights/best.pt
#
#  Requirements:
#      pip install roboflow ultralytics
#
#  Notes
#  -----
#  The script first tries to download the dataset in "yolov8-seg" format
#  (polygon masks).  If the Roboflow project only has bounding-box annotations,
#  it falls back to "yolov8" detection format.  Training a segmentation model
#  on detection-only data still works — the model learns to detect surgical
#  tools accurately; the segmentation head produces reasonable masks using
#  the pretrained priors from yolov8n-seg.pt.
# ******************************************************************************

from roboflow import Roboflow
from ultralytics import YOLO

ROBOFLOW_API_KEY = "eQff4X3fd5dkiQUCaKm0"
WORKSPACE        = "covoice19-workspace"
PROJECT          = "surgical-tools-s93bt"
VERSION          = 1

# ---------------------------------------------------------------------------
# Step 1 — Download dataset from Roboflow
# ---------------------------------------------------------------------------

print("Connecting to Roboflow...")
rf      = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)

dataset = None

# Try segmentation format first (polygon masks)
try:
    print("Attempting to download in yolov8-seg format (polygon masks)...")
    dataset = version.download("yolov8-seg")
    print(f"Segmentation dataset downloaded to: {dataset.location}")
except Exception as e:
    print(f"  yolov8-seg not available ({e})")
    print("  Falling back to yolov8 detection format...")

# Fall back to detection format
if dataset is None:
    dataset = version.download("yolov8")
    print(f"Detection dataset downloaded to: {dataset.location}")

print(f"Classes: {dataset.classes}")

# ---------------------------------------------------------------------------
# Step 2 — Fine-tune YOLOv8-nano segmentation model
# ---------------------------------------------------------------------------
# yolov8n-seg.pt is the nano segmentation model — fast to train, good on CPU.
# Swap for yolov8s-seg.pt or yolov8m-seg.pt for more accuracy if you have
# a GPU and more time.

print("\nStarting segmentation model fine-tuning...")

model = YOLO("yolov8n-seg.pt")   # downloads automatically if absent

results = model.train(
    data     = f"{dataset.location}/data.yaml",
    epochs   = 50,
    imgsz    = 640,
    batch    = 16,          # lower to 8 if you run out of RAM
    name     = "surgical_tools_seg",
    project  = "runs/segment",
    exist_ok = True,
    patience = 15,
    verbose  = True,
)

# ---------------------------------------------------------------------------
# Step 3 — Report
# ---------------------------------------------------------------------------

best = "runs/segment/surgical_tools_seg/weights/best.pt"
print("\n" + "=" * 60)
print("Training complete.")
print(f"Best segmentation model saved to: {best}")
print("combined_viewer_v5.py will load this automatically on next launch.")
print("=" * 60)
