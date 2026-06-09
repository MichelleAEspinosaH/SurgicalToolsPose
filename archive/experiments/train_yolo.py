# ******************************************************************************
#  train_yolo.py
#
#  Fine-tunes a YOLOv8 detection model on the Roboflow surgical-tools dataset,
#  then saves the best weights so combined_viewer_v2.py can load them.
#
#  Run this ONCE before launching combined_viewer_v2.py:
#      python3 train_yolo.py
#
#  The trained model is saved to:
#      runs/detect/surgical_tools/weights/best.pt
#
#  Requirements (install if missing):
#      pip install roboflow ultralytics
# ******************************************************************************

from roboflow import Roboflow
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Step 1 — Download the dataset from Roboflow
# ---------------------------------------------------------------------------
# Workspace : covoice19-workspace
# Project   : surgical-tools-s93bt
# Format    : yolov8  (Ultralytics-compatible folder structure with data.yaml)
# ---------------------------------------------------------------------------

print("Connecting to Roboflow and downloading dataset...")

rf      = Roboflow(api_key="eQff4X3fd5dkiQUCaKm0")
project = rf.workspace("covoice19-workspace").project("surgical-tools-s93bt")

# Download the latest available version in YOLOv8 format.
# If the project has multiple versions, change the number below (e.g. version(2)).
version = project.version(1)
dataset = version.download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")
print(f"Classes: {dataset.classes}")

# ---------------------------------------------------------------------------
# Step 2 — Fine-tune YOLOv8 on the downloaded dataset
# ---------------------------------------------------------------------------
# We start from the official YOLOv8-nano weights ('yolov8n.pt') so training
# is fast even without a GPU.  Increase 'epochs' or switch to 'yolov8s.pt'
# for better accuracy if you have more time / compute.
#
# The data.yaml file (inside dataset.location) tells YOLO where the images
# and labels are and what the class names are.
# ---------------------------------------------------------------------------

print("\nStarting fine-tuning...")

model = YOLO("yolov8n.pt")   # base weights — downloads automatically if absent

results = model.train(
    data    = f"{dataset.location}/data.yaml",
    epochs  = 50,              # increase for better accuracy (e.g. 100-200)
    imgsz   = 640,             # standard YOLOv8 input size
    batch   = 16,              # lower to 8 if you run out of RAM
    name    = "surgical_tools",
    project = "runs/detect",
    exist_ok= True,            # overwrite previous run with the same name
    patience= 15,              # stop early if no improvement for 15 epochs
    verbose = True,
)

# ---------------------------------------------------------------------------
# Step 3 — Report where the best model was saved
# ---------------------------------------------------------------------------

best_weights = "runs/detect/surgical_tools/weights/best.pt"
print("\n" + "=" * 60)
print("Training complete.")
print(f"Best model saved to: {best_weights}")
print("Update YOLO_MODEL_PATH in combined_viewer_v2.py if the path differs.")
print("=" * 60)
