# ******************************************************************************
#  train_yolo_seg.py
#
#  Fine-tunes a YOLOv8 SEGMENTATION model on TWO Roboflow surgical-tools
#  datasets merged into one.  Run this ONCE before launching v4/v5:
#      python3 train_yolo_seg.py
#
#  Datasets used
#  -------------
#  1. covoice19-workspace / surgical-tools-s93bt  (original)
#  2. esra-m / surgical-tools-ks5sk               (added)
#
#  The two datasets are downloaded, their class lists are unified, labels from
#  dataset 2 are remapped to the unified IDs, and all images+labels are merged
#  into a single directory before training.
#
#  Output
#  ------
#  runs/segment/surgical_tools_seg/weights/best.pt
#  (combined_viewer_v4.py and v5.py load this automatically on next launch)
#
#  Requirements
#  ------------
#  pip install roboflow ultralytics pyyaml
# ******************************************************************************

import os
import shutil
from pathlib import Path

import yaml
from roboflow import Roboflow
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

ROBOFLOW_API_KEY = "eQff4X3fd5dkiQUCaKm0"

DATASETS = [
    dict(workspace="covoice19-workspace", project="surgical-tools-s93bt", version=1),
    dict(workspace="esra-m",              project="surgical-tools-ks5sk",  version=1),
]

MERGED_DIR = Path("merged_dataset")

# ---------------------------------------------------------------------------
# Helper: load data.yaml from a downloaded Roboflow dataset
# ---------------------------------------------------------------------------

def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _class_names(data_yaml: dict) -> list:
    """Return the list of class name strings from a data.yaml dict."""
    names = data_yaml.get("names", [])
    if isinstance(names, dict):          # some Roboflow YAMLs use {0: 'cls', ...}
        names = [names[k] for k in sorted(names)]
    return list(names)


# ---------------------------------------------------------------------------
# Helper: copy images and remap labels from one split into the merged dir
# ---------------------------------------------------------------------------

SPLIT_ALIASES = {
    "train": ["train"],
    "valid": ["valid", "val", "validation"],
    "test":  ["test"],
}

def _copy_split(src_root: Path, prefix: str, split: str,
                id_map: dict, out_root: Path):
    """
    Copy all images and (remapped) labels from one dataset split.
    prefix is prepended to every filename so files from different datasets
    never collide (e.g. 'ds0_', 'ds1_').
    """
    candidates = SPLIT_ALIASES.get(split, [split])
    src_img_dir = src_lbl_dir = None
    for alias in candidates:
        p = src_root / alias / "images"
        if p.exists():
            src_img_dir = p
            src_lbl_dir = src_root / alias / "labels"
            break
    if src_img_dir is None:
        return 0   # this split not present in dataset

    out_img_dir = out_root / split / "images"
    out_lbl_dir = out_root / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path in src_img_dir.glob("*"):
        dst_img = out_img_dir / f"{prefix}{img_path.name}"
        shutil.copy(img_path, dst_img)

        lbl_path = src_lbl_dir / (img_path.stem + ".txt") if src_lbl_dir else None
        dst_lbl  = out_lbl_dir / f"{prefix}{img_path.stem}.txt"

        if lbl_path and lbl_path.exists():
            lines = lbl_path.read_text().strip().splitlines()
            new_lines = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                old_id = int(parts[0])
                new_id = id_map.get(old_id, old_id)
                new_lines.append(f"{new_id} " + " ".join(parts[1:]))
            dst_lbl.write_text("\n".join(new_lines))
        else:
            # No label file → empty (background image)
            dst_lbl.write_text("")

        copied += 1
    return copied


# ---------------------------------------------------------------------------
# Step 1 — Download both datasets
# ---------------------------------------------------------------------------

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
downloaded = []

for ds in DATASETS:
    print(f"\nDownloading {ds['workspace']}/{ds['project']} v{ds['version']}…")
    project = rf.workspace(ds["workspace"]).project(ds["project"])
    version = project.version(ds["version"])

    dataset = None
    try:
        print("  Trying yolov8-seg format (polygon masks)…")
        dataset = version.download("yolov8-seg")
        print(f"  Downloaded (seg) → {dataset.location}")
    except Exception as e:
        print(f"  yolov8-seg unavailable ({e}) — falling back to yolov8 detection format")

    if dataset is None:
        dataset = version.download("yolov8")
        print(f"  Downloaded (det) → {dataset.location}")

    downloaded.append(dataset)

# ---------------------------------------------------------------------------
# Step 2 — Build unified class list and per-dataset ID maps
# ---------------------------------------------------------------------------

print("\nBuilding unified class list…")

all_names = []   # unified ordered list
id_maps   = []   # list of {old_id → new_id} dicts, one per dataset

for ds in downloaded:
    yaml_dict = _load_yaml(os.path.join(ds.location, "data.yaml"))
    names     = _class_names(yaml_dict)
    id_map    = {}
    for old_id, name in enumerate(names):
        if name not in all_names:
            all_names.append(name)
        id_map[old_id] = all_names.index(name)
    id_maps.append(id_map)

print(f"  {len(all_names)} unified classes: {all_names}")

# ---------------------------------------------------------------------------
# Step 3 — Merge all images + labels into MERGED_DIR
# ---------------------------------------------------------------------------

print(f"\nMerging datasets into {MERGED_DIR} …")

if MERGED_DIR.exists():
    shutil.rmtree(MERGED_DIR)

total = 0
for idx, (ds, id_map) in enumerate(zip(downloaded, id_maps)):
    prefix   = f"ds{idx}_"
    src_root = Path(ds.location)
    for split in ["train", "valid", "test"]:
        n = _copy_split(src_root, prefix, split, id_map, MERGED_DIR)
        if n:
            print(f"  ds{idx}  {split}: {n} images")
            total += n

print(f"  Total: {total} images merged.")

# Write merged data.yaml
merged_yaml = {
    "train": str((MERGED_DIR / "train" / "images").resolve()),
    "val":   str((MERGED_DIR / "valid" / "images").resolve()),
    "test":  str((MERGED_DIR / "test"  / "images").resolve()),
    "nc":    len(all_names),
    "names": all_names,
}
merged_yaml_path = MERGED_DIR / "data.yaml"
with open(merged_yaml_path, "w") as f:
    yaml.dump(merged_yaml, f, default_flow_style=False, allow_unicode=True)

print(f"  Wrote {merged_yaml_path}")

# ---------------------------------------------------------------------------
# Step 4 — Fine-tune YOLOv8-nano segmentation model on merged dataset
# ---------------------------------------------------------------------------

print("\nStarting segmentation model fine-tuning on merged dataset…")

model   = YOLO("yolov8n-seg.pt")   # downloads automatically if absent
results = model.train(
    data     = str(merged_yaml_path),
    epochs   = 60,        # more epochs to handle larger combined dataset
    imgsz    = 640,
    batch    = 16,        # lower to 8 if you run out of RAM
    name     = "surgical_tools_seg",
    project  = "runs/segment",
    exist_ok = True,
    patience = 15,
    verbose  = True,
)

# ---------------------------------------------------------------------------
# Step 5 — Report
# ---------------------------------------------------------------------------

best = "runs/segment/surgical_tools_seg/weights/best.pt"
print("\n" + "=" * 60)
print("Training complete.")
print(f"Best segmentation model: {best}")
print(f"Unified classes ({len(all_names)}): {all_names}")
print("combined_viewer_v4.py and v5.py will load this automatically.")
print("=" * 60)
