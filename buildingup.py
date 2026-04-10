from roboflow import Roboflow
from ultralytics import YOLO
import os
import shutil
from pathlib import Path
import yaml

# Roboflow
API_KEY = "eQff4X3fd5dkiQUCaKm0"
DATASETS = [
    {"workspace": "covoice19-workspace", "project": "surgical-tools-s93bt", "version": 1},
    {"workspace": "esra-m", "project": "surgical-tools-ks5sk", "version": 1},
]

# Paths
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_ROOT = ROOT / "dataset_sources"
MERGED_ROOT = ROOT / "dataset_merged"
MERGED_YAML = MERGED_ROOT / "data.yaml"

SPLIT_ALIASES = {
    "train": ["train"],
    "val": ["val", "valid", "validation"],
    "test": ["test"],
}


def class_names(data_yaml: dict) -> list:
    names = data_yaml.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def resolve_split_dirs(ds_root: Path, split: str):
    for alias in SPLIT_ALIASES[split]:
        img_dir = ds_root / alias / "images"
        lbl_dir = ds_root / alias / "labels"
        if img_dir.exists():
            return img_dir, lbl_dir
    return None, None


def ensure_out_dirs(split: str):
    out_img = MERGED_ROOT / split / "images"
    out_lbl = MERGED_ROOT / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    return out_img, out_lbl


def remap_and_copy_split(ds_root: Path, ds_prefix: str, split: str, id_map: dict):
    src_img, src_lbl = resolve_split_dirs(ds_root, split)
    if src_img is None:
        return 0

    out_img, out_lbl = ensure_out_dirs(split)
    count = 0

    for img_path in src_img.glob("*"):
        if not img_path.is_file():
            continue
        dst_img = out_img / f"{ds_prefix}_{img_path.name}"
        shutil.copy(img_path, dst_img)

        src_txt = src_lbl / f"{img_path.stem}.txt" if src_lbl else None
        dst_txt = out_lbl / f"{ds_prefix}_{img_path.stem}.txt"

        if src_txt and src_txt.exists():
            lines = src_txt.read_text().strip().splitlines()
            remapped = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                old_id = int(parts[0])
                new_id = id_map[old_id]
                remapped.append(f"{new_id} " + " ".join(parts[1:]))
            dst_txt.write_text("\n".join(remapped))
        else:
            dst_txt.write_text("")

        count += 1

    return count


def main():
    print("Downloading both Roboflow datasets...")
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=API_KEY)
    downloaded = []

    for i, ds in enumerate(DATASETS):
        target = DOWNLOAD_ROOT / f"ds{i}_{ds['project']}"
        target.mkdir(parents=True, exist_ok=True)
        dataset = rf.workspace(ds["workspace"]).project(ds["project"]).version(ds["version"]).download(
            "yolov8",
            location=str(target),
            overwrite=True,
        )
        downloaded.append(Path(dataset.location))
        print(f"  Downloaded {ds['workspace']}/{ds['project']} -> {dataset.location}")

    print("Building unified class map...")
    unified_names = []
    per_dataset_id_maps = []

    for ds_root in downloaded:
        yaml_path = ds_root / "data.yaml"
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        names = class_names(cfg)
        id_map = {}
        for old_id, cls_name in enumerate(names):
            if cls_name not in unified_names:
                unified_names.append(cls_name)
            id_map[old_id] = unified_names.index(cls_name)
        per_dataset_id_maps.append(id_map)

    print(f"  Unified classes ({len(unified_names)}): {unified_names}")

    if MERGED_ROOT.exists():
        shutil.rmtree(MERGED_ROOT)

    total = 0
    for i, (ds_root, id_map) in enumerate(zip(downloaded, per_dataset_id_maps)):
        prefix = f"ds{i}"
        for split in ("train", "val", "test"):
            n = remap_and_copy_split(ds_root, prefix, split, id_map)
            total += n
            if n:
                print(f"  {prefix} {split}: {n} images")

    print(f"Merged total images: {total}")

    merged_cfg = {
        "train": str((MERGED_ROOT / "train" / "images").resolve()),
        "val": str((MERGED_ROOT / "val" / "images").resolve()),
        "nc": len(unified_names),
        "names": unified_names,
    }
    test_dir = MERGED_ROOT / "test" / "images"
    if test_dir.exists() and any(test_dir.iterdir()):
        merged_cfg["test"] = str(test_dir.resolve())

    with open(MERGED_YAML, "w") as f:
        yaml.dump(merged_cfg, f)

    print(f"Wrote merged YAML: {MERGED_YAML}")

    print("Starting YOLO fine-tuning...")
    model = YOLO("yolov8s.pt")
    model.train(
        data=str(MERGED_YAML),
        epochs=10,
        imgsz=640,
        batch=16,
        device="mps",
        workers=0,
        project="surgical_tools_ft",
        name="run1",
        exist_ok=True,
    )

    print("\nDone. Weights saved to: runs/detect/surgical_tools_ft/run1/weights/best.pt")


if __name__ == "__main__":
    main()
import argparse
import os
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

ESC_KEY = 27
PRINT_INTERVAL = 1
MIN_DEPTH = 20
MAX_DEPTH = 10000

MERGED_DIR = Path("merged_dataset_buildingup")
ROBOFLOW_DATASETS = [
    dict(workspace="esra-m", project="surgical-tools-ks5sk", version_key="v_ks5sk"),
    dict(workspace="covoice19-workspace", project="surgical-tools-s93bt", version_key="v_s93bt"),
]
SPLIT_ALIASES = {
    "train": ["train"],
    "valid": ["valid", "val", "validation"],
    "test": ["test"],
}


def default_weights_path():
    candidates = (
        Path("runs/detect/buildingup/weights/best.pt"),
        Path("runs/segment/buildingup/weights/best.pt"),
        Path("yolov8s.pt"),
    )
    for path in candidates:
        if path.is_file():
            return str(path)
    return "yolov8s.pt"


def draw_yolo_boxes(image, model, conf=0.15, iou=0.5, max_det=300):
    if model is None:
        return image
    result = model(image, verbose=False, conf=conf, iou=iou, max_det=max_det)
    out = image.copy()
    if not result or len(result[0].boxes) == 0:
        return out
    r = result[0]
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = r.names[cls_id]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{name} {conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


class DepthReader:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.latest_image = None
        self.lock = threading.Lock()
        self.running = True
        self.temporal_filter = TemporalFilter(alpha=0.5)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        last_print = time.time()
        while self.running:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None or depth_frame.get_format() != OBFormat.Y16:
                continue

            w, h = depth_frame.get_width(), depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
            data = data.astype(np.float32) * scale
            data = np.where((data > MIN_DEPTH) & (data < MAX_DEPTH), data, 0).astype(np.uint16)
            data = self.temporal_filter.process(data)

            now = time.time()
            if now - last_print >= PRINT_INTERVAL:
                print("Center distance:", data[h // 2, w // 2], "cm")
                last_print = now

            img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            with self.lock:
                self.latest_image = img

    def get_latest(self):
        with self.lock:
            return self.latest_image

    def stop(self):
        self.running = False
        self._thread.join()


def _load_data_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _class_names_list(data_yaml: dict) -> list:
    names = data_yaml.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def _copy_split_merged(src_root: Path, prefix: str, split: str, id_map: dict, out_root: Path):
    candidates = SPLIT_ALIASES.get(split, [split])
    src_img_dir = None
    src_lbl_dir = None
    for alias in candidates:
        p = src_root / alias / "images"
        if p.exists():
            src_img_dir = p
            src_lbl_dir = src_root / alias / "labels"
            break
    if src_img_dir is None:
        return 0

    out_img_dir = out_root / split / "images"
    out_lbl_dir = out_root / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path in src_img_dir.glob("*"):
        if not img_path.is_file():
            continue
        dst_img = out_img_dir / f"{prefix}{img_path.name}"
        shutil.copy(img_path, dst_img)

        lbl_path = src_lbl_dir / (img_path.stem + ".txt") if src_lbl_dir else None
        dst_lbl = out_lbl_dir / f"{prefix}{img_path.stem}.txt"

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
            dst_lbl.write_text("")

        copied += 1
    return copied


def run_roboflow_training(args):
    from roboflow import Roboflow

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY or pass --api-key.")

    rf = Roboflow(api_key=api_key)
    downloaded = []
    for ds in ROBOFLOW_DATASETS:
        ver = getattr(args, ds["version_key"])
        print(f"\nDownloading {ds['workspace']}/{ds['project']} v{ver}...")
        project = rf.workspace(ds["workspace"]).project(ds["project"])
        version = project.version(ver)
        try:
            dataset = version.download("yolov8-seg")
            print(f"  Downloaded (seg) -> {dataset.location}")
        except Exception:
            dataset = version.download("yolov8")
            print(f"  Downloaded (det) -> {dataset.location}")
        downloaded.append(dataset)

    all_names = []
    id_maps = []
    for ds in downloaded:
        yaml_dict = _load_data_yaml(os.path.join(ds.location, "data.yaml"))
        names = _class_names_list(yaml_dict)
        id_map = {}
        for old_id, name in enumerate(names):
            if name not in all_names:
                all_names.append(name)
            id_map[old_id] = all_names.index(name)
        id_maps.append(id_map)

    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    for idx, (ds, id_map) in enumerate(zip(downloaded, id_maps)):
        prefix = f"ds{idx}_"
        src_root = Path(ds.location)
        for split in ("train", "valid", "test"):
            _copy_split_merged(src_root, prefix, split, id_map, MERGED_DIR)

    merged_yaml_path = MERGED_DIR / "data.yaml"
    merged = {
        "train": str((MERGED_DIR / "train" / "images").resolve()),
        "val": str((MERGED_DIR / "valid" / "images").resolve()),
        "nc": len(all_names),
        "names": all_names,
    }
    test_img = MERGED_DIR / "test" / "images"
    if test_img.exists() and any(test_img.iterdir()):
        merged["test"] = str(test_img.resolve())
    with open(merged_yaml_path, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, allow_unicode=True)

    model = YOLO(args.model)
    task = "segment" if args.model.endswith("-seg.pt") or "seg" in args.model.lower() else "detect"
    project_dir = "runs/segment" if task == "segment" else "runs/detect"
    model.train(
        data=str(merged_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="buildingup",
        project=project_dir,
        device=args.device,
        workers=args.workers,
        exist_ok=True,
        patience=args.patience,
        verbose=True,
    )


def run_live(weights=None, conf=0.15, iou=0.5, max_det=300):
    weights_path = weights or default_weights_path()
    try:
        yolo_model = YOLO(weights_path)
        print("YOLO model loaded:", weights_path)
        if weights_path in {"yolov8s.pt", "yolov8n.pt", "yolo26n-seg.pt"}:
            print("Warning: running generic model (not your fine-tuned Roboflow weights).")
    except Exception as e:
        print("Could not load YOLO model:", e)
        yolo_model = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open RGB camera.")
        return

    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None
        depth_profile = profile_list.get_default_video_stream_profile()
        assert depth_profile is not None
        config.enable_stream(depth_profile)
    except Exception as e:
        print("Could not configure depth stream:", e)
        cap.release()
        return

    pipeline.start(config)
    depth_reader = DepthReader(pipeline)

    print("Press q or ESC to quit.")
    while True:
        ret, color_image = cap.read()
        if not ret:
            break
        color_image = cv2.rotate(color_image, cv2.ROTATE_180)
        rgb_with_boxes = draw_yolo_boxes(
            color_image, yolo_model, conf=conf, iou=iou, max_det=max_det
        )
        cv2.imshow("RGB Camera", rgb_with_boxes)

        depth_image = depth_reader.get_latest()
        if depth_image is not None:
            depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
            cv2.imshow("Depth Viewer", depth_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ESC_KEY:
            break

    depth_reader.stop()
    cap.release()
    cv2.destroyAllWindows()
    pipeline.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Read RGB+depth (180 deg) or train YOLO on two Roboflow datasets."
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--v-ks5sk", type=int, default=1)
    parser.add_argument("--v-s93bt", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    # Training execution disabled to prevent accidental retraining.
    # if args.train:
    #     run_roboflow_training(args)
    # else:
    #     run_live(weights=args.weights, conf=args.conf, iou=args.iou, max_det=args.max_det)
    run_live(weights=args.weights, conf=args.conf, iou=args.iou, max_det=args.max_det)


if __name__ == "__main__":
    main()
