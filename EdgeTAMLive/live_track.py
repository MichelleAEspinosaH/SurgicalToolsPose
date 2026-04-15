#!/usr/bin/env python3
"""
Real-time surgical tool tracking with EdgeTAM.

Opens the Orbbec RGB camera (or any camera index), lets you click seed
points on the first frame, then streams live masks + oriented 3D cubes.

Usage:
    .venv/bin/python live_track.py                   # camera 0, default settings
    .venv/bin/python live_track.py --camera 1        # different camera index
    .venv/bin/python live_track.py --axis-smooth 0.9 # heavier axis smoothing
    .venv/bin/python live_track.py --no-half         # disable half-precision
    .venv/bin/python live_track.py --output out.mp4  # save output video
"""

import argparse
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SIZE = (640, 360)
EDGETAM_REPO = Path(__file__).parent / "EdgeTAM"
CHECKPOINT = EDGETAM_REPO / "checkpoints" / "edgetam.pt"
MODEL_CFG = "configs/edgetam.yaml"

_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_IMG_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

# ---------------------------------------------------------------------------
# EdgeTAM loader
# ---------------------------------------------------------------------------

def _load_predictor(device: str):
    repo = str(EDGETAM_REPO.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore
    return build_sam2_video_predictor(MODEL_CFG, str(CHECKPOINT), device=device)


def _autocast_config(device: str, use_half: bool) -> tuple[str, torch.dtype, bool]:
    device_type = device.split(":")[0]
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float16
    enabled = use_half and device_type != "cpu"
    return device_type, dtype, enabled


def choose_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray) -> np.ndarray:
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    return cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)

# ---------------------------------------------------------------------------
# Point-picking UI
# ---------------------------------------------------------------------------

def pick_points(first_frame: np.ndarray) -> list[tuple[int, float, float]]:
    win = "Select EdgeTAM points"
    points: list[tuple[int, float, float]] = []

    def draw() -> np.ndarray:
        vis = first_frame.copy()
        for obj_id, px_f, py_f in points:
            px, py = int(px_f), int(py_f)
            cv2.circle(vis, (px, py), 6, (0, 255, 255), -1)
            cv2.circle(vis, (px, py), 9, (255, 255, 255), 1)
            cv2.putText(vis, f"ID{obj_id}", (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(
            vis,
            "Left click: new object point | Backspace: undo | c: clear | Enter: start",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )
        return vis

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((len(points) + 1, float(x), float(y)))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    cancelled = False
    while True:
        cv2.imshow(win, draw())
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and points:
            break
        elif k in (8, 127) and points:
            points.pop()
        elif k == ord("c"):
            points.clear()
        elif k in (ord("q"), 27):
            cancelled = True
            break
    cv2.destroyWindow(win)
    return [] if cancelled else points

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def point_color(obj_id: int) -> tuple[int, int, int]:
    hue = (obj_id * 47 + 20) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _mask_to_2d_bool(m: np.ndarray, fh: int, fw: int) -> np.ndarray:
    x = np.squeeze(np.asarray(m, dtype=np.float32))
    while x.ndim > 2:
        x = x[0]
    if x.ndim != 2:
        return np.zeros((fh, fw), dtype=bool)
    if x.shape != (fh, fw):
        x = cv2.resize(x, (fw, fh), interpolation=cv2.INTER_NEAREST)
    return x > 0.0


def overlay_masks(frame, obj_ids, masks, alpha=0.45):
    vis = frame.copy().astype(np.float32)
    fh, fw = frame.shape[:2]
    masks_np = masks.detach().cpu().numpy()
    for i in range(min(len(obj_ids), masks_np.shape[0])):
        binm = _mask_to_2d_bool(masks_np[i], fh, fw)
        if not np.any(binm):
            continue
        c = np.array(point_color(int(obj_ids[i])), dtype=np.float32)
        vis[binm] = vis[binm] * (1 - alpha) + c * alpha
    return vis.astype(np.uint8)


def _draw_3d_cube_from_mask(
    vis: np.ndarray,
    mask_bool: np.ndarray,
    obj_id: int,
    state: dict | None,
    smooth_alpha: float = 0.8,
) -> dict | None:
    if not np.any(mask_bool):
        return state

    # Reuse PCA/ICP orientation state so cube follows object rotation over time.
    m = cv2.erode(mask_bool.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    clean = m > 0
    ctr = _resample_contour(clean, num_pts=100)
    if ctr is None:
        return state

    c_cur = ctr.mean(0)
    x_cur = ctr - c_cur

    if state is None:
        pca = _pca_axes(clean)
        if pca is None:
            return None
        _, u1, u2 = pca
        state = {
            "ref_centered": x_cur,
            "u1": u1.astype(np.float64),
            "u2": u2.astype(np.float64),
            "prev_R": np.eye(2, dtype=np.float64),
            "prev_t": np.zeros(2, dtype=np.float64),
        }
        r = np.eye(2, dtype=np.float64)
    else:
        x_ref = state["ref_centered"]
        if x_ref.shape[0] != x_cur.shape[0]:
            return state
        r, t = _icp(x_ref, x_cur, state.get("prev_R"), state.get("prev_t"))
        state["prev_R"], state["prev_t"] = r, t

    raw_angle = np.arctan2(r[1, 0], r[0, 0])
    if "smoothed_angle" not in state:
        state["smoothed_angle"] = raw_angle
    elif smooth_alpha > 0.0:
        da = (raw_angle - state["smoothed_angle"] + np.pi) % (2 * np.pi) - np.pi
        state["smoothed_angle"] += (1.0 - smooth_alpha) * da
    else:
        state["smoothed_angle"] = raw_angle
    a = state["smoothed_angle"]
    r_s = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float64)

    if "smoothed_center" not in state:
        state["smoothed_center"] = c_cur.copy()
    elif smooth_alpha > 0.0:
        state["smoothed_center"] = smooth_alpha * state["smoothed_center"] + (1.0 - smooth_alpha) * c_cur
    else:
        state["smoothed_center"] = c_cur.copy()

    c = state["smoothed_center"]
    u1 = r_s @ state["u1"]; u1 /= np.linalg.norm(u1) + 1e-12
    u2 = r_s @ state["u2"]; u2 /= np.linalg.norm(u2) + 1e-12
    u3 = u1 + u2
    if np.linalg.norm(u3) < 1e-6:
        u3 = np.array([1.0, -1.0], dtype=np.float64)
    u3 /= np.linalg.norm(u3) + 1e-12

    # Build the front face as the minimum-area rectangle in image space.
    ys, xs = np.where(clean)
    if len(xs) < 20:
        return state
    p32 = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    rect = cv2.minAreaRect(p32.reshape(-1, 1, 2))
    front = cv2.boxPoints(rect).astype(np.float64)  # (4,2), minimum-area box
    w, h = rect[1]
    depth = max(8.0, 0.35 * min(float(w), float(h)))

    # Keep corner order consistent (clockwise around centroid) for stable edge linking.
    fc = front.mean(axis=0)
    ang = np.arctan2(front[:, 1] - fc[1], front[:, 0] - fc[0])
    front = front[np.argsort(ang)]
    f0, f1, f2, f3 = front[0], front[1], front[2], front[3]

    # Back face shifted along projected depth direction.
    shift = -depth * u3
    b0, b1, b2, b3 = f0 + shift, f1 + shift, f2 + shift, f3 + shift

    def _to_i32(poly):
        return np.array([[int(round(v[0])), int(round(v[1]))] for v in poly], dtype=np.int32)

    front_i = _to_i32([f0, f1, f2, f3])
    back_i = _to_i32([b0, b1, b2, b3])

    col = point_color(obj_id)
    col_back = (max(0, col[0] - 80), max(0, col[1] - 80), max(0, col[2] - 80))
    cv2.polylines(vis, [back_i], True, col_back, 2, cv2.LINE_AA)
    cv2.polylines(vis, [front_i], True, col, 2, cv2.LINE_AA)
    for i in range(4):
        cv2.line(vis, tuple(front_i[i]), tuple(back_i[i]), col, 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        f"CUBE{obj_id}",
        tuple(front_i[0] + np.array([4, -4], dtype=np.int32)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        col,
        2,
        lineType=cv2.LINE_AA,
    )
    return state

# ---------------------------------------------------------------------------
# ICP pose axes
# ---------------------------------------------------------------------------

def _resample_contour(mask_bool, num_pts):
    m = mask_bool.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    pts_all, w_all = [], []
    for c in cnts:
        if len(c) < 5:
            continue
        p = c.reshape(-1, 2).astype(np.float64)
        prev, nxt = np.roll(p, 1, 0), np.roll(p, -1, 0)
        v1, v2 = p - prev, nxt - p
        n1 = np.linalg.norm(v1, axis=1) + 1e-9
        n2 = np.linalg.norm(v2, axis=1) + 1e-9
        curv = np.arccos(np.clip(np.sum(v1 * v2, axis=1) / (n1 * n2), -1, 1))
        pts_all.append(p)
        w_all.append(1.0 + 3.0 * curv / np.pi)
    if not pts_all:
        return None
    pts = np.vstack(pts_all)
    w = np.concatenate(w_all)
    if len(pts) <= num_pts:
        return pts
    prob = w / (w.sum() + 1e-12)
    return pts[np.random.default_rng(0).choice(len(pts), num_pts, replace=False, p=prob)]


def _pca_axes(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) < 12:
        return None
    p = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    c = p.mean(0)
    cov = ((p - c).T @ (p - c)) / max(len(p), 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)
    return c, v[:, order[-1]] / (np.linalg.norm(v[:, order[-1]]) + 1e-12), \
              v[:, order[0]]  / (np.linalg.norm(v[:, order[0]])  + 1e-12)


def _kabsch(P, Q):
    mu_p, mu_q = P.mean(0), Q.mean(0)
    h = (P - mu_p).T @ (Q - mu_q)
    u, _, vt = np.linalg.svd(h)
    if np.linalg.det(u @ vt) < 0:
        vt = vt.copy(); vt[1] *= -1
    r = u @ vt
    return r.astype(np.float64), (mu_q - mu_p @ r.T).astype(np.float64)


def _icp(P, Q, prev_R=None, prev_t=None, max_iter=8, tol=1e-4):
    if P.shape[0] < 3 or Q.shape[0] < 3:
        return np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
    r = prev_R.copy() if prev_R is not None else np.eye(2, dtype=np.float64)
    t = prev_t.copy() if prev_t is not None else Q.mean(0) - (P @ r.T).mean(0)
    tree = cKDTree(Q) if cKDTree is not None else None
    prev_err = np.inf
    for _ in range(max_iter):
        p_t = P @ r.T + t
        if tree is not None:
            dist, idx = tree.query(p_t)
            err = float(np.mean(dist * dist))
        else:
            d2 = np.sum((p_t[:, None] - Q[None]) ** 2, axis=2)
            idx, err = np.argmin(d2, axis=1), float(np.mean(np.min(d2, axis=1)))
        ri, ti = _kabsch(p_t, Q[idx])
        r, t = ri @ r, t @ ri.T + ti
        if prev_err < np.inf and abs(prev_err - err) < tol:
            break
        prev_err = err
    return r, t


# ---------------------------------------------------------------------------
# Live camera frame provider
# ---------------------------------------------------------------------------

class LiveFrameProvider:
    """Always-latest frame provider for real-time inference.

    The capture thread continuously overwrites a single 'latest' slot.
    When propagate_in_video requests frame idx, it receives whatever the
    newest camera frame is at that moment — no queue ever builds up.
    Each idx is cached once so get_raw(idx) returns the same frame the
    model saw.
    """

    def __init__(self, cap: cv2.VideoCapture, image_size: int):
        self.cap = cap
        self.image_size = image_size
        # Rolling cache: model_idx -> (tensor, raw). Bounded to last 32 frames.
        self._cache: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
        self._latest_tensor: torch.Tensor | None = None
        self._latest_raw: np.ndarray | None = None
        self._lock = threading.Lock()

    def _encode(self, frame: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        frame = preprocess(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(
            cv2.resize(rgb, (self.image_size, self.image_size))
        ).float().div(255.0).permute(2, 0, 1)
        return (t - _IMG_MEAN) / _IMG_STD, frame

    def capture_next(self) -> bool:
        # Drain any buffered camera frames so we always get the newest one.
        frame = None
        for _ in range(4):
            ok, f = self.cap.read()
            if ok:
                frame = f
        if frame is None:
            return False
        t, raw = self._encode(frame)
        with self._lock:
            self._latest_tensor = t
            self._latest_raw = raw
        return True

    def __len__(self):
        return 1_000_000  # always tell SAM2 there are more frames

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Block until the camera has produced at least one frame.
        while True:
            with self._lock:
                if self._latest_tensor is not None:
                    if idx not in self._cache:
                        # Snapshot the current latest for this model index.
                        self._cache[idx] = (self._latest_tensor, self._latest_raw)
                        # Evict entries older than 32 frames to bound memory.
                        if len(self._cache) > 32:
                            oldest = min(self._cache)
                            del self._cache[oldest]
                    return self._cache[idx][0]
            time.sleep(0.001)

    def get_raw(self, idx: int) -> np.ndarray:
        with self._lock:
            entry = self._cache.get(idx)
            return entry[1] if entry is not None else self._latest_raw

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args) -> None:
    device = choose_device(args.device)
    print(f"Device: {device}")

    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT}")
        print("Expected: EdgeTAM/checkpoints/edgetam.pt")
        return

    print("Loading EdgeTAM …")
    predictor = _load_predictor(device)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize camera-side frame queue

    image_size = predictor.image_size
    provider = LiveFrameProvider(cap, image_size)

    if not provider.capture_next():
        print("No frame from camera.")
        cap.release()
        return

    points = pick_points(provider.get_raw(0))
    if not points:
        print("No points selected. Exiting.")
        cap.release()
        return

    # Background capture thread
    stop_flag = threading.Event()

    def _capture_loop():
        while not stop_flag.is_set():
            if not provider.capture_next():
                stop_flag.set()

    threading.Thread(target=_capture_loop, daemon=True).start()

    # Init EdgeTAM state from single-frame temp folder, then swap in live provider
    tmp = tempfile.mkdtemp(prefix="edgetam_live_")
    cv2.imwrite(os.path.join(tmp, "000000.jpg"), provider.get_raw(0))
    state = predictor.init_state(tmp, async_loading_frames=False)
    shutil.rmtree(tmp, ignore_errors=True)

    state["images"] = provider
    state["num_frames"] = 1_000_000

    for obj_id, x, y in points:
        predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=int(obj_id),
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

    writer = None
    if args.output:
        h, w = TARGET_SIZE[1], TARGET_SIZE[0]
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))

    com_trails: dict[int, list[tuple[int, int]]] = {}  # obj_id -> list of (x, y) COM positions
    cube_states: dict[int, dict | None] = {}
    fps_t0 = time.perf_counter()
    fps_frames = 0
    ac_device, ac_dtype, ac_enabled = _autocast_config(device, args.half)

    print("Live tracking started. Press 'q' or ESC to stop.")
    try:
        with torch.autocast(device_type=ac_device, dtype=ac_dtype, enabled=ac_enabled):
            for fi, obj_ids, masks in predictor.propagate_in_video(state):
                frame = provider.get_raw(fi)
                if hasattr(obj_ids, "tolist"):
                    ids = [int(x) for x in obj_ids.tolist()]
                else:
                    ids = [int(x) for x in obj_ids]

                vis = overlay_masks(frame, ids, masks, alpha=args.alpha)
                fh, fw = frame.shape[:2]
                masks_np = masks.detach().cpu().numpy()
                for i in range(min(len(ids), masks_np.shape[0])):
                    oid = ids[i]
                    binm = _mask_to_2d_bool(masks_np[i], fh, fw)

                    if np.any(binm):
                        ys, xs = np.where(binm)
                        cx, cy = int(xs.mean()), int(ys.mean())
                        if oid not in com_trails:
                            com_trails[oid] = []
                        com_trails[oid].append((cx, cy))

                        cube_states[oid] = _draw_3d_cube_from_mask(
                            vis, binm, oid, cube_states.get(oid), smooth_alpha=args.axis_smooth
                        )

                # Draw COM trails
                for oid, trail in com_trails.items():
                    col = point_color(oid)
                    for j in range(1, len(trail)):
                        cv2.line(vis, trail[j - 1], trail[j], col, 2, lineType=cv2.LINE_AA)
                    if trail:
                        cv2.circle(vis, trail[-1], 5, col, -1)
                        cv2.circle(vis, trail[-1], 7, (255, 255, 255), 1)

                if fi == 0:
                    for oid, px, py in points:
                        cv2.circle(vis, (int(px), int(py)), 5, (0, 255, 255), -1)
                        cv2.putText(vis, f"ID{oid}", (int(px) + 8, int(py) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                cv2.imshow("EdgeTAM Live", vis)
                if writer is not None:
                    writer.write(vis)

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if stop_flag.is_set():
                    break
                fps_frames += 1
                now = time.perf_counter()
                dt = now - fps_t0
                if dt >= 1.0:
                    print(f"FPS: {fps_frames / dt:.2f}")
                    fps_t0 = now
                    fps_frames = 0
    finally:
        stop_flag.set()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time EdgeTAM tracking on Orbbec RGB stream.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--alpha", type=float, default=0.45,
                        help="Mask overlay alpha")
    parser.add_argument("--axis-smooth", type=float, default=0.8, metavar="ALPHA",
                        help="EMA smoothing for axes (0=none, 0.99=heavy, default 0.8)")
    parser.add_argument("--half", action="store_true", default=True,
                        help="Use half-precision autocast (default: on)")
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--output", default="",
                        help="Optional path to save output video (e.g. out.mp4)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
