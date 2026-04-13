#!/usr/bin/env python3
import argparse
import os
import tempfile
import threading
import time
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


TARGET_SIZE = (640, 360)
AXIS_LEN_PX = 40.0

# ImageNet normalization constants (same as SAM2's load_video_frames)
_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_IMG_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def rotate_frame_180(frame: np.ndarray) -> np.ndarray:
    return cv2.rotate(frame, cv2.ROTATE_180)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    frame = rotate_frame_180(frame)
    return cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def pick_points(first_frame: np.ndarray) -> list[tuple[int, float, float]]:
    win = "Select SAM2 points"
    points: list[tuple[int, float, float]] = []

    def draw() -> np.ndarray:
        vis = first_frame.copy()
        for obj_id, px_f, py_f in points:
            px, py = int(px_f), int(py_f)
            cv2.circle(vis, (px, py), 6, (0, 255, 255), -1)
            cv2.circle(vis, (px, py), 9, (255, 255, 255), 1)
            cv2.putText(
                vis,
                f"ID{obj_id}",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        cv2.putText(
            vis,
            "Left click: add NEW object point (new ID) | Backspace: undo | c: clear | Enter: start",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return vis

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            obj_id = len(points) + 1
            points.append((obj_id, float(x), float(y)))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    cancelled = False
    while True:
        cv2.imshow(win, draw())
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):  # ENTER
            if points:
                break
        elif k in (8, 127):  # Backspace / Delete
            if points:
                points.pop()
        elif k == ord("c"):
            points.clear()
        elif k in (ord("q"), 27):
            cancelled = True
            break
    cv2.destroyWindow(win)
    return [] if cancelled else points


def point_color(obj_id: int) -> tuple[int, int, int]:
    hue = (obj_id * 47 + 20) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _mask_to_2d_bool(m: np.ndarray, fh: int, fw: int) -> np.ndarray:
    """SAM2 may return (H,W), (1,H,W), (1,1,H,W), etc.; normalize to (fh,fw) bool."""
    x = np.asarray(m, dtype=np.float32)
    x = np.squeeze(x)
    while x.ndim > 2:
        x = x[0]
    if x.ndim != 2:
        return np.zeros((fh, fw), dtype=bool)
    if x.shape[0] != fh or x.shape[1] != fw:
        x = cv2.resize(x, (fw, fh), interpolation=cv2.INTER_NEAREST)
    return x > 0.0


def overlay_masks_with_ids(
    frame: np.ndarray,
    obj_ids: list[int],
    masks: torch.Tensor,
    alpha: float = 0.45,
) -> np.ndarray:
    vis = frame.copy().astype(np.float32)
    fh, fw = frame.shape[:2]
    masks_np = masks.detach().cpu().numpy()
    n = min(len(obj_ids), masks_np.shape[0])
    for i in range(n):
        obj_id = obj_ids[i]
        binm = _mask_to_2d_bool(masks_np[i], fh, fw)
        if not np.any(binm):
            continue
        c = np.array(point_color(int(obj_id)), dtype=np.float32)
        vis[binm] = vis[binm] * (1.0 - alpha) + c * alpha
    return vis.astype(np.uint8)


def _resample_contour_xy(mask_bool: np.ndarray, num_pts: int) -> np.ndarray | None:
    m = (mask_bool.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    pts_all = []
    w_all = []
    for c in cnts:
        if len(c) < 5:
            continue
        p = c.reshape(-1, 2).astype(np.float64)
        prev = np.roll(p, 1, axis=0)
        nxt = np.roll(p, -1, axis=0)
        v1 = p - prev
        v2 = nxt - p
        n1 = np.linalg.norm(v1, axis=1) + 1e-9
        n2 = np.linalg.norm(v2, axis=1) + 1e-9
        cosang = np.sum(v1 * v2, axis=1) / (n1 * n2)
        cosang = np.clip(cosang, -1.0, 1.0)
        curv = np.arccos(cosang)
        w = 1.0 + 3.0 * (curv / np.pi)
        pts_all.append(p)
        w_all.append(w)
    if not pts_all:
        return None
    pts = np.vstack(pts_all)
    w = np.concatenate(w_all)
    if len(pts) <= num_pts:
        return pts
    prob = w / (np.sum(w) + 1e-12)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=num_pts, replace=False, p=prob)
    return pts[idx]


def _mask_pca_axes_2d(mask_bool: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    ys, xs = np.where(mask_bool)
    if len(xs) < 12:
        return None
    p = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    c = p.mean(axis=0)
    pc = p - c
    cov = (pc.T @ pc) / max(len(pc), 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)
    u2 = v[:, order[0]]
    u1 = v[:, order[-1]]
    u1 = u1 / (np.linalg.norm(u1) + 1e-12)
    u2 = u2 / (np.linalg.norm(u2) + 1e-12)
    return c, u1, u2


def _kabsch_rigid_rows_2d(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu_p = P.mean(axis=0)
    mu_q = Q.mean(axis=0)
    pc = P - mu_p
    qc = Q - mu_q
    h = pc.T @ qc
    u, _, vt = np.linalg.svd(h)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[1, :] *= -1.0
        r = u @ vt
    t = mu_q - mu_p @ r.T
    return r.astype(np.float64), t.astype(np.float64)


def _icp_2d(
    P: np.ndarray,
    Q: np.ndarray,
    prev_R: np.ndarray | None = None,
    prev_t: np.ndarray | None = None,
    max_iter: int = 8,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    if P.shape[0] < 3 or Q.shape[0] < 3:
        return np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
    r_acc = prev_R.copy() if prev_R is not None else np.eye(2, dtype=np.float64)
    if prev_t is not None:
        t_acc = prev_t.copy()
    else:
        t_acc = Q.mean(axis=0) - (P @ r_acc.T).mean(axis=0)
    prev_err = np.inf
    tree = cKDTree(Q) if cKDTree is not None else None
    for _ in range(max_iter):
        p_t = (P @ r_acc.T) + t_acc
        if tree is not None:
            dist, idx = tree.query(p_t)
            q_near = Q[idx]
            err = float(np.mean(dist * dist))
        else:
            diff = p_t[:, None, :] - Q[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            idx = np.argmin(d2, axis=1)
            q_near = Q[idx]
            err = float(np.mean(np.min(d2, axis=1)))
        ri, ti = _kabsch_rigid_rows_2d(p_t, q_near)
        r_acc = ri @ r_acc
        t_acc = t_acc @ ri.T + ti
        if prev_err < np.inf and abs(prev_err - err) < tol:
            break
        prev_err = err
    return r_acc, t_acc


def draw_object_axes_icp(
    vis: np.ndarray,
    mask_bool: np.ndarray,
    obj_id: int,
    state: dict | None,
    contour_pts: int = 100,
) -> dict | None:
    fh, fw = vis.shape[:2]
    if not np.any(mask_bool):
        return state
    m = (mask_bool.astype(np.uint8) * 255)
    m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    clean_mask = m > 0
    ctr = _resample_contour_xy(clean_mask, contour_pts)
    if ctr is None:
        return state
    c_cur = ctr.mean(axis=0)
    x_cur = ctr - c_cur

    if state is None:
        pca = _mask_pca_axes_2d(clean_mask)
        if pca is None:
            return None
        _, u1, u2 = pca
        state = {
            "ref_contour_centered": ctr - c_cur,
            "u1": u1.astype(np.float64),
            "u2": u2.astype(np.float64),
            "prev_R": np.eye(2, dtype=np.float64),
            "prev_t": np.zeros(2, dtype=np.float64),
        }
        r = np.eye(2, dtype=np.float64)
    else:
        x_ref = state["ref_contour_centered"]
        if x_ref.shape[0] != x_cur.shape[0]:
            return state
        r, t = _icp_2d(
            x_ref,
            x_cur,
            prev_R=state.get("prev_R"),
            prev_t=state.get("prev_t"),
        )
        state["prev_R"] = r
        state["prev_t"] = t

    u1t = r @ state["u1"]
    u2t = r @ state["u2"]
    u1t = u1t / (np.linalg.norm(u1t) + 1e-12)
    u2t = u2t / (np.linalg.norm(u2t) + 1e-12)
    u3t = u1t + u2t
    if np.linalg.norm(u3t) < 1e-6:
        u3t = np.array([1.0, 0.0], dtype=np.float64)
    u3t = u3t / (np.linalg.norm(u3t) + 1e-12)

    c = np.array([float(c_cur[0]), float(c_cur[1])], dtype=np.float64)
    axes = [
        (u1t, (0, 0, 255), "X"),
        (u2t, (0, 255, 0), "Y"),
        (u3t, (255, 0, 0), "Z"),
    ]
    for u, col, label in axes:
        p0 = c - u * AXIS_LEN_PX
        p1 = c + u * AXIS_LEN_PX
        p0 = (int(np.clip(p0[0], 0, fw - 1)), int(np.clip(p0[1], 0, fh - 1)))
        p1 = (int(np.clip(p1[0], 0, fw - 1)), int(np.clip(p1[1], 0, fh - 1)))
        cv2.line(vis, p0, p1, col, 2, lineType=cv2.LINE_AA)
        cv2.putText(
            vis,
            f"{label}{obj_id}",
            (p1[0] + 3, p1[1] + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
        )
    return state


def resolve_model_paths(repo_root: Path, model_size: str) -> tuple[str, str]:
    model_size = model_size.lower()
    if model_size == "tiny":
        return (
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            str(repo_root / "checkpoints" / "sam2.1_hiera_tiny.pt"),
        )
    if model_size == "small":
        return (
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            str(repo_root / "checkpoints" / "sam2.1_hiera_small.pt"),
        )
    if model_size in {"base", "base_plus", "b+"}:
        return (
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            str(repo_root / "checkpoints" / "sam2.1_hiera_base_plus.pt"),
        )
    if model_size == "large":
        return (
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            str(repo_root / "checkpoints" / "sam2.1_hiera_large.pt"),
        )
    raise ValueError(f"Unsupported model size: {model_size}")


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare_sam2_input_as_jpeg_folder(input_path: str) -> tuple[str, str]:
    """
    Build a temporary JPEG frame folder for SAM2 init_state().
    This avoids the optional decord dependency required for direct video-file loading.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = preprocess_frame(frame)
        out_path = os.path.join(tmp_dir, f"{frame_i:06d}.jpg")
        cv2.imwrite(out_path, frame)
        frame_i += 1
    cap.release()
    if frame_i == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Input video contained no readable frames.")
    return tmp_dir, tmp_dir


# ---------------------------------------------------------------------------
# Live camera streaming support
# ---------------------------------------------------------------------------

class LiveCameraFrameProvider:
    """Thread-safe frame buffer for SAM2 live streaming.

    Implements __len__ and __getitem__ so it can be assigned directly to
    inference_state["images"]. __getitem__ blocks until the requested frame
    index has been captured by the background capture thread.
    """

    def __init__(self, cap: cv2.VideoCapture, image_size: int):
        self.cap = cap
        self.image_size = image_size
        self._tensors: list[torch.Tensor] = []
        self._raw: list[np.ndarray] = []
        self._lock = threading.Lock()

    def capture_next(self) -> bool:
        """Capture and preprocess the next camera frame. Returns False on failure."""
        ok, frame = self.cap.read()
        if not ok:
            return False
        frame = preprocess_frame(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.image_size, self.image_size))
        t = torch.from_numpy(resized).float().div(255.0).permute(2, 0, 1)
        t = (t - _IMG_MEAN) / _IMG_STD
        with self._lock:
            self._tensors.append(t)
            self._raw.append(frame)
        return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._tensors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return normalized tensor for frame idx; blocks until available."""
        while True:
            with self._lock:
                if idx < len(self._tensors):
                    return self._tensors[idx]
            time.sleep(0.001)

    def get_raw(self, idx: int) -> np.ndarray:
        with self._lock:
            return self._raw[idx]


# ---------------------------------------------------------------------------
# Shared rendering helper
# ---------------------------------------------------------------------------

def _render_frame(
    frame_idx: int,
    obj_ids,
    masks: torch.Tensor,
    frame: np.ndarray,
    axis_states: dict,
    args,
    writer,
    seed_points: list | None = None,
) -> None:
    if hasattr(obj_ids, "tolist"):
        obj_ids_list = [int(x) for x in obj_ids.tolist()]
    else:
        obj_ids_list = [int(i) for i in obj_ids]

    vis = overlay_masks_with_ids(frame, obj_ids_list, masks, alpha=args.alpha)
    fh, fw = frame.shape[:2]
    masks_np = masks.detach().cpu().numpy()
    for i in range(min(len(obj_ids_list), masks_np.shape[0])):
        oid = obj_ids_list[i]
        binm = _mask_to_2d_bool(masks_np[i], fh, fw)
        axis_states[oid] = draw_object_axes_icp(vis, binm, oid, axis_states.get(oid))

    if seed_points and frame_idx == 0:
        for obj_id, px_f, py_f in seed_points:
            px, py = int(px_f), int(py_f)
            cv2.circle(vis, (px, py), 5, (0, 255, 255), -1)
            cv2.putText(
                vis,
                f"ID{obj_id}",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

    cv2.imshow("SAM2", vis)
    if writer is not None:
        writer.write(vis)


def _make_writer(args, fps: float = 30.0):
    if not args.output:
        return None
    h, w = TARGET_SIZE[1], TARGET_SIZE[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(args.output, fourcc, fps, (w, h))


# ---------------------------------------------------------------------------
# Pre-recorded video path
# ---------------------------------------------------------------------------

def run_video(args, predictor, device: str) -> None:
    if not os.path.exists(args.input):
        print(f"Input video not found: {args.input}")
        print("Pass a valid path with --input")
        return

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Could not open input video: {args.input}")
        return

    ok, first = cap.read()
    if not ok:
        print("Could not read first frame.")
        cap.release()
        return
    first = preprocess_frame(first)
    cap.release()

    points = pick_points(first)
    if not points:
        print("No points selected. Exiting.")
        return

    sam2_video_path, temp_frames_dir = prepare_sam2_input_as_jpeg_folder(args.input)
    print(f"Prepared temporary JPEG frame folder for SAM2: {sam2_video_path}")
    # async_loading_frames uses float64 internally, which MPS doesn't support
    use_async = (device != "mps")
    state = predictor.init_state(video_path=sam2_video_path, async_loading_frames=use_async)

    for obj_id, x, y in points:
        predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=int(obj_id),
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    writer = _make_writer(args, fps)
    cur_idx = -1
    axis_states: dict[int, dict | None] = {}
    frame = None

    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        while cur_idx < frame_idx:
            ok, frame = cap.read()
            if not ok:
                frame = None
                break
            frame = preprocess_frame(frame)
            cur_idx += 1
        if frame is None:
            break

        _render_frame(frame_idx, obj_ids, masks, frame, axis_states, args, writer,
                      seed_points=points)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    if writer is not None:
        writer.release()
    if temp_frames_dir and os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Live camera path
# ---------------------------------------------------------------------------

def run_live_camera(args, predictor) -> None:
    cam_idx = int(args.input)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Could not open camera index {cam_idx}")
        return

    image_size = predictor.image_size
    provider = LiveCameraFrameProvider(cap, image_size)

    # Capture first frame for point selection
    if not provider.capture_next():
        print("No frame from camera.")
        cap.release()
        return

    points = pick_points(provider.get_raw(0))
    if not points:
        print("No points selected. Exiting.")
        cap.release()
        return

    # Start background capture thread
    stop_flag = threading.Event()

    def _capture_loop():
        while not stop_flag.is_set():
            if not provider.capture_next():
                stop_flag.set()
                break

    capture_thread = threading.Thread(target=_capture_loop, daemon=True)
    capture_thread.start()

    # Init SAM2 with a single-frame temp folder, then swap in the live provider
    tmp = tempfile.mkdtemp(prefix="sam2_live_")
    cv2.imwrite(os.path.join(tmp, "000000.jpg"), provider.get_raw(0))
    state = predictor.init_state(tmp, async_loading_frames=False)
    shutil.rmtree(tmp, ignore_errors=True)

    # Patch state to stream from the live provider indefinitely
    state["images"] = provider
    state["num_frames"] = 1_000_000

    # Seed tracking on frame 0
    for obj_id, x, y in points:
        predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=int(obj_id),
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

    writer = _make_writer(args)
    axis_states: dict[int, dict | None] = {}

    # Run propagate_in_video as a single long-running generator.
    # provider.__getitem__(N) blocks until frame N is captured, so the
    # generator naturally paces itself to the camera — no manual frame
    # counting needed.
    print("Live tracking started. Press 'q' or ESC to stop.")
    try:
        for fi, obj_ids, masks in predictor.propagate_in_video(state):
            _render_frame(fi, obj_ids, masks, provider.get_raw(fi),
                          axis_states, args, writer,
                          seed_points=points if fi == 0 else None)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if stop_flag.is_set():
                break
    finally:
        stop_flag.set()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply official Meta SAM2 video tracking from manual point prompts."
    )
    parser.add_argument("--input", default="movie.mp4.mov",
                        help="Input video path, or camera index (e.g. 0, 1) for live mode")
    parser.add_argument(
        "--sam2-repo",
        default="segment-anything-2",
        help="Path to cloned SAM2 repo",
    )
    parser.add_argument(
        "--model-size",
        default="base_plus",
        choices=["tiny", "small", "base_plus", "large"],
        help="Official SAM2.1 model size",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay alpha")
    parser.add_argument("--output", default="", help="Optional output video path")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Apply torch.compile for faster inference (MPS/CUDA). "
             "First inference will be slow (~30s) while compiling.",
    )
    args = parser.parse_args()

    repo_root = Path(args.sam2_repo).expanduser().resolve()
    model_cfg, ckpt_path = resolve_model_paths(repo_root, args.model_size)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run: (cd segment-anything-2/checkpoints && ./download_ckpts.sh)")
        return

    device = choose_device(args.device)
    print(f"Using device={device}, model={args.model_size}")
    predictor = build_sam2_video_predictor(model_cfg, ckpt_path, device=device)

    if args.compile:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True  # MPS may have unsupported ops
        predictor = torch.compile(predictor, mode="reduce-overhead")
        print("torch.compile applied (first inference will be slower while compiling)")

    is_live = args.input.isdigit()
    if is_live:
        run_live_camera(args, predictor)
    else:
        run_video(args, predictor, device)


if __name__ == "__main__":
    main()
