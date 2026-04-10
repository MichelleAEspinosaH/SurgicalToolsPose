# ******************************************************************************
#  sam2_yolo_track.py
#
#  YOLO box prompts + Ultralytics SAM 2.1 video-style tracking
#  (SAM2DynamicInteractivePredictor: memory across frames).
#
#  Requires: pip install -U ultralytics
#  Weights download on first use (e.g. sam2.1_t.pt, YOLO best.pt).
# ******************************************************************************

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

# Forward type for Ultralytics Results
YoloResults = Any


def _boxes_xyxy_from_yolo_result(r, names: dict, excluded: set[str]) -> list[list[float]]:
    if r.boxes is None or len(r.boxes) == 0:
        return []
    boxes = r.boxes
    out = []
    for i in range(len(boxes)):
        cid = int(boxes.cls[i])
        label = str(names.get(cid, "")).lower()
        if label in excluded:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
        out.append([float(x1), float(y1), float(x2), float(y2)])
    return out


def overlay_sam_on_bgr(
    bgr: np.ndarray,
    sam_results: list[Any] | None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend SAM instance masks onto BGR image with distinct colors."""
    if not sam_results or sam_results[0].masks is None:
        return bgr
    m = sam_results[0].masks
    if m.data is None:
        return bgr
    masks = m.data.cpu().numpy()
    h, w = bgr.shape[:2]
    vis = bgr.copy()
    n = masks.shape[0]
    for i in range(n):
        mask = masks[i]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        binm = mask > 0.5
        hue = (i * 47 + 30) % 180
        hsv = np.uint8([[[hue, 200, 255]]])
        col = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        col = col.astype(np.float32)
        vis_f = vis.astype(np.float32)
        for c in range(3):
            vis_f[:, :, c] = np.where(
                binm,
                vis_f[:, :, c] * (1 - alpha) + col[c] * alpha,
                vis_f[:, :, c],
            )
        vis = vis_f.astype(np.uint8)
    return vis


@dataclass
class SamYoloVideoTracker:
    """Holds YOLO + SAM2 dynamic predictor; call `step()` or `step_with_yolo_result()`."""

    yolo_weights: str = ""
    yolo_conf: float = 0.4
    yolo_model: YOLO | None = None
    sam_model: str = "sam2.1_t.pt"
    sam_imgsz: int = 512
    max_obj_num: int = 10
    excluded_classes: set[str] = field(
        default_factory=lambda: {"human", "person", "hand", "arm"}
    )
    refresh_every: int = 20

    _yolo: YOLO | None = field(default=None, init=False, repr=False)
    _sam: SAM2DynamicInteractivePredictor | None = field(default=None, init=False, repr=False)
    _prev_n: int = field(default=-1, init=False, repr=False)
    _frame_i: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.yolo_model is not None:
            self._yolo = self.yolo_model
        elif self.yolo_weights:
            self._yolo = YOLO(self.yolo_weights)
        else:
            raise ValueError("Provide yolo_model or yolo_weights")
        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            imgsz=self.sam_imgsz,
            model=self.sam_model,
            save=False,
        )
        self._sam = SAM2DynamicInteractivePredictor(
            overrides=overrides,
            max_obj_num=max(self.max_obj_num, 3),
        )

    def reset_sam(self) -> None:
        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            imgsz=self.sam_imgsz,
            model=self.sam_model,
            save=False,
        )
        self._sam = SAM2DynamicInteractivePredictor(
            overrides=overrides,
            max_obj_num=max(self.max_obj_num, 3),
        )
        self._prev_n = -1
        self._frame_i = 0

    def _sam_step(
        self,
        frame_bgr: np.ndarray,
        yolo_r: YoloResults,
    ) -> tuple[np.ndarray, list[Any] | None]:
        assert self._yolo is not None and self._sam is not None
        yolo_vis = yolo_r.plot()
        bboxes = _boxes_xyxy_from_yolo_result(
            yolo_r, self._yolo.names, self.excluded_classes
        )
        n = len(bboxes)
        sam_out = None

        if n == 0:
            if self._prev_n > 0:
                self.reset_sam()
            self._prev_n = 0
            self._frame_i += 1
            return yolo_vis, None

        obj_ids = list(range(n))
        need_reseed = n != self._prev_n or self._frame_i == 0 or (
            self.refresh_every > 0
            and self._frame_i > 0
            and self._frame_i % self.refresh_every == 0
        )

        if need_reseed:
            sam_out = self._sam(
                source=frame_bgr,
                bboxes=bboxes,
                obj_ids=obj_ids,
                update_memory=True,
            )
            self._prev_n = n
        else:
            sam_out = self._sam(source=frame_bgr)

        self._frame_i += 1
        return yolo_vis, sam_out

    def step(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list[Any] | None]:
        """Run YOLO on frame, then SAM2 (use when YOLO is not run elsewhere)."""
        assert self._yolo is not None
        yolo_r = self._yolo(frame_bgr, conf=self.yolo_conf, verbose=False)[0]
        return self._sam_step(frame_bgr, yolo_r)

    def step_with_yolo_result(
        self, frame_bgr: np.ndarray, yolo_r: YoloResults
    ) -> list[Any] | None:
        """
        Use an existing ultralytics Results object (same frame). Returns SAM
        results only, or None if no boxes / SAM skipped.
        """
        _, sam_out = self._sam_step(frame_bgr, yolo_r)
        return sam_out
