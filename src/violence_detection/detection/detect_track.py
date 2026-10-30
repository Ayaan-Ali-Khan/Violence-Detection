"""Person detection and per-video ByteTrack helpers built on Ultralytics YOLO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameTracks:
    """All person detections for one frame, in pixel coordinates."""

    boxes_xyxy: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    track_ids: np.ndarray


def load_yolo(weights: str | Path) -> Any:
    """Load a pretrained Ultralytics YOLO detector from a local or named weight."""
    from ultralytics import YOLO

    return YOLO(str(weights), task="detect")


def reset_tracker(model: Any) -> None:
    """Clear ByteTrack state so IDs start fresh for the next video."""
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    for tracker in getattr(predictor, "trackers", []):
        tracker.reset()


def track_frame(
    model: Any,
    frame: np.ndarray,
    *,
    tracker: str | Path,
    imgsz: int = 640,
    conf: float = 0.1,
    iou: float = 0.7,
    device: str = "auto",
) -> FrameTracks:
    """Detect COCO persons and associate them with persistent ByteTrack IDs."""
    result = model.track(
        source=frame,
        persist=True,
        tracker=str(tracker),
        classes=[0],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=None if device == "auto" else device,
        verbose=False,
    )[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return FrameTracks(
            boxes_xyxy=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            class_ids=np.empty((0,), dtype=np.int64),
            track_ids=np.empty((0,), dtype=np.int64),
        )

    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
    scores = boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
    class_ids = boxes.cls.detach().cpu().numpy().astype(np.int64, copy=False)
    ids = boxes.id
    track_ids = (
        ids.detach().cpu().numpy().astype(np.int64, copy=False)
        if ids is not None
        else np.full((len(xyxy),), -1, dtype=np.int64)
    )
    return FrameTracks(xyxy, scores, class_ids, track_ids)


def iter_video_tracks(
    model: Any,
    video_path: str | Path,
    *,
    tracker: str | Path,
    imgsz: int = 640,
    conf: float = 0.1,
    iou: float = 0.7,
    device: str = "auto",
    frame_stride: int = 1,
):
    """Yield ``(frame_index, tracks)`` for decoded frames in a video.

    ``frame_stride`` defaults to every frame. If increased for a speed-limited
    run, the frame indices remain source indices so downstream code can see gaps.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {path}")
    reset_tracker(model)
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % frame_stride == 0:
                yield frame_index, track_frame(
                    model,
                    frame,
                    tracker=tracker,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=device,
                )
            frame_index += 1
    finally:
        capture.release()
