"""Temporal interaction crop construction with explicit full-frame fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from violence_detection.crops.pairs import PairSelection, select_clip_pair


@dataclass(frozen=True)
class CropBoxes:
    """One XYXY crop per source frame plus masks describing its provenance."""

    boxes_xyxy: np.ndarray
    fallback_mask: np.ndarray
    observed_mask: np.ndarray
    interpolated_mask: np.ndarray

    @property
    def fallback_rate(self) -> float:
        """Fraction of frames assigned the full-frame fallback."""
        return float(np.mean(self.fallback_mask)) if len(self.fallback_mask) else 0.0


@dataclass(frozen=True)
class InteractionCropResult:
    """Crop boxes and locked-person metadata for one clip."""

    crops: CropBoxes
    selection: PairSelection | None


def _box_inside_frame(
    box: np.ndarray, frame_width: int, frame_height: int, margin: float
) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 0 or box_height <= 0:
        raise ValueError("crop source box must have positive width and height")
    side = min(
        max(box_width, box_height) * (1.0 + 2.0 * margin), frame_width, frame_height
    )
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = side / 2.0
    center_x = min(max(center_x, half), frame_width - half)
    center_y = min(max(center_y, half), frame_height - half)
    left, top = center_x - half, center_y - half
    right, bottom = center_x + half, center_y + half
    return np.asarray(
        [
            max(0, int(np.floor(left))),
            max(0, int(np.floor(top))),
            min(frame_width, int(np.ceil(right))),
            min(frame_height, int(np.ceil(bottom))),
        ],
        dtype=np.int32,
    )


def _fill_short_gaps(boxes: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    filled = boxes.copy()
    interpolated = np.zeros(len(boxes), dtype=bool)
    known = np.flatnonzero(np.isfinite(boxes).all(axis=1))
    for left, right in zip(known[:-1], known[1:], strict=True):
        missing = right - left - 1
        if 0 < missing <= max_gap:
            for index in range(left + 1, right):
                ratio = (index - left) / (right - left)
                filled[index] = boxes[left] * (1.0 - ratio) + boxes[right] * ratio
                interpolated[index] = True
    return filled, interpolated


def _smooth_valid_segments(boxes: np.ndarray, window: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes.copy()
    if window <= 1:
        return boxes.copy()
    smoothed = boxes.copy()
    finite = np.isfinite(boxes).all(axis=1)
    starts = np.flatnonzero(finite & np.r_[True, ~finite[:-1]])
    ends = np.flatnonzero(finite & np.r_[~finite[1:], True]) + 1
    radius = window // 2
    for start, end in zip(starts, ends, strict=True):
        for index in range(start, end):
            left, right = max(start, index - radius), min(end, index + radius + 1)
            smoothed[index] = np.mean(boxes[left:right], axis=0)
    return smoothed


def build_crop_boxes(
    frame_count: int,
    frame_indices: np.ndarray,
    boxes_xyxy: np.ndarray,
    track_ids: np.ndarray,
    selected_track_ids: tuple[int, ...] | list[int] | set[int],
    *,
    frame_width: int,
    frame_height: int,
    margin: float = 0.15,
    smoothing_window: int = 5,
    max_gap: int = 3,
) -> CropBoxes:
    """Make square, smoothed group crops and use full frames for long gaps.

    Short interior gaps up to ``max_gap`` frames are linearly interpolated.
    Leading/trailing gaps and longer interior gaps remain full-frame fallbacks.
    """
    if frame_count < 0 or frame_width < 1 or frame_height < 1:
        raise ValueError(
            "frame_count must be non-negative and frame dimensions positive"
        )
    if margin < 0 or smoothing_window < 1 or max_gap < 0:
        raise ValueError(
            "margin/max_gap must be non-negative and smoothing_window positive"
        )
    indices = np.asarray(frame_indices, dtype=np.int64)
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    ids = np.asarray(track_ids)
    if (
        boxes.ndim != 2
        or boxes.shape[1:] != (4,)
        or len(indices) != len(boxes)
        or len(ids) != len(boxes)
    ):
        raise ValueError(
            "frame_indices, boxes_xyxy and track_ids must have matching lengths"
        )

    selected = {int(track_id) for track_id in selected_track_ids}
    observed_boxes = np.full((frame_count, 4), np.nan, dtype=np.float64)
    for frame, box, track_id in zip(indices, boxes, ids, strict=True):
        if int(track_id) not in selected or frame < 0 or frame >= frame_count:
            continue
        if not np.isfinite(box).all():
            continue
        x1 = min(max(float(box[0]), 0.0), float(frame_width))
        y1 = min(max(float(box[1]), 0.0), float(frame_height))
        x2 = min(max(float(box[2]), 0.0), float(frame_width))
        y2 = min(max(float(box[3]), 0.0), float(frame_height))
        if x2 <= x1 or y2 <= y1:
            continue
        if np.isfinite(observed_boxes[frame]).all():
            observed_boxes[frame] = [
                min(observed_boxes[frame, 0], x1),
                min(observed_boxes[frame, 1], y1),
                max(observed_boxes[frame, 2], x2),
                max(observed_boxes[frame, 3], y2),
            ]
        else:
            observed_boxes[frame] = [x1, y1, x2, y2]

    observed_mask = np.isfinite(observed_boxes).all(axis=1)
    filled_boxes, interpolated_mask = _fill_short_gaps(observed_boxes, max_gap)
    smoothed = _smooth_valid_segments(filled_boxes, smoothing_window)
    fallback_mask = ~np.isfinite(smoothed).all(axis=1)
    crop_boxes = np.tile(
        np.asarray([0, 0, frame_width, frame_height], dtype=np.int32),
        (frame_count, 1),
    )
    for frame in np.flatnonzero(~fallback_mask):
        crop_boxes[frame] = _box_inside_frame(
            smoothed[frame], frame_width, frame_height, margin
        )
    return CropBoxes(crop_boxes, fallback_mask, observed_mask, interpolated_mask)


def build_interaction_crops(
    frame_count: int,
    frame_indices: np.ndarray,
    boxes_xyxy: np.ndarray,
    track_ids: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    scores: np.ndarray | None = None,
    tau: float = 1.75,
    min_pair_frames: int = 3,
    group_tau: float | None = None,
    margin: float = 0.15,
    smoothing_window: int = 5,
    max_gap: int = 3,
) -> InteractionCropResult:
    """Select one clip-stable person group and create its crops per frame."""
    selection = select_clip_pair(
        frame_indices,
        boxes_xyxy,
        track_ids,
        scores,
        tau=tau,
        min_pair_frames=min_pair_frames,
        group_tau=group_tau,
    )
    crops = build_crop_boxes(
        frame_count,
        frame_indices,
        boxes_xyxy,
        track_ids,
        selection.group_track_ids if selection else (),
        frame_width=frame_width,
        frame_height=frame_height,
        margin=margin,
        smoothing_window=smoothing_window,
        max_gap=max_gap,
    )
    return InteractionCropResult(crops=crops, selection=selection)
