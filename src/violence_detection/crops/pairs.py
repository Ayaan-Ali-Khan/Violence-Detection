"""Stable person-pair selection from per-frame detector/tracker output."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class PairSelection:
    """The clip-locked pair and nearby tracked people selected for a crop."""

    pair_track_ids: tuple[int, int]
    group_track_ids: tuple[int, ...]
    support_frames: int
    support_fraction: float
    median_distance: float
    mean_distance: float


def normalized_pair_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Return center distance divided by the mean person-box height.

    Boxes use pixel ``(x1, y1, x2, y2)`` coordinates. A value of 1 means that
    the box centers are one average person height apart, independent of video
    resolution.
    """
    a = np.asarray(box_a, dtype=np.float64)
    b = np.asarray(box_b, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("each box must have shape (4,) in xyxy format")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("boxes must contain only finite coordinates")
    width_a = float(a[2] - a[0])
    width_b = float(b[2] - b[0])
    height_a = float(a[3] - a[1])
    height_b = float(b[3] - b[1])
    mean_height = (height_a + height_b) / 2.0
    if width_a <= 0 or width_b <= 0 or height_a <= 0 or height_b <= 0:
        raise ValueError("boxes must have positive width and height")
    center_a = np.array([(a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0])
    center_b = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
    return float(np.linalg.norm(center_a - center_b) / mean_height)


def select_clip_pair(
    frame_indices: np.ndarray,
    boxes_xyxy: np.ndarray,
    track_ids: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    tau: float = 1.75,
    min_pair_frames: int = 3,
    group_tau: float | None = None,
) -> PairSelection | None:
    """Lock the most consistently close pair over a clip and expand its group.

    A pair is eligible on a frame when its normalized center distance is at
    most ``tau``. The pair with the most eligible frames wins; ties prefer the
    smaller median distance, higher mean detector confidence, and lower IDs.
    Other tracks join the locked group when they stay within ``group_tau`` of
    either locked person for at least ``min_pair_frames`` frames. This avoids
    switching the selected pair from frame to frame.
    """
    frame_indices = np.asarray(frame_indices)
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    ids = np.asarray(track_ids)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError("boxes_xyxy must have shape (N, 4)")
    if (
        frame_indices.ndim != 1
        or ids.ndim != 1
        or len(frame_indices) != len(boxes)
        or len(ids) != len(boxes)
    ):
        raise ValueError(
            "frame_indices, boxes_xyxy and track_ids must have equal lengths"
        )
    if scores is None:
        confidence = np.ones(len(boxes), dtype=np.float64)
    else:
        confidence = np.asarray(scores, dtype=np.float64)
        if confidence.ndim != 1 or len(confidence) != len(boxes):
            raise ValueError("scores must have one value per box")
    if tau <= 0 or min_pair_frames < 1:
        raise ValueError("tau and min_pair_frames must be positive")
    group_tau = tau * 1.5 if group_tau is None else float(group_tau)
    if group_tau < tau:
        raise ValueError("group_tau must be greater than or equal to tau")

    by_frame: dict[int, dict[int, tuple[np.ndarray, float]]] = defaultdict(dict)
    for frame, box, track_id, score in zip(
        frame_indices, boxes, ids, confidence, strict=True
    ):
        tid = int(track_id)
        if (
            tid < 0
            or not np.isfinite(box).all()
            or box[3] <= box[1]
            or box[2] <= box[0]
        ):
            continue
        by_frame[int(frame)][tid] = (box, float(score) if np.isfinite(score) else 0.0)

    observations: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for people in by_frame.values():
        for id_a, id_b in combinations(sorted(people), 2):
            try:
                distance = normalized_pair_distance(people[id_a][0], people[id_b][0])
            except ValueError:
                continue
            if distance <= tau:
                mean_score = (people[id_a][1] + people[id_b][1]) / 2.0
                observations[(id_a, id_b)].append((distance, mean_score))

    eligible = [
        (pair, values)
        for pair, values in observations.items()
        if len(values) >= min_pair_frames
    ]
    if not eligible:
        return None
    pair, values = min(
        eligible,
        key=lambda item: (
            -len(item[1]),
            float(np.median([value[0] for value in item[1]])),
            -float(np.mean([value[1] for value in item[1]])),
            item[0],
        ),
    )

    nearby_support: dict[int, int] = defaultdict(int)
    selected = set(pair)
    for people in by_frame.values():
        present_pair = [tid for tid in pair if tid in people]
        if not present_pair:
            continue
        for tid, (box, _) in people.items():
            if tid in selected:
                continue
            if any(
                normalized_pair_distance(box, people[locked_id][0]) <= group_tau
                for locked_id in present_pair
            ):
                nearby_support[tid] += 1
    group = tuple(
        sorted(
            selected
            | {
                tid
                for tid, support in nearby_support.items()
                if support >= min_pair_frames
            }
        )
    )
    distances = np.asarray([value[0] for value in values], dtype=np.float64)
    supporting_frame_count = len(values)
    support_fraction = supporting_frame_count / max(1, len(by_frame))
    return PairSelection(
        pair_track_ids=pair,
        group_track_ids=group,
        support_frames=supporting_frame_count,
        support_fraction=float(support_fraction),
        median_distance=float(np.median(distances)),
        mean_distance=float(np.mean(distances)),
    )
