"""Unit tests for the interaction crop and clip-pair rules."""

import numpy as np
import pytest

from violence_detection.crops import (
    build_crop_boxes,
    build_interaction_crops,
    normalized_pair_distance,
    select_clip_pair,
)


def test_normalized_pair_distance_is_resolution_independent() -> None:
    first = np.asarray([0, 0, 20, 100])
    second = np.asarray([100, 0, 120, 100])
    doubled = np.asarray([value * 2 for value in (first, second)])

    assert normalized_pair_distance(first, second) == pytest.approx(1.0)
    assert normalized_pair_distance(*doubled) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "box_a, box_b",
    [([0, 0, 10, 0], [10, 0, 20, 10]), ([0, 0, 1, 1, 2], [0, 0, 1, 1])],
)
def test_normalized_pair_distance_rejects_invalid_boxes(box_a, box_b) -> None:
    with pytest.raises(ValueError):
        normalized_pair_distance(np.asarray(box_a), np.asarray(box_b))


def _three_person_tracks():
    frames = np.repeat(np.arange(5), 3)
    boxes = []
    ids = []
    for frame in range(5):
        # IDs 1 and 2 stay close and form the locked pair; ID 3 is a nearby
        # group member. A fourth person would be too distant to join.
        boxes.extend(
            [
                [20 + frame, 20, 50 + frame, 120],
                [100 + frame, 20, 130 + frame, 120],
                [265 + frame, 20, 295 + frame, 120],
            ]
        )
        ids.extend([1, 2, 3])
    return frames, np.asarray(boxes, dtype=np.float32), np.asarray(ids)


def test_pair_is_locked_across_clip_and_group_expands() -> None:
    frames, boxes, ids = _three_person_tracks()

    result = select_clip_pair(frames, boxes, ids, tau=1.5, group_tau=2.0)

    assert result is not None
    assert result.pair_track_ids == (1, 2)
    assert result.group_track_ids == (1, 2, 3)
    assert result.support_frames == 5


def test_pair_selection_requires_minimum_persistent_support() -> None:
    frames = np.asarray([0, 1])
    boxes = np.asarray([[0, 0, 20, 100], [30, 0, 50, 100]], dtype=np.float32)
    ids = np.asarray([1, 2])

    assert select_clip_pair(frames, boxes, ids, min_pair_frames=2) is None


def test_crop_boxes_interpolate_only_short_internal_gaps_and_fallback_elsewhere() -> (
    None
):
    boxes = np.asarray(
        [[10, 10, 40, 80], [20, 10, 50, 80], [30, 10, 60, 80]],
        dtype=np.float32,
    )
    result = build_crop_boxes(
        frame_count=8,
        frame_indices=np.asarray([1, 3, 7]),
        boxes_xyxy=boxes,
        track_ids=np.asarray([1, 1, 1]),
        selected_track_ids=(1,),
        frame_width=100,
        frame_height=100,
        margin=0.1,
        smoothing_window=1,
        max_gap=1,
    )

    assert result.interpolated_mask.tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert result.fallback_mask.tolist() == [
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]
    assert result.boxes_xyxy[0].tolist() == [0, 0, 100, 100]
    assert (
        result.boxes_xyxy[1, 2] - result.boxes_xyxy[1, 0]
        == result.boxes_xyxy[1, 3] - result.boxes_xyxy[1, 1]
    )
    assert result.fallback_rate == pytest.approx(4 / 8)


def test_crop_box_margin_squares_and_clamps_near_frame_edge() -> None:
    result = build_crop_boxes(
        frame_count=1,
        frame_indices=np.asarray([0]),
        boxes_xyxy=np.asarray([[0, 10, 20, 60]], dtype=np.float32),
        track_ids=np.asarray([8]),
        selected_track_ids=(8,),
        frame_width=100,
        frame_height=80,
        margin=0.2,
        smoothing_window=1,
    )
    x1, y1, x2, y2 = result.boxes_xyxy[0]

    assert x1 >= 0 and y1 >= 0 and x2 <= 100 and y2 <= 80
    assert x2 - x1 == y2 - y1
    assert x1 == 0


def test_unavailable_pair_uses_full_frame_for_every_frame() -> None:
    result = build_interaction_crops(
        frame_count=3,
        frame_indices=np.asarray([0]),
        boxes_xyxy=np.asarray([[10, 10, 40, 80]], dtype=np.float32),
        track_ids=np.asarray([5]),
        frame_width=100,
        frame_height=80,
    )

    assert result.selection is None
    assert result.crops.fallback_mask.all()
    assert (result.crops.boxes_xyxy == np.asarray([0, 0, 100, 80])).all()
