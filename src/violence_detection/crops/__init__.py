"""Person-pair selection and temporal interaction crop helpers."""

from violence_detection.crops.crop_builder import (
    CropBoxes,
    InteractionCropResult,
    build_crop_boxes,
    build_interaction_crops,
)
from violence_detection.crops.pairs import (
    PairSelection,
    normalized_pair_distance,
    select_clip_pair,
)

__all__ = [
    "CropBoxes",
    "InteractionCropResult",
    "PairSelection",
    "build_crop_boxes",
    "build_interaction_crops",
    "normalized_pair_distance",
    "select_clip_pair",
]
