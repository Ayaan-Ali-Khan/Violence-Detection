"""Bounded-CPU video decoding and fixed-length uint8 clip preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class PreprocessSpec:
    """Model-specific input settings loaded from ``configs/preprocess.yaml``."""

    model_name: str
    input_tag: str
    num_frames: int
    cache_frames: int
    resolution: int
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    color: str = "RGB"
    storage_dtype: str = "uint8"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecodedClip:
    """Frames and original source indices for one video."""

    full_frames: np.ndarray
    frame_indices: np.ndarray
    crop_frames: np.ndarray | None = None
    crop_found_mask: np.ndarray | None = None


def load_preprocess_spec(config_path: str | Path, model_name: str) -> PreprocessSpec:
    """Load and validate one model's preprocessing profile."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    try:
        model_config = config["models"][model_name]
    except KeyError as exc:
        raise ValueError(f"No preprocessing profile for model {model_name!r}") from exc
    num_frames = int(model_config["num_frames"])
    cache_frames = int(model_config["cache_frames"])
    if num_frames < 1 or cache_frames < num_frames:
        raise ValueError("cache_frames must be at least num_frames, both positive")
    if cache_frames % num_frames:
        raise ValueError("cache_frames must divide evenly into temporal segments")
    if cache_frames // num_frames != int(
        config["augmentation"]["temporal_shift_candidates"]
    ):
        raise ValueError("cache sampling and temporal-shift candidate counts disagree")
    resolution = int(model_config["resolution"])
    input_size = int(model_config["input_size"])
    if resolution < 1 or input_size < 1:
        raise ValueError("resolution and input_size must be positive")
    if (
        config.get("color", "RGB") != "RGB"
        or config.get("storage_dtype", "uint8") != "uint8"
    ):
        raise ValueError("the tensor cache currently supports RGB uint8 frames only")
    mean = tuple(float(value) for value in model_config["mean"])
    std = tuple(float(value) for value in model_config["std"])
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError(
            "mean and std must each have 3 channels and std must be positive"
        )
    return PreprocessSpec(
        model_name=model_name,
        input_tag=str(model_config["input_tag"]),
        num_frames=num_frames,
        cache_frames=cache_frames,
        resolution=resolution,
        input_size=input_size,
        mean=mean,
        std=std,
    )


def sample_frame_indices(total_frames: int, cache_frames: int) -> np.ndarray:
    """Return deterministic center-of-bin frame indices spanning the video."""
    if total_frames < 1 or cache_frames < 1:
        raise ValueError("total_frames and cache_frames must be positive")
    positions = (np.arange(cache_frames, dtype=np.float64) + 0.5) * (
        total_frames / cache_frames
    )
    return np.minimum(positions.astype(np.int64), total_frames - 1)


def _crop_frame(frame: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    if box is None:
        return frame
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = min(max(x1, 0), frame_width)
    x2 = min(max(x2, 0), frame_width)
    y1 = min(max(y1, 0), frame_height)
    y2 = min(max(y2, 0), frame_height)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def decode_video(
    video_path: str | Path,
    spec: PreprocessSpec,
    *,
    include_crop: bool = False,
    crop_boxes_xyxy: np.ndarray | None = None,
    crop_fallback_mask: np.ndarray | None = None,
) -> DecodedClip:
    """Decode a video once, sampling a bounded number of RGB uint8 frames.

    The source frames are resized to a fixed square resolution, matching the
    historical X3D notebook's input convention. Crop mode uses Phase 3's
    source-resolution crop boxes before resizing. Frames are kept as uint8 on
    disk and transferred to GPU for augmentation and normalization.
    """
    if include_crop and crop_boxes_xyxy is None:
        raise ValueError("crop boxes are required when include_crop=True")
    if include_crop and crop_fallback_mask is None:
        raise ValueError("crop fallback mask is required when include_crop=True")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    cv2.setNumThreads(1)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        capture.release()
        raise OSError(f"Video reports no decodable frames: {video_path}")
    wanted = sample_frame_indices(total_frames, spec.cache_frames)
    positions_by_frame: dict[int, list[int]] = {}
    for position, frame_index in enumerate(wanted.tolist()):
        positions_by_frame.setdefault(frame_index, []).append(position)

    full_output: list[np.ndarray | None] = [None] * spec.cache_frames
    crop_output: list[np.ndarray | None] = [None] * spec.cache_frames
    crop_found_output = np.zeros(spec.cache_frames, dtype=np.bool_)
    last_full: np.ndarray | None = None
    last_crop: np.ndarray | None = None
    last_sampled_index = -1
    last_sampled_crop_found = False
    try:
        frame_index = 0
        while frame_index <= int(wanted.max()):
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break
            if frame_index in positions_by_frame:
                last_sampled_index = frame_index
                last_sampled_crop_found = (
                    bool(
                        frame_index < len(crop_fallback_mask)
                        and not crop_fallback_mask[frame_index]
                    )
                    if include_crop
                    else False
                )
                full_rgb = cv2.cvtColor(
                    cv2.resize(
                        frame_bgr,
                        (spec.resolution, spec.resolution),
                        interpolation=cv2.INTER_AREA,
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                last_full = full_rgb
                if include_crop:
                    source_crop = _crop_frame(
                        frame_bgr,
                        crop_boxes_xyxy[frame_index]
                        if frame_index < len(crop_boxes_xyxy)
                        else None,
                    )
                    resized_crop = cv2.resize(
                        source_crop,
                        (spec.resolution, spec.resolution),
                        interpolation=cv2.INTER_AREA,
                    )
                    last_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                for position in positions_by_frame[frame_index]:
                    full_output[position] = last_full
                    if include_crop:
                        crop_output[position] = last_crop
                        crop_found_output[position] = last_sampled_crop_found
            frame_index += 1
    finally:
        capture.release()

    if last_full is None:
        raise OSError(f"Could not decode any sampled frames from {video_path}")
    actual_indices = wanted.copy()
    for position in range(spec.cache_frames):
        if full_output[position] is None:
            full_output[position] = last_full
            actual_indices[position] = last_sampled_index
        if include_crop and crop_output[position] is None:
            crop_output[position] = last_crop if last_crop is not None else last_full
            crop_found_output[position] = last_sampled_crop_found

    full_frames = np.stack(full_output).astype(np.uint8, copy=False)
    crop_frames = (
        np.stack(crop_output).astype(np.uint8, copy=False) if include_crop else None
    )
    return DecodedClip(
        full_frames=full_frames,
        frame_indices=actual_indices.astype(np.int32, copy=False),
        crop_frames=crop_frames,
        crop_found_mask=crop_found_output if include_crop else None,
    )
