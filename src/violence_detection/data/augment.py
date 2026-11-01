"""Light, clip-consistent augmentations applied on the GPU."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from violence_detection.data.preprocess import PreprocessSpec


def _to_model_layout(
    videos: torch.Tensor,
    spec: PreprocessSpec,
    *,
    training: bool,
    flip_mask: torch.Tensor,
    brightness_delta: torch.Tensor,
) -> torch.Tensor:
    # CPU batches arrive as B,T,H,W,C uint8. Transfer, then convert to B,C,T,H,W.
    value = videos.to(device=flip_mask.device, non_blocking=True)
    value = value.permute(0, 4, 1, 2, 3).to(dtype=torch.float32).div_(255.0)
    batch, channels, frames, height, width = value.shape
    if height != spec.input_size or width != spec.input_size:
        value = F.interpolate(
            value.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, channels, height, width
            ),
            size=(spec.input_size, spec.input_size),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, channels, spec.input_size, spec.input_size)
        value = value.permute(0, 2, 1, 3, 4)
    if training:
        value = torch.where(flip_mask.view(batch, 1, 1, 1, 1), value.flip(-1), value)
        value = (value + brightness_delta.view(batch, 1, 1, 1, 1)).clamp_(0.0, 1.0)
    mean = value.new_tensor(spec.mean).view(1, 3, 1, 1, 1)
    std = value.new_tensor(spec.std).view(1, 3, 1, 1, 1)
    return ((value - mean) / std).contiguous(memory_format=torch.channels_last_3d)


def prepare_video_batch(
    videos: torch.Tensor | Mapping[str, torch.Tensor],
    spec: PreprocessSpec,
    *,
    training: bool,
    augmentation: Mapping[str, float],
    device: torch.device,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Resize, augment, and normalize one batch on the target device.

    A single random flip and brightness adjustment is shared across all frames
    in a clip. For ``both`` view batches the same random values apply to each
    view, keeping their transformations aligned.
    """
    batch_size = (
        next(iter(videos.values())).shape[0]
        if isinstance(videos, Mapping)
        else videos.shape[0]
    )
    if training:
        flip = torch.rand(batch_size, device=device) < float(
            augmentation.get("horizontal_flip_probability", 0.5)
        )
        delta = (torch.rand(batch_size, device=device) * 2.0 - 1.0) * float(
            augmentation.get("brightness_delta", 0.1)
        )
    else:
        flip = torch.zeros(batch_size, dtype=torch.bool, device=device)
        delta = torch.zeros(batch_size, dtype=torch.float32, device=device)
    if isinstance(videos, Mapping):
        return {
            name: _to_model_layout(
                view,
                spec,
                training=training,
                flip_mask=flip,
                brightness_delta=delta,
            )
            for name, view in videos.items()
        }
    return _to_model_layout(
        videos,
        spec,
        training=training,
        flip_mask=flip,
        brightness_delta=delta,
    )
