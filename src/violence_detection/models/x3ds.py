"""X3D model construction and Kinetics-pretrained backbone initialization."""

from __future__ import annotations

import torch
from torch import nn


def _factory(variant: str):
    if variant not in {"x3d_s", "x3d_m"}:
        raise ValueError(f"Unsupported X3D variant: {variant!r}")
    try:
        from pytorchvideo.models.hub import x3d_m, x3d_s
    except ImportError as exc:
        raise ImportError(
            "X3D requires pytorchvideo. Install requirements-kaggle.txt or the video extra."
        ) from exc
    return {"x3d_s": x3d_s, "x3d_m": x3d_m}[variant]


def build_x3d(variant: str = "x3d_s", num_classes: int = 2) -> nn.Module:
    """Build an uninitialized X3D model and replace its classification layer."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    model = _factory(variant)(pretrained=False)
    head = model.blocks[5]
    if not isinstance(head.proj, nn.Linear):
        raise TypeError("Expected X3D classification projection to be nn.Linear")
    head.proj = nn.Linear(head.proj.in_features, num_classes)
    return model


def initialize_pretrained_backbone(model: nn.Module, variant: str) -> list[str]:
    """Load Kinetics weights while retaining the newly initialized class head.

    Call on rank 0 before wrapping the model in DDP. DDP then broadcasts the
    initialized backbone to the second GPU without duplicate downloads.
    """
    pretrained_model = _factory(variant)(pretrained=True)
    target = model.state_dict()
    source = pretrained_model.state_dict()
    compatible: dict[str, torch.Tensor] = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    allowed_missing = {"blocks.5.proj.weight", "blocks.5.proj.bias"}
    unexpected_missing = set(missing) - allowed_missing
    if unexpected or unexpected_missing:
        raise RuntimeError(
            "Pretrained X3D checkpoint does not match this model: "
            f"missing={sorted(unexpected_missing)}, unexpected={sorted(unexpected)}"
        )
    return sorted(allowed_missing & set(missing))


def split_parameter_groups(
    model: nn.Module, backbone_lr: float, head_lr: float
) -> list[dict[str, object]]:
    """Return separate backbone/head optimizer groups for stable fine-tuning."""
    head_parameters = list(model.blocks[5].proj.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_ids
    ]
    return [
        {"params": backbone_parameters, "lr": backbone_lr},
        {"params": head_parameters, "lr": head_lr},
    ]
