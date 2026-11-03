"""X3D model construction and Kinetics-pretrained backbone initialization."""

from __future__ import annotations

from collections.abc import Mapping

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
    if variant == "x3d_m":
        # PyTorchVideo's X3D-M head pools with a fixed 7x7 spatial window,
        # which fails for our 160px input (the final feature map is 5x5).
        # Adaptive global pooling preserves the pretrained classifier's
        # expected channel count and supports the project's cached resolution.
        head.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    if not isinstance(head.proj, nn.Linear):
        raise TypeError("Expected X3D classification projection to be nn.Linear")
    head.proj = nn.Linear(head.proj.in_features, num_classes)
    return model


class LateFusionX3D(nn.Module):
    """Independent full-frame and crop X3D streams with a learned logit head."""

    def __init__(self, variant: str = "x3d_s", num_classes: int = 2) -> None:
        super().__init__()
        self.variant = variant
        self.full_stream = build_x3d(variant, num_classes)
        self.crop_stream = build_x3d(variant, num_classes)
        self.fusion_head = nn.Linear(num_classes * 2, num_classes)

    def forward(self, views: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(views) != {"full", "crop"}:
            raise ValueError("LateFusionX3D requires full and crop video batches")
        full_logits = self.full_stream(views["full"])
        crop_logits = self.crop_stream(views["crop"])
        return self.fusion_head(torch.cat((full_logits, crop_logits), dim=1))


class Swin3DViewAveraging(nn.Module):
    """One Swin3D-T classifier applied to both paired views, with mean logits."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        from torchvision.models.video import swin3d_t

        self.backbone = swin3d_t(weights=None)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)

    def forward(
        self, videos: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(videos, Mapping):
            if set(videos) != {"full", "crop"}:
                raise ValueError("Swin3D-T requires full and crop paired views")
            # Avoid materializing the paired views as one doubled input batch.
            return (self.backbone(videos["full"]) + self.backbone(videos["crop"])) / 2
        return self.backbone(videos)


def initialize_pretrained_swin3d_t(model: Swin3DViewAveraging) -> list[str]:
    """Load the torchvision Kinetics-400 Swin3D-T weights except its class head."""
    from torchvision.models.video import Swin3D_T_Weights, swin3d_t

    pretrained = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
    target = model.backbone.state_dict()
    compatible = {
        key: value
        for key, value in pretrained.state_dict().items()
        if key in target and target[key].shape == value.shape
    }
    missing, unexpected = model.backbone.load_state_dict(compatible, strict=False)
    allowed_missing = {"head.weight", "head.bias"}
    unexpected_missing = set(missing) - allowed_missing
    if unexpected or unexpected_missing:
        raise RuntimeError(
            "Pretrained Swin3D-T checkpoint does not match this model: "
            f"missing={sorted(unexpected_missing)}, unexpected={sorted(unexpected)}"
        )
    return sorted(allowed_missing & set(missing))


def swin3d_parameter_groups(
    model: Swin3DViewAveraging, backbone_lr: float, head_lr: float
) -> list[dict[str, object]]:
    """Use a smaller learning rate for the pretrained Swin3D-T backbone."""
    head_parameters = list(model.backbone.head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_ids
    ]
    return [
        {"params": backbone_parameters, "lr": backbone_lr},
        {"params": head_parameters, "lr": head_lr},
    ]


def initialize_fusion_streams(
    model: LateFusionX3D,
    full_checkpoint: str,
    crop_checkpoint: str,
) -> None:
    """Initialize the two streams from their independently trained specialists."""
    for name, path, stream, expected_mode in (
        ("full", full_checkpoint, model.full_stream, "full"),
        ("crop", crop_checkpoint, model.crop_stream, "crop"),
    ):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture", "x3d") != "x3d":
            raise ValueError(f"{name} stream checkpoint is not a single-stream X3D")
        if checkpoint.get("mode") != expected_mode:
            raise ValueError(
                f"{name} stream checkpoint mode must be {expected_mode!r}, "
                f"got {checkpoint.get('mode')!r}"
            )
        checkpoint_variant = checkpoint.get("model_variant")
        if checkpoint_variant != model.variant:
            raise ValueError(
                f"{name} stream variant {checkpoint_variant!r} does not match "
                f"fusion variant {model.variant!r}"
            )
        stream.load_state_dict(checkpoint["model_state_dict"], strict=True)


def initialize_pretrained_fusion(model: LateFusionX3D) -> list[list[str]]:
    """Initialize both late-fusion streams from the matching Kinetics weights."""
    return [
        initialize_pretrained_backbone(stream, model.variant)
        for stream in (model.full_stream, model.crop_stream)
    ]


def fusion_parameter_groups(
    model: LateFusionX3D, backbone_lr: float, head_lr: float
) -> list[dict[str, object]]:
    """Use lower LR for both video encoders and higher LR for classifier heads."""
    backbones: list[nn.Parameter] = []
    heads: list[nn.Parameter] = list(model.fusion_head.parameters())
    for stream in (model.full_stream, model.crop_stream):
        stream_head = list(stream.blocks[5].proj.parameters())
        head_ids = {id(parameter) for parameter in stream_head}
        heads.extend(stream_head)
        backbones.extend(
            parameter
            for parameter in stream.parameters()
            if id(parameter) not in head_ids
        )
    return [
        {"params": backbones, "lr": backbone_lr},
        {"params": heads, "lr": head_lr},
    ]


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
