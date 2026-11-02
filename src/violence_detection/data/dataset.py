"""Low-CPU PyTorch datasets backed by memory-mapped uint8 ``.npy`` tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from violence_detection.data.preprocess import PreprocessSpec, load_preprocess_spec


class CachedVideoDataset(Dataset):
    """Read full-frame, interaction-crop, or paired views without loading RAM copies.

    The full/crop arrays have ``(N, candidate_frames, H, W, 3)`` layout and
    stay memory mapped. Training randomly chooses one temporal candidate from
    each segment; evaluation always chooses the first for deterministic input.
    """

    def __init__(
        self,
        cache_root: str | Path,
        input_tag: str,
        split: str,
        *,
        model_name: str = "x3d_s",
        preprocess_config: str | Path = "configs/preprocess.yaml",
        mode: str = "full",
        training: bool = False,
        manifest_path: str | Path | None = None,
        return_crop_status: bool = False,
    ) -> None:
        if mode not in {"full", "crop", "both"}:
            raise ValueError("mode must be one of: full, crop, both")
        self.spec: PreprocessSpec = load_preprocess_spec(preprocess_config, model_name)
        self.mode = mode
        self.training = training
        self.return_crop_status = return_crop_status
        self.split_dir = Path(cache_root) / input_tag / split
        metadata_path = self.split_dir / "cache_meta.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Tensor cache metadata not found: {metadata_path}")
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        required_views = ["full", "crop"] if mode == "both" else [mode]
        available_views = set(metadata.get("views", []))
        missing_views = set(required_views) - available_views
        if missing_views:
            raise ValueError(
                f"Cache at {self.split_dir} lacks {sorted(missing_views)} view(s); "
                f"available: {sorted(available_views)}"
            )
        self.labels = np.load(self.split_dir / "labels.npy", mmap_mode="r")
        self.views = {
            view: np.load(self.split_dir / f"{view}_videos.npy", mmap_mode="r")
            for view in required_views
        }
        crop_status_path = self.split_dir / "crop_found_mask.npy"
        self.crop_found_mask = None
        if crop_status_path.is_file():
            self.crop_found_mask = np.load(crop_status_path, mmap_mode="r")
            if (
                self.crop_found_mask.shape
                != (
                    len(self.labels),
                    self.spec.cache_frames,
                )
                or self.crop_found_mask.dtype != np.bool_
            ):
                raise ValueError(f"Invalid crop-found mask at {crop_status_path}")
        if return_crop_status and self.crop_found_mask is None:
            raise ValueError(
                f"Crop-found status was requested but is missing at {crop_status_path}"
            )
        self.completed = np.load(self.split_dir / "completed.npy", mmap_mode="r")
        if not bool(np.all(self.completed)):
            incomplete = int((~self.completed).sum())
            raise ValueError(
                f"Tensor cache at {self.split_dir} has {incomplete} incomplete rows"
            )
        for name, array in self.views.items():
            if array.dtype != np.uint8 or array.ndim != 5 or array.shape[-1] != 3:
                raise ValueError(f"Invalid {name} tensor layout at {self.split_dir}")
        if any(len(array) != len(self.labels) for array in self.views.values()):
            raise ValueError("video tensor and label counts differ")
        self.rows = None
        if manifest_path is not None:
            manifest = pd.read_csv(manifest_path)
            manifest = manifest.loc[
                (manifest["dataset"] == metadata["dataset"])
                & (manifest["split"] == split)
            ].reset_index(drop=True)
            if len(manifest) != len(self.labels):
                raise ValueError("manifest row count does not match cached tensor rows")
            if not np.array_equal(
                manifest["label"].to_numpy(dtype=np.int64),
                np.asarray(self.labels, dtype=np.int64),
            ):
                raise ValueError("manifest labels do not match cached tensor labels")
            self.rows = manifest

    def __len__(self) -> int:
        return len(self.labels)

    def _sample_indices(self) -> np.ndarray:
        base = np.arange(self.spec.num_frames, dtype=np.int64) * (
            self.spec.cache_frames // self.spec.num_frames
        )
        if self.training:
            candidates = self.spec.cache_frames // self.spec.num_frames
            offsets = torch.randint(candidates, (self.spec.num_frames,)).numpy()
            base += offsets
        return base

    def _sample_view(
        self, array: np.ndarray, index: int, frame_indices: np.ndarray
    ) -> torch.Tensor:
        clip = np.ascontiguousarray(array[index, frame_indices])
        return torch.from_numpy(clip)

    def __getitem__(self, index: int):
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        frame_indices = self._sample_indices()
        crop_status = (
            torch.from_numpy(
                np.asarray(self.crop_found_mask[index, frame_indices]).copy()
            )
            if self.return_crop_status
            else None
        )
        if self.mode == "both":
            video = {
                "full": self._sample_view(self.views["full"], index, frame_indices),
                "crop": self._sample_view(self.views["crop"], index, frame_indices),
            }
        else:
            video = self._sample_view(self.views[self.mode], index, frame_indices)
        if self.return_crop_status:
            return video, label, crop_status
        return video, label

    def record(self, index: int) -> dict[str, Any]:
        """Return manifest metadata for prediction/report output."""
        if self.rows is None:
            return {"row": index, "label": int(self.labels[index])}
        return self.rows.iloc[index].to_dict()
