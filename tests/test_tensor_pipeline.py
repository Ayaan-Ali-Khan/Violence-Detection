"""Tests for uint8 preprocessing, resumable tensor caches, and mmap datasets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from violence_detection.data.augment import prepare_video_batch
from violence_detection.data.cache_tensors import process_split
from violence_detection.data.dataset import CachedVideoDataset
from violence_detection.data.preprocess import (
    load_preprocess_spec,
    sample_frame_indices,
)


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    assert writer.isOpened()
    for index in range(8):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :16] = (index * 20, 10, 230)
        frame[:, 16:] = (10, index * 20, 30)
        writer.write(frame)
    writer.release()


def _run_verifier(cache_root: Path, input_tag: str, manifest: Path, report_dir: Path):
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "verify_tensor_cache.py"),
            "--cache-dir",
            str(cache_root),
            "--input-tag",
            input_tag,
            "--manifest",
            str(manifest),
            "--report-dir",
            str(report_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = pd.read_csv(report_dir / "verification_report.csv")
    return result.iloc[0].to_dict()


def test_uniform_sampling_spans_clip_and_handles_short_videos() -> None:
    indices = sample_frame_indices(total_frames=8, cache_frames=32)

    assert indices.shape == (32,)
    assert indices.min() == 0
    assert indices.max() == 7
    assert np.all(indices[1:] >= indices[:-1])


def test_model_preprocessing_profile_has_expected_temporal_candidates() -> None:
    spec = load_preprocess_spec("configs/preprocess.yaml", "x3d_s")

    assert spec.num_frames == 16
    assert spec.cache_frames == 32
    assert spec.resolution == 176
    assert spec.input_tag == "rwf_full_f16_s160_v1"


def test_resume_safe_cache_and_mmap_full_crop_both_modes(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    video_path = raw_root / "rwf2000" / "train" / "Fight" / "sample.avi"
    video_path.parent.mkdir(parents=True)
    _write_test_video(video_path)
    relative_path = "data/raw/rwf2000/train/Fight/sample.avi"
    row = {
        "dataset": "rwf2000",
        "split": "train",
        "relative_path": relative_path,
        "label": 1,
        "label_name": "violent",
        "sha256": "test-video-hash",
        "file_size_bytes": video_path.stat().st_size,
        "frame_count": 8,
    }
    rows = pd.DataFrame([row])
    manifest_path = tmp_path / "manifest.csv"
    rows.to_csv(manifest_path, index=False)
    spec = load_preprocess_spec("configs/preprocess.yaml", "x3d_s")
    crop_root = tmp_path / "crop_cache"
    crop_path = (
        crop_root / "rwf2000" / "train" / "train" / "Fight" / "sample.avi.crop.npz"
    )
    crop_path.parent.mkdir(parents=True)
    crop_boxes = np.tile(np.asarray([0, 0, 16, 24], dtype=np.int32), (8, 1))
    with crop_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            crop_boxes_xyxy=crop_boxes,
            fallback_mask=np.zeros(8, dtype=np.uint8),
        )

    summary = process_split(
        rows,
        "train",
        spec=spec,
        config_hash="test-config",
        output_root=tmp_path / "tensor_cache",
        data_root=raw_root,
        mode="both",
        crop_cache_dir=crop_root,
        workers=0,
        max_clips=None,
        force=False,
    )
    split_dir = tmp_path / "tensor_cache" / spec.input_tag / "train"
    assert summary["completed"] == 1
    assert np.load(split_dir / "crop_found_mask.npy").all()
    first_verification = _run_verifier(
        tmp_path / "tensor_cache", spec.input_tag, manifest_path, tmp_path / "verify1"
    )
    assert first_verification["status"] == "ok"

    for mode in ("full", "crop", "both"):
        dataset = CachedVideoDataset(
            tmp_path / "tensor_cache",
            spec.input_tag,
            "train",
            model_name="x3d_s",
            preprocess_config="configs/preprocess.yaml",
            mode=mode,
            training=False,
            manifest_path=manifest_path,
        )
        views, label = dataset[0]
        assert int(label) == 1
        if mode == "both":
            batch = {name: view.unsqueeze(0) for name, view in views.items()}
        else:
            batch = views.unsqueeze(0)
        prepared = prepare_video_batch(
            batch,
            spec,
            training=False,
            augmentation={},
            device=torch.device("cpu"),
        )
        if mode == "both":
            assert prepared["full"].shape == (1, 3, 16, 160, 160)
            assert prepared["crop"].shape == (1, 3, 16, 160, 160)
        else:
            assert prepared.shape == (1, 3, 16, 160, 160)

    # A completed row with a bad checksum is invalidated and decoded again.
    full = np.load(split_dir / "full_videos.npy", mmap_mode="r+")
    full[0, 0, 0, 0, 0] ^= 255
    full.flush()
    resumed = process_split(
        rows,
        "train",
        spec=spec,
        config_hash="test-config",
        output_root=tmp_path / "tensor_cache",
        data_root=raw_root,
        mode="both",
        crop_cache_dir=crop_root,
        workers=0,
        max_clips=None,
        force=False,
    )
    assert resumed["completed"] == 1
    second_verification = _run_verifier(
        tmp_path / "tensor_cache", spec.input_tag, manifest_path, tmp_path / "verify2"
    )
    assert second_verification["status"] == "ok"
