"""Create resume-safe memory-mapped uint8 video tensors from a split manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

from violence_detection.data.preprocess import (
    PreprocessSpec,
    decode_video,
    load_preprocess_spec,
)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_hash(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()


def resolve_video_path(path_value: str, dataset: str, data_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() or data_root == Path("."):
        return path
    parts = path.parts
    if len(parts) >= 3 and parts[:2] == ("data", "raw"):
        return data_root.joinpath(*parts[2:])
    return data_root / dataset / path


def resolve_crop_cache_path(
    crop_cache_dir: Path, dataset: str, split: str, relative_path: str
) -> Path:
    relative = Path(relative_path)
    parts = relative.parts
    if len(parts) >= 3 and parts[:2] == ("data", "raw"):
        relative = Path(*parts[3:])
    standard = (
        crop_cache_dir / dataset / split / relative.parent / f"{relative.name}.crop.npz"
    )
    if standard.is_file():
        return standard
    # Some Kaggle Dataset uploads flatten the duplicated split directory.
    flattened = crop_cache_dir / dataset / relative.parent / f"{relative.name}.crop.npz"
    return flattened if flattened.is_file() else standard


def _decode_worker(
    job: tuple[int, str, dict[str, Any], bool, str | None, str | None],
):
    row_index, video_path, spec_data, include_crop, crop_path, crop_settings_hash = job
    try:
        spec = PreprocessSpec(
            **{
                **spec_data,
                "mean": tuple(spec_data["mean"]),
                "std": tuple(spec_data["std"]),
            }
        )
        crop_boxes = None
        crop_fallback_mask = None
        if include_crop:
            if crop_path is None or not Path(crop_path).is_file():
                raise FileNotFoundError(
                    f"Interaction crop cache not found: {crop_path}"
                )
            with np.load(crop_path, allow_pickle=False) as archive:
                crop_boxes = archive["crop_boxes_xyxy"]
                if "fallback_mask" not in archive:
                    raise ValueError(
                        f"Crop cache has no fallback mask for audit: {crop_path}"
                    )
                crop_fallback_mask = archive["fallback_mask"].astype(bool)
                if crop_settings_hash is not None:
                    crop_metadata = json.loads(str(archive["metadata"].item()))
                    if crop_metadata.get("settings_sha256") != crop_settings_hash:
                        raise ValueError(
                            "interaction crop cache settings do not match the selected "
                            f"tau/margin profile: {crop_path}"
                        )
        decoded = decode_video(
            video_path,
            spec,
            include_crop=include_crop,
            crop_boxes_xyxy=crop_boxes,
            crop_fallback_mask=crop_fallback_mask,
        )
        return row_index, decoded, None
    except Exception as exc:  # Return errors to the parent so other clips continue.
        return row_index, None, f"{type(exc).__name__}: {exc}"


def _initialize_array(path: Path, shape: tuple[int, ...], dtype: np.dtype):
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    array[...] = 0
    array.flush()
    return array


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def process_split(
    rows: pd.DataFrame,
    split: str,
    *,
    spec: PreprocessSpec,
    config_hash: str,
    output_root: Path,
    data_root: Path,
    mode: str,
    crop_cache_dir: Path | None,
    crop_settings_hash: str | None = None,
    workers: int,
    max_clips: int | None,
    force: bool,
) -> dict[str, Any]:
    rows = rows.reset_index(drop=True).copy()
    split_dir = output_root / spec.input_tag / split
    meta_path = split_dir / "cache_meta.json"
    split_dir.mkdir(parents=True, exist_ok=True)

    view_names = ["full", "crop"] if mode == "both" else [mode]
    shape = (
        len(rows),
        spec.cache_frames,
        spec.resolution,
        spec.resolution,
        3,
    )
    manifest_records = rows.to_dict("records")
    spec_metadata = json.loads(json.dumps(spec.as_dict()))
    manifest_fingerprint = _json_hash(
        [
            {
                "relative_path": str(row["relative_path"]),
                "label": int(row["label"]),
                "sha256": str(row.get("sha256", "")),
                "file_size_bytes": int(row.get("file_size_bytes", 0) or 0),
            }
            for row in manifest_records
        ]
    )
    expected_meta = {
        "dataset": str(rows.iloc[0]["dataset"]) if len(rows) else "rwf2000",
        "split": split,
        "input_tag": spec.input_tag,
        "model_profile": spec.model_name,
        "spec": spec_metadata,
        "config_sha256": config_hash,
        "manifest_sha256": manifest_fingerprint,
        "rows": len(rows),
        "shape": list(shape),
        "dtype": "uint8",
        "views": view_names,
        "view_mode": mode,
    }
    if crop_settings_hash is not None:
        expected_meta["crop_settings_sha256"] = crop_settings_hash
    if force:
        shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
    elif meta_path.exists():
        saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved_meta != expected_meta:
            raise ValueError(
                f"Existing cache metadata differs at {split_dir}; use --force to rebuild."
            )
    elif any(split_dir.iterdir()):
        raise ValueError(
            f"Unrecognized files already exist in {split_dir}; inspect or use --force."
        )

    view_arrays = {}
    checksum_arrays = {}
    for view in view_names:
        tensor_path = split_dir / f"{view}_videos.npy"
        checksum_path = split_dir / f"{view}_sha256.npy"
        if tensor_path.exists():
            array = np.load(tensor_path, mmap_mode="r+")
            if array.shape != shape or array.dtype != np.uint8:
                raise ValueError(f"Unexpected shape or dtype in {tensor_path}")
        else:
            array = _initialize_array(tensor_path, shape, np.dtype(np.uint8))
        view_arrays[view] = array
        if checksum_path.exists():
            checksums = np.load(checksum_path, mmap_mode="r+")
            if checksums.shape != (len(rows),) or checksums.dtype.kind != "S":
                raise ValueError(f"Unexpected checksum array in {checksum_path}")
        else:
            checksums = _initialize_array(checksum_path, (len(rows),), np.dtype("S64"))
        checksum_arrays[view] = checksums

    crop_found_mask = crop_found_sha256 = None
    if "crop" in view_names:
        crop_found_path = split_dir / "crop_found_mask.npy"
        crop_found_hash_path = split_dir / "crop_found_sha256.npy"
        if crop_found_path.exists():
            crop_found_mask = np.load(crop_found_path, mmap_mode="r+")
            if (
                crop_found_mask.shape != (len(rows), spec.cache_frames)
                or crop_found_mask.dtype != np.bool_
            ):
                raise ValueError(
                    f"Unexpected crop-found mask layout at {crop_found_path}"
                )
        else:
            crop_found_mask = _initialize_array(
                crop_found_path, (len(rows), spec.cache_frames), np.dtype(np.bool_)
            )
        if crop_found_hash_path.exists():
            crop_found_sha256 = np.load(crop_found_hash_path, mmap_mode="r+")
            if (
                crop_found_sha256.shape != (len(rows),)
                or crop_found_sha256.dtype.kind != "S"
            ):
                raise ValueError(
                    f"Unexpected crop-found checksum layout at {crop_found_hash_path}"
                )
        else:
            crop_found_sha256 = _initialize_array(
                crop_found_hash_path, (len(rows),), np.dtype("S64")
            )

    labels_path = split_dir / "labels.npy"
    if labels_path.exists():
        labels = np.load(labels_path, mmap_mode="r+")
        if labels.shape != (len(rows),) or labels.dtype != np.int64:
            raise ValueError(f"Unexpected labels array at {labels_path}")
    else:
        labels = np.asarray(rows["label"], dtype=np.int64)
        np.save(labels_path, labels)
    labels[...] = np.asarray(rows["label"], dtype=np.int64)

    source_indices_path = split_dir / "source_frame_indices.npy"
    source_shape = (len(rows), spec.cache_frames)
    if source_indices_path.exists():
        source_indices = np.load(source_indices_path, mmap_mode="r+")
        if source_indices.shape != source_shape or source_indices.dtype != np.int32:
            raise ValueError(
                f"Unexpected source-frame indices at {source_indices_path}"
            )
    else:
        source_indices = _initialize_array(
            source_indices_path, source_shape, np.dtype(np.int32)
        )

    complete_path = split_dir / "completed.npy"
    if complete_path.exists():
        completed = np.load(complete_path, mmap_mode="r+")
        if completed.shape != (len(rows),) or completed.dtype != np.bool_:
            raise ValueError(f"Unexpected completion bitmap at {complete_path}")
    else:
        completed = _initialize_array(complete_path, (len(rows),), np.dtype(np.bool_))
    for view in view_names:
        for row_index in np.flatnonzero(completed):
            expected_hash = (
                bytes(checksum_arrays[view][row_index]).decode().rstrip("\x00")
            )
            if (
                not expected_hash
                or _file_hash(view_arrays[view][row_index]) != expected_hash
            ):
                completed[row_index] = False
    if crop_found_mask is not None:
        for row_index in np.flatnonzero(completed):
            expected_hash = bytes(crop_found_sha256[row_index]).decode().rstrip("\x00")
            if (
                not expected_hash
                or _file_hash(crop_found_mask[row_index]) != expected_hash
            ):
                completed[row_index] = False

    if not meta_path.exists():
        _atomic_json(meta_path, expected_meta)
    rows.to_csv(split_dir / "manifest.csv", index=False)

    pending_indices = [int(i) for i in np.flatnonzero(~completed)]
    if max_clips is not None:
        pending_indices = pending_indices[:max_clips]
    jobs = []
    for row_index in pending_indices:
        row = manifest_records[row_index]
        video_path = resolve_video_path(
            str(row["relative_path"]), str(row["dataset"]), data_root
        )
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        expected_size = int(row.get("file_size_bytes", 0) or 0)
        if expected_size and video_path.stat().st_size != expected_size:
            raise ValueError(
                f"File size mismatch for {video_path}: manifest={expected_size}, actual={video_path.stat().st_size}"
            )
        crop_path = (
            resolve_crop_cache_path(
                crop_cache_dir,
                str(row["dataset"]),
                str(row["split"]),
                str(row["relative_path"]),
            )
            if "crop" in view_names and crop_cache_dir is not None
            else None
        )
        jobs.append(
            (
                row_index,
                str(video_path),
                spec.as_dict(),
                "crop" in view_names,
                str(crop_path) if crop_path else None,
                crop_settings_hash,
            )
        )

    errors: list[dict[str, Any]] = []
    completed_since_flush = 0

    def save_result(row_index: int, decoded, error: str | None) -> None:
        nonlocal completed_since_flush
        row = manifest_records[row_index]
        if error:
            errors.append(
                {
                    "row": row_index,
                    "relative_path": row["relative_path"],
                    "error": error,
                }
            )
            return
        if "full" in view_names:
            view_arrays["full"][row_index] = decoded.full_frames
            checksum_arrays["full"][row_index] = _file_hash(
                decoded.full_frames
            ).encode()
        if "crop" in view_names:
            if decoded.crop_frames is None:
                errors.append(
                    {
                        "row": row_index,
                        "relative_path": row["relative_path"],
                        "error": "crop view requested but no crop frames were decoded",
                    }
                )
                return
            view_arrays["crop"][row_index] = decoded.crop_frames
            checksum_arrays["crop"][row_index] = _file_hash(
                decoded.crop_frames
            ).encode()
            if decoded.crop_found_mask is None:
                errors.append(
                    {
                        "row": row_index,
                        "relative_path": row["relative_path"],
                        "error": "crop-found mask was not decoded",
                    }
                )
                return
            crop_found_mask[row_index] = decoded.crop_found_mask
            crop_found_sha256[row_index] = _file_hash(decoded.crop_found_mask).encode()
        source_indices[row_index] = decoded.frame_indices
        completed[row_index] = True
        completed_since_flush += 1
        if completed_since_flush >= 8:
            flush_arrays()
            completed_since_flush = 0
        if int(completed.sum()) % 50 == 0:
            print(
                f"{split}: {int(completed.sum())}/{len(rows)} clips cached", flush=True
            )

    def flush_arrays() -> None:
        for array in [
            *view_arrays.values(),
            *checksum_arrays.values(),
            labels,
            source_indices,
            *(
                [crop_found_mask, crop_found_sha256]
                if crop_found_mask is not None
                else []
            ),
        ]:
            array.flush() if hasattr(array, "flush") else None
        # Commit the completion bitmap only after data/checksum pages are flushed.
        completed.flush()

    if workers > 0 and jobs:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            job_iterator = iter(jobs)
            active = set()
            for _ in range(min(len(jobs), workers * 2)):
                active.add(executor.submit(_decode_worker, next(job_iterator)))
            while active:
                done, active = concurrent.futures.wait(
                    active, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    row_index, decoded, error = future.result()
                    save_result(row_index, decoded, error)
                    try:
                        active.add(executor.submit(_decode_worker, next(job_iterator)))
                    except StopIteration:
                        pass
    else:
        for job in jobs:
            row_index, decoded, error = _decode_worker(job)
            save_result(row_index, decoded, error)
    if completed_since_flush:
        flush_arrays()

    if errors:
        pd.DataFrame(errors).to_csv(split_dir / "cache_errors.csv", index=False)
    elif (split_dir / "cache_errors.csv").exists():
        (split_dir / "cache_errors.csv").unlink()

    summary = {
        "split": split,
        "clips": len(rows),
        "completed": int(completed.sum()),
        "pending": int((~completed).sum()),
        "errors": len(errors),
        "directory": split_dir.as_posix(),
        "bytes": sum(
            path.stat().st_size for path in split_dir.iterdir() if path.is_file()
        ),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if errors or (summary["pending"] and max_clips is None):
        raise RuntimeError(
            f"Tensor cache for {split} is incomplete; review {split_dir / 'cache_errors.csv'} and rerun to resume."
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/splits/rwf2000.csv")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    parser.add_argument("--model", choices=["x3d_s", "x3d_m"], default="x3d_s")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/tensor_cache/rwf2000")
    )
    parser.add_argument("--crop-cache-dir", type=Path)
    parser.add_argument("--mode", choices=["full", "crop", "both"], default="full")
    parser.add_argument(
        "--workers", type=int, default=2, help="Bounded CPU video-decode processes"
    )
    parser.add_argument(
        "--max-clips", type=int, help="Optional smoke/resume limit per split"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate cache files in each selected split",
    )
    args = parser.parse_args()
    if args.workers < 0 or (args.max_clips is not None and args.max_clips < 1):
        raise ValueError("workers must be non-negative and max-clips positive")
    if args.mode in {"crop", "both"} and args.crop_cache_dir is None:
        parser.error("--crop-cache-dir is required for crop or both cache modes")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = load_preprocess_spec(args.config, args.model)
    crop_settings = config.get("crop_settings")
    crop_settings_hash = None
    if args.mode in {"crop", "both"}:
        if not isinstance(crop_settings, dict):
            parser.error("crop/both cache mode requires crop_settings in the config")
        tau = float(crop_settings["tau"])
        settings = {
            "tau": tau,
            "min_pair_frames": int(crop_settings["min_pair_frames"]),
            "group_tau": tau * float(crop_settings["group_tau_multiplier"]),
            "margin": float(crop_settings["margin"]),
            "smoothing_window": int(crop_settings["smoothing_window"]),
            "max_gap": int(crop_settings["max_gap"]),
        }
        crop_settings_hash = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()
        ).hexdigest()
    config_hash = _json_hash(
        {
            "version": config.get("version", 1),
            "dataset": config.get("dataset", "rwf2000"),
            "spec": spec.as_dict(),
            "mode": args.mode,
            "crop_settings_sha256": crop_settings_hash,
        }
    )
    manifest = pd.read_csv(args.manifest)
    required = {"dataset", "split", "relative_path", "label", "label_name"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{args.manifest} is missing columns: {sorted(missing)}")
    manifest = manifest.loc[manifest["dataset"] == "rwf2000"]
    if manifest.empty:
        raise ValueError("Manifest contains no RWF-2000 rows")

    split_order = [
        name
        for name in ("train", "validation", "test")
        if name in set(manifest["split"])
    ]
    summaries = []
    for split in split_order:
        rows = manifest.loc[manifest["split"] == split]
        summaries.append(
            process_split(
                rows,
                split,
                spec=spec,
                config_hash=config_hash,
                output_root=args.cache_dir,
                data_root=args.data_root,
                mode=args.mode,
                crop_cache_dir=args.crop_cache_dir,
                crop_settings_hash=crop_settings_hash,
                workers=args.workers,
                max_clips=args.max_clips,
                force=args.force,
            )
        )
    total_clips = sum(summary["clips"] for summary in summaries)
    total_bytes = sum(summary["bytes"] for summary in summaries)
    print(
        f"Cached {total_clips} RWF clips; tensor cache size: {total_bytes / (1024**3):.2f} GiB"
    )


if __name__ == "__main__":
    cv2.setNumThreads(1)
    main()
