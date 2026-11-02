"""Verify tensor-cache completeness, shapes, checksums, and storage size."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def verify_split(split_dir: Path, expected_rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "split": split_dir.name,
        "directory": split_dir.as_posix(),
        "status": "ok",
        "issues": [],
    }
    if not split_dir.is_dir():
        return {**result, "status": "missing", "issues": ["split directory missing"]}
    metadata_path = split_dir / "cache_meta.json"
    metadata = None
    if not metadata_path.is_file():
        result["issues"].append("cache_meta.json missing")
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            metadata = None
            result["issues"].append(f"invalid cache metadata: {exc}")
        if metadata:
            result["metadata_rows"] = int(metadata.get("rows", -1))
            result["expected_shape"] = metadata.get("shape")
            result["expected_dtype"] = metadata.get("dtype")
            result["views"] = metadata.get("views", [])
            if (
                expected_rows is not None
                and int(metadata.get("rows", -1)) != expected_rows
            ):
                result["issues"].append(
                    f"manifest count {expected_rows} != cache row count {metadata.get('rows')}"
                )

    for filename in ("labels.npy", "source_frame_indices.npy", "completed.npy"):
        if not (split_dir / filename).is_file():
            result["issues"].append(f"{filename} missing")

    try:
        completed = np.load(split_dir / "completed.npy", mmap_mode="r")
        result["completed_rows"] = int(completed.sum())
        result["incomplete_rows"] = int((~completed).sum())
        if result["incomplete_rows"]:
            result["issues"].append(f"{result['incomplete_rows']} rows are incomplete")
    except (OSError, ValueError) as exc:
        completed = None
        result["issues"].append(f"completed.npy is corrupt: {exc}")

    if metadata:
        expected_shape = tuple(int(value) for value in metadata.get("shape", []))
        rows = expected_shape[0] if expected_shape else 0
        for view in metadata.get("views", []):
            tensor_path = split_dir / f"{view}_videos.npy"
            checksum_path = split_dir / f"{view}_sha256.npy"
            try:
                tensor = np.load(tensor_path, mmap_mode="r")
                if tensor.shape != expected_shape:
                    result["issues"].append(
                        f"{tensor_path.name} shape {tensor.shape} != {expected_shape}"
                    )
                if tensor.dtype != np.uint8:
                    result["issues"].append(
                        f"{tensor_path.name} dtype {tensor.dtype} != uint8"
                    )
                result[f"{view}_shape"] = str(tensor.shape)
                result[f"{view}_dtype"] = str(tensor.dtype)
                checksums = np.load(checksum_path, mmap_mode="r")
                if checksums.shape != (rows,):
                    result["issues"].append(
                        f"{checksum_path.name} shape {checksums.shape} != {(rows,)}"
                    )
                elif completed is not None:
                    checked = 0
                    corrupt = 0
                    for row_index in np.flatnonzero(completed):
                        expected = bytes(checksums[row_index]).decode().rstrip("\x00")
                        if not expected or _digest(tensor[row_index]) != expected:
                            corrupt += 1
                        checked += 1
                    result[f"{view}_checksums_checked"] = checked
                    result[f"{view}_corrupt_rows"] = corrupt
                    if corrupt:
                        result["issues"].append(
                            f"{corrupt} {view} tensor rows have missing or mismatched checksums"
                        )
            except (OSError, ValueError, KeyError) as exc:
                result["issues"].append(
                    f"{view} tensor/checksum file missing or corrupt: {exc}"
                )

        if "crop" in metadata.get("views", []):
            mask_path = split_dir / "crop_found_mask.npy"
            checksum_path = split_dir / "crop_found_sha256.npy"
            try:
                masks = np.load(mask_path, mmap_mode="r")
                checksums = np.load(checksum_path, mmap_mode="r")
                expected_mask_shape = (rows, expected_shape[1])
                if masks.shape != expected_mask_shape or masks.dtype != np.bool_:
                    result["issues"].append(
                        f"crop_found_mask.npy shape/dtype invalid: {masks.shape}/{masks.dtype}"
                    )
                if checksums.shape != (rows,):
                    result["issues"].append(
                        f"crop_found_sha256.npy shape {checksums.shape} != {(rows,)}"
                    )
                elif completed is not None:
                    checked = 0
                    corrupt = 0
                    for row_index in np.flatnonzero(completed):
                        expected = bytes(checksums[row_index]).decode().rstrip("\x00")
                        if not expected or _digest(masks[row_index]) != expected:
                            corrupt += 1
                        checked += 1
                    result["crop_status_checksums_checked"] = checked
                    result["crop_status_corrupt_rows"] = corrupt
                    if corrupt:
                        result["issues"].append(
                            f"{corrupt} crop-found status rows have missing or mismatched checksums"
                        )
                result["mean_candidate_crop_found_fraction"] = float(masks.mean())
            except (OSError, ValueError) as exc:
                result["issues"].append(
                    f"crop-found mask/checksum missing or corrupt: {exc}"
                )

    for filename in ("labels.npy", "source_frame_indices.npy"):
        path = split_dir / filename
        if path.is_file():
            try:
                array = np.load(path, mmap_mode="r")
                if filename == "labels.npy" and (
                    array.dtype != np.int64 or len(array) != result.get("metadata_rows")
                ):
                    result["issues"].append("labels.npy shape or dtype is invalid")
                if filename == "source_frame_indices.npy" and array.dtype != np.int32:
                    result["issues"].append("source_frame_indices.npy dtype is invalid")
            except (OSError, ValueError) as exc:
                result["issues"].append(f"{filename} is corrupt: {exc}")

    result["disk_bytes"] = sum(
        path.stat().st_size for path in split_dir.rglob("*") if path.is_file()
    )
    if result["issues"]:
        result["status"] = "issues"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/tensor_cache/rwf2000")
    )
    parser.add_argument("--input-tag", default="rwf_full_f16_s160_v1")
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/splits/rwf2000.csv")
    )
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest = manifest.loc[manifest["dataset"] == "rwf2000"]
    expected = manifest.groupby("split").size().to_dict()
    root = args.cache_dir / args.input_tag
    split_dirs = [
        root / split for split in ("train", "validation", "test") if split in expected
    ]
    results = [verify_split(path, int(expected[path.name])) for path in split_dirs]
    report_dir = args.report_dir or root
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "verification_report.csv"
    md_path = report_dir / "verification_report.md"
    pd.DataFrame(results).to_csv(csv_path, index=False)

    total_bytes = sum(int(item.get("disk_bytes", 0)) for item in results)
    total_clips = sum(int(item.get("metadata_rows", 0)) for item in results)
    ok = all(item["status"] == "ok" for item in results)
    lines = [
        "# Tensor-cache verification",
        "",
        f"Input profile: `{args.input_tag}`",
        f"Cache root: `{root.as_posix()}`",
        f"Overall status: **{'PASS' if ok else 'ISSUES FOUND'}**",
        f"Clips represented: {total_clips}",
        f"Disk size: {total_bytes / (1024**3):.3f} GiB ({total_bytes:,} bytes)",
        "",
        pd.DataFrame(results).to_string(index=False),
        "",
        f"Detailed report: `{csv_path.as_posix()}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
