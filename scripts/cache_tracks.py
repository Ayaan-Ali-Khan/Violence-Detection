"""Run person detection and ByteTrack over one or more video manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from violence_detection.detection.detect_track import iter_video_tracks, load_yolo


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(path_value: str, dataset: str, data_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if data_root == Path("."):
        return path
    parts = path.parts
    if len(parts) >= 3 and parts[:2] == ("data", "raw"):
        return data_root.joinpath(*parts[2:])
    return data_root / dataset / path


def output_path_for(cache_dir: Path, dataset: str, split: str, source_name: str) -> Path:
    rel = Path(source_name)
    parts = rel.parts
    if len(parts) >= 3 and parts[:2] == ("data", "raw"):
        rel = Path(*parts[3:])
    return cache_dir / dataset / split / rel.with_name(f"{rel.name}.npz")


def has_current_cache(path: Path, expected: dict[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            saved = json.loads(str(archive["metadata"].item()))
        return all(saved.get(key) == value for key, value in expected.items())
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def write_cache(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(
            stream,
            **arrays,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--weights")
    parser.add_argument("--tracker", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/detections/yolo11n"))
    parser.add_argument("--stats-output", type=Path, default=Path("data/detection/track_stats.csv"))
    parser.add_argument("--device")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--limit-clips", type=int, help="Optional smoke-run limit across all manifests")
    parser.add_argument("--force", action="store_true", help="Recompute valid existing cache files")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    weights = args.weights or config["weights"]
    tracker = args.tracker or Path(config["tracker"])
    device = args.device or str(config.get("device", "auto"))
    imgsz = args.imgsz or int(config["imgsz"])
    confidence = args.confidence if args.confidence is not None else float(config["confidence"])
    iou = args.iou if args.iou is not None else float(config["iou"])
    frame_stride = (
        args.frame_stride
        if args.frame_stride is not None
        else int(config.get("frame_stride", 1))
    )
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if not tracker.exists():
        raise FileNotFoundError(f"Tracker config not found: {tracker}")

    model = load_yolo(weights)
    model_checkpoint = Path(str(getattr(model, "ckpt_path", weights)))
    weights_hash = sha256_file(model_checkpoint) if model_checkpoint.is_file() else str(weights)
    settings = {
        "cache_schema_version": 1,
        "weights": str(weights),
        "weights_sha256": weights_hash,
        "tracker_sha256": sha256_file(tracker),
        "imgsz": imgsz,
        "confidence": confidence,
        "iou": iou,
        "device": device,
        "frame_stride": frame_stride,
    }
    config_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    manifest_rows = []
    for manifest in args.manifests:
        table = pd.read_csv(manifest)
        required = {"dataset", "split", "relative_path", "label", "label_name"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{manifest} is missing required columns: {sorted(missing)}")
        table["manifest"] = str(manifest)
        manifest_rows.extend(table.to_dict("records"))

    results = []
    started = time.perf_counter()
    processed = 0
    for row in manifest_rows:
        if args.limit_clips is not None and processed >= args.limit_clips:
            break
        dataset = str(row["dataset"])
        split = str(row["split"])
        source_path = resolve_source(str(row["relative_path"]), dataset, args.data_root)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Video not found: {source_path}. Set --data-root to the mounted dataset directory."
            )
        source_stat = source_path.stat()
        target = output_path_for(args.cache_dir, dataset, split, str(row["relative_path"]))
        expected = {
            "config_sha256": config_hash,
            "source_sha256": str(row.get("sha256", "")),
            "source_size_bytes": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
        }
        if not args.force and has_current_cache(target, expected):
            print(f"SKIP {dataset}/{split}: {source_path.name}")
            processed += 1
            continue

        frame_indices = []
        boxes = []
        scores = []
        classes = []
        track_ids = []
        per_frame_counts = []
        frame_count = 0
        for frame_index, tracked in iter_video_tracks(
            model,
            source_path,
            tracker=tracker,
            imgsz=imgsz,
            conf=confidence,
            iou=iou,
            device=device,
            frame_stride=frame_stride,
        ):
            per_frame_counts.append(len(tracked.boxes_xyxy))
            frame_count += 1
            if len(tracked.boxes_xyxy):
                frame_indices.extend([frame_index] * len(tracked.boxes_xyxy))
                boxes.extend(tracked.boxes_xyxy)
                scores.extend(tracked.scores)
                classes.extend(tracked.class_ids)
                track_ids.extend(tracked.track_ids)

        arrays = {
            "frame_indices": np.asarray(frame_indices, dtype=np.int32),
            "boxes_xyxy": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            "scores": np.asarray(scores, dtype=np.float32),
            "class_ids": np.asarray(classes, dtype=np.int16),
            "track_ids": np.asarray(track_ids, dtype=np.int32),
            "people_per_frame": np.asarray(per_frame_counts, dtype=np.int16),
        }
        metadata = {
            **expected,
            "dataset": dataset,
            "split": split,
            "relative_path": str(row["relative_path"]),
            "label": int(row["label"]),
            "label_name": str(row["label_name"]),
            "decoded_sampled_frames": frame_count,
            "manifest_frame_count": row.get("frame_count"),
            "model_settings": settings,
        }
        write_cache(target, arrays, metadata)
        results.append(
            {
                **metadata,
                "cache_path": target.as_posix(),
                "detections": len(boxes),
                "people_per_frame_mean": float(np.mean(per_frame_counts)) if per_frame_counts else 0.0,
                "no_detection_rate": float(np.mean(np.asarray(per_frame_counts) == 0)) if per_frame_counts else 1.0,
                "unique_track_ids": len(set(t for t in track_ids if t >= 0)),
            }
        )
        processed += 1
        if processed % 10 == 0:
            elapsed = time.perf_counter() - started
            print(f"Processed {processed}/{len(manifest_rows)} clips in {elapsed / 60:.1f} min")

    if results:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        new_stats = pd.DataFrame(results)
        if args.stats_output.exists() and not args.force:
            old = pd.read_csv(args.stats_output)
            old = old.loc[~old["cache_path"].isin(new_stats["cache_path"])]
            new_stats = pd.concat([old, new_stats], ignore_index=True)
        new_stats.to_csv(args.stats_output, index=False)
    elapsed = time.perf_counter() - started
    print(f"Visited {processed} clips in {elapsed / 60:.1f} min; cache root: {args.cache_dir}")
    if not results:
        print("No new clips were computed (all selected caches were already current).")


if __name__ == "__main__":
    main()
