"""Aggregate person-per-frame and no-detection statistics from track caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_cache(path: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        people_per_frame = archive["people_per_frame"].astype(np.int32, copy=False)
        track_ids = archive["track_ids"].astype(np.int64, copy=False)
    return metadata, people_per_frame, track_ids


def summarize(
    group: list[dict[str, object]], *, aggregate_labels: bool = False
) -> dict[str, object]:
    people = np.concatenate([entry["people_per_frame"] for entry in group])
    total_tracks = sum(int(entry["unique_track_ids"]) for entry in group)
    return {
        "dataset": group[0]["dataset"],
        "split": group[0]["split"],
        "label_name": (
            "all"
            if aggregate_labels or len({entry["label_name"] for entry in group}) != 1
            else group[0]["label_name"]
        ),
        "clips": len(group),
        "frames_processed": len(people),
        "person_detections": int(people.sum()),
        "mean_people_per_frame": float(people.mean()) if len(people) else 0.0,
        "median_people_per_frame": float(np.median(people)) if len(people) else 0.0,
        "p95_people_per_frame": float(np.percentile(people, 95)) if len(people) else 0.0,
        "no_detection_frames": int(np.count_nonzero(people == 0)),
        "no_detection_frame_rate": float(np.mean(people == 0)) if len(people) else 1.0,
        "clips_with_no_detections": sum(not np.any(entry["people_per_frame"]) for entry in group),
        "no_detection_clip_rate": float(
            np.mean([not np.any(entry["people_per_frame"]) for entry in group])
        ),
        "track_instances": total_tracks,
        "mean_track_instances_per_clip": total_tracks / len(group) if group else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/detections/yolo11n"))
    parser.add_argument("--output", type=Path, default=Path("data/detection/detection_stats.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/detection/detection_stats.md"))
    args = parser.parse_args()

    caches = sorted(args.cache_dir.rglob("*.npz"))
    if not caches:
        raise FileNotFoundError(f"No track caches found under {args.cache_dir}")

    entries = []
    for path in caches:
        metadata, people_per_frame, track_ids = read_cache(path)
        entries.append(
            {
                "dataset": metadata["dataset"],
                "split": metadata["split"],
                "label_name": metadata["label_name"],
                "people_per_frame": people_per_frame,
                "unique_track_ids": len(set(int(track_id) for track_id in track_ids if track_id >= 0)),
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for entry in entries:
        grouped.setdefault(
            (str(entry["dataset"]), str(entry["split"]), str(entry["label_name"])), []
        ).append(entry)
    rows = [summarize(group) for _, group in sorted(grouped.items())]
    all_grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for entry in entries:
        all_grouped.setdefault((str(entry["dataset"]), str(entry["split"])), []).append(entry)
    rows.extend(
        summarize(group, aggregate_labels=True)
        for _, group in sorted(all_grouped.items())
    )
    summary = pd.DataFrame(rows).sort_values(["dataset", "split", "label_name"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    report = [
        "# Detection and tracking statistics",
        "",
        f"Cache directory: `{args.cache_dir.as_posix()}`",
        f"Cache files included: {len(caches)}",
        "",
        "Statistics are calculated over processed frames. If a cache used a frame stride greater than one, the rates apply to sampled frames.",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "`no_detection_frame_rate` is the fraction of processed frames with zero person boxes. `track_instances` sums unique nonnegative ByteTrack IDs within each clip; IDs reset for each clip.",
        "",
        f"CSV: `{args.output.as_posix()}`",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
