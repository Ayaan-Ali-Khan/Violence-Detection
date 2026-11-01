"""Build per-frame pair/group crop-box caches from detection/track caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from violence_detection.crops import build_interaction_crops


def cache_path_for(cache_dir: Path, row: dict[str, object]) -> Path:
    relative = Path(str(row["relative_path"]))
    parts = relative.parts
    if len(parts) >= 3 and parts[:2] == ("data", "raw"):
        relative = Path(*parts[3:])
    return (
        cache_dir
        / str(row["dataset"])
        / str(row["split"])
        / relative.with_name(f"{relative.name}.npz")
    )


def crop_output_path(output_dir: Path, cache_dir: Path, cache_path: Path) -> Path:
    relative = cache_path.relative_to(cache_dir)
    return (
        output_dir / relative.parent / f"{relative.name.removesuffix('.npz')}.crop.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/detections/yolo11n")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("cache/interaction_crops/yolo11n")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/interaction_crops.yaml")
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("data/interaction_crops/crop_stats.csv"),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tau = float(config["tau"])
    min_pair_frames = int(config["min_pair_frames"])
    group_tau = tau * float(config["group_tau_multiplier"])
    margin = float(config["margin"])
    smoothing_window = int(config["smoothing_window"])
    max_gap = int(config["max_gap"])
    settings = {
        "tau": tau,
        "min_pair_frames": min_pair_frames,
        "group_tau": group_tau,
        "margin": margin,
        "smoothing_window": smoothing_window,
        "max_gap": max_gap,
    }
    settings_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()

    rows: list[dict[str, object]] = []
    for manifest_path in args.manifests:
        manifest = pd.read_csv(manifest_path)
        required = {
            "dataset",
            "split",
            "relative_path",
            "label",
            "label_name",
            "width",
            "height",
        }
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
        for row in manifest.to_dict("records"):
            detection_path = cache_path_for(args.cache_dir, row)
            if not detection_path.is_file():
                raise FileNotFoundError(f"Detection cache not found: {detection_path}")
            with np.load(detection_path, allow_pickle=False) as archive:
                detections = {
                    key: archive[key]
                    for key in (
                        "frame_indices",
                        "boxes_xyxy",
                        "scores",
                        "track_ids",
                        "people_per_frame",
                    )
                }
            frame_count = len(detections["people_per_frame"])
            width, height = int(row["width"]), int(row["height"])
            result = build_interaction_crops(
                frame_count,
                detections["frame_indices"],
                detections["boxes_xyxy"],
                detections["track_ids"],
                frame_width=width,
                frame_height=height,
                scores=detections["scores"],
                tau=tau,
                min_pair_frames=min_pair_frames,
                group_tau=group_tau,
                margin=margin,
                smoothing_window=smoothing_window,
                max_gap=max_gap,
            )
            target = crop_output_path(args.output_dir, args.cache_dir, detection_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "dataset": str(row["dataset"]),
                "split": str(row["split"]),
                "relative_path": str(row["relative_path"]),
                "label": int(row["label"]),
                "frame_width": width,
                "frame_height": height,
                "frame_count": frame_count,
                "tau": tau,
                "margin": margin,
                "settings_sha256": settings_hash,
                "pair_track_ids": list(result.selection.pair_track_ids)
                if result.selection
                else [],
                "group_track_ids": list(result.selection.group_track_ids)
                if result.selection
                else [],
                "pair_support_frames": result.selection.support_frames
                if result.selection
                else 0,
                "pair_support_fraction": result.selection.support_fraction
                if result.selection
                else 0.0,
            }
            with target.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    crop_boxes_xyxy=result.crops.boxes_xyxy,
                    fallback_mask=result.crops.fallback_mask.astype(np.uint8),
                    observed_mask=result.crops.observed_mask.astype(np.uint8),
                    interpolated_mask=result.crops.interpolated_mask.astype(np.uint8),
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
            nonfallback = ~result.crops.fallback_mask
            widths = result.crops.boxes_xyxy[:, 2] - result.crops.boxes_xyxy[:, 0]
            heights = result.crops.boxes_xyxy[:, 3] - result.crops.boxes_xyxy[:, 1]
            area_fraction = (widths * heights) / float(width * height)
            selection = result.selection
            rows.append(
                {
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "relative_path": row["relative_path"],
                    "label_name": row["label_name"],
                    "crop_cache_path": target.as_posix(),
                    "pair_track_ids": ";".join(map(str, selection.pair_track_ids))
                    if selection
                    else "",
                    "group_track_ids": ";".join(map(str, selection.group_track_ids))
                    if selection
                    else "",
                    "pair_support_frames": selection.support_frames if selection else 0,
                    "pair_support_fraction": selection.support_fraction
                    if selection
                    else 0.0,
                    "fallback_frames": int(result.crops.fallback_mask.sum()),
                    "fallback_rate": result.crops.fallback_rate,
                    "interpolated_frames": int(result.crops.interpolated_mask.sum()),
                    "mean_crop_area_fraction": float(area_fraction[nonfallback].mean())
                    if nonfallback.any()
                    else 1.0,
                }
            )

    stats = pd.DataFrame(rows)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.stats_output, index=False)
    report_path = args.stats_output.with_suffix(".md")
    grouped = stats.groupby(["dataset", "split"], as_index=False).agg(
        clips=("relative_path", "count"),
        clips_without_pair=("pair_track_ids", lambda values: int((values == "").sum())),
        mean_fallback_rate=("fallback_rate", "mean"),
        mean_crop_area_fraction=("mean_crop_area_fraction", "mean"),
    )
    report_path.write_text(
        "# Interaction crop cache summary\n\n"
        f"Pair threshold: `{tau}` normalized box heights; margin: `{margin}`; "
        f"smoothing window: `{smoothing_window}` frames; interpolation limit: `{max_gap}` frames.\n\n"
        f"Crop-box caches written: {len(stats)}\n\n"
        + grouped.to_string(index=False)
        + f"\n\nPer-clip details: `{args.stats_output.as_posix()}`\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(stats)} interaction crop caches to {args.output_dir}")
    print(f"Wrote per-clip stats to {args.stats_output}")
    print(f"Wrote summary to {report_path}")


if __name__ == "__main__":
    main()
