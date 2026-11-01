"""Sweep pair distance and crop margin on RWF-2000 validation only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from violence_detection.crops import build_crop_boxes, select_clip_pair


def detection_cache_path(cache_dir: Path, row: dict[str, object]) -> Path:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/splits/rwf2000.csv")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/detections/yolo11n")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/interaction_crops.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interaction_crops/validation_tuning.csv"),
    )
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0],
    )
    parser.add_argument(
        "--margins", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25, 0.30]
    )
    parser.add_argument(
        "--limit", type=int, help="Optional limit for a quick validation-only iteration"
    )
    args = parser.parse_args()

    table = pd.read_csv(args.manifest)
    validation = table.loc[
        (table["dataset"] == "rwf2000") & (table["split"] == "validation")
    ]
    if validation.empty:
        raise ValueError(
            "The manifest has no RWF-2000 validation rows; tuning is validation-only."
        )
    rows = validation.to_dict("records")
    if args.limit is not None:
        rows = rows[: args.limit]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    min_pair_frames = int(config["min_pair_frames"])
    group_tau_multiplier = float(config["group_tau_multiplier"])
    smoothing_window = int(config["smoothing_window"])
    max_gap = int(config["max_gap"])

    cached = []
    for row in rows:
        cache_path = detection_cache_path(args.cache_dir, row)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Validation detection cache not found: {cache_path}"
            )
        with np.load(cache_path, allow_pickle=False) as archive:
            data = {
                key: archive[key]
                for key in (
                    "frame_indices",
                    "boxes_xyxy",
                    "scores",
                    "track_ids",
                    "people_per_frame",
                )
            }
        cached.append((row, data))

    results = []
    for tau in args.taus:
        selected = []
        for row, data in cached:
            pair = select_clip_pair(
                data["frame_indices"],
                data["boxes_xyxy"],
                data["track_ids"],
                data["scores"],
                tau=tau,
                min_pair_frames=min_pair_frames,
                group_tau=tau * group_tau_multiplier,
            )
            selected.append((row, data, pair))
        for margin in args.margins:
            clip_rows = []
            for row, data, pair in selected:
                crops = build_crop_boxes(
                    len(data["people_per_frame"]),
                    data["frame_indices"],
                    data["boxes_xyxy"],
                    data["track_ids"],
                    pair.group_track_ids if pair else (),
                    frame_width=int(row["width"]),
                    frame_height=int(row["height"]),
                    margin=margin,
                    smoothing_window=smoothing_window,
                    max_gap=max_gap,
                )
                nonfallback = ~crops.fallback_mask
                width = int(row["width"])
                height = int(row["height"])
                crop = crops.boxes_xyxy
                area_fraction = (
                    (crop[:, 2] - crop[:, 0]) * (crop[:, 3] - crop[:, 1])
                ) / float(width * height)
                clip_rows.append(
                    {
                        "has_pair": pair is not None,
                        "fallback_rate": crops.fallback_rate,
                        "crop_area_fraction": float(area_fraction[nonfallback].mean())
                        if nonfallback.any()
                        else 1.0,
                        "group_size": len(pair.group_track_ids) if pair else 0,
                        "pair_support_fraction": pair.support_fraction if pair else 0.0,
                    }
                )
            frame = pd.DataFrame(clip_rows)
            results.append(
                {
                    "tau": tau,
                    "margin": margin,
                    "validation_clips": len(frame),
                    "clips_with_pair": int(frame["has_pair"].sum()),
                    "pair_coverage": float(frame["has_pair"].mean()),
                    "mean_fallback_rate": float(frame["fallback_rate"].mean()),
                    "median_fallback_rate": float(frame["fallback_rate"].median()),
                    "mean_nonfallback_crop_area_fraction": float(
                        frame["crop_area_fraction"].mean()
                    ),
                    "median_group_size": float(frame["group_size"].median()),
                    "mean_pair_support_fraction": float(
                        frame.loc[frame["has_pair"], "pair_support_fraction"].mean()
                    )
                    if frame["has_pair"].any()
                    else 0.0,
                }
            )

    output = pd.DataFrame(results).sort_values(["tau", "margin"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    report = args.output.with_suffix(".md")
    report.write_text(
        "# Interaction crop validation tuning\n\n"
        f"Source: `{args.manifest}` filtered strictly to `rwf2000 / validation` "
        f"({len(rows)} clips). No train, test, RLVS, or Hockey clips were used.\n\n"
        "The sweep reports pair coverage, full-frame fallback rate, and crop area as "
        "engineering proxies. The datasets have no person boxes or interaction-pair "
        "ground truth, so these metrics do not measure violence-recognition accuracy.\n\n"
        + output.to_string(index=False)
        + f"\n\nCSV: `{args.output.as_posix()}`\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(output)} tau/margin candidates over {len(rows)} validation clips to {args.output}"
    )
    print(f"Wrote report to {report}")


if __name__ == "__main__":
    main()
