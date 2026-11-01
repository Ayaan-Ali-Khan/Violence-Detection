"""Create deterministic validation-set overlays for interaction-crop review."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from violence_detection.crops import build_interaction_crops


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
        "--output-dir",
        type=Path,
        default=Path("cache/interaction_crops/validation_review"),
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=Path("data/interaction_crops/visual_review_notes.csv"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--margin", type=float)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    validation = manifest.loc[
        (manifest["dataset"] == "rwf2000") & (manifest["split"] == "validation")
    ]
    if validation.empty:
        raise ValueError(
            "The manifest has no RWF validation rows; visual review is validation-only."
        )
    sample_size = min(args.sample_size, len(validation))
    rng = random.Random(args.seed)
    selected_rows = rng.sample(validation.to_dict("records"), sample_size)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tau = float(args.tau if args.tau is not None else config["tau"])
    margin = float(args.margin if args.margin is not None else config["margin"])
    group_tau = tau * float(config["group_tau_multiplier"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    notes = []

    for clip_number, row in enumerate(selected_rows, start=1):
        cache_path = detection_cache_path(args.cache_dir, row)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Validation detection cache not found: {cache_path}"
            )
        with np.load(cache_path, allow_pickle=False) as archive:
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
            min_pair_frames=int(config["min_pair_frames"]),
            group_tau=group_tau,
            margin=margin,
            smoothing_window=int(config["smoothing_window"]),
            max_gap=int(config["max_gap"]),
        )
        video_path = args.project_root / str(row["relative_path"])
        if not video_path.is_file():
            raise FileNotFoundError(f"Validation video not found: {video_path}")
        sample_indices = sorted(
            set([0, max(0, frame_count // 2), max(0, frame_count - 1)])
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"Could not open validation video: {video_path}")
        frame_panels = []
        try:
            for frame_index in sample_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame_tracks = detections["frame_indices"] == frame_index
                selected_ids = (
                    set(result.selection.group_track_ids) if result.selection else set()
                )
                for box, track_id in zip(
                    detections["boxes_xyxy"][frame_tracks],
                    detections["track_ids"][frame_tracks],
                    strict=True,
                ):
                    x1, y1, x2, y2 = map(int, box)
                    color = (
                        (40, 210, 40)
                        if int(track_id) in selected_ids
                        else (230, 170, 30)
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID {int(track_id)}",
                        (x1, max(16, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                x1, y1, x2, y2 = result.crops.boxes_xyxy[frame_index]
                color = (
                    (0, 0, 255)
                    if result.crops.fallback_mask[frame_index]
                    else (255, 40, 220)
                )
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                title = f"Frame {frame_index} | fallback={bool(result.crops.fallback_mask[frame_index])}"
                cv2.putText(
                    frame,
                    title,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                panel = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
                frame_panels.append(panel)
        finally:
            capture.release()
        while len(frame_panels) < 3:
            frame_panels.append(np.zeros((270, 480, 3), dtype=np.uint8))
        pair_ids = result.selection.pair_track_ids if result.selection else ()
        group_ids = result.selection.group_track_ids if result.selection else ()
        header = np.zeros((58, 1440, 3), dtype=np.uint8)
        description = (
            f"{clip_number:02d}/{sample_size} {Path(str(row['relative_path'])).name} "
            f"label={row['label_name']} pair={pair_ids} group={group_ids} "
            f"fallback={result.crops.fallback_rate:.1%}"
        )
        cv2.putText(
            header,
            description[:160],
            (12, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        canvas = np.vstack([header, np.hstack(frame_panels)])
        output_path = (
            args.output_dir
            / f"{clip_number:02d}_{Path(str(row['relative_path'])).stem}.jpg"
        )
        cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        notes.append(
            {
                "clip_number": clip_number,
                "dataset": row["dataset"],
                "split": row["split"],
                "relative_path": row["relative_path"],
                "label_name": row["label_name"],
                "pair_track_ids": ";".join(map(str, pair_ids)),
                "group_track_ids": ";".join(map(str, group_ids)),
                "fallback_rate": result.crops.fallback_rate,
                "overlay_path": output_path.as_posix(),
                "review_status": "pending_manual_review",
                "review_notes": "Inspect pair identity, group coverage, and crop boundary across the three sampled frames.",
            }
        )

    args.notes.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(notes).to_csv(args.notes, index=False)
    print(f"Wrote {sample_size} validation overlay images to {args.output_dir}")
    print(f"Wrote review checklist to {args.notes}")


if __name__ == "__main__":
    main()
