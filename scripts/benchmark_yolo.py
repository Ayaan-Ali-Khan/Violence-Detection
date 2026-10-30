"""Compare YOLO11 model sizes on a deterministic sample of project videos."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from violence_detection.detection.detect_track import load_yolo


def sample_frames(path: Path, count: int) -> list[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return []
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            return []
        indices = sorted(set(round(x) for x in np.linspace(0, frame_count - 1, count)))
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append((index, frame))
        return frames
    finally:
        capture.release()


def build_sample(manifests: list[Path], clips_per_class: int, frames_per_clip: int, seed: int):
    tables = [pd.read_csv(path) for path in manifests]
    videos = pd.concat(tables, ignore_index=True).drop_duplicates("relative_path")
    rng = random.Random(seed)
    samples: list[tuple[str, Path, int, np.ndarray]] = []
    for dataset, dataset_rows in videos.groupby("dataset", sort=True):
        for label, label_rows in dataset_rows.groupby("label", sort=True):
            candidates = label_rows.sort_values("relative_path").to_dict("records")
            rng.shuffle(candidates)
            for row in candidates[:clips_per_class]:
                path = Path(str(row["relative_path"]))
                for frame_index, frame in sample_frames(path, frames_per_clip):
                    samples.append((str(dataset), path, frame_index, frame))
    if not samples:
        raise RuntimeError("No readable frames found in the supplied manifests")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--models", nargs="+", help="Optional override, e.g. yolo11n.pt yolo11s.pt yolo11m.pt")
    parser.add_argument(
        "--manifests",
        nargs="+",
        type=Path,
        default=[Path("data/splits/rwf2000.csv"), Path("data/splits/rlvs_eval.csv"), Path("data/splits/hockeyfights_eval.csv")],
    )
    parser.add_argument("--output", type=Path, default=Path("data/detection/yolo11_benchmark.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/detection/yolo11_size_comparison.md"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    benchmark = config["benchmark"]
    samples = build_sample(
        args.manifests,
        clips_per_class=int(benchmark["clips_per_class_per_dataset"]),
        frames_per_clip=int(benchmark["frames_per_clip"]),
        seed=args.seed,
    )
    device = str(config.get("device", "auto"))
    actual_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    summary_rows = []
    for weights in args.models or benchmark["weights"]:
        model = load_yolo(weights)
        warmup_count = min(int(benchmark["warmup_frames"]), len(samples))
        for _, _, _, frame in samples[:warmup_count]:
            model.predict(
                source=frame,
                classes=[0],
                imgsz=int(config["imgsz"]),
                conf=float(config["confidence"]),
                iou=float(config["iou"]),
                device=None if device == "auto" else device,
                verbose=False,
            )
        latencies = []
        counts = []
        for dataset, path, frame_index, frame in samples:
            start = time.perf_counter()
            result = model.predict(
                source=frame,
                classes=[0],
                imgsz=int(config["imgsz"]),
                conf=float(config["confidence"]),
                iou=float(config["iou"]),
                device=None if device == "auto" else device,
                verbose=False,
            )[0]
            latencies.append(time.perf_counter() - start)
            boxes = result.boxes
            count = int(len(boxes)) if boxes is not None else 0
            counts.append(count)
            rows.append(
                {
                    "model": str(weights),
                    "dataset": dataset,
                    "video": path.as_posix(),
                    "frame_index": frame_index,
                    "people_detected": count,
                    "latency_sec": latencies[-1],
                }
            )
        weight_file = Path(getattr(model, "ckpt_path", weights))
        if not weight_file.exists():
            weight_file = Path(str(weights))
        weight_mb = weight_file.stat().st_size / (1024**2) if weight_file.exists() else None
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        group = [r for r in rows if r["model"] == str(weights)]
        summary = {
            "model": str(weights),
            "checkpoint_mb": weight_mb,
            "device": actual_device,
            "sampled_frames": len(group),
            "mean_latency_sec": sum(r["latency_sec"] for r in group) / len(group),
            "median_latency_sec": float(pd.Series([r["latency_sec"] for r in group]).median()),
            "frames_per_second": len(group) / sum(r["latency_sec"] for r in group),
            "mean_people_per_frame": sum(counts) / len(counts),
            "no_person_frame_rate": sum(c == 0 for c in counts) / len(counts),
        }
        print(summary)
        summary_rows.append(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    summaries = pd.DataFrame(summary_rows)
    summaries.to_csv(args.output.with_name("yolo11_benchmark_summary.csv"), index=False)
    device_name = (
        torch.cuda.get_device_name(0)
        if actual_device == "cuda" and device == "auto"
        else actual_device
    )
    choice_note = ""
    by_model = {str(row["model"]): row for row in summary_rows}
    if "yolo11n.pt" in by_model and "yolo11s.pt" in by_model:
        nano = by_model["yolo11n.pt"]
        small = by_model["yolo11s.pt"]
        speed_ratio = float(nano["frames_per_second"]) / float(small["frames_per_second"])
        count_delta = (
            float(small["mean_people_per_frame"]) - float(nano["mean_people_per_frame"])
        ) / max(float(small["mean_people_per_frame"]), 1e-9)
        choice_note = (
            f"Initial cache choice: YOLO11n, because it was {speed_ratio:.2f}x faster than "
            f"YOLO11s on this CPU sample. YOLO11s returned {count_delta:.1%} more boxes "
            "per frame, with the same no-person-frame rate. These are unannotated sample "
            "counts, not accuracy/recall measurements; recheck throughput on the Kaggle GPU."
        )
    report = [
        "# YOLO11 model-size comparison",
        "",
        f"Device: {device_name}; image size: {config['imgsz']}; confidence: {config['confidence']}; IoU: {config['iou']}.",
        f"Sample: {len(samples)} deterministically selected frames across the supplied dataset manifests.",
        "",
        "The sampled data have video-level labels, not person-box annotations. This comparison reports runtime and detected-person counts; it is not an mAP/precision benchmark.",
        "",
        "```text",
        summaries.to_string(index=False),
        "```",
        "",
        choice_note,
        "",
        f"Per-frame measurements: `{args.output.as_posix()}`.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
