"""Evaluate a saved video-classifier checkpoint on held-out RWF test clips."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from violence_detection.data.augment import prepare_video_batch
from violence_detection.data.dataset import CachedVideoDataset
from violence_detection.data.preprocess import load_preprocess_spec
from violence_detection.models.x3ds import (
    LateFusionX3D,
    Swin3DViewAveraging,
    build_x3d,
)
from violence_detection.utils.seed import seed_worker


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def _predict(
    model, loader, spec, augmentation, device, amp, warmup_batches, threshold
):
    predictions: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    losses: list[float] = []
    crop_statuses: list[np.ndarray] = []
    timed_clips = 0
    timed_batches = 0
    inference_seconds = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if len(batch) == 3:
                videos, labels, crop_found = batch
                crop_statuses.append(crop_found.numpy())
            else:
                videos, labels = batch
            labels_device = labels.to(device, non_blocking=True)
            if batch_index >= warmup_batches and device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            inputs = prepare_video_batch(
                videos,
                spec,
                training=False,
                augmentation=augmentation,
                device=device,
            )
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                if isinstance(inputs, dict):
                    unwrapped = model.module if hasattr(model, "module") else model
                    if isinstance(unwrapped, (LateFusionX3D, Swin3DViewAveraging)):
                        logits = model(inputs)
                    else:
                        batch_size = inputs["full"].shape[0]
                        logits_all = model(torch.cat([inputs["full"], inputs["crop"]]))
                        logits = (logits_all[:batch_size] + logits_all[batch_size:]) / 2
                else:
                    logits = model(inputs)
            if batch_index >= warmup_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - started
                timed_clips += int(labels.size(0))
                timed_batches += 1
            loss = criterion(logits, labels_device)
            probability = logits.softmax(dim=1)[:, 1]
            probability_numpy = probability.float().cpu().numpy()
            probabilities.append(probability_numpy)
            predictions.append((probability_numpy >= threshold).astype(np.int64))
            labels_all.append(labels.cpu().numpy())
            losses.append(float(loss.item()) * labels.size(0))
    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(predictions)
    y_probability = np.concatenate(probabilities)
    try:
        auc = float(roc_auc_score(y_true, y_probability))
    except ValueError:
        auc = float("nan")
    metrics = {
        "loss": float(sum(losses) / max(1, len(y_true))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_violent": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_violent": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_violent": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "positive_class": "violent (Fight, label=1)",
        "num_clips": int(len(y_true)),
        "decision_threshold_violent": float(threshold),
    }
    status = np.concatenate(crop_statuses) if crop_statuses else None
    timing = {
        "warmup_batches": int(warmup_batches),
        "timed_batches": timed_batches,
        "timed_clips": timed_clips,
        "elapsed_seconds": inference_seconds,
        "clips_per_second": timed_clips / inference_seconds
        if inference_seconds > 0
        else float("nan"),
        "milliseconds_per_clip": inference_seconds * 1000 / timed_clips
        if timed_clips > 0
        else float("nan"),
        "batch_size": int(loader.batch_size or 1),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
    return metrics, y_true, y_pred, y_probability, status, timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/train_x3d.yaml"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Override the manifest in the model config for cross-dataset evaluation.",
    )
    parser.add_argument("--input-tag")
    parser.add_argument("--mode", choices=["full", "crop", "both"])
    parser.add_argument(
        "--include-crop-status",
        action="store_true",
        help="Require and export crop-found/fallback groups from the tensor cache",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed violent-class probability threshold selected from validation.",
    )
    parser.add_argument(
        "--device",
        help="Evaluation device, such as cuda:0 or cuda:1; defaults to the first GPU.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU evaluation for a small smoke run",
    )
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = config["model"]
    data_config = config["data"]
    variant = str(checkpoint.get("model_variant", model_config["variant"]))
    mode = args.mode or str(checkpoint.get("mode", data_config.get("mode", "full")))
    input_tag = args.input_tag or str(
        checkpoint.get("input_tag", data_config["input_tag"])
    )
    cache_root = args.cache_dir or Path(data_config["cache_dir"])

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Test evaluation requires CUDA. Run the Kaggle GPU runner.")
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device or default_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise ValueError(
            f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
        )
    torch.set_num_threads(int(config["training"]["cpu_threads_per_rank"]))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    spec = load_preprocess_spec(data_config["preprocess_config"], variant)
    dataset = CachedVideoDataset(
        cache_root,
        input_tag,
        args.split,
        model_name=variant,
        preprocess_config=data_config["preprocess_config"],
        mode=mode,
        training=False,
        manifest_path=args.manifest or data_config["manifest"],
        return_crop_status=args.include_crop_status,
    )
    batch_size = args.batch_size or int(config["training"]["per_gpu_batch_size"])
    if batch_size < 1 or args.warmup_batches < 0:
        raise ValueError("batch size must be positive and warmup batches non-negative")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"]["workers_per_rank"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        worker_init_fn=seed_worker,
        prefetch_factor=int(config["training"]["prefetch_factor"])
        if int(config["training"]["workers_per_rank"]) > 0
        else None,
    )

    architecture = str(checkpoint.get("architecture", "x3d"))
    if architecture == "late_fusion":
        model = LateFusionX3D(variant, int(model_config.get("num_classes", 2)))
    elif architecture == "x3d":
        model = build_x3d(variant, int(model_config.get("num_classes", 2)))
    elif architecture == "swin3d_t":
        model = Swin3DViewAveraging(int(model_config.get("num_classes", 2)))
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if torch.cuda.is_available():
        model.to(memory_format=torch.channels_last_3d)
    augmentation = yaml.safe_load(
        Path(data_config["preprocess_config"]).read_text(encoding="utf-8")
    ).get("augmentation", {})
    metrics, y_true, y_pred, y_probability, crop_status, timing = _predict(
        model,
        loader,
        spec,
        augmentation,
        device,
        bool(config["training"]["amp"] and torch.cuda.is_available()),
        args.warmup_batches,
        args.threshold,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    dataset_name = (
        str(dataset.record(0).get("dataset", "dataset"))
        if len(dataset)
        else "dataset"
    )
    if args.output_dir is None:
        model_name = variant if architecture == variant else f"{variant}_{architecture}"
        args.output_dir = (
            Path("experiments") / "evaluation" / f"{dataset_name}_{args.split}" / model_name
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    if args.split == "test":
        (args.output_dir / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
        )
    timing["model_architecture"] = architecture
    timing["model_variant"] = variant
    (args.output_dir / "inference_speed.json").write_text(
        json.dumps(timing, indent=2, allow_nan=False), encoding="utf-8"
    )
    prediction_rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        record = dataset.record(index)
        found_count = int(crop_status[index].sum()) if crop_status is not None else None
        status_name = "unavailable"
        if found_count is not None:
            status_name = (
                "crop_found"
                if found_count == spec.num_frames
                else "fallback"
                if found_count == 0
                else "mixed"
            )
        prediction_rows.append(
            {
                "relative_path": record.get("relative_path", index),
                "dataset": record.get("dataset", "unknown"),
                "split": record.get("split", args.split),
                "label": int(y_true[index]),
                "label_name": "violent" if y_true[index] else "nonviolent",
                "prediction": int(y_pred[index]),
                "prediction_name": "violent" if y_pred[index] else "nonviolent",
                "violent_probability": float(y_probability[index]),
                "decision_threshold_violent": float(args.threshold),
                "crop_found_frames": found_count,
                "sampled_frames": spec.num_frames if found_count is not None else None,
                "crop_status": status_name,
            }
        )
    predictions_frame = pd.DataFrame(prediction_rows)
    predictions_frame.to_csv(args.output_dir / "predictions.csv", index=False)
    if args.split == "test":
        predictions_frame.to_csv(args.output_dir / "test_predictions.csv", index=False)
    matrix_frame = pd.DataFrame(
        matrix,
        index=["actual_nonviolent", "actual_violent"],
        columns=["pred_nonviolent", "pred_violent"],
    )
    matrix_frame.to_csv(args.output_dir / "confusion_matrix.csv")
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Nonviolent", "Violent"],
        yticklabels=["Nonviolent", "Violent"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"{dataset_name} {args.split} confusion matrix",
    )
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    metrics_table = _markdown_table(
        list(metrics),
        [[str(value) for value in metrics.values()]],
    )
    matrix_table = _markdown_table(
        ["actual \\ predicted", "Nonviolent", "Violent"],
        [
            ["Nonviolent", str(matrix[0, 0]), str(matrix[0, 1])],
            ["Violent", str(matrix[1, 0]), str(matrix[1, 1])],
        ],
    )
    input_label = {
        "full": "full-frame",
        "crop": "interaction-crop",
        "both": "learned late fusion"
        if architecture == "late_fusion"
        else "shared-model paired views",
    }[mode]
    report = (
        f"# {architecture} {input_label}: {dataset_name} {args.split}\n\n"
        "Positive class: violent (Fight, label 1). Precision, recall, and F1 are binary metrics for this class.\n\n"
        + metrics_table
        + "\n\nInference speed (single visible GPU, warmup excluded): "
        + f"{timing['clips_per_second']:.3f} clips/s, "
        + f"{timing['milliseconds_per_clip']:.2f} ms/clip, "
        + f"batch size {timing['batch_size']} on {timing['gpu']}.\n"
        + "\n\nConfusion matrix (rows=actual, columns=predicted):\n\n"
        + matrix_table
        + "\n"
    )
    (args.output_dir / f"{dataset_name}_{args.split}_results.md").write_text(
        report, encoding="utf-8"
    )
    if args.split == "test":
        (args.output_dir / "test_results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
