"""Evaluate a saved X3D checkpoint on the held-out RWF test split."""

from __future__ import annotations

import argparse
import json
import os
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
from violence_detection.models.x3ds import LateFusionX3D, build_x3d
from violence_detection.utils.seed import seed_worker


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def _predict(model, loader, spec, augmentation, device, amp):
    predictions: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    losses: list[float] = []
    crop_statuses: list[np.ndarray] = []
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 3:
                videos, labels, crop_found = batch
                crop_statuses.append(crop_found.numpy())
            else:
                videos, labels = batch
            labels_device = labels.to(device, non_blocking=True)
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
                    if isinstance(unwrapped, LateFusionX3D):
                        logits = model(inputs)
                    else:
                        batch_size = inputs["full"].shape[0]
                        logits_all = model(torch.cat([inputs["full"], inputs["crop"]]))
                        logits = (logits_all[:batch_size] + logits_all[batch_size:]) / 2
                else:
                    logits = model(inputs)
                loss = criterion(logits, labels_device)
            probability = logits.softmax(dim=1)[:, 1]
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            probabilities.append(probability.float().cpu().numpy())
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
    }
    status = np.concatenate(crop_statuses) if crop_statuses else None
    return metrics, y_true, y_pred, y_probability, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/train_x3d.yaml"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--input-tag")
    parser.add_argument("--mode", choices=["full", "crop", "both"])
    parser.add_argument(
        "--include-crop-status",
        action="store_true",
        help="Require and export crop-found/fallback groups from the tensor cache",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/x3d_s_rwf2000/test")
    )
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU evaluation for a small smoke run",
    )
    args = parser.parse_args()

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
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
        manifest_path=data_config["manifest"],
        return_crop_status=args.include_crop_status,
    )
    batch_size = int(config["training"]["per_gpu_batch_size"]) * max(1, gpu_count)
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
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if torch.cuda.is_available():
        model.to(memory_format=torch.channels_last_3d)
        if gpu_count > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(gpu_count)))
    augmentation = yaml.safe_load(
        Path(data_config["preprocess_config"]).read_text(encoding="utf-8")
    ).get("augmentation", {})
    metrics, y_true, y_pred, y_probability, crop_status = _predict(
        model,
        loader,
        spec,
        augmentation,
        device,
        bool(config["training"]["amp"] and torch.cuda.is_available()),
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
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
                "label": int(y_true[index]),
                "label_name": "violent" if y_true[index] else "nonviolent",
                "prediction": int(y_pred[index]),
                "prediction_name": "violent" if y_pred[index] else "nonviolent",
                "violent_probability": float(y_probability[index]),
                "crop_found_frames": found_count,
                "sampled_frames": spec.num_frames if found_count is not None else None,
                "crop_status": status_name,
            }
        )
    pd.DataFrame(prediction_rows).to_csv(
        args.output_dir / "test_predictions.csv", index=False
    )
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
        title="RWF-2000 test confusion matrix",
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
        f"# X3D {input_label}: RWF-2000 test\n\n"
        "Positive class: violent (Fight, label 1). Precision, recall, and F1 are binary metrics for this class.\n\n"
        + metrics_table
        + "\n\nConfusion matrix (rows=actual, columns=predicted):\n\n"
        + matrix_table
        + "\n"
    )
    (args.output_dir / "test_results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
