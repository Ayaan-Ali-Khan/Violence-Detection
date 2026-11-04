"""Recompute saved RWF test decisions at a validation-locked threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    frame = pd.read_csv(args.predictions)
    required = {"label", "violent_probability", "relative_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    labels = frame["label"].to_numpy(dtype=np.int64)
    probabilities = frame["violent_probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Violent probabilities must be finite and within [0, 1]")
    predictions = (probabilities >= args.threshold).astype(np.int64)
    frame["dataset"] = "rwf2000"
    frame["split"] = "test"
    frame["prediction"] = predictions
    frame["prediction_name"] = np.where(predictions == 1, "violent", "nonviolent")
    frame["label_name"] = np.where(labels == 1, "violent", "nonviolent")
    frame["decision_threshold_violent"] = float(args.threshold)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = float("nan")
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_violent": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall_violent": float(recall_score(labels, predictions, zero_division=0)),
        "f1_violent": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": auc,
        "positive_class": "violent (Fight, label=1)",
        "decision_threshold_violent": float(args.threshold),
        "num_clips": int(len(frame)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        matrix,
        index=["actual_nonviolent", "actual_violent"],
        columns=["pred_nonviolent", "pred_violent"],
    ).to_csv(args.output_dir / "confusion_matrix.csv")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
