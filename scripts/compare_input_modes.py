"""Compare full-frame, crop-only, and late-fusion RWF test predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _metric_row(
    frame: pd.DataFrame, input_name: str, group_name: str
) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    predictions = frame["prediction"].to_numpy(dtype=np.int64)
    probabilities = frame["violent_probability"].to_numpy(dtype=np.float64)
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = float("nan")
    return {
        "input": input_name,
        "crop_status_group": group_name,
        "num_clips": len(frame),
        "accuracy": float(accuracy_score(labels, predictions))
        if len(frame)
        else np.nan,
        "precision_violent": float(
            precision_score(labels, predictions, zero_division=0)
        )
        if len(frame)
        else np.nan,
        "recall_violent": float(recall_score(labels, predictions, zero_division=0))
        if len(frame)
        else np.nan,
        "f1_violent": float(f1_score(labels, predictions, zero_division=0))
        if len(frame)
        else np.nan,
        "roc_auc": auc,
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "input",
        "crop_status_group",
        "num_clips",
        "accuracy",
        "precision_violent",
        "recall_violent",
        "f1_violent",
        "roc_auc",
    ]
    display = frame[columns].copy()
    for column in columns[3:]:
        display[column] = display[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:.4f}"
        )
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-predictions", type=Path, required=True)
    parser.add_argument("--crop-predictions", type=Path, required=True)
    parser.add_argument("--fusion-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    predictions = {
        name: pd.read_csv(path)
        for name, path in (
            ("full_frame", args.full_predictions),
            ("crop_only", args.crop_predictions),
            ("late_fusion", args.fusion_predictions),
        )
    }
    required = {
        "relative_path",
        "label",
        "prediction",
        "violent_probability",
        "crop_status",
    }
    reference_paths = None
    reference_labels = None
    for name, frame in predictions.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions lack columns: {sorted(missing)}")
        if frame["relative_path"].duplicated().any():
            raise ValueError(f"{name} predictions contain duplicate video paths")
        paths = set(frame["relative_path"])
        labels = frame.set_index("relative_path")["label"].sort_index()
        if reference_paths is None:
            reference_paths = paths
            reference_labels = labels
        elif paths != reference_paths or not labels.equals(reference_labels):
            raise ValueError(f"{name} predictions do not align with the same test set")

    status_by_path = predictions["late_fusion"].set_index("relative_path")[
        "crop_status"
    ]
    for name, frame in predictions.items():
        current_status = frame.set_index("relative_path")["crop_status"]
        if not current_status.sort_index().equals(status_by_path.sort_index()):
            raise ValueError(f"{name} predictions have inconsistent crop statuses")
    allowed_statuses = {"crop_found", "mixed", "fallback"}
    if not set(status_by_path.unique()).issubset(allowed_statuses):
        raise ValueError("Unexpected crop_status values in prediction files")

    groups = ["overall", "crop_found", "mixed", "fallback"]
    rows: list[dict[str, Any]] = []
    for input_name, frame in predictions.items():
        joined = frame.set_index("relative_path").join(
            status_by_path.rename("shared_crop_status")
        )
        for group in groups:
            subset = (
                joined
                if group == "overall"
                else joined.loc[joined["shared_crop_status"] == group]
            )
            rows.append(_metric_row(subset.reset_index(), input_name, group))
    results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "input_comparison.csv", index=False)
    status_counts = status_by_path.value_counts().reindex(
        ["crop_found", "mixed", "fallback"], fill_value=0
    )
    status_counts.rename_axis("crop_status_group").rename("num_clips").to_csv(
        args.output_dir / "crop_status_counts.csv"
    )
    report = (
        "# RWF-2000 test input comparison\n\n"
        "All three inputs use the same 400 held-out clips and the selected "
        "validation checkpoints. `crop_found` means all 16 deterministic model "
        "frames have a non-fallback interaction crop; `fallback` means all 16 "
        "use full-frame fallback boxes; `mixed` means both occur in the clip. "
        "Precision, recall, and F1 use violent (label 1) as the positive class.\n\n"
        + _markdown_table(results)
        + "\n\nClip counts by crop availability:\n\n"
        + "| crop_status_group | num_clips |\n| --- | ---: |\n"
        + "\n".join(f"| {group} | {count} |" for group, count in status_counts.items())
        + "\n\nMachine-readable metrics: `input_comparison.csv`.\n"
    )
    (args.output_dir / "input_comparison.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
