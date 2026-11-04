"""Summarize cross-dataset performance drops and export FP/FN review lists."""

from __future__ import annotations

import argparse
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


def _parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=DIR, got {value!r}")
    name, path_value = value.split("=", maxsplit=1)
    if not name or not path_value:
        raise ValueError(f"Expected NAME=DIR, got {value!r}")
    return name, Path(path_value)


def _prediction_path(directory: Path) -> Path:
    for name in ("predictions.csv", "test_predictions.csv"):
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No predictions.csv in {directory}")


def _markdown_table(frame: pd.DataFrame, *, float_format: str = ".4f") -> str:
    headers = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                formatted.append(format(value, float_format))
            else:
                formatted.append(str(value))
        rows.append(formatted)
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def _load_predictions(name: str, directory: Path, reference: bool) -> pd.DataFrame:
    frame = pd.read_csv(_prediction_path(directory))
    required = {"label", "prediction", "violent_probability", "relative_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{directory} predictions are missing {sorted(missing)}")
    if "dataset" not in frame:
        if not reference:
            raise ValueError(f"{directory} predictions do not identify their dataset")
        frame["dataset"] = "rwf2000"
    if "split" not in frame:
        frame["split"] = "test" if reference else "evaluation"
    if frame["dataset"].nunique() != 1:
        raise ValueError(f"{directory} must contain exactly one dataset")
    if frame["relative_path"].duplicated().any():
        raise ValueError(f"{directory} contains duplicate video paths")
    frame = frame.copy()
    frame["model"] = name
    frame["is_reference"] = reference
    frame["label"] = frame["label"].astype(int)
    frame["prediction"] = frame["prediction"].astype(int)
    frame["violent_probability"] = frame["violent_probability"].astype(float)
    return frame


def _metrics(frame: pd.DataFrame) -> dict[str, float | int | str]:
    labels = frame["label"].to_numpy(dtype=int)
    predicted = frame["prediction"].to_numpy(dtype=int)
    probabilities = frame["violent_probability"].to_numpy(dtype=float)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = float("nan")
    return {
        "clips": int(len(frame)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision_violent": float(
            precision_score(labels, predicted, zero_division=0)
        ),
        "recall_violent": float(recall_score(labels, predicted, zero_division=0)),
        "f1_violent": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": auc,
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "false_alarm_rate_nonviolent": float(
            matrix[0, 1] / max(1, matrix[0].sum())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="MODEL=DIR",
        help="Cross-dataset evaluation directory; repeat for each model and dataset.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="MODEL=DIR",
        help="Matching RWF-2000 test result directory; repeat for each model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.result:
        parser.error("At least one --result is required")

    frames = []
    for value in args.reference:
        name, directory = _parse_mapping(value)
        frame = _load_predictions(name, directory, reference=True)
        if set(frame["dataset"].astype(str)) != {"rwf2000"}:
            raise ValueError(f"Reference {directory} must be the RWF-2000 test set")
        frames.append(frame)
    for value in args.result:
        name, directory = _parse_mapping(value)
        frame = _load_predictions(name, directory, reference=False)
        if "test" in set(frame["split"].astype(str)):
            raise ValueError(f"Cross-dataset result {directory} cannot be the RWF test split")
        frames.append(frame)

    references = {
        str(frame.iloc[0]["model"]): frame
        for frame in frames
        if bool(frame.iloc[0]["is_reference"])
    }
    for frame in frames:
        if bool(frame.iloc[0]["is_reference"]):
            continue
        reference = references.get(str(frame.iloc[0]["model"]))
        if reference is None:
            continue
        if (
            "decision_threshold_violent" in frame
            and "decision_threshold_violent" in reference
        ):
            current = frame["decision_threshold_violent"].to_numpy(dtype=float)
            reference_threshold = reference[
                "decision_threshold_violent"
            ].to_numpy(dtype=float)
            if not np.allclose(current, current[0]) or not np.allclose(
                reference_threshold, current[0]
            ):
                raise ValueError(
                    "Cross-dataset and RWF reference decisions must use the same "
                    f"locked threshold for {frame.iloc[0]['model']}"
                )
    rows = []
    false_positives = []
    false_negatives = []
    for frame in frames:
        dataset = str(frame.iloc[0]["dataset"])
        split = str(frame.iloc[0]["split"])
        values = _metrics(frame)
        row = {"model": str(frame.iloc[0]["model"]), "dataset": dataset, "split": split, **values}
        reference = references.get(row["model"])
        if reference is not None and dataset != "rwf2000":
            reference_metrics = _metrics(reference)
            row["accuracy_drop_vs_rwf"] = reference_metrics["accuracy"] - values["accuracy"]
            row["f1_drop_vs_rwf"] = reference_metrics["f1_violent"] - values["f1_violent"]
            row["roc_auc_drop_vs_rwf"] = reference_metrics["roc_auc"] - values["roc_auc"]
        else:
            row["accuracy_drop_vs_rwf"] = np.nan
            row["f1_drop_vs_rwf"] = np.nan
            row["roc_auc_drop_vs_rwf"] = np.nan
        rows.append(row)

        errors = frame.loc[frame["label"] != frame["prediction"]].copy()
        errors["error_type"] = np.where(
            errors["label"] == 0, "false_positive", "false_negative"
        )
        false_positives.append(errors.loc[errors["label"] == 0])
        false_negatives.append(errors.loc[errors["label"] == 1])

    summary = pd.DataFrame(rows).sort_values(["model", "dataset"])
    fp_frame = pd.concat(false_positives, ignore_index=True)
    fn_frame = pd.concat(false_negatives, ignore_index=True)
    hockey_fp = fp_frame.loc[fp_frame["dataset"] == "hockeyfights"].sort_values(
        "violent_probability", ascending=False
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "cross_dataset_metrics.csv", index=False)
    fp_frame.sort_values(
        ["model", "dataset", "violent_probability"],
        ascending=[True, True, False],
    ).to_csv(args.output_dir / "false_positives.csv", index=False)
    fn_frame.sort_values(
        ["model", "dataset", "violent_probability"],
        ascending=[True, True, True],
    ).to_csv(args.output_dir / "false_negatives.csv", index=False)
    hockey_fp.to_csv(args.output_dir / "hockey_sports_false_alarms.csv", index=False)

    lines = [
        "# Cross-dataset error analysis",
        "",
        "Each evaluation uses the model-specific threshold selected on the RWF-2000 validation split.",
        "Accuracy/F1/AUC drops are computed as RWF-2000 official test score minus the cross-dataset score; positive values are drops.",
        "",
        _markdown_table(summary),
        "",
        "## Hockey non-fight false alarms",
        "",
        "Every false positive in Hockey Fights is a non-fight sports clip classified as violent. Review the exported rows against the videos to determine whether fast skating, checks, pileups, or camera motion triggered the alarm; the manifest does not annotate specific motion types.",
        "",
    ]
    if hockey_fp.empty:
        lines.append("No Hockey Fights false positives were found.")
    else:
        lines.append(
            hockey_fp[["model", "relative_path", "violent_probability"]]
            .head(50)
            .pipe(_markdown_table)
        )
    lines.extend(
        [
            "",
            "## Misclassified video files",
            "",
            f"False positives: `{(args.output_dir / 'false_positives.csv').name}`",
            f"False negatives: `{(args.output_dir / 'false_negatives.csv').name}`",
            f"Hockey non-fight false alarms: `{(args.output_dir / 'hockey_sports_false_alarms.csv').name}`",
            "",
        ]
    )
    report_path = args.output_dir / "cross_dataset_error_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
