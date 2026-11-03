"""Compare Phase 5/6 classifiers on the same held-out clips and inference setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_result(name: str, root: Path) -> dict[str, object]:
    metrics_path = root / "test_metrics.json"
    speed_path = root / "inference_speed.json"
    predictions_path = root / "test_predictions.csv"
    for path in (metrics_path, speed_path, predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing evaluation output: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    speed = json.loads(speed_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    if predictions["relative_path"].duplicated().any():
        raise ValueError(f"{name} has duplicate test video paths")
    return {
        "model": name,
        "accuracy": float(metrics["accuracy"]),
        "precision_violent": float(metrics["precision_violent"]),
        "recall_violent": float(metrics["recall_violent"]),
        "f1_violent": float(metrics["f1_violent"]),
        "roc_auc": float(metrics["roc_auc"]),
        "clips_per_second": float(speed["clips_per_second"]),
        "milliseconds_per_clip": float(speed["milliseconds_per_clip"]),
        "batch_size": int(speed["batch_size"]),
        "gpu": str(speed["gpu"]),
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Evaluation result directory; repeat for each model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--f1-tolerance", type=float, default=0.005)
    args = parser.parse_args()
    if args.f1_tolerance < 0:
        raise ValueError("--f1-tolerance must be non-negative")

    parsed: list[tuple[str, Path]] = []
    for item in args.result:
        if "=" not in item:
            raise ValueError(f"Expected NAME=DIR, got: {item!r}")
        name, directory = item.split("=", maxsplit=1)
        if not name or not directory:
            raise ValueError(f"Expected NAME=DIR, got: {item!r}")
        parsed.append((name, Path(directory)))
    if len({name for name, _ in parsed}) != len(parsed):
        raise ValueError("Model names must be unique")

    results = [_read_result(name, directory) for name, directory in parsed]
    reference = results[0]["predictions"]
    reference = reference.set_index("relative_path")["label"].sort_index()
    for result in results[1:]:
        current = result["predictions"].set_index("relative_path")["label"].sort_index()
        if not current.equals(reference):
            raise ValueError(
                f"{result['model']} predictions do not align with the same test set"
            )
    batch_sizes = {int(result["batch_size"]) for result in results}
    gpu_names = {str(result["gpu"]) for result in results}
    if len(batch_sizes) != 1 or len(gpu_names) != 1:
        raise ValueError(
            "Inference timing must use the same batch size and GPU for every model"
        )

    rows = [
        {key: value for key, value in result.items() if key != "predictions"}
        for result in results
    ]
    frame = pd.DataFrame(rows).sort_values(
        ["f1_violent", "accuracy", "clips_per_second"],
        ascending=[False, False, False],
    )
    best_f1 = float(frame["f1_violent"].max())
    eligible = frame.loc[frame["f1_violent"] >= best_f1 - args.f1_tolerance]
    selected = eligible.sort_values(
        ["clips_per_second", "f1_violent", "accuracy"],
        ascending=[False, False, False],
    ).iloc[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "model_comparison.csv", index=False)
    decision = {
        "recommended_model": str(selected["model"]),
        "rule": (
            f"Keep models within {args.f1_tolerance:.3f} absolute violent-class F1 "
            "of the highest F1, then select the fastest measured inference model."
        ),
        "best_f1": best_f1,
        "f1_tolerance": args.f1_tolerance,
        "selected_f1": float(selected["f1_violent"]),
        "selected_clips_per_second": float(selected["clips_per_second"]),
        "test_clips": int(len(reference)),
        "timing_setup": (
            f"single GPU, batch size {int(selected['batch_size'])}, "
            "warmup batches excluded"
        ),
    }
    (args.output_dir / "model_recommendation.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    visible = frame.drop(columns=["gpu"]).copy()
    for column in (
        "accuracy",
        "precision_violent",
        "recall_violent",
        "f1_violent",
        "roc_auc",
        "clips_per_second",
        "milliseconds_per_clip",
    ):
        visible[column] = visible[column].map(lambda value: f"{value:.4f}")
    columns = list(visible.columns)
    table_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    table_lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in visible.itertuples(index=False, name=None)
    )
    markdown = (
        "# Phase 6 model comparison\n\n"
        "All models were evaluated on the same held-out RWF-2000 test clips. "
        "Inference timing uses one visible GPU, batch size 1, and excludes warmup.\n\n"
        + "\n".join(table_lines)
        + "\n\n## Selection\n\n"
        + f"Recommended classifier: **{decision['recommended_model']}**. "
        + decision["rule"]
        + f"\n\nTest clips: {decision['test_clips']}."
        + "\n"
    )
    (args.output_dir / "model_comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
