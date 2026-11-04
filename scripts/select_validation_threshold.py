"""Select a binary decision threshold using only the RWF validation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="rwf2000")
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    required = {"dataset", "split", "label", "violent_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    if set(frame["dataset"].astype(str)) != {args.dataset}:
        raise ValueError("Threshold calibration must use only the requested dataset")
    if set(frame["split"].astype(str)) != {args.split}:
        raise ValueError("Threshold calibration must use only the requested split")
    if frame.empty or frame["label"].nunique() != 2:
        raise ValueError("Validation predictions must contain both binary classes")

    labels = frame["label"].to_numpy(dtype=np.int64)
    probabilities = frame["violent_probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Violent probabilities must be finite and within [0, 1]")

    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scored = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        scored.append(
            (
                float(f1_score(labels, predictions, zero_division=0)),
                -abs(float(threshold) - 0.5),
                float(threshold),
                float(precision_score(labels, predictions, zero_division=0)),
                float(recall_score(labels, predictions, zero_division=0)),
            )
        )
    best = max(scored)
    fixed_default_predictions = probabilities >= 0.5
    result = {
        "threshold": best[2],
        "calibration_dataset": args.dataset,
        "calibration_split": args.split,
        "selection_metric": "violent-class F1",
        "tie_break": "closest threshold to 0.5",
        "num_clips": int(len(labels)),
        "validation_f1_at_threshold": best[0],
        "validation_precision_at_threshold": best[3],
        "validation_recall_at_threshold": best[4],
        "validation_f1_at_0_5": float(
            f1_score(labels, fixed_default_predictions, zero_division=0)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
