"""Create a duplicate-clean Hockey Fights evaluation manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/hockeyfights.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/hockeyfights_eval.csv"))
    parser.add_argument(
        "--exclusions", type=Path, default=Path("data/splits/hockeyfights_eval_exclusions.csv")
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.metadata)
    required = {"relative_path", "label", "sha256", "decode_ok"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    unreadable = frame.loc[~frame["decode_ok"].astype(bool)].copy()
    if not unreadable.empty:
        unreadable["exclusion_reason"] = "unreadable_video"

    eligible = frame.loc[frame["decode_ok"].astype(bool)].copy()
    exclusions = [unreadable] if not unreadable.empty else []
    keep_indices: set[int] = set()
    for _, group in eligible.groupby("sha256", sort=True, dropna=False):
        if group["label"].nunique() > 1:
            conflicted = group.copy()
            conflicted["exclusion_reason"] = "identical_video_has_conflicting_labels"
            exclusions.append(conflicted)
            continue
        group = group.sort_values("relative_path")
        keep_indices.add(int(group.index[0]))
        duplicates = group.iloc[1:].copy()
        if not duplicates.empty:
            duplicates["exclusion_reason"] = "duplicate_video_same_label"
            exclusions.append(duplicates)

    kept = frame.loc[frame.index.isin(keep_indices)].copy()
    kept["split"] = "evaluation"
    kept = kept.sort_values("relative_path").reset_index(drop=True)
    excluded = pd.concat(exclusions, ignore_index=True) if exclusions else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.exclusions.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(args.output, index=False)
    excluded.to_csv(args.exclusions, index=False)
    print(f"Evaluation videos retained: {len(kept)}")
    print(f"Excluded duplicate/unreadable records: {len(excluded)}")
    if len(kept):
        print(kept["label_name"].value_counts().sort_index().to_string())
    print(f"Wrote {args.output}")
    print(f"Wrote {args.exclusions}")


if __name__ == "__main__":
    main()
