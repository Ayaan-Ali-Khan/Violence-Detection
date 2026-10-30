"""Create deterministic train/validation/test manifests from RWF-2000."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/rwf2000.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/rwf2000.csv"))
    parser.add_argument("--exclusions", type=Path, default=Path("data/splits/rwf2000_exclusions.csv"))
    parser.add_argument(
        "--reviewed-duplicates",
        type=Path,
        default=Path("data/metadata/leakage_manual_review.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    required = {"relative_path", "label", "split", "decode_ok", "sha256"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    unreadable = metadata.loc[~metadata["decode_ok"].astype(bool)]
    if not unreadable.empty:
        raise ValueError(f"Cannot split: {len(unreadable)} RWF clips are unreadable")

    official_train = metadata.loc[metadata["split"] == "train"].copy()
    official_test = metadata.loc[metadata["split"] == "val"].copy()
    if official_train.empty or official_test.empty:
        raise ValueError("Expected RWF directories named train/ and val/")

    exclusions = []
    test_hashes = set(official_test["sha256"].dropna().astype(str))
    test_overlap = official_train[official_train["sha256"].astype(str).isin(test_hashes)].copy()
    if not test_overlap.empty:
        test_overlap["exclusion_reason"] = "exact_duplicate_of_official_test"
        exclusions.append(test_overlap)

    test_paths = set(official_test["relative_path"].astype(str))
    train_paths = set(official_train["relative_path"].astype(str))
    confirmed_near_paths: set[str] = set()
    if args.reviewed_duplicates.exists():
        reviewed = pd.read_csv(args.reviewed_duplicates)
        confirmed = reviewed.loc[reviewed["decision"] == "same_source_video"]
        for _, pair in confirmed.iterrows():
            path_a, path_b = str(pair["path_a"]), str(pair["path_b"])
            if path_a in test_paths and path_b in train_paths:
                confirmed_near_paths.add(path_b)
            elif path_b in test_paths and path_a in train_paths:
                confirmed_near_paths.add(path_a)
    already_excluded = set(test_overlap["relative_path"].astype(str))
    confirmed_near_paths -= already_excluded
    near_overlap = official_train[
        official_train["relative_path"].astype(str).isin(confirmed_near_paths)
    ].copy()
    if not near_overlap.empty:
        near_overlap["exclusion_reason"] = "visually_confirmed_duplicate_of_official_test"
        exclusions.append(near_overlap)
    all_test_overlap_paths = already_excluded | confirmed_near_paths
    candidates = official_train.loc[
        ~official_train["relative_path"].astype(str).isin(all_test_overlap_paths)
    ].copy()

    # Repeated training files inflate the apparent sample count. Keep the first
    # deterministically sorted record for each byte-identical content hash.
    duplicate_train = candidates[candidates.duplicated("sha256", keep="first")].copy()
    if not duplicate_train.empty:
        duplicate_train["exclusion_reason"] = "duplicate_training_file_same_sha256"
        exclusions.append(duplicate_train)
    candidates = candidates.drop_duplicates("sha256", keep="first")

    train_rows, validation_rows = train_test_split(
        candidates,
        test_size=args.validation_size,
        random_state=args.seed,
        shuffle=True,
        stratify=candidates["label"],
    )
    train_rows = train_rows.assign(split="train")
    validation_rows = validation_rows.assign(split="validation")
    official_test = official_test.assign(split="test")
    result = pd.concat([train_rows, validation_rows, official_test], ignore_index=True)
    result.insert(0, "split_seed", args.seed)
    result = result.sort_values(["split", "relative_path"]).reset_index(drop=True)

    if result["relative_path"].duplicated().any():
        raise ValueError("Duplicate paths found in split manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    excluded = pd.concat(exclusions, ignore_index=True) if exclusions else pd.DataFrame()
    args.exclusions.parent.mkdir(parents=True, exist_ok=True)
    excluded.to_csv(args.exclusions, index=False)
    print(f"Wrote {args.output}")
    print(f"Excluded {len(excluded)} duplicate training records to {args.exclusions}")
    print(result.groupby(["split", "label_name"]).size().to_string())


if __name__ == "__main__":
    main()
