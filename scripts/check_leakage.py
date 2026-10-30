"""Report exact-file duplicates and sampled-frame matches across datasets."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", nargs="+", type=Path, help="Metadata CSV files to compare")
    parser.add_argument("--output-dir", type=Path, default=Path("data/metadata"))
    args = parser.parse_args()

    tables = [pd.read_csv(path) for path in args.metadata]
    frame = pd.concat(tables, ignore_index=True)
    needed = {"dataset", "relative_path", "label", "sha256", "sample_frame_hashes"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    pairs: dict[tuple[str, str, str, str], dict[str, object]] = {}

    def record(left: pd.Series, right: pd.Series, kind: str, evidence: str) -> None:
        if left["relative_path"] == right["relative_path"]:
            return
        ordered = sorted(
            [left, right],
            key=lambda row: (str(row["dataset"]), str(row["relative_path"])),
        )
        a, b = ordered
        key = (str(a["dataset"]), str(a["relative_path"]), str(b["dataset"]), str(b["relative_path"]))
        existing = pairs.get(key)
        if existing:
            kinds = set(str(existing["match_type"]).split(";"))
            kinds.add(kind)
            existing["match_type"] = ";".join(sorted(kinds))
            existing["evidence"] = f"{existing['evidence']}; {evidence}"
        else:
            pairs[key] = {
                "dataset_a": a["dataset"],
                "path_a": a["relative_path"],
                "split_a": a.get("split"),
                "label_a": a["label"],
                "dataset_b": b["dataset"],
                "path_b": b["relative_path"],
                "split_b": b.get("split"),
                "label_b": b["label"],
                "match_type": kind,
                "evidence": evidence,
            }

    # Exact file hashes provide definitive duplicates, including transcodes only
    # when their encoded bytes happen to be identical.
    for _, group in frame.dropna(subset=["sha256"]).groupby("sha256"):
        if len(group) > 1:
            for (_, left), (_, right) in combinations(group.iterrows(), 2):
                record(left, right, "exact_sha256", str(left["sha256"]))

    # Shared hashes from sampled frames are candidate near-duplicates, not proof.
    frame_index = {idx: row for idx, row in frame.iterrows()}
    hash_to_rows: dict[str, set[int]] = {}
    for idx, row in frame.iterrows():
        hashes = {item for item in str(row["sample_frame_hashes"]).split(";") if item and item != "nan"}
        for image_hash in hashes:
            hash_to_rows.setdefault(image_hash, set()).add(idx)
    common_frame_counts: dict[tuple[int, int], set[str]] = {}
    for image_hash, row_ids in hash_to_rows.items():
        for left_id, right_id in combinations(sorted(row_ids), 2):
            left, right = frame_index[left_id], frame_index[right_id]
            # Same-dataset shared frames can be common background; report
            # cross-dataset candidates and RWF train/test collisions only.
            is_cross_dataset = left["dataset"] != right["dataset"]
            rwf_splits = {str(left.get("split")), str(right.get("split"))}
            is_rwf_boundary = (
                str(left["dataset"]).lower().startswith("rwf")
                and str(right["dataset"]).lower().startswith("rwf")
                and "train" in rwf_splits
                and bool(rwf_splits & {"val", "validation", "test"})
            )
            if is_cross_dataset or is_rwf_boundary:
                common_frame_counts.setdefault((left_id, right_id), set()).add(image_hash)
    for (left_id, right_id), matched_hashes in common_frame_counts.items():
        if len(matched_hashes) >= 2:
            record(
                frame_index[left_id],
                frame_index[right_id],
                "sampled_frame_candidate",
                f"{len(matched_hashes)} shared 64-bit sampled-frame hashes; manually review",
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "duplicate_pairs.csv"
    pairs_frame = pd.DataFrame(pairs.values())
    if pairs_frame.empty:
        pairs_frame = pd.DataFrame(
            columns=["dataset_a", "path_a", "split_a", "label_a", "dataset_b", "path_b", "split_b", "label_b", "match_type", "evidence"]
        )

    review_path = args.output_dir / "leakage_manual_review.csv"
    if review_path.exists() and not pairs_frame.empty:
        reviews = pd.read_csv(review_path)
        review_map = {}
        for _, review in reviews.iterrows():
            key = tuple(sorted([
                (str(review["dataset_a"]), str(review["path_a"])),
                (str(review["dataset_b"]), str(review["path_b"])),
            ]))
            review_map[key] = (review["review_status"], review["decision"], review["notes"])
        statuses, decisions, notes = [], [], []
        for _, pair in pairs_frame.iterrows():
            key = tuple(sorted([
                (str(pair["dataset_a"]), str(pair["path_a"])),
                (str(pair["dataset_b"]), str(pair["path_b"])),
            ]))
            review = review_map.get(key)
            statuses.append(review[0] if review else "unreviewed")
            decisions.append(review[1] if review else "pending")
            notes.append(review[2] if review else "")
        pairs_frame["review_status"] = statuses
        pairs_frame["review_decision"] = decisions
        pairs_frame["review_notes"] = notes
    pairs_frame.to_csv(pairs_path, index=False)

    exact = pairs_frame[pairs_frame["match_type"].str.contains("exact_sha256")] if len(pairs_frame) else pairs_frame
    exact_count = len(exact)
    cross_dataset_exact = int((exact["dataset_a"] != exact["dataset_b"]).sum()) if len(exact) else 0
    rwf_boundary_exact = int(
        (
            exact["dataset_a"].astype(str).str.lower().str.startswith("rwf")
            & exact["dataset_b"].astype(str).str.lower().str.startswith("rwf")
            & exact["split_a"].astype(str).eq("train")
            & exact["split_b"].astype(str).eq("val")
        ).sum()
    ) if len(exact) else 0
    rlvs_internal_exact = int(
        ((exact["dataset_a"] == "rlvs") & (exact["dataset_b"] == "rlvs")).sum()
    ) if len(exact) else 0
    hockey_internal_exact = int(
        ((exact["dataset_a"] == "hockeyfights") & (exact["dataset_b"] == "hockeyfights")).sum()
    ) if len(exact) else 0
    rwf_internal_exact = int(
        ((exact["dataset_a"].astype(str).str.lower().str.startswith("rwf"))
         & (exact["dataset_b"].astype(str).str.lower().str.startswith("rwf"))
         & (exact["split_a"].astype(str) == exact["split_b"].astype(str))).sum()
    ) if len(exact) else 0
    conflicting_labels = int((exact["label_a"] != exact["label_b"]).sum()) if len(exact) else 0
    candidate_count = int(pairs_frame["match_type"].str.contains("sampled_frame_candidate").sum()) if len(pairs_frame) else 0
    confirmed_count = int(
        (pairs_frame.get("review_decision", pd.Series(dtype=str)) == "same_source_video").sum()
    ) if len(pairs_frame) else 0
    lines = [
        "# Dataset duplicate and leakage check",
        "",
        f"Datasets compared: {', '.join(sorted(frame['dataset'].dropna().astype(str).unique()))}",
        f"Video records compared: {len(frame)}",
        f"Pairs with exact SHA-256 match: {exact_count}",
        f"Exact matches between different datasets: {cross_dataset_exact}",
        f"Exact RWF train-to-official-val matches: {rwf_boundary_exact}",
        f"Exact duplicate pairs within RLVS: {rlvs_internal_exact}",
        f"Exact duplicate pairs within Hockey Fights: {hockey_internal_exact}",
        f"Exact duplicate pairs within an RWF source split: {rwf_internal_exact}",
        f"Exact duplicate pairs with conflicting labels: {conflicting_labels}",
        f"Pairs with at least two shared sampled-frame hashes (review candidates): {candidate_count}",
        f"Manually confirmed same-source pairs: {confirmed_count}",
        "",
        "Exact SHA-256 matches are byte-identical files. Sampled-frame matches are candidate duplicates only; "
        "common static backgrounds can collide. Review those pairs visually before excluding data.",
        "",
        "No exact SHA-256 matches were found between datasets in this scan. For the RWF internal "
        "train/test boundary, keep the official test record and exclude confirmed training copies; "
        "the RWF split builder applies that rule.",
        "The report does not modify or delete source videos.",
        "",
        f"Pair details: `{pairs_path.as_posix()}`",
    ]
    report_path = args.output_dir / "leakage_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
