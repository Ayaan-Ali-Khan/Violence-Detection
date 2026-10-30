"""Build a video metadata CSV for a dataset rooted on disk."""

from __future__ import annotations

import argparse
from pathlib import Path

from violence_detection.data.metadata import build_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Stable dataset key, e.g. rwf2000")
    parser.add_argument("--root", required=True, type=Path, help="Dataset root directory")
    parser.add_argument("--output", type=Path, help="Output CSV (defaults to data/metadata/<dataset>.csv)")
    args = parser.parse_args()

    output = args.output or Path("data/metadata") / f"{args.dataset.lower()}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_metadata(args.dataset, args.root)
    frame.to_csv(output, index=False)
    failed = int((~frame["decode_ok"].astype(bool)).sum()) if len(frame) else 0
    print(f"Dataset: {args.dataset}")
    print(f"Videos: {len(frame)}; unreadable: {failed}")
    if len(frame):
        print("Class counts:")
        print(frame["label_name"].value_counts().sort_index().to_string())
        if "split" in frame and frame["split"].notna().any():
            print("Counts by split and class:")
            print(frame.groupby(["split", "label_name"], dropna=False).size().to_string())
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
