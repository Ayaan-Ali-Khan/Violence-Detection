"""Classify one video clip and render person detections/tracks to an MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from violence_detection.inference import DEFAULT_CHECKPOINT, ViolencePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--detector-weights", type=Path)
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default=None, help="For example cuda:0 or cpu.")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()

    predictor = ViolencePredictor(
        checkpoint=args.checkpoint
        if args.checkpoint is not None
        else DEFAULT_CHECKPOINT,
        detector_weights=args.detector_weights,
        threshold=args.threshold,
        device=args.device,
        cpu_threads=args.cpu_threads,
    )
    result = predictor.predict(args.video, output_video=args.output_video)
    rendered = json.dumps(result, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
