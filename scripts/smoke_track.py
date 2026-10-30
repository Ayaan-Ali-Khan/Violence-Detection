"""Run ByteTrack on one short clip and write ID/overlay sanity-check outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from violence_detection.detection.detect_track import load_yolo, reset_tracker, track_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--tracker", type=Path, default=Path("configs/bytetrack.yaml"))
    parser.add_argument("--output-video", type=Path, default=Path("cache/previews/bytetrack_smoke.mp4"))
    parser.add_argument("--output-json", type=Path, default=Path("cache/previews/bytetrack_smoke.json"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create overlay video: {args.output_video}")

    model = load_yolo(args.weights)
    reset_tracker(model)
    id_frames: dict[int, list[int]] = defaultdict(list)
    counts = []
    frame_index = 0
    try:
        while frame_index < args.max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            tracked = track_frame(
                model,
                frame,
                tracker=args.tracker,
                imgsz=args.imgsz,
                conf=args.confidence,
                device=args.device,
            )
            counts.append(len(tracked.boxes_xyxy))
            for box, score, track_id in zip(
                tracked.boxes_xyxy, tracked.scores, tracked.track_ids, strict=True
            ):
                x1, y1, x2, y2 = box.astype(int)
                color = (30, 220, 40) if track_id >= 0 else (0, 180, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID {track_id} {score:.2f}",
                    (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                if track_id >= 0:
                    id_frames[int(track_id)].append(frame_index)
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    tracks = []
    for track_id, indices in sorted(id_frames.items()):
        tracks.append(
            {
                "track_id": track_id,
                "frames_seen": len(indices),
                "first_frame": min(indices),
                "last_frame": max(indices),
                "span_frames": max(indices) - min(indices) + 1,
                "continuity_ratio": len(indices) / (max(indices) - min(indices) + 1),
            }
        )
    summary = {
        "video": args.video.as_posix(),
        "processed_frames": frame_index,
        "frames_with_person_detections": sum(count > 0 for count in counts),
        "mean_people_per_frame": sum(counts) / len(counts) if counts else 0.0,
        "unique_track_ids": len(tracks),
        "tracks": tracks,
        "overlay_video": args.output_video.as_posix(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
