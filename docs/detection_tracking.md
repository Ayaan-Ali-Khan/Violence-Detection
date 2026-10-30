# Phase 2: detection and tracking

The detection stage uses COCO person class `0` from pretrained YOLO11 weights and
Ultralytics ByteTrack. The current defaults live in `configs/detection.yaml` and
`configs/bytetrack.yaml`: 640-pixel inference, confidence 0.10, IoU 0.70, and every
source frame. Tracker state is reset at each clip boundary so IDs are unique only
within a clip.

## Compare YOLO11 sizes

From the repository root, with the project dependencies installed and internet
enabled for the first run:

```bash
PYTHONPATH=src python scripts/benchmark_yolo.py
```

The default command samples the same deterministic clips/frames for nano and small
models. Add `--models yolo11n.pt yolo11s.pt yolo11m.pt` to include medium when its
checkpoint is available. It writes per-frame measurements to
`data/detection/yolo11_benchmark.csv`, a summary CSV beside it, and a short comparison
report. The datasets have video labels rather than person-box annotations, so this
benchmark compares checkpoint size, speed, and number of people found; it does not
measure mAP or person-detection accuracy.

## Check tracking IDs on a clip

```bash
PYTHONPATH=src python scripts/smoke_track.py \
  --video data/raw/rwf2000/val/Fight/0Ow4cotKOuw_0.avi
```

This writes a JSON summary and an annotated MP4 under ignored `cache/previews/`.
Green boxes have a ByteTrack ID; unassigned detections are orange. The JSON reports
track lengths and continuity ratios for a quick identity sanity check.

## Create/resume track caches

Cache the RWF training/validation/test selection and both external evaluation sets:

```bash
PYTHONPATH=src python scripts/cache_tracks.py data/splits/rwf2000.csv
PYTHONPATH=src python scripts/cache_tracks.py \
  data/splits/rlvs_eval.csv data/splits/hockeyfights_eval.csv
PYTHONPATH=src python scripts/summarize_detection_stats.py
```

Track arrays are written as compressed per-clip NPZ files below
`cache/detections/yolo11n/`, which is ignored by Git. The cache includes frame indices,
XYXY boxes, confidence, class IDs, ByteTrack IDs, and people counts for each processed
frame. A source/settings fingerprint makes reruns skip valid outputs and recompute
stale or incomplete files. Use `--force` to intentionally overwrite caches and
`--limit-clips N` for a small smoke run. Change `--cache-dir` when comparing models.

For Kaggle, attach the raw datasets as notebook inputs and point `--data-root` at the
directory containing folders named `rwf2000`, `rlvs`, and `hockeyfights` (often
`/kaggle/input`). Keep caches and reports under `/kaggle/working`, for example:

```bash
PYTHONPATH=src python scripts/cache_tracks.py \
  data/splits/rwf2000.csv data/splits/rlvs_eval.csv data/splits/hockeyfights_eval.csv \
  --data-root /kaggle/input \
  --cache-dir /kaggle/working/detections/yolo11n \
  --stats-output /kaggle/working/detections/per_clip_stats.csv \
  --device 0
PYTHONPATH=src python scripts/summarize_detection_stats.py \
  --cache-dir /kaggle/working/detections/yolo11n \
  --output /kaggle/working/detections/detection_stats.csv \
  --report /kaggle/working/detections/detection_stats.md
```

`detection_stats.csv` and `.md` report average/median/p95 people per processed frame,
no-detection frame rate, fully missed clips, detection totals, and track instances by
dataset/split/class. If a frame stride is used, the statistics refer to sampled frames.
