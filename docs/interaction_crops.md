# Phase 3: interaction crops

## Reusable API

`violence_detection.crops` exposes:

- `normalized_pair_distance(box_a, box_b)`: Euclidean distance between box
  centers divided by their mean height.
- `select_clip_pair(...)`: chooses the most persistently close pair across a
  clip, locks those IDs for the clip, then adds tracks that stay near either
  locked person for the required number of frames.
- `build_crop_boxes(...)`: unions the selected people per frame, fills only
  short internal gaps, smooths box edges, makes margin-expanded square crops,
  and marks every unavailable frame as a full-frame fallback.
- `build_interaction_crops(...)`: composes pair selection and crop generation
  and returns the selected pair/group alongside per-frame crop metadata.

Crop arrays are XYXY integer boxes in the source-video coordinate system. The
fallback, directly observed and interpolated masks are kept separately so later
training code can audit or filter each kind of frame.

## Selected settings

`configs/interaction_crops.yaml` sets `tau=2.0` normalized person-box heights,
`margin=0.10`, a 5-frame smoothing window, a 3-frame interpolation limit, and a
group-expansion radius of `1.5 * tau`. The 318-row tuning sweep and the 50-clip
overlay review used only RWF-2000's `validation` split. At `tau=2.0`, the sweep
found a persistent pair in 263/318 clips (82.7%); the mean frame fallback rate
was 33.1%, and the mean crop area on nonfallback frames was 47.0% of the full
frame at a 10% margin. Tau values above 2.0 added only one validation clip at
2.25 and no further clips at 2.5 or 3.0, so 2.0 was selected at the coverage
elbow. The 10% margin was the smallest reviewed candidate that retained visible
space around the selected group.

These are engineering coverage and crop-size proxies, not interaction accuracy:
the datasets do not provide person boxes or pair annotations. Manual inspection
of the deterministic 50-clip start/middle/end overlays found the selected boxes
generally covered the selected tracks. Full-frame fallbacks appear when no
persistent pair exists or when selected tracks are absent; these are intentional.
Some crowded scenes produce broad group crops, so the selected threshold should
be re-evaluated when trained action-classification results become available.

## Reproduce

Build crop-box NPZ files for the three manifests after the detection caches are
available:

```bash
PYTHONPATH=src python scripts/build_interaction_crops.py \
  data/splits/rwf2000.csv \
  data/splits/rlvs_eval.csv \
  data/splits/hockeyfights_eval.csv \
  --cache-dir cache/detections/yolo11n
```

Tune only on RWF validation clips:

```bash
PYTHONPATH=src python scripts/tune_interaction_crops.py \
  --cache-dir cache/detections/yolo11n
```

Create the deterministic visual review set:

```bash
PYTHONPATH=src python scripts/review_interaction_crops.py \
  --cache-dir cache/detections/yolo11n
```

The build script writes per-clip crop boxes below `cache/interaction_crops/`
and a fallback/crop-size summary under `data/interaction_crops/`. The review
script writes three overlay frames per sampled clip to the ignored cache folder
and a tracked checklist CSV. Re-run the Kaggle detection runner's Phase 3 cell
to create crop caches beside the Kaggle detection outputs.
