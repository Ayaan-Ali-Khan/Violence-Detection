# Clip-level inference

The packaged prediction API uses the Phase 6 selected classifier, Swin3D-T
paired views, plus YOLO11n/ByteTrack person tracks and Phase 3-style
clip-stable interaction crops. The crop path uses the same `tau`, margin,
smoothing, short-gap interpolation, and full-frame fallback as the training
cache. Frames are sampled deterministically using the first temporal candidate
in each segment, matching evaluation. Each call returns one clip-level verdict
and writes an MP4 showing every detected person, their ByteTrack ID, a short
track trail, and the final clip verdict banner.

## Install

Use Python 3.11 or 3.12 for a local environment, or a Kaggle GPU image that has
compatible Torch and TorchVision already installed. Install the video extra
without replacing a Kaggle CUDA build:

```bash
python -m pip install -e '.[video]'
```

The selected classifier checkpoint
(`model_checkpoints/phase6/swin3d_t_rwf2000_paired_views/best.pt`) must be
available locally or passed by path. The predictor uses the local YOLO11n
weights when present; otherwise Ultralytics downloads its named `yolo11n.pt`
weight, so the first run needs Internet access. The predictor defaults to CUDA
when available and otherwise runs on CPU. It sets OpenCV to one thread and
Torch to two CPU threads by default.

## Python API

```python
from violence_detection.inference import ViolencePredictor

predictor = ViolencePredictor(device="cuda:0")
result = predictor.predict("path/to/clip.avi")
print(result)
```

Create the predictor once and reuse it for multiple clips; model and detector
weights are loaded at initialization. By default, the annotated MP4 is saved to
`experiments/inference/<clip-stem>_annotated.mp4`; pass `output_video=` to
choose another path. If a file with that name already exists, it is replaced.
The result dictionary includes the final `Violent`/`NonViolent` verdict, both
class probabilities, the RWF-validation-selected threshold, the output video
path, detection/track counts, selected pair IDs, crop fallback rate, and
crop-found rate among sampled frames. A clip with no stable person pair uses
full-frame fallback crops.

The video includes box rectangles, person IDs, detection scores, short center-
point trails for tracked people, and the same final verdict banner on every
frame. The classifier evaluates the clip as a whole; the banner is not a
frame-by-frame violence timeline.

## Command-line inference

```bash
python scripts/predict_video.py \
  --video path/to/clip.avi \
  --checkpoint model_checkpoints/phase6/swin3d_t_rwf2000_paired_views/best.pt \
  --device cuda:0 \
  --output-video experiments/inference/clip_annotated.mp4 \
  --output-json experiments/inference/clip_result.json
```

If `--output-video` is omitted, the default output path above is used. The
command prints the result JSON as well as optionally saving it with
`--output-json`.

The convenience function is also available:

```python
from violence_detection.inference import predict

result = predict("path/to/clip.avi")
```

This convenience form constructs a predictor for that call. For a video stream,
buffer a short clip, then call the reusable predictor on each completed clip;
the current trained classifier makes clip-level decisions and does not yet
produce per-frame alerts.

The default violent threshold is `0.94775390625`, calibrated for the selected
Swin3D-T checkpoint on the RWF-2000 validation split. It is loaded from the
saved Phase 7 calibration artifact when available and falls back to the same
recorded value in a clean checkout. It is fixed for cross-dataset evaluation
and may not be optimal for a new camera domain. Pass an explicit `threshold=`
when constructing `ViolencePredictor` only when the operating point has been
selected using an appropriate validation set.

## Crop-only helper

```python
from violence_detection.detection.detect_track import load_yolo
from violence_detection.inference import build_clip_crops

detector = load_yolo("yolo11n.pt")
crop_result = build_clip_crops("path/to/clip.avi", detector=detector)
print(crop_result.crops.fallback_rate, crop_result.selection)
```

`crop_result.crops.boxes_xyxy` has one source-resolution XYXY box per source
frame. When no interaction pair is found, the box spans the full frame. The
helper returns pair/group IDs and crop provenance for auditability.

## Selected checkpoint result context

Swin3D-T was selected in Phase 6 using the project rule: retain models within
0.005 absolute violent-class F1 of the highest RWF-2000 test F1, then choose the
fastest. X3D-S late fusion had better RLVS and Hockey Fights performance in
Phase 7. Keep those cross-dataset differences visible when presenting the
selected model; selection does not imply universal domain superiority.
