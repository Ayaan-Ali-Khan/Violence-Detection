# Phase 6: stronger classifiers

Phase 6 compares the Phase 5 X3D-S late-fusion checkpoint with X3D-M late
fusion and an optional Swin3D-T classifier. All candidates use the paired
full-frame and interaction-crop views from RWF-2000, the same train/validation
split, and the same official 400-clip test split.

## Candidate models

- **X3D-M late fusion (Task 37):** two independent Kinetics-pretrained X3D-M
  streams process the full frame and interaction crop. Their class logits feed
  a learned fusion layer. The fixed inner 16x7x7 average pool is replaced by
  adaptive pooling for the project's 160px input; the pretrained projection
  layers remain in place. It uses mixed precision and one clip per GPU to keep
  T4 memory bounded.
- **Swin3D-T (Task 38, optional):** a Kinetics-400-pretrained Swin3D-T processes
  each view sequentially and averages the two logits. It uses one clip per GPU
  and avoids concatenating both views into a doubled input batch. Set
  `RUN_SWIN3D = False` in the Kaggle notebook to skip it when the session time
  budget is tight.
- **X3D-S late fusion baseline:** the Phase 5 best checkpoint is evaluated
  again in the same session, on the same cache, using the same timing method.

The X3D-M and Swin3D-T models reuse the paired uint8 tensor cache. The cache is
temporary under `/kaggle/temp`; the saved notebook output contains model
checkpoints and evaluation artifacts rather than another multi-gigabyte copy
of training tensors.

## Kaggle run

Use a Kaggle GPU notebook with T4 x2 and Internet enabled. Attach the RWF-2000
videos, the Phase 3 crop-cache dataset, and the Phase 5 output dataset with
`x3d_s_rwf2000_late_fusion/best.pt`. The runner clones the public repository,
installs only `requirements-kaggle.txt`, builds and verifies the paired cache,
trains candidates with DistributedDataParallel, AMP, two data-loader workers
per rank, and two CPU threads per rank.

The runner writes experiment records under `/kaggle/working/experiments/phase6/`
and reusable weights under `/kaggle/working/model_checkpoints/phase6/`:

- `experiments/phase6/`: resolved configs, validation history, learning curves,
  train/validation gap reports, evaluations, comparison, and cache verification.
- `model_checkpoints/phase6/`: `best.pt` and `last.pt` for X3D-M and Swin3D-T.
- `evaluation/`: test metrics, predictions, confusion matrices, and
  `inference_speed.json` for each model.
- `comparison/` (under `experiments/phase6/`): aligned metrics, speed comparison, and a classifier
  recommendation.

## Comparison method

Evaluation uses the same test clips, batch size 1, one visible GPU, AMP, and
five warmup batches excluded from the measurement. The reported speed includes
input transfer and model forward execution, but excludes DataLoader wait time
and metric/loss calculation. This makes the figure useful for comparing model
execution on the same Kaggle hardware; it is not an end-to-end deployed
surveillance pipeline latency.

The comparison script recommends the fastest model whose violent-class F1 is
within 0.005 absolute points of the best F1. It also reports accuracy,
precision, recall, ROC AUC, and milliseconds per clip. The recommendation is
produced only after the Kaggle runs complete; this local setup does not have
the T4 hardware needed to produce valid Phase 6 speed measurements.

## Phase 5 starting point

On the 400-clip RWF-2000 test split, Phase 5 measured X3D-S late fusion at
89.0% accuracy, 0.889 violent-class F1, and 0.957 ROC AUC. Phase 6 compared
X3D-M and Swin3D-T against this baseline before selecting the leading candidate.

## Completed Phase 6 results

The Kaggle T4 run completed both stronger-model training jobs, evaluated all
three candidates on the same 400 official RWF-2000 test clips, and produced
checkpoints, training histories, predictions, confusion matrices, and timing
reports. The comparison results are:

| model | accuracy | violent F1 | ROC AUC | clips/s | ms/clip |
| --- | ---: | ---: | ---: | ---: | ---: |
| X3D-S late fusion (Phase 5) | 0.8900 | 0.8894 | 0.9572 | 23.14 | 43.21 |
| X3D-M late fusion | 0.8650 | 0.8657 | 0.9414 | 23.62 | 42.34 |
| Swin3D-T paired views | 0.8975 | 0.8967 | 0.9565 | 15.40 | 64.92 |

Using the predefined rule (within 0.005 absolute violent-class F1 of the best,
then choose the fastest qualifying model), **Swin3D-T is the selected
classifier**. It is the only model within that F1 tolerance of the top score.
X3D-S is about 21.8 ms faster per clip but its F1 is 0.0073 below Swin3D-T,
so it does not satisfy the specified tolerance. X3D-M is faster still by a
small amount, but its test F1 is lower than both alternatives.

These are clip-level classifier measurements, not end-to-end surveillance
pipeline latency: the timing excludes data-loader wait, tracking, and the
alerting/web-app path. The test set contains 400 clips, so treat the modest
metric differences as a model-selection result for this project rather than
proof of broad generalization. The model comparison and recommendation JSON
under `experiments/phase6/comparison/` are the run's machine-readable
source of truth.
