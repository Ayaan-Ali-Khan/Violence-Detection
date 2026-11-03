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

The runner writes results under `/kaggle/working/phase6/`:

- `experiments/`: best/last checkpoints, resolved configs, validation history,
  learning curves, and train/validation gap reports.
- `evaluation/`: test metrics, predictions, confusion matrices, and
  `inference_speed.json` for each model.
- `comparison/`: aligned metrics, speed comparison, and a classifier
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
89.0% accuracy, 0.889 violent-class F1, and 0.957 ROC AUC. X3D-M and Swin3D-T
must be compared against that result before selecting the final classifier.
