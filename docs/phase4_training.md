# Phase 4: Tensor cache and X3D baseline

This phase builds a reproducible full-frame X3D-S baseline from the RWF-2000
manifest. It keeps decoded frames as RGB `uint8` in memory-mapped NumPy arrays;
the training process selects 16 of 32 evenly spaced candidates, transfers each
batch to the GPU, resizes to the model's 160×160 input, augments, and normalizes
there. The preprocessing profile follows the earlier X3D notebook's 176px
cached frames (`round(160 × 1.1)`) and Kinetics normalization values.

## Kaggle run

1. Upload `notebooks/kaggle_training_runner.ipynb` to Kaggle, attach the
   RWF-2000 dataset containing `train/{Fight,NonFight}` and
   `val/{Fight,NonFight}`, select the **T4 x2** accelerator, enable Internet,
   and add `WANDB_API_KEY` as a Kaggle secret.
2. Set `GIT_REF` in the first code cell to a commit SHA when you need a
   reportable, pinned run. The notebook clones or checks out that code.
3. Run all cells. Caching, verification, training, and held-out test evaluation
   save under `/kaggle/working` so they are available from notebook output.
4. Download `tensor_cache/reports/verification_report.md`,
   `experiments/x3d_s_rwf2000/` and its `test/` subdirectory. Preserve these
   artifacts with the W&B run link for the report.

The full-frame cache is expected to use about 5.5 GiB for 1,986 clips, plus
small metadata arrays. It can resume after interruption. `--max-clips` is
available for a small cache smoke run; do not use that option for the final
training run.

## Resource settings

Training uses one DDP process per visible GPU, FP16 autocast and gradient
scaling, channels-last-3D model inputs, and global batch size 16 on two GPUs
(8 per GPU). Each GPU process uses two loader workers and two Torch CPU threads;
decoding uses two bounded processes. This keeps CPU concurrency capped while
the model and GPU-side resize/normalization use both T4s. Reduce batch size to
4 per GPU if Kaggle reports CUDA out-of-memory; retain both GPUs.

The training script selects checkpoints by validation loss and early stops.
It writes `best.pt`, `last.pt`, `history.csv`, `learning_curves.png`, and
`train_val_gap.json`. It never reads the test split. The separate evaluator
reports test accuracy, violent-class precision/recall/F1, ROC AUC, and the
confusion matrix, with per-clip predictions.

## Main files

- `configs/preprocess.yaml`: versioned X3D-S/M frame, resolution, normalization,
  and augmentation profiles with distinct input tags.
- `src/violence_detection/data/preprocess.py` and
  `src/violence_detection/data/cache_tensors.py`: bounded decoding,
  deterministic sampling, RGB uint8 preprocessing, and resumable checksummed
  mmap tensor caches. `scripts/cache_tensors.py` is the CLI entry point used by
  the Kaggle notebook.
- `scripts/verify_tensor_cache.py`: cache completeness, layout, labels, checksum,
  and disk-size report.
- `src/violence_detection/data/dataset.py` and `augment.py`: mmap-backed full,
  crop, or paired views; temporal sampling and GPU-side transforms.
- `src/violence_detection/models/x3ds.py`: X3D construction, pretrained backbone
  initialization, and separate head/backbone learning rates.
- `src/violence_detection/train.py`: DDP training, validation, checkpointing,
  tracking, learning curves, and train/validation gap.
- `src/violence_detection/evaluate.py`: held-out test metrics and artifacts.
- `configs/train_x3d.yaml` and `notebooks/kaggle_training_runner.ipynb`:
  baseline settings and Kaggle orchestration.

The Kaggle cache, trained baseline, validation gap, and held-out test results
are generated artifacts. They are not included in the repository until the
notebook has completed successfully.
