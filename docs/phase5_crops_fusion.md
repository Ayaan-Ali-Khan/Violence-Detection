# Phase 5: interaction crops and late fusion

Phase 5 reuses the RWF-2000 validation-selected Phase 3 crop settings
(`tau=2.0`, margin `0.10`). The crop input tag includes both settings so the
derived tensors cannot be mistaken for another crop recipe. The cache stores
full and interaction-crop RGB `uint8` frames, plus a checksummed boolean mask
for each of the 32 candidate frames. The mask is true when Phase 3 supplied a
non-fallback crop box and false when the crop view uses the full frame.

## Kaggle prerequisites and run

1. Push the project code and upload `notebooks/kaggle_phase5_runner.ipynb` to
   Kaggle. Select **T4 x2**, enable Internet, and add `WANDB_API_KEY` as a
   notebook secret.
2. Attach the RWF-2000 video dataset and a Kaggle Dataset containing the Phase 3
   RWF crop `.npz` files. Keep the `rwf2000/...` hierarchy when possible; the
   runner also accepts a flattened split-directory upload. Each archive must include `crop_boxes_xyxy`,
   `fallback_mask`, and the settings metadata. The notebook locates these
   inputs and fails early when it cannot match the train crop cache.
3. Optionally attach the Phase 4 output as another Kaggle Dataset and set
   `FULL_BASELINE_CHECKPOINT` in the notebook to its `best.pt`. Otherwise, the
   notebook retrains the full-frame baseline on the paired Phase 5 cache.
4. Run the notebook top to bottom. It caches both views once, trains the
   crop-only specialist, trains the late-fusion model, evaluates all three
   inputs on the same RWF test clips, and makes the subgroup comparison.

The paired tensor cache is approximately 11 GiB for 1,986 clips (about 5.5 GiB
per view), plus masks and metadata. It is under `/kaggle/working/tensor_cache/`.
Training artifacts are under `/kaggle/working/experiments/`; download those
model and comparison outputs separately from the tensor cache if output size is
a concern. In the repository workspace, keep the downloaded Phase 5 output
under `cache/training/phase5/` to match the existing Phase 4 artifact layout.

## Models and resource limits

- **Full-frame:** the Phase 4 best checkpoint, or a re-trained full-frame
  baseline if one is not attached.
- **Crop-only:** one X3D-S stream trained on interaction-crop tensors, starting
  from Kinetics-pretrained weights.
- **Late fusion:** independent full-frame and crop X3D-S streams initialized
  from their trained specialist checkpoints. Each stream emits class logits;
  a trainable linear fusion head combines the four logits. Both streams and the
  fusion head are fine-tuned end to end.

Training uses one DDP process per visible GPU, FP16 autocast and gradient
scaling, and two DataLoader workers plus two Torch threads per process. The
single-stream runs use batch size 8 per GPU; the dual-stream fusion run uses 4
per GPU to bound activation memory. Video decoding is capped at two worker
processes. The full/crop data are already decoded and mmap-backed, so training
does not perform per-frame CPU decoding.

## Fallback-aware comparison

Evaluation deterministically selects the first of two cached temporal
candidates in each of 16 segments. A test clip is categorized as:

- `crop_found`: all 16 selected frames have a non-fallback interaction crop.
- `fallback`: all 16 selected frames use full-frame fallback boxes.
- `mixed`: the selected frames include both real crop boxes and fallback boxes.

`scripts/compare_input_modes.py` requires aligned predictions and labels from
all three runs, then reports accuracy, violent-class precision/recall/F1, and
ROC AUC for each input overall and within each availability group. It writes
`input_comparison.csv`, `crop_status_counts.csv`, and
`input_comparison.md`. The official RWF validation-as-test split is only used
for final reporting; crop thresholds and model selection remain validation
driven.

## Main implementation files

- `configs/preprocess_crops.yaml`: model preprocessing profile and explicit
  tau/margin-tagged cache ID.
- `src/violence_detection/data/preprocess.py` and `cache_tensors.py`:
  one-pass frame decoding, crop generation from Phase 3 boxes, and resumable
  paired caches with crop-found masks and row checksums.
- `src/violence_detection/data/dataset.py`: mmap full/crop/both loading and
  deterministic crop-status selection for evaluation.
- `src/violence_detection/models/x3ds.py`: independent two-stream X3D and
  trainable logit fusion head.
- `src/violence_detection/train.py` and `evaluate.py`: DDP training and
  test-prediction export for all three inputs.
- `scripts/compare_input_modes.py`: overall and fallback-stratified comparison.
- `notebooks/kaggle_phase5_runner.ipynb`: end-to-end Kaggle orchestration.
