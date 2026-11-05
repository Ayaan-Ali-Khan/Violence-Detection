# Phase 8 packaging and reproducibility

## Inference package

The reusable clip API is documented in [inference.md](inference.md). It loads
the selected Phase 6 Swin3D-T paired-view checkpoint once, runs person
detection and ByteTrack, builds the clip-stable interaction crop, applies the
same deterministic sampling and normalization as evaluation, and returns
probabilities and crop provenance. The inference function reports a clip-level
decision and writes a video annotated with person detections, track IDs, short
movement trails, and the final clip-level verdict.

Use [the Kaggle inference smoke notebook](../notebooks/kaggle_phase8_inference_smoke.ipynb)
to run a few RWF-2000 clips and inspect the verdict JSON and rendered videos.
Attach the RWF-2000 dataset and the selected Swin3D-T Kaggle Model; if automatic
checkpoint discovery does not find its project `.pt` checkpoint, set the path
override in the notebook.

## Clean-checkout reproduction

For a clean reproduction, first commit the source changes and pin the Kaggle
runner's `GIT_REF` to that commit SHA. The `main` branch is intentionally
convenient for development but moves over time and is not a reproducible run
reference. Start with a fresh clone and the same pinned code:

```bash
git clone https://github.com/Ayaan-Ali-Khan/Violence-Detection.git
cd Violence-Detection
git checkout <recorded-commit-sha>
python -m pip install -r requirements.txt
python scripts/write_reproducibility_manifest.py --require-clean \
  --checkpoint-root /path/to/model_checkpoints \
  --output experiments/phase8/reproducibility_manifest.json
```

On Kaggle, choose the same GPU accelerator as the experiment, attach the
versioned dataset inputs, set the notebook `GIT_REF` to the same SHA, and run
the saved Phase 7 runner. The runner writes a reproducibility manifest
containing Python, Torch, CUDA, GPU, package, source hash, config, and selected
checkpoint hashes beside the evaluation results. Keep that JSON beside the
experiment results. `requirements-kaggle.txt` is for Kaggle sessions that
already provide CUDA-enabled PyTorch; it avoids replacing Kaggle's Torch build.

The Phase 6 RWF test results informed the model-selection rule, so they are
selection results. The Phase 7 RLVS and Hockey Fights evaluations used thresholds
selected only on RWF validation. Exact bitwise training reproduction is not
promised because Kaggle GPU hardware and nondeterministic video kernels can
vary; the manifest records the environment needed to interpret each rerun.

## Results tables and figures

Regenerate the Phase 8 tables and report figures with:

```bash
python scripts/build_phase8_results.py
```

Outputs are written to `experiments/phase8/`. Source CSVs from Phases 6 and 7
remain the canonical metric records; the Phase 8 files are report-ready views
derived from them.
