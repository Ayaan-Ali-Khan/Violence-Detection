# Phase 7: cross-dataset evaluation

Phase 7 measures transfer of the three Phase 6 finalists without training on or
tuning against the external evaluation datasets:

- X3D-S late fusion (Phase 5 checkpoint)
- X3D-M late fusion (Phase 6 checkpoint)
- Swin3D-T paired views (Phase 6 checkpoint)

The Kaggle runner is [kaggle_phase7_runner.ipynb](../notebooks/kaggle_phase7_runner.ipynb).
It expects RWF-2000, RLVS, Hockey Fights, Phase 3 paired crop caches for those
datasets, the model checkpoints, and the Phase 6 RWF test evaluation outputs
as Kaggle inputs. It stores temporary tensors in `/kaggle/temp` and final
reports under `/kaggle/working/experiments/phase7`.

## Locked evaluation protocol

The input profile is shared with Phase 6: paired full-frame and interaction
crop views, 16 sampled model frames from 32 cached RGB candidates, 160-pixel
model input, and the Phase 3 `tau=2.0`, `margin=0.10` crop profile. Crop
fallback masks are retained in the tensor cache and prediction files.

For each model, the runner evaluates the RWF train-derived `validation` split,
then selects the violent-class probability threshold that maximizes violent
F1 on that split. Ties prefer a threshold closest to 0.5. The threshold JSON
is saved before external evaluation and reused unchanged for RLVS and Hockey
Fights. Neither external set is used to tune thresholds, checkpoints, crop
settings, or model choice.

The runner reapplies each locked threshold to the saved RWF official-test
probabilities before calculating score drops. This keeps the RWF reference and
the external evaluations on the same decision threshold without rerunning the
RWF test videos.

Cache construction decodes videos with two bounded CPU workers. For each
dataset, inference runs two model evaluations concurrently, one on each
visible GPU, with batch size one and AMP enabled. The final model runs after
the first pair. Each external tensor cache is removed after its evaluations
finish to stay within Kaggle disk limits. The pipeline does not train models
in Phase 7.

## Error analysis

The runner reports accuracy, violent-class precision/recall/F1, ROC AUC,
false positives, false negatives, and nonviolent false-alarm rate for each
model and dataset. It computes metric drops relative to each model's official
RWF-2000 test result. False-positive and false-negative CSVs include filenames
and violent probabilities for review. All Hockey Fights non-fight false
positives are also exported as `hockey_sports_false_alarms.csv`.

The Hockey manifest labels non-fight clips as nonviolent but does not annotate
whether a particular false alarm came from skating, a body check, a pileup, or
camera motion. Those clip-level causes require visual review of the exported
shortlist; the report does not infer them from filenames.

## Dataset availability

The project currently has RWF-2000, RLVS, and Hockey Fights. These cover the
RWF training/test domain and two external evaluation sets. The separate todo
item for selecting an optional third dataset was previously deferred, so task
44 is skipped by the runner until a third dataset is selected and its manifest
and crop cache are added. Do not count RWF-2000 as an external third dataset.

## Output layout

- `cache_verification/`: shape, dtype, completeness, checksum, and disk-size
  reports for the RWF validation, RLVS, and Hockey Fights caches.
- `calibration/`: RWF validation probabilities and locked per-model thresholds.
- `evaluation/`: per-model and per-dataset metrics, predictions, and confusion
  matrices.
- `error_analysis/`: cross-dataset metric table, metric drops, false-positive
  and false-negative lists, and Hockey non-fight false alarms.
- `logs/`: per-evaluation process logs.
