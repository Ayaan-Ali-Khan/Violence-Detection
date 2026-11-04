# Experiment outputs

Keep experiment metadata and evaluation artifacts here. Each phase gets its own
folder:

```text
experiments/
  phase4/      # baseline run, curves, evaluation, reports
  phase5/      # crop/fusion runs, comparison tables
  phase6/      # stronger-model runs, test evaluations, model comparison
  phase7/      # cross-dataset evaluations and error analysis
```

Best and resumable model weights belong in the matching `model_checkpoints/`
phase folder. Kaggle runners write both trees under `/kaggle/working`; when
downloading a run, place each downloaded tree at the same path in the project.
Video and tensor caches belong under `cache/`, not in experiment folders.
