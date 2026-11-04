# Model checkpoints

Store reusable or resumable weights here, grouped by phase and run:

```text
model_checkpoints/
  phase4/x3d_s_rwf2000/best.pt
  phase5/x3d_s_rwf2000_late_fusion/best.pt
  phase6/swin3d_t_rwf2000_paired_views/best.pt
```

The directory is ignored by Git because checkpoint files are large. Keep
`best.pt` for inference and comparisons; retain `last.pt` when resuming a run.
Experiment histories, configs, curves, predictions, and reports go in
`experiments/`.
