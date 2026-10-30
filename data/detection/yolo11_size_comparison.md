# YOLO11 model-size comparison

Device: cpu; image size: 640; confidence: 0.1; IoU: 0.7.
Sample: 36 deterministically selected frames across the supplied dataset manifests.

The sampled data have video-level labels, not person-box annotations. This comparison reports runtime and detected-person counts; it is not an mAP/precision benchmark.

```text
     model  checkpoint_mb device  sampled_frames  mean_latency_sec  median_latency_sec  frames_per_second  mean_people_per_frame  no_person_frame_rate
yolo11n.pt       5.353703    cpu              36          0.105454            0.093402           9.482817               7.055556              0.027778
yolo11s.pt      18.419010    cpu              36          0.240532            0.202866           4.157453               7.361111              0.027778
```

Initial cache choice: YOLO11n, because it was 2.28x faster than YOLO11s on this CPU sample. YOLO11s returned 4.2% more boxes per frame, with the same no-person-frame rate. These are unannotated sample counts, not accuracy/recall measurements; recheck throughput on the Kaggle GPU.

Per-frame measurements: `data/detection/yolo11_benchmark.csv`.
