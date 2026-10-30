# Detection and tracking statistics

Cache directory: `cache/detections/yolo11n-smoke`
Cache files included: 2

Statistics are calculated over processed frames. If a cache used a frame stride greater than one, the rates apply to sampled frames.

```text
dataset split label_name  clips  frames_processed  person_detections  mean_people_per_frame  median_people_per_frame  p95_people_per_frame  no_detection_frames  no_detection_frame_rate  clips_with_no_detections  no_detection_clip_rate  track_instances  mean_track_instances_per_clip
rwf2000  test        all      2               300               1870               6.233333                      7.0                 10.05                    0                      0.0                         0                     0.0               59                           29.5
rwf2000  test    violent      2               300               1870               6.233333                      7.0                 10.05                    0                      0.0                         0                     0.0               59                           29.5
```

`no_detection_frame_rate` is the fraction of processed frames with zero person boxes. `track_instances` sums unique nonnegative ByteTrack IDs within each clip; IDs reset for each clip.

CSV: `data/detection/detection_stats_smoke.csv`
