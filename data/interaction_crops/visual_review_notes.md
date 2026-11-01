# Visual review notes

Reviewed a deterministic random sample of 50 RWF-2000 validation clips (seed 42). Each contact panel shows the start, middle, and final decoded sampled frame with person-track boxes and the generated crop/fallback boundary. This is a visual spot check at three times per clip, not frame-by-frame annotation.

- Clips reviewed: 50
- Clips with no persistent pair and full-frame fallback: 13
- Mean fallback rate in this sample: 42.1%
- The crop boundary generally covered the selected tracked people in sampled frames. Full-frame fallbacks were visible in clips where no stable pair was found or the selected tracks were absent at that time.
- Crowded scenes sometimes expand to many track IDs and broad crops; this is noted per clip in the CSV for follow-up after classifier evaluation.

Detailed checklist: `data/interaction_crops/visual_review_notes.csv`. Overlay images: `cache/interaction_crops/validation_review/`.
