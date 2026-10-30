# Local dataset verification

Verification was run against the files currently present in `data/raw/`.

| Dataset | Files | Class counts | Readability check |
|---|---:|---|---|
| RWF-2000 | 2,000 | Fight 1,000 (train 800, official val 200); NonFight 1,000 (train 800, official val 200) | All 2,000 opened and yielded a first frame; sampled-frame fingerprinting ran. |
| RLVS | 2,000 | Violence 1,000; NonViolence 1,000 | All 2,000 opened and yielded a first frame; sampled-frame fingerprinting ran. |
| Hockey Fights | 1,000 | Fight 500; NonFight 500 | All 1,000 opened and yielded a first frame; sampled-frame fingerprinting ran. |

The decode check records whether OpenCV could open each clip and read its first frame;
sampled frames are also used to build the perceptual fingerprint. This is not an
exhaustive decode of every frame. OpenCV emitted an H.264 macroblock warning while
scanning RLVS, but the affected clip still passed the current open/first-frame check.
Re-run a full sequential decode audit before final model evaluation if strict media
integrity is required.

RWF counts agree with the expected 1,600/400 train/official-validation layout and
balanced classes. RLVS counts agree with the 1,000/1,000 class-directory layout.
Hockey Fights counts agree with the expected balanced 1,000-video collection. Its
evaluation manifest removes three byte-identical duplicate copies; see
`data/splits/hockeyfights_eval_exclusions.csv`.
