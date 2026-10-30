# Dataset source, terms, and readiness notes

This file records what is known about the local dataset copies. A directory name or
third-party mirror is not proof that video redistribution is permitted. Keep source
videos out of Git; retain only metadata and split manifests.

## RWF-2000

- Local root: `data/raw/rwf2000/`
- Local structure: `train/{Fight,NonFight}` and `val/{Fight,NonFight}`.
- Observed: 2,000 videos, 1,600 in train and 400 in official validation; balanced labels.
- The checked-in notebook references the Kaggle distribution `vulamnguyen/rwf2000`.
- Source paper: [RWF-2000 paper](https://arxiv.org/abs/1911.05913).
- The Kaggle listing is [here](https://www.kaggle.com/datasets/vulamnguyen/rwf2000), but
  Kaggle listings carry dataset-specific self-reported terms. The exact local download
  provenance and applicable license have not been confirmed.
- No license or terms file was present alongside the local videos. Confirm the terms
  associated with the exact distribution before sharing any video or derived dataset.

## Real-Life Violence Situations (RLVS)

- Local root: `data/raw/rlvs/`
- Local structure: `Violence/` and `NonViolence/`, 1,000 files per class.
- The NonViolence directory contains both AVI and MP4 files.
- The original source of this local copy is not recorded in the repository, and no
  license or terms file was present beside the videos. Record the exact download page
  and review its terms before redistributing any videos.
- A commonly used distribution is [the Kaggle RLVS listing](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset);
  its terms cannot be attributed to this local copy without confirming that it was the
  source used for this download.
- Local exact-hash checks found duplicate files, including one identical video placed
  in both labels. See `leakage_report.md`; the generated evaluation manifest excludes
  duplicate copies and the conflicting-label pair.

## Hockey Fights

- Local root: `data/raw/hockeyfights/`
- Local structure: flat directory; original-style filename prefixes `fi` (fight) and
  `no` (non-fight) are used to infer labels. The current inventory has 1,000 AVI files,
  expected to be 500 per class; see `hockeyfights.csv` for inspected counts.
- The local download source was not recorded. The dataset authors' [VisiLab project
  page](https://visilab.etsii.uclm.es/?page_id=1397) identifies the collection, while the
  [Academic Torrents entry](https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89)
  describes a user-hosted copy and lists no license terms. These references identify the
  dataset but do not establish redistribution rights for this exact local copy.
- Do not distribute the videos. Record the exact source and review its terms before
  publishing or sharing raw clips. The dataset contains NHL broadcast footage, so
  downstream video rights should not be assumed from the academic paper's availability.

## Optional third evaluation set: VID

- Selected candidate: [VID: A Comprehensive Dataset for Violence Detection in Various
  Contexts](https://doi.org/10.7910/DVN/N4LNZD), hosted by Harvard Dataverse.
- It adds domestic, street, classroom, and other violence contexts; the paper describes
  raw video and pose-estimated/masked variants and states that the data are available
  for academic, research, or instructional use.
- The paper is published under CC BY-NC, but that article license is not automatically
  the dataset license. Verify the dataset record's own terms before use or redistribution.
- Download was not completed: the required network access request to Harvard Dataverse
  was rejected, so no dataset files were fetched.
