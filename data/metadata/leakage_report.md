# Dataset duplicate and leakage check

- Datasets compared: hockeyfights, rlvs, rwf2000
- Video records compared: 5000
- Pairs with exact SHA-256 match: 28
- Exact matches between different datasets: 0
- Exact RWF train-to-official-val matches: 6
- Exact duplicate pairs within RLVS: 14
- Exact duplicate pairs within Hockey Fights: 3
- Exact duplicate pairs within an RWF source split: 5
- Exact duplicate pairs with conflicting labels: 1
- Pairs with at least two shared sampled-frame hashes (review candidates): 9
- Manually confirmed same-source pairs: 3

Exact SHA-256 matches are byte-identical files. Sampled-frame matches are candidate duplicates only; common static backgrounds can collide. Review those pairs visually before excluding data.

No exact SHA-256 matches were found between datasets in this scan. For the RWF internal train/test boundary, keep the official test record and exclude confirmed training copies; the RWF split builder applies that rule.
The report does not modify or delete source videos.

Pair details: `data/metadata/duplicate_pairs.csv`
