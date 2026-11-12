# Decoder sensitivity (paper Table V)

One decoder constant is varied at a time around its final value; for each setting the whole chain, including threshold and path selection on labeled data, is rerun and the target is scored on the evaluation data. CPU only; no neural network is trained.

| Script | Dataset | Constants varied |
|---|---|---|
| `cuhkx.py` | CUHK-X (3 outer folds) | λ, trigram back-off, session gap |
| `mmact.py` | MMAct (F0) | λ, transition smoothing |
| `mmfit.py` | MM-Fit (test) | λ, transition smoothing, session gap |

Each script first reproduces the paper numbers with the final constants (CUHK-X 74.97 / 79.95, MMAct 88.5 / 99.9 on the official cross-subject split, MM-Fit 94.6 / 94.0, Direct / PRISM macro-F1). Run `python experiments/sensitivity/<script>.py --help` for the input paths. Per-setting results are in `results/`.
