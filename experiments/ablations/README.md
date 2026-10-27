# Formal component controls

These entries construct the paper's prespecified controls from your normalized expert banks, then evaluate the frozen predictions for Figure 4. Data and pretrained weights are prepared separately.

## Build predictions

Use the normalized JSON schema in [the reproduction guide](../../docs/REPRODUCIBILITY.md). Arrays are JSON lists; `expert_bank_dtype` records the original bank dtype. CUHK-X source inputs additionally provide `inner_fold`, `users`, `dates`, and `starts`. MMAct inputs provide availability, normalized geometry, and interval metadata; MM-Fit inputs provide workout, time, and descriptor arrays.

```bash
python experiments/ablations/build.py --config configs/cuhkx.yaml \
  --source-input ../private-cache/fold0_source.json \
  --posterior-input ../private-cache/fold0_target.json \
  --fold 0 --output ../prism-output/controls_fold0
```

Repeat for outer folds 1 and 2 with their corresponding source/target banks. For MMAct and MM-Fit, use the matching config and add `--selection-input` with a separate labeled selection bank.

Construction reads only source/selection labels. The target JSON accepts only the normalized label-free schema. Original main-path selection runs first; controls do not participate in choosing the selected path. Outputs are `frozen_predictions.npz`, `selection.json`, and `PRE_EVALUATION_FREEZE.json` in a new external directory.

## Evaluate and plot

Evaluation labels use `{"record_id": [...], "label": [...]}`. Evaluate the three CUHK-X outer folds together:

```bash
python experiments/ablations/evaluate.py --dataset CUHK-X \
  --predictions ../prism-output/controls_fold0/frozen_predictions.npz \
                ../prism-output/controls_fold1/frozen_predictions.npz \
                ../prism-output/controls_fold2/frozen_predictions.npz \
  --labels ../private-cache/evaluation_labels.json \
  --output ../prism-output/controls_evaluation
```

Use one prediction file for MMAct or MM-Fit. The evaluator verifies construction hashes before reading labels and computes paired session/workout bootstrap metrics with 2,000 repetitions and seed 20261015. It exports `metrics_long.csv` and a label-free `pooled_predictions.npz`; the latter is also the input for CUHK-X analysis scripts.

Combine the three datasets' metric CSV files with a single header, then run:

```bash
python experiments/analysis/fig4_gains.py \
  --metrics ../prism-output/metrics_long.csv \
  --out ../prism-output/fig4.pdf
```

All calculated controls and five random-peer seeds are retained. MMAct's formal component table uses `direct_sequence` for its ungated first-order MAP path and `direct_sequence_gated` for the selected-margin version; the main PRISM API keeps its existing path names. `direct_sequence_nogate` provides the first-order decoder without a gate on MMAct and MM-Fit.
