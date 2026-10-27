# Paper analysis and figures

Install `pip install -e ".[analysis]"` from the repository root. All inputs and generated figures stay in your local storage outside the source directory.

| Script | Output |
|---|---|
| `analysis_stats.py` | Aggregate statistics used by Figures 3 and 5–7 |
| `fig3_folds.py` | Figure 3, CUHK-X teacher targets per fold |
| `fig4_gains.py` | Figure 4, target gains and confidence intervals |
| `fig5_7_analysis.py` | Figures 5–7, scatter, lollipop and stacked bars |
| `analysis_pairs.py` | Session co-occurrence and confusion counts used in the text |

Every script supports `--help`. The commands below run from the repository root.

## Teacher prediction statistics

The local prediction NPZ contains `evaluation_ids`, `evaluation_clusters`, `evaluation_outer_fold`, and integer predictions keyed by `evaluation_prediction__METHOD`. Required methods are `direct`, `equal_mean`, `group`, `sequence`, `direct_sequence`, `direct_viterbi_naive` and `online_forward`. `sequence` denotes D→G→S. The selected teacher uses this path on fold 0 and D→S on folds 1 and 2.

When `evaluation_prediction__selected` is present, analysis uses that source-selected path directly. The metadata CSV has `sample_id`, `class_id`, `user_id`. The class manifest CSV has `class_id`, `class_label`, with the dataset's 40-class ordering.

```bash
python experiments/analysis/analysis_stats.py ../private-cache/cuhkx/frozen_predictions.npz ../private-cache/cuhkx/metadata.csv ../private-cache/cuhkx/classes.csv ../prism-output/analysis.json
python experiments/analysis/fig3_folds.py ../prism-output/analysis.json ../prism-output/figures
python experiments/analysis/fig5_7_analysis.py ../prism-output/analysis.json ../prism-output/figures
python experiments/analysis/analysis_pairs.py ../private-cache/cuhkx/frozen_predictions.npz ../private-cache/cuhkx/metadata.csv ../private-cache/cuhkx/classes.csv
```

## Figure 4

The input is a locally computed CSV with columns `dataset`, `experiment`, `split`, `method`, `delta_macro_f1`, `macro_f1_ci95_lower`, `macro_f1_ci95_upper`. Values are fractions; the plot converts them to percentage points.

| Dataset | Experiment | Split | Selected method |
|---|---|---|---|
| CUHK-X | `E1R_clean23` | `all_2470` | `selected` from `E5_selected_path_completed`, split `all_2470` |
| MMAct | `E3_component_ablation` | `F0` | `direct_sequence` |
| MM-Fit | `E2_information_decomposition` | `TEST` | `direct_sequence` |

Method names follow `ROWS` in the plotting script. Confidence intervals come from the paired session/workout bootstrap in `prism/eval/bootstrap.py`.

```bash
python experiments/analysis/fig4_gains.py --metrics ../prism-output/metrics_long.csv --out ../prism-output/figures/fig4_gains_en.pdf
```

Figure 4 reads the supplied metrics and requires a selected-path row for each dataset. Font selection uses the local font registry. The scripts save vector PDF and PNG previews to the specified output location.
