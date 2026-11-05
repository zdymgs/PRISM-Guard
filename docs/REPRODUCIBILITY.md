# Reproduction workflow

Run commands from the repository root with Python 3.12. Prepare data and pretrained weights from their providers and store generated artifacts in a separate workspace.

## Order of operations

1. Use [the teacher entry points](../experiments/teachers/README.md) to prepare clip features and source-fitted expert banks.
2. Export aligned source, selection and unlabeled target bundles in the [bank schema](DATA.md). Preserve expert order and original array dtype.
3. Run `scripts/build_targets.py --path auto` with the matching dataset configuration and storage paths. Source/selection labels fit and select the path and gate.
4. Train the source student or provide its C0 checkpoint, then run `scripts/distill_student.py` on the frozen targets. CUHK-X uses a distinct source model and expert bank for each outer fold.
5. Read evaluation labels after predictions are frozen. Pool CUHK-X clip predictions across all three folds before computing macro-F1.
6. Run the [formal analysis scripts](../experiments/analysis/README.md) on your aggregate predictions and metrics.

## Configuration and artifacts

`configs/cuhkx.yaml`, `configs/mmact.yaml` and `configs/mmfit.yaml` record the dataset-specific decoder and gate settings. CLI path overrides affect storage, while the method settings stay in the configuration and native numerical kernels.

Target construction writes a canonical configuration snapshot and SHA256 hashes. `scripts/evaluate.py` verifies them before evaluation. Use the same path overrides for construction and evaluation and a fresh output directory for each run.

The paired bootstrap implementation in `prism/eval/bootstrap.py` uses the paper's session/workout resampling unit, 2,000 repetitions and seed 20261015. Generated arrays, checkpoints, predictions and figures remain in your output workspace.

## Paper settings

The supplementary material of the paper refers to these keys.

| Setting | File and key | Value |
|---|---|---|
| eta | `configs/{cuhkx,mmact,mmfit}.yaml`: `eta` | 0.0005 |
| stacker_C | `configs/cuhkx.yaml`: `stacker_C`; MMAct / MM-Fit | 0.03; — / — |
| CUHK-X emission weight | `configs/cuhkx.yaml`: `emission_alpha` | 0.65 |
| MMAct emission weights | `configs/mmact.yaml`: `direct_sequence_emission_alpha` (D→S), `group_sequence_emission_alpha` (D→G→S) | 1.0; 0.65 |
| MM-Fit emission weight | `configs/mmfit.yaml`: `emission_scale` | 0.2 |
| transition_lambda | `configs/{cuhkx,mmact,mmfit}.yaml`: `transition_lambda` | CUHK-X: 0.45; MMAct: 0.45; MM-Fit: 0.1 |
| beam_width | `configs/cuhkx.yaml`: `beam_width`; MMAct / MM-Fit | 50; — / — |
| session_gap_seconds | `configs/cuhkx.yaml`: `session_gap_seconds`; MMAct / MM-Fit | 30; — / — |
| CUHK-X peer similarity threshold | `configs/cuhkx.yaml`: `repeat_graph.minimum_probability_similarity` | 0.75 |
| MMAct peer similarity threshold | `configs/mmact.yaml`: `match_threshold` | 0.55 |
| MM-Fit peer similarity threshold | `configs/mmfit.yaml`: `bout_peers.threshold` | 0.85 |
| Skip cost | `configs/cuhkx.yaml`: `repeat_graph.alignment_gap_penalty`; `configs/mmact.yaml`: `skip_penalty`; MM-Fit | 0.20; 0.20; — |
| Group gate grid | `configs/cuhkx.yaml`: `group_gate_grid.{start,stop,count}` | -0.10; 0.90; 201 |
| Sequence gate grid | `configs/cuhkx.yaml`: `sequence_gate_grid.{start,stop,count}` | -1.0; 1.0; 401 |
| KD-soft optimizer and weight decay | `configs/baselines/kd_soft.yaml`: `optimizer`, `weight_decay` | AdamW; 0.02 |
| KD-soft learning rates | `configs/baselines/kd_soft.yaml`: `motion_fusion_lr`, `visual_head_lr`, `minimum_lr` | 0.0001; 5.0e-05; 5.0e-06 |
| KD-soft training | `configs/baselines/kd_soft.yaml`: `batch_size`, `epochs`, `temperature`, `scope` | 64; 40; 1; `heads_motion_encoder` |
| KD-soft target and loss | `configs/baselines/kd_soft.yaml`: `target`, `loss` | `DIRECT_content_teacher_soft_posterior`; `forward_KL_no_smoothing` |
| KD-soft precision | `configs/baselines/kd_soft.yaml`: `cuhkx_precision`, `other_precision` | `AMP_float16`; `float32` |
| Confidence PL threshold | `configs/baselines/conf_pl.yaml`: `confidence_threshold` | 0.9 |
| bootstrap_repetitions | `configs/{cuhkx,mmact,mmfit}.yaml`: `bootstrap_repetitions` | 2000 |
| bootstrap_seed | `configs/{cuhkx,mmact,mmfit}.yaml`: `bootstrap_seed` | 20261015 |

## Commands

The [home page](../README.md#usage) provides the target, distillation and evaluation commands. Source student stages and the KD-soft and confidence PL interfaces are documented beside their code. Teacher commands support an explicit local workspace, data root, pretrained root and external backbone root.

## Challenge inference

The [challenge runtime](../competition/README.md) rebuilds inputs from raw clips in a fresh temporary workspace. Supply your own deployment bundle and data, then check output row identities and ordering against the input manifest.
