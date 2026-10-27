# Local data interfaces

Obtain the datasets from their official providers and prepare pretrained weights locally. Store datasets, caches, posterior banks and outputs outside this repository.

## Expert banks for target construction

Each JSON bundle has aligned `record_id` and an `expert_bank` array of shape `[N, E, C]`. Set `expert_bank_dtype` to the original array dtype (`float32` or `float64`). Keep expert order identical across source, selection and target bundles.

| Dataset | Classes | Paper bank size | Additional fields |
|---|---:|---:|---|
| CUHK-X | 40 | 23 experts | `users`, `dates`, `starts` |
| MMAct | 37 | 15 slots | `expert_names`, `availability`, `geometry`, `subject`, `scene`, `session`, `start_ms`, `end_ms`, `ordinal` |
| MM-Fit | 11 | 6 experts | `workout`, `time`, `descriptor` |

`source.json` additionally contains source `label`; CUHK-X also supplies its existing three-way `inner_fold` partition. MMAct and MM-Fit supply `selection.json` with selection `label`. The target `posterior.json` contains no labels. CUHK-X `starts` are seconds; MMAct timing fields are milliseconds.

For MMAct, retain all 15 bank slots and their availability masks; the configured Direct mixture and relational features use the same bank. MM-Fit expert order is `rgb`, `pose`, `imu`, `video_mae_large`, `video_jepa`, `joint`.

For a Direct-only input, use `record_id` and `direct_probability[N,C]`, with optional `session` and `position`. Relational construction uses the full expert bank and its protocol fields.

## Storage and command options

By default, the target CLI reads `source.json`, `selection.json` and `posterior.json` from its configured cache root. Set paths with `--source-input`, `--selection-input`, `--posterior-input`, `--cache-root`, `--data-root` and `--output-root`, or the root environment variables `PRISM_DATA_ROOT`, `PRISM_CACHE_ROOT`, `PRISM_OUTPUT_ROOT`.

`build_targets.py` writes `targets.json`, `target_freeze.json` and the configuration snapshot to the output directory. `targets.json` contains aligned `record_id`, `prediction`, `targets[N,C]`, `class_count` and `clusters`. `targets` is the smoothed distribution used for distillation.

## Student inputs

The student CLI accepts a `prism.student.v1` JSON manifest with `dataset`, aligned `record_id` and a `tensors` map. CUHK-X requires `fold` and `subject_id`; MM-Fit source visual training also requires per-record `workout` for its original stride-five sampling. Each tensor entry has a local `path`; NPZ entries also specify `key`. Paths are relative to the manifest or the chosen cache root. An optional `indices` array selects rows from an existing local cache.

The tensor first dimension is the clip count `N`; the shapes below omit it.

| Input | CUHK-X shape | MMAct / MM-Fit shape |
|---|---|---|
| `backbone_sequence` | `[2,3,16,512]` | `[2,3,16,512]` |
| `view_valid`, `view_quality` | `[2,16,3]` | `[2,3,16]` |
| `global_time_position` | `[2,16]` | `[2,16]` |
| `skeleton_features`, `skeleton_feature_mask` | `[2,16,17,13]` | `[2,16,17,13]` |
| `skeleton_joint_mask` | `[2,16,17]` | `[2,16,17]` |
| `skeleton_relations`, `skeleton_relation_mask` | `[2,16,18]` | `[2,16,18]` |
| `skeleton_frame_quality` | `[2,16]` | `[2,16]` |
| `imu_sequences` | `[2,16,5,4,16]` | `[2,16,5,4,6]` |
| `imu_sequence_mask` | `[2,16,5,4]` | `[2,16,5,4,6]` |
| `imu_bin_statistics` | `[2,16,5,52]` | `[2,16,5,30]` |
| `imu_bin_mask` | `[2,16,5]` | `[2,16,5,4]` |
| `imu_global_statistics` | `[5,48]` | `[2,5,30]` |
| `imu_global_mask` | `[5,2]` | `[2,5]` |

See [student instructions](../experiments/students_cuhkx_3fold/README.md) for source training, C0 checkpoints and raw-image inputs used by the baseline.

## Evaluation-only labels

Evaluation labels use a separate JSON object with aligned `record_id` and integer `label`. Reference predictions use `record_id` and `prediction`. The evaluation command reads these after target construction and freezing.

Analysis scripts consume locally generated clip metadata and predictions in the schemas described in [analysis instructions](../experiments/analysis/README.md).
