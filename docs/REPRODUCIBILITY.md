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

## Commands

The [home page](../README.md#usage) provides the target, distillation and evaluation commands. Source student stages and the KD-soft, confidence PL and Tent interfaces are documented beside their code. Teacher commands support an explicit local workspace, data root, pretrained root and external backbone root.

## Challenge inference

The [challenge runtime](../competition/README.md) rebuilds inputs from raw clips in a fresh temporary workspace. Supply your own deployment bundle and data, then check output row identities and ordering against the input manifest.
