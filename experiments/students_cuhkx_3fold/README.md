# Paper students

Install the `student` extra. Prepare the dataset, pretrained MC3 weights, source supervision and native tensor caches locally. The complete visual, skeleton, IMU and fusion models are in `prism/student/`.

## Distillation

```bash
python scripts/distill_student.py --dataset cuhkx --fold 0 --inputs "$PRISM_CACHE_ROOT/student_fold0.json" --targets "$PRISM_OUTPUT_ROOT/targets_fold0/targets.json" --c0 "$PRISM_CACHE_ROOT/c0_fold0.pt" --output "$PRISM_OUTPUT_ROOT/student_fold0"
python experiments/students_cuhkx_3fold/train.py --fold 1 --inputs "$PRISM_CACHE_ROOT/student_fold1.json" --targets "$PRISM_OUTPUT_ROOT/targets_fold1/targets.json" --c0 "$PRISM_CACHE_ROOT/c0_fold1.pt" --output "$PRISM_OUTPUT_ROOT/student_fold1"
python scripts/distill_student.py --dataset mmact --inputs "$PRISM_CACHE_ROOT/student_mmact.json" --targets "$PRISM_OUTPUT_ROOT/targets_mmact/targets.json" --c0 "$PRISM_CACHE_ROOT/c0_mmact.pt" --output "$PRISM_OUTPUT_ROOT/student_mmact"
python scripts/distill_student.py --dataset mmfit --inputs "$PRISM_CACHE_ROOT/student_mmfit.json" --targets "$PRISM_OUTPUT_ROOT/targets_mmfit/targets.json" --c0 "$PRISM_CACHE_ROOT/c0_mmfit.pt" --output "$PRISM_OUTPUT_ROOT/student_mmfit"
```

CUHK-X fold `2` denotes H2. Supply each fold's source-trained C0 and selected target file. Fold 0 uses the full chain; folds 1 and 2 use D→S. All folds use the original 40-epoch training function. MMAct and MM-Fit use their dataset-native models and training settings.

The default adaptation batch is 64, with AdamW, weight decay 0.02, motion/fusion learning rate `1e-4`, visual-head learning rate `5e-5`, minimum learning rate `5e-6`, temperature 1 and gradient clipping at 1. Seeds are `20260826`, `20260922` and `20260927` for CUHK-X, MMAct and MM-Fit. The MC3 backbone remains frozen. Outputs include the trained checkpoint and label-free predictions.

## Input files

The input JSON uses `schema: "prism.student.v1"`, `dataset`, unique `record_id`, and a `tensors` map. CUHK-X also requires matching `fold` and `subject_id`. Each tensor entry is `{"path":"local_array.npy"}` or `{"path":"local_archive.npz","key":"array_name"}`. Relative paths resolve beside the manifest, or under `--cache-root` / `PRISM_CACHE_ROOT`. Optional `indices` maps manifest rows to existing cache rows. See [tensor shapes](../../docs/DATA.md).

Distillation reads `record_id` and `targets[N,C]` from the target construction output. It does not open source teacher tensors or ground-truth labels. Checkpoints with `model_state`, `model`, or `state_dict` are loaded with strict key and shape matching.

## Source initialization

Use `experiments/students/train_source.py` for the `visual`, `motion` and `fusion` stages:

```bash
python experiments/students/train_source.py --dataset cuhkx --fold 0 --stage visual --inputs "$PRISM_CACHE_ROOT/source_pixels_fold0.json" --labels "$PRISM_CACHE_ROOT/source_labels_fold0.json" --pretrained-weights "$PRISM_CACHE_ROOT/mc3_18_kinetics.pth" --output "$PRISM_OUTPUT_ROOT/c0_fold0_visual"
python experiments/students/train_source.py --dataset cuhkx --fold 0 --stage motion --inputs "$PRISM_CACHE_ROOT/source_motion_fold0.json" --labels "$PRISM_CACHE_ROOT/source_labels_fold0.json" --output "$PRISM_OUTPUT_ROOT/c0_fold0_motion"
python experiments/students/train_source.py --dataset cuhkx --fold 0 --stage fusion --inputs "$PRISM_CACHE_ROOT/source_sequences_fold0.json" --labels "$PRISM_CACHE_ROOT/source_labels_fold0.json" --visual-checkpoint "$PRISM_OUTPUT_ROOT/c0_fold0_visual/final.pt" --motion-checkpoint "$PRISM_OUTPUT_ROOT/c0_fold0_motion/final.pt" --output "$PRISM_OUTPUT_ROOT/c0_fold0"
```

Source labels are supplied separately as `{"record_id":[...],"label":[...]}`. Source manifests may additionally reference `teacher_logits`, `teacher_features`, `imu_teacher_logits`, `imu_teacher_valid` and cached `anchor_logits`. These contain source-fitted teacher supervision. For CUHK-X, source logits are the log of the mean clean inner-OOF posterior. Visual/fusion teacher features have shape `[N,2,3,1024]`; motion training averages the three views to `[N,2,1024]`.

The visual stage reads raw uint8 images: CUHK-X `[N,2,16,3,H,W]`, RGB datasets `[N,2,16,3,3,H,W]`. Fusion and adaptation read the supplied post-MC3 sequence cache. Cache construction must use the matching visual source checkpoint.

MMAct source training additionally requires `--validation-inputs` and `--validation-labels` for its F1 selection split. MM-Fit fusion uses the same options for its validation split. RGB source logits use `2*log(Direct_probability)`; source teacher features use the dataset-native 1024-dimensional descriptors. MM-Fit visual source manifests include a `workout` string for each record, preserving the original stride-five sampling within each workout.

| Dataset | Source seed | Visual | Motion | Fusion |
|---|---:|---:|---:|---|
| CUHK-X | 20260811 | 16 epochs | 24 epochs | 4 + 20 epochs |
| MMAct | 20260922 | 16 epochs | 24 epochs | 4 + 20 epochs |
| MM-Fit | 20260927 | 2 epochs | 24 epochs | 24 epochs |

Source stages save `final.pt`; RGB stages with selection data also save `best.pt`. Supply the selected fusion checkpoint as the common C0 for compared adaptation methods.

## Frozen student evaluation

Evaluate the saved logits with separate labels. Supply all three CUHK-X folds together to compute pooled macro-F1:

```bash
python scripts/evaluate_student.py --dataset cuhkx --predictions ../prism-output/cuhkx/fold0/student/predictions.npz ../prism-output/cuhkx/fold1/student/predictions.npz ../prism-output/cuhkx/fold2/student/predictions.npz --labels ../private-cache/cuhkx/evaluation_labels.json --output ../prism-output/cuhkx/student_metrics.json
```

For MMAct and MM-Fit, use the matching dataset name and one prediction file. Label JSON contains aligned `record_id` and `label`; identity sets are checked before scoring.
