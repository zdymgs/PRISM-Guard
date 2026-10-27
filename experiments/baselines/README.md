# Student baselines

Install the `baselines` extra. These entry points use the same native model, C0 and unlabeled clip batch as the Direct-distilled student.

```bash
python experiments/baselines/run.py --dataset cuhkx --fold 0 --method kd-soft --inputs "$PRISM_CACHE_ROOT/student_fold0.json" --targets "$PRISM_CACHE_ROOT/direct_posteriors_fold0.json" --c0 "$PRISM_CACHE_ROOT/c0_fold0.pt" --output "$PRISM_OUTPUT_ROOT/kd_soft_fold0"
python experiments/baselines/run.py --dataset mmact --method confidence-pl --inputs "$PRISM_CACHE_ROOT/student_mmact.json" --targets "$PRISM_CACHE_ROOT/direct_posteriors_mmact.json" --c0 "$PRISM_CACHE_ROOT/c0_mmact.pt" --output "$PRISM_OUTPUT_ROOT/confidence_pl_mmact"
python experiments/baselines/run.py --dataset cuhkx --fold 2 --method tent --inputs "$PRISM_CACHE_ROOT/student_pixels_fold2.json" --targets "$PRISM_CACHE_ROOT/direct_posteriors_fold2.json" --c0 "$PRISM_CACHE_ROOT/c0_fold2.pt" --output "$PRISM_OUTPUT_ROOT/tent_fold2"
```

The posterior file contains aligned `record_id` and **unsmoothed** `direct_probability[N,C]`. KD-soft fits this distribution with forward KL. Confidence pseudo-labeling uses cross-entropy on records whose maximum Direct probability is at least `0.9`. Evaluation predictions cover the complete batch. Both run for 40 epochs using the [student settings](../students_cuhkx_3fold/README.md).

Tent is the reported CUHK-X baseline: one epoch of entropy minimization with Adam at `1e-3`, updating BN/LN affine parameters. It reads raw pixels and full motion tensors; normalization uses microbatches of four with gradients accumulated to a logical batch of 64. Its visual backbone uses checkpointed blocks and full-precision computation.

Each command requires an empty output directory and writes the model, training log and unlabeled predictions there. Source labels are not read by these baselines.
