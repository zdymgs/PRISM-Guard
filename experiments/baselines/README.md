# Student baselines

Install the `baselines` extra. These entry points use the same native model, C0 and unlabeled clip batch as the Direct-distilled student.

```bash
python experiments/baselines/run.py --dataset cuhkx --fold 0 --method kd-soft --inputs "$PRISM_CACHE_ROOT/student_fold0.json" --targets "$PRISM_CACHE_ROOT/direct_posteriors_fold0.json" --c0 "$PRISM_CACHE_ROOT/c0_fold0.pt" --output "$PRISM_OUTPUT_ROOT/kd_soft_fold0"
python experiments/baselines/run.py --dataset mmact --method confidence-pl --inputs "$PRISM_CACHE_ROOT/student_mmact.json" --targets "$PRISM_CACHE_ROOT/direct_posteriors_mmact.json" --c0 "$PRISM_CACHE_ROOT/c0_mmact.pt" --output "$PRISM_OUTPUT_ROOT/confidence_pl_mmact"
```

The posterior file contains aligned `record_id` and **unsmoothed** `direct_probability[N,C]`. KD-soft fits this distribution with forward KL. Confidence pseudo-labeling uses cross-entropy on records whose maximum Direct probability is at least `0.9`. Evaluation predictions cover the complete batch. Both run for 40 epochs using the [student settings](../students_cuhkx_3fold/README.md).

Each command requires an empty output directory and writes the model, training log and unlabeled predictions there. Source labels are not read by these baselines.
