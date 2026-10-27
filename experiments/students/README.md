# Native source training and adaptation

The [student instructions](../students_cuhkx_3fold/README.md) describe the input manifest, source stages, checkpoint formats and commands.

`source_visual.py`, `source_motion.py`, `source_fusion.py` and `adapt_cuhkx.py` retain the CUHK-X training functions used by the paper. The RGB source modules contain the native MMAct/MM-Fit objectives and visual epoch routine. `runtime.py` connects the complete dataset-native models to explicitly supplied tensor caches.

The `train_source.py` entry runs source initialization; `scripts/distill_student.py` runs label-free adaptation from C0. Both expose their arguments through `--help`.
