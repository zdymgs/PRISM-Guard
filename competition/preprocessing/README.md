# the challenge student raw-input preprocessing

The public raw-data entry point is:

```bash
python competition/student/runtime.py --help
```

It audits trial directories, builds a temporary manifest, extracts pose/ROI geometry, constructs visual and motion inputs, and invokes the fixed student. The original `student/modules/` layout is retained because the runtime loads builders dynamically and shares their alignment utilities.

## Stages

| Stage | Module in `student/modules/` | Output created locally |
|---|---|---|
| Automatic pose | `build_adaptive_yolo11_pose_skeleton_cache.py` | Per-trial pose geometry from YOLO11n-pose |
| Person/workspace regions | `build_multiscale_dir_rois.py` | Three-view crop geometry |
| Visual inputs | `build_test_pixel_cache.py` | Two windows × three views × 16 grayscale-replicated frames |
| Skeleton and IMU sources | `build_test_motion_source_cache.py` | Aligned raw motion features and masks |
| Motion windows | `build_motion_window_cache.py` | Motion bins aligned to selected visual times |
| Student forward | `../infer_student.py` | MC3 sequences, logits, and ordered predictions |

IR windows use normalized intervals `[0, 0.70]` and `[0.30, 1]`. The runtime keeps original frame timestamps, uses masks for unavailable inputs, and aligns skeleton and five IMU streams to the selected visual times. `Depth_Color` indices supply the common-frame contract. Full construction is illustrated in [the preprocessing diagram](../docs/figures/fig04_input_preprocessing.svg).

## Builder interfaces

For advanced use, each raw builder exposes its actual command-line options:

```bash
python competition/student/modules/build_adaptive_yolo11_pose_skeleton_cache.py --help
python competition/student/modules/build_multiscale_dir_rois.py --help
python competition/student/modules/build_test_pixel_cache.py --help
python competition/student/modules/build_test_motion_source_cache.py --help
python competition/student/modules/build_motion_window_cache.py --help
python competition/student/infer_student.py --help
```

Use `runtime.py` for complete raw-trial inference: its stage adapter preserves the timestamp and manifest handling needed by the fixed model. Intermediate arrays are generated in the user's temporary workspace and are not distributed with the code. `--keep-work` retains that workspace for local inspection.
