from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from audit_yolo11_pose_skeleton import frame_map
from build_shared_dir_roi_feature_cache import read_ir
from build_videomae_cache import safe_relative, square_crop
from build_visual_pixel_cache import (
    VIEW_NAMES,
    WINDOW_BOUNDS,
    WINDOW_NAMES,
    create_or_open,
    window_indices,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data/test_union_manifest.csv"
DEFAULT_ROI = PROJECT_DIR / "runs/dir_multiscale_roi_test"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/test_pixel_cache_t16_r160"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a label-free 405-row challenge Test pixel cache. Unreadable IR is "
            "represented explicitly by an all-false view mask; no Large fallback."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roi-run", type=Path, default=DEFAULT_ROI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    source.sort(key=lambda row: row["official_sample_id"])
    if len(source) != 405:
        raise RuntimeError(f"official Test universe changed: {len(source)}")
    if len({row["official_sample_id"] for row in source}) != len(source):
        raise RuntimeError("duplicate official Test sample_id")
    return source


def cache_rows(source: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "sample_id": row["official_sample_id"],
            "source_id": row["sample_id"],
            "user_id": "anonymous",
            "class_id": "-1",
        }
        for row in source
    ]


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.frames < 2 or args.resolution < 64:
        raise ValueError("frames must be >=2 and resolution must be >=64")
    source_rows = read_manifest(args.manifest.resolve())
    if args.max_trials > 0:
        source_rows = source_rows[: args.max_trials]
    rows = cache_rows(source_rows)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    images, completed, valid_out, quality_out, source_index_out = create_or_open(
        output, rows, args.frames, args.resolution, args.overwrite
    )
    roi_run = args.roi_run.resolve()
    started = time.perf_counter()
    built = skipped = missing_visual = 0

    for row_index, (source, row) in enumerate(zip(source_rows, rows, strict=True)):
        if bool(completed[row_index]) and bool(np.all(source_index_out[row_index] >= 0)):
            skipped += 1
            continue
        roi_path = (
            roi_run
            / "trial_roi_cache"
            / safe_relative(source["sample_id"]).with_suffix(".npz")
        )
        if roi_path.is_file():
            with np.load(roi_path, allow_pickle=False) as data:
                frame_ids = np.asarray(data["frame_ids"]).astype(str)
                region_names = tuple(np.asarray(data["region_names"]).astype(str))
                boxes = np.asarray(data["roi_boxes_xyxy"], dtype=np.float32)
                roi_valid = np.asarray(data["roi_valid"], dtype=bool)
                roi_quality = np.asarray(data["roi_quality"], dtype=np.float32)
            person_index = region_names.index("full_body")
            workspace_index = region_names.index("hand_workspace")
        else:
            # The four known IR-corrupt recordings still have a complete Skeleton
            # timeline. It defines the shared visual/motion grid while pixels and
            # view masks stay exactly zero.
            skeleton_frames = frame_map(Path(source["skeleton_path"]), "skeleton")
            frame_ids = np.asarray(sorted(skeleton_frames), dtype=str)
            if not len(frame_ids):
                raise RuntimeError(f"no Skeleton frames for {row['sample_id']}")
            boxes = np.empty((len(frame_ids), 0, 4), dtype=np.float32)
            roi_valid = np.empty((len(frame_ids), 0), dtype=bool)
            roi_quality = np.empty((len(frame_ids), 0), dtype=np.float32)
            person_index = workspace_index = -1

        chosen_windows = [
            window_indices(len(frame_ids), low, high, args.frames)
            for low, high in WINDOW_BOUNDS
        ]
        source_index_out[row_index] = np.stack(chosen_windows).astype(np.int32)
        if not roi_path.is_file():
            images[row_index] = 0
            valid_out[row_index] = 0
            quality_out[row_index] = 0
            completed[row_index] = 1
            missing_visual += 1
            built += 1
            continue

        ir_paths = frame_map(Path(source["ir_path"]), "ir")
        unique_indices = sorted(set(np.concatenate(chosen_windows).tolist()))
        frame_images: dict[int, np.ndarray] = {}
        for frame_index in unique_indices:
            frame_id = frame_ids[frame_index]
            if frame_id not in ir_paths:
                raise RuntimeError(f"missing IR frame {frame_id}: {row['sample_id']}")
            frame_images[frame_index] = read_ir(ir_paths[frame_id])[:, :, 0]

        for window_index, chosen in enumerate(chosen_windows):
            for time_index, frame_index in enumerate(chosen):
                image = frame_images[int(frame_index)]
                person_valid = bool(roi_valid[frame_index, person_index])
                workspace_valid = bool(roi_valid[frame_index, workspace_index])
                person_box = (
                    boxes[frame_index, person_index]
                    if person_valid
                    else np.full(4, np.nan, dtype=np.float32)
                )
                workspace_box = (
                    boxes[frame_index, workspace_index] if workspace_valid else person_box
                )
                crops = (
                    image,
                    square_crop(image, person_box, scale=1.15),
                    square_crop(image, workspace_box, scale=1.40),
                )
                for view_index, crop in enumerate(crops):
                    images[row_index, window_index, time_index, view_index] = cv2.resize(
                        crop,
                        (args.resolution, args.resolution),
                        interpolation=cv2.INTER_AREA,
                    )
                valid_out[row_index, window_index, time_index] = (
                    1,
                    int(person_valid),
                    int(workspace_valid or person_valid),
                )
                quality_out[row_index, window_index, time_index] = (
                    1.0,
                    roi_quality[frame_index, person_index] if person_valid else 0.25,
                    roi_quality[frame_index, workspace_index]
                    if workspace_valid
                    else (
                        roi_quality[frame_index, person_index] * 0.5
                        if person_valid
                        else 0.20
                    ),
                )
        completed[row_index] = 1
        built += 1
        if built % 25 == 0 or row_index + 1 == len(rows):
            for array in (images, completed, valid_out, quality_out, source_index_out):
                array.flush()
            print(
                json.dumps(
                    {
                        "processed": row_index + 1,
                        "total": len(rows),
                        "built": built,
                        "skipped": skipped,
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )

    for array in (images, completed, valid_out, quality_out, source_index_out):
        array.flush()
    visual_available = np.asarray(valid_out, dtype=bool).any(axis=(1, 2, 3))
    if not np.asarray(completed, dtype=bool).all():
        raise RuntimeError("challenge Test pixel cache is incomplete")
    summary = {
        "stage": "S_label_free_test_pixel_cache",
        "trials": len(rows),
        "windows": list(WINDOW_NAMES),
        "window_bounds": WINDOW_BOUNDS,
        "frames_per_window": args.frames,
        "views": list(VIEW_NAMES),
        "resolution": args.resolution,
        "shape": list(images.shape),
        "visual_available": int(visual_available.sum()),
        "visual_missing": int((~visual_available).sum()),
        "visual_missing_sample_ids": [
            rows[index]["sample_id"] for index in np.flatnonzero(~visual_available)
        ],
        "missing_visual_policy": (
            "all-zero pixels and quality plus all-false view mask; Skeleton timeline "
            "defines source indices; no Large-teacher or legacy-prediction fallback"
        ),
        "labels_read": False,
        "large_videomae_input": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
