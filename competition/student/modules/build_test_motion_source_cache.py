from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from audit_yolo11_pose_skeleton import frame_map, load_raw_skeleton
from skeleton_imu_preprocessing import (
    COMMON_PART_NAMES,
    H36M_JOINT_NAMES,
    IMU_CHANNEL_NAMES,
    IMU_DEVICE_NAMES,
    SKELETON_FEATURE_NAMES,
    SKELETON_RELATION_NAMES,
    assign_points_to_frame_intervals,
    build_skeleton_features,
    frame_times_from_ids,
    load_full_imu_trial,
    safe_trial_path,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data/test_union_manifest.csv"
DEFAULT_ROI = PROJECT_DIR / "runs/dir_multiscale_roi_test"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/test_motion_source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build label-free compatible Skeleton/IMU source caches for all "
            "405 Test recordings, independently of IR availability."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roi-run", type=Path, default=DEFAULT_ROI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: row["official_sample_id"])
    if len(rows) != 405:
        raise RuntimeError(f"official Test universe changed: {len(rows)}")
    return rows


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_skeleton(
    row: dict[str, str], roi_run: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = frame_map(Path(row["skeleton_path"]), "skeleton")
    roi_path = (
        roi_run
        / "trial_roi_cache"
        / safe_trial_path(row["sample_id"]).with_suffix(".npz")
    )
    if roi_path.is_file():
        # Match the visual timeline exactly. One Test recording contains extra
        # Skeleton frames that are intentionally absent from the valid IR/roi axis.
        with np.load(roi_path, allow_pickle=False) as data:
            frame_ids = np.asarray(data["frame_ids"]).astype(str)
        missing = [frame_id for frame_id in frame_ids if frame_id not in paths]
        if missing:
            raise RuntimeError(
                f"roi frame missing from Skeleton: {row['official_sample_id']}/{missing[0]}"
            )
    else:
        frame_ids = np.asarray(sorted(paths), dtype=str)
    if not len(frame_ids):
        raise RuntimeError(f"empty Skeleton recording: {row['official_sample_id']}")
    values: list[np.ndarray] = []
    people: list[int] = []
    for frame_id in frame_ids:
        raw, count = load_raw_skeleton(paths[frame_id])
        values.append(raw)
        people.append(count)
    return frame_ids, np.stack(values).astype(np.float32), np.asarray(people, dtype=np.uint8)


def build_trial(row: dict[str, str], roi_run: Path, output: Path) -> dict[str, Any]:
    sample_id = row["sample_id"]
    frame_ids, skeleton_raw, skeleton_people = load_skeleton(row, roi_run)
    absolute_frame_times = frame_times_from_ids(frame_ids)
    frame_times = (absolute_frame_times - absolute_frame_times[0]).astype(np.float32)
    skeleton = build_skeleton_features(skeleton_raw, absolute_frame_times)
    skeleton_quality = np.asarray(skeleton["frame_quality"], dtype=np.float32)
    skeleton_quality *= (skeleton_people > 0).astype(np.float32)

    imu_text = row.get("imu_path", "").strip()
    imu_path = Path(imu_text) if imu_text else Path("__missing_imu_path__")
    by_device, imu_audit = load_full_imu_trial(imu_path)
    value_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    offsets = [0]
    interval_counts = np.zeros((len(frame_ids), len(IMU_DEVICE_NAMES)), dtype=np.uint32)
    device_mask = np.zeros(len(IMU_DEVICE_NAMES), dtype=bool)
    before_points = after_points = 0
    absolute_offsets: list[np.ndarray] = []

    for device_index, device in enumerate(IMU_DEVICE_NAMES):
        if device not in by_device:
            offsets.append(offsets[-1])
            continue
        point_times, values = by_device[device]
        frame_index = assign_points_to_frame_intervals(point_times, absolute_frame_times)
        counts = np.bincount(frame_index, minlength=len(frame_ids)).astype(np.uint32)
        if int(counts.sum()) != len(values):
            raise AssertionError(f"IMU interval assignment lost points: {sample_id}/{device}")
        interval_counts[:, device_index] = counts
        device_mask[device_index] = len(values) > 0
        value_parts.append(values.astype(np.float32))
        time_parts.append((point_times - absolute_frame_times[0]).astype(np.float32))
        frame_parts.append(frame_index.astype(np.int16))
        offsets.append(offsets[-1] + len(values))
        before_points += int((point_times < absolute_frame_times[0]).sum())
        after_points += int((point_times > absolute_frame_times[-1]).sum())
        absolute_offsets.append(np.abs(point_times - absolute_frame_times[frame_index]))

    if value_parts:
        imu_values = np.concatenate(value_parts)
        imu_times = np.concatenate(time_parts)
        imu_frame_index = np.concatenate(frame_parts)
        alignment_offsets = np.concatenate(absolute_offsets)
    else:
        imu_values = np.empty((0, len(IMU_CHANNEL_NAMES)), dtype=np.float32)
        imu_times = np.empty(0, dtype=np.float32)
        imu_frame_index = np.empty(0, dtype=np.int16)
        alignment_offsets = np.empty(0, dtype=np.float64)
    offsets_array = np.asarray(offsets, dtype=np.int32)
    if int(interval_counts.sum()) != len(imu_values) or int(offsets_array[-1]) != len(imu_values):
        raise AssertionError(f"trial-level IMU accounting differs: {sample_id}")
    if int(interval_counts.max(initial=0)) > np.iinfo(np.uint16).max:
        raise OverflowError(f"too many IMU points in one interval: {sample_id}")

    destination = output / "trial_motion_cache" / safe_trial_path(sample_id).with_suffix(".npz")
    atomic_npz(
        destination,
        frame_ids=frame_ids,
        frame_time_seconds=frame_times,
        skeleton_features=np.asarray(skeleton["features"], dtype=np.float32),
        skeleton_feature_mask=np.asarray(skeleton["feature_mask"], dtype=bool),
        skeleton_joint_mask=np.asarray(skeleton["joint_mask"], dtype=bool),
        skeleton_relations=np.asarray(skeleton["relations"], dtype=np.float32),
        skeleton_relation_mask=np.asarray(skeleton["relation_mask"], dtype=bool),
        skeleton_frame_quality=skeleton_quality,
        skeleton_person_count=skeleton_people,
        skeleton_clip_scale=np.asarray(float(skeleton["clip_scale"]), dtype=np.float32),
        imu_values=imu_values,
        imu_time_seconds=imu_times,
        imu_frame_index=imu_frame_index,
        imu_device_offsets=offsets_array,
        imu_interval_counts=interval_counts.astype(np.uint16),
        imu_device_mask=device_mask,
        joint_names=np.asarray(H36M_JOINT_NAMES),
        common_part_names=np.asarray(COMMON_PART_NAMES),
        skeleton_feature_names=np.asarray(SKELETON_FEATURE_NAMES),
        skeleton_relation_names=np.asarray(SKELETON_RELATION_NAMES),
        imu_device_names=np.asarray(IMU_DEVICE_NAMES),
        imu_channel_names=np.asarray(IMU_CHANNEL_NAMES),
    )
    covered = interval_counts.sum(axis=1) > 0
    return {
        "sample_id": sample_id,
        "official_sample_id": row["official_sample_id"],
        "class_id": -1,
        "class_name": "",
        "user_id": "anonymous",
        "trial_id": row["trial_id"],
        "frames": len(frame_ids),
        "frame_time_mode": "absolute_timestamp",
        "skeleton_valid_frames": int(np.asarray(skeleton["joint_mask"]).any(axis=1).sum()),
        "skeleton_mean_joint_valid_rate": float(np.asarray(skeleton["joint_mask"]).mean()),
        "skeleton_clip_scale": float(skeleton["clip_scale"]),
        "imu_path_present": int(bool(imu_text)),
        "imu_csv_files": imu_audit.csv_files,
        "imu_data_rows": imu_audit.data_rows,
        "imu_accepted_points": len(imu_values),
        "imu_rejected_rows": imu_audit.rejected_rows,
        "imu_unknown_device_rows": imu_audit.unknown_device_rows,
        "imu_device_count": int(device_mask.sum()),
        "imu_complete_five_devices": int(device_mask.all()),
        "imu_covered_frames": int(covered.sum()),
        "imu_frame_coverage_rate": float(covered.mean()),
        "imu_points_before_camera_span": before_points,
        "imu_points_after_camera_span": after_points,
        "imu_mean_abs_frame_offset_seconds": (
            float(alignment_offsets.mean()) if len(alignment_offsets) else 0.0
        ),
        "imu_p95_abs_frame_offset_seconds": (
            float(np.percentile(alignment_offsets, 95)) if len(alignment_offsets) else 0.0
        ),
        "cache_bytes": destination.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest.resolve())
    if args.max_trials > 0:
        rows = rows[: args.max_trials]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    roi_run = args.roi_run.resolve()
    summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        destination = output / "trial_motion_cache" / safe_trial_path(row["sample_id"]).with_suffix(".npz")
        summary_path = output / "trial_summary" / safe_trial_path(row["sample_id"]).with_suffix(".json")
        if destination.is_file() and summary_path.is_file() and not args.overwrite:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = build_trial(row, roi_run, output)
            atomic_json(summary_path, summary)
        summaries.append(summary)
        if index % 50 == 0 or index == len(rows):
            print(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(rows),
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
    write_csv(output / "trial_summary.csv", summaries)
    final = {
        "stage": "S_label_free_test_motion_source_cache",
        "selected_trials": len(rows),
        "completed_trials": len(summaries),
        "completed_frames": int(sum(int(row["frames"]) for row in summaries)),
        "skeleton_usable_trials": int(
            sum(int(row["skeleton_valid_frames"]) > 0 for row in summaries)
        ),
        "imu_usable_trials": int(sum(int(row["imu_device_count"]) > 0 for row in summaries)),
        "imu_missing_sample_ids": [
            row["official_sample_id"] for row in summaries if int(row["imu_device_count"]) == 0
        ],
        "labels_read": False,
        "ir_required": False,
        "visual_timeline_contract": (
            "Use exact roi frame_ids when IR is readable; use full Skeleton timeline "
            "only for the four IR-unreadable recordings"
        ),
        "large_videomae_required": False,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    atomic_json(output / "summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
