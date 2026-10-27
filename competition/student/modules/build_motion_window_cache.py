from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from skeleton_imu_preprocessing import IMU_DEVICE_NAMES, safe_trial_path
from event_preprocessing import device_relative_imu, rotate_skeleton_cache


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PIXELS = PROJECT_DIR / "runs/visual_pixel_cache_t16_r160_v12"
DEFAULT_MOTION = PROJECT_DIR / "runs/skeleton_imu_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/motion_window_cache_t16"
IMU_POINTS_PER_BIN = 4
IMU_TOKEN_CHANNELS = 16  # raw acc/gyro 6 + compensated acc/gyro 6 + quaternion 4
IMU_BIN_STAT_CHANNELS = 52  # mean 16 + std/rms/delta for the 12 vector channels
IMU_GLOBAL_STAT_CHANNELS = 48  # device: 8 statistics x 6 raw channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align motion_source Skeleton/IMU to the exact student two-window visual time grid. "
            "This deterministic cache is valid for both fast audits and final inference."
        )
    )
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--motion-source-run", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--points-per-bin", type=int, default=IMU_POINTS_PER_BIN)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bin_assignments(
    center_times: np.ndarray, point_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map every point inside a visual window to its nearest unique frame centre.

    Duplicate visual frame indices share the same bin. The first and last selected
    visual timestamps are the inclusive window boundaries, so no point outside the
    visual evidence window leaks into the token.
    """

    centers = np.asarray(center_times, dtype=np.float64)
    points = np.asarray(point_times, dtype=np.float64)
    if centers.ndim != 1 or len(centers) == 0:
        raise ValueError("center_times must be a non-empty vector")
    if np.any(np.diff(centers) < 0):
        raise ValueError("center_times must be monotonic")
    unique_centers, slot_to_unique = np.unique(centers, return_inverse=True)
    if len(unique_centers) == 1:
        # Short trials can repeat one source frame for every visual slot. Assign
        # the nearest available IMU points without pretending there are 16 times.
        if len(points):
            nearest = np.argmin(np.abs(points - unique_centers[0]))
            point_to_unique = np.full(len(points), -1, dtype=np.int64)
            point_to_unique[nearest] = 0
        else:
            point_to_unique = np.empty(0, dtype=np.int64)
        return slot_to_unique.astype(np.int64), point_to_unique
    boundaries = (unique_centers[:-1] + unique_centers[1:]) * 0.5
    point_to_unique = np.searchsorted(boundaries, points, side="right").astype(np.int64)
    in_window = (points >= unique_centers[0]) & (points <= unique_centers[-1])
    point_to_unique[~in_window] = -1
    return slot_to_unique.astype(np.int64), point_to_unique


def resample_points(values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if count < 1:
        raise ValueError("points-per-bin must be positive")
    output = np.zeros((count, IMU_TOKEN_CHANNELS), dtype=np.float32)
    mask = np.zeros(count, dtype=bool)
    if not len(values):
        return output, mask
    chosen = np.rint(np.linspace(0, len(values) - 1, min(count, len(values)))).astype(
        np.int64
    )
    output[: len(chosen)] = values[chosen]
    mask[: len(chosen)] = True
    return output, mask


def imu_bin_statistics(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not len(values):
        return np.zeros(IMU_BIN_STAT_CHANNELS, dtype=np.float32)
    vectors = values[:, :12]
    delta = vectors[-1] - vectors[0] if len(vectors) > 1 else np.zeros(12)
    return np.concatenate(
        (
            values.mean(axis=0),
            vectors.std(axis=0),
            np.sqrt(np.mean(vectors**2, axis=0)),
            delta,
        )
    ).astype(np.float32)


def device_statistics(raw_vectors: np.ndarray) -> np.ndarray:
    """Preserve the proven device per-device full-trial statistics as global tokens."""

    values = np.asarray(raw_vectors, dtype=np.float32)
    output: list[float] = []
    for channel in range(6):
        signal = values[:, channel] if len(values) else np.empty(0, dtype=np.float32)
        if not len(signal):
            output.extend([0.0] * 8)
            continue
        difference = np.diff(signal)
        output.extend(
            (
                float(signal.mean()),
                float(signal.std()),
                float(np.sqrt(np.mean(signal**2))),
                float(signal.min()),
                float(signal.max()),
                float(np.ptp(signal)),
                float(np.mean(np.abs(difference))) if len(difference) else 0.0,
                float(np.mean(difference**2)) if len(difference) else 0.0,
            )
        )
    return np.asarray(output, dtype=np.float32)


def aggregate_imu(
    frame_times: np.ndarray,
    source_indices: np.ndarray,
    flat_values: np.ndarray,
    flat_times: np.ndarray,
    offsets: np.ndarray,
    points_per_bin: int,
) -> dict[str, np.ndarray]:
    windows, steps = source_indices.shape
    devices = len(IMU_DEVICE_NAMES)
    sequences = np.zeros(
        (windows, steps, devices, points_per_bin, IMU_TOKEN_CHANNELS), dtype=np.float32
    )
    sequence_mask = np.zeros((windows, steps, devices, points_per_bin), dtype=bool)
    statistics = np.zeros(
        (windows, steps, devices, IMU_BIN_STAT_CHANNELS), dtype=np.float32
    )
    bin_mask = np.zeros((windows, steps, devices), dtype=bool)
    global_statistics = np.zeros(
        (devices, IMU_GLOBAL_STAT_CHANNELS), dtype=np.float32
    )
    global_mask = np.zeros((devices, 2), dtype=np.float32)

    for device in range(devices):
        start, end = int(offsets[device]), int(offsets[device + 1])
        raw_device = np.asarray(flat_values[start:end], dtype=np.float32)
        device_times = np.asarray(flat_times[start:end], dtype=np.float64)
        transformed = device_relative_imu(raw_device)
        combined = np.concatenate(
            (
                transformed["raw_vectors"],
                transformed["values"][:, :6],
                transformed["values"][:, 6:10],
            ),
            axis=1,
        )
        global_statistics[device] = device_statistics(
            transformed["raw_vectors"]
        )
        global_mask[device] = (float(len(raw_device) > 0), 0.0)
        if len(raw_device) and len(frame_times) > 1:
            global_mask[device, 1] = float(
                np.mean(
                    (device_times >= float(frame_times[0]))
                    & (device_times <= float(frame_times[-1]))
                )
            )
        for window in range(windows):
            centers = frame_times[source_indices[window]]
            slot_to_unique, point_to_unique = bin_assignments(centers, device_times)
            for slot, unique_index in enumerate(slot_to_unique):
                selected = combined[point_to_unique == unique_index]
                sequence, mask = resample_points(selected, points_per_bin)
                sequences[window, slot, device] = sequence
                sequence_mask[window, slot, device] = mask
                statistics[window, slot, device] = imu_bin_statistics(selected)
                bin_mask[window, slot, device] = bool(len(selected))
    return {
        "sequences": sequences,
        "sequence_mask": sequence_mask,
        "statistics": statistics,
        "bin_mask": bin_mask,
        "global_statistics": global_statistics,
        "global_mask": global_mask,
    }


def create_memmaps(
    output: Path,
    trials: int,
    windows: int,
    steps: int,
    points_per_bin: int,
    overwrite: bool,
) -> dict[str, np.memmap]:
    contracts: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
        "skeleton_features": (np.dtype(np.float16), (trials, windows, steps, 17, 13)),
        "skeleton_feature_mask": (np.dtype(np.uint8), (trials, windows, steps, 17, 13)),
        "skeleton_joint_mask": (np.dtype(np.uint8), (trials, windows, steps, 17)),
        "skeleton_relations": (np.dtype(np.float16), (trials, windows, steps, 18)),
        "skeleton_relation_mask": (np.dtype(np.uint8), (trials, windows, steps, 18)),
        "skeleton_frame_quality": (np.dtype(np.float16), (trials, windows, steps)),
        "imu_sequences": (
            np.dtype(np.float16),
            (trials, windows, steps, 5, points_per_bin, IMU_TOKEN_CHANNELS),
        ),
        "imu_sequence_mask": (
            np.dtype(np.uint8),
            (trials, windows, steps, 5, points_per_bin),
        ),
        "imu_bin_statistics": (
            np.dtype(np.float16),
            (trials, windows, steps, 5, IMU_BIN_STAT_CHANNELS),
        ),
        "imu_bin_mask": (np.dtype(np.uint8), (trials, windows, steps, 5)),
        "imu_global_statistics": (
            # Squared-difference statistics can legitimately exceed FP16 range
            # on noisy sensors. Keep this tiny array lossless instead of clipping.
            np.dtype(np.float32),
            (trials, 5, IMU_GLOBAL_STAT_CHANNELS),
        ),
        "imu_global_mask": (np.dtype(np.float16), (trials, 5, 2)),
        "completed": (np.dtype(np.uint8), (trials,)),
    }
    result: dict[str, np.memmap] = {}
    for name, (dtype, shape) in contracts.items():
        path = output / f"{name}.npy"
        if path.exists() and not overwrite:
            array = np.lib.format.open_memmap(path, mode="r+")
            if array.dtype != dtype or array.shape != shape:
                raise RuntimeError(f"motion cache contract changed for {path}")
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            array[:] = 0
            array.flush()
        result[name] = array
    return result


def main() -> None:
    args = parse_args()
    if args.points_per_bin < 1:
        raise ValueError("--points-per-bin must be positive")
    pixel_cache = args.pixel_cache.resolve()
    motion_source_run = args.motion_source_run.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(pixel_cache / "rows.csv")
    if args.max_trials:
        rows = rows[: args.max_trials]
    source_indices = np.load(pixel_cache / "source_frame_indices.npy", mmap_mode="r")
    source_indices = source_indices[: len(rows)]
    if source_indices.ndim != 3:
        raise RuntimeError("student source_frame_indices must be [N,2,T]")
    windows, steps = source_indices.shape[1:]

    motion_rows = read_csv(motion_source_run / "trial_summary.csv")
    motion_by_sample = {row["sample_id"]: row for row in motion_rows}
    if len(motion_by_sample) != len(motion_rows):
        raise RuntimeError("duplicate motion_source sample_id")
    missing = [row["source_id"] for row in rows if row["source_id"] not in motion_by_sample]
    if missing:
        raise RuntimeError(f"student source_id missing from motion_source cache: {missing[:3]}")

    arrays = create_memmaps(
        output,
        len(rows),
        windows,
        steps,
        args.points_per_bin,
        args.overwrite,
    )
    started = time.perf_counter()
    built = skipped = 0
    for index, row in enumerate(rows):
        if arrays["completed"][index]:
            skipped += 1
            continue
        trial_path = (
            motion_source_run
            / "trial_motion_cache"
            / safe_trial_path(row["source_id"]).with_suffix(".npz")
        )
        with np.load(trial_path, allow_pickle=False) as trial:
            frame_times = np.asarray(trial["frame_time_seconds"], dtype=np.float64)
            chosen = np.asarray(source_indices[index], dtype=np.int64)
            if chosen.min() < 0 or chosen.max() >= len(frame_times):
                raise RuntimeError(f"visual/motion frame mismatch: {row['source_id']}")
            rotated = rotate_skeleton_cache(
                trial["skeleton_features"],
                trial["skeleton_feature_mask"],
                trial["skeleton_joint_mask"],
                trial["skeleton_relations"],
                trial["skeleton_relation_mask"],
                frame_times,
            )
            arrays["skeleton_features"][index] = rotated["features"][chosen]
            arrays["skeleton_feature_mask"][index] = rotated["feature_mask"][chosen]
            arrays["skeleton_joint_mask"][index] = trial["skeleton_joint_mask"][chosen]
            arrays["skeleton_relations"][index] = rotated["relations"][chosen]
            arrays["skeleton_relation_mask"][index] = rotated["relation_mask"][chosen]
            arrays["skeleton_frame_quality"][index] = trial[
                "skeleton_frame_quality"
            ][chosen]
            imu = aggregate_imu(
                frame_times,
                chosen,
                trial["imu_values"],
                trial["imu_time_seconds"],
                trial["imu_device_offsets"],
                args.points_per_bin,
            )
        arrays["imu_sequences"][index] = imu["sequences"]
        arrays["imu_sequence_mask"][index] = imu["sequence_mask"]
        arrays["imu_bin_statistics"][index] = imu["statistics"]
        arrays["imu_bin_mask"][index] = imu["bin_mask"]
        arrays["imu_global_statistics"][index] = imu["global_statistics"]
        arrays["imu_global_mask"][index] = imu["global_mask"]
        arrays["completed"][index] = 1
        built += 1
        if built % 50 == 0 or index + 1 == len(rows):
            for array in arrays.values():
                array.flush()
            print(
                json.dumps(
                    {
                        "processed": index + 1,
                        "total": len(rows),
                        "built": built,
                        "skipped": skipped,
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )

    with (output / "rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_id", "source_id", "user_id", "class_id")
        )
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    total_bytes = sum((output / f"{name}.npy").stat().st_size for name in arrays)
    summary = {
        "stage": "exact_visual_grid_motion_cache",
        "trials": len(rows),
        "completed": int(np.asarray(arrays["completed"]).sum()),
        "windows": windows,
        "steps": steps,
        "points_per_imu_bin": args.points_per_bin,
        "join_contract": "student rows.source_id == motion_source trial_summary.sample_id",
        "skeleton_coordinate": "video body-local; camera axes excluded",
        "imu_channels": "raw6 + device-relative compensated6 + relative quaternion4",
        "imu_global_statistics": "exact device 8x6 per-device statistics",
        "total_bytes": total_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "accuracy_contract": (
            "Deterministic alignment/cache only; it does not reduce student frames, views, "
            "resolution, epochs or final end-to-end trainability."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
