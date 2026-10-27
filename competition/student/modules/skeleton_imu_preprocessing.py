from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


H36M_JOINT_NAMES = (
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)
H36M_PARENTS = np.asarray(
    (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15),
    dtype=np.int64,
)

COMMON_PART_NAMES = (
    "global",
    "head",
    "torso",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "hand_workspace",
)
PART_JOINTS = (
    tuple(range(17)),
    (8, 9, 10),
    (0, 1, 4, 7, 8, 9),
    (11, 12, 13),
    (14, 15, 16),
    (4, 5, 6),
    (1, 2, 3),
    (9, 10, 13, 16),
)

IMU_DEVICE_NAMES = ("WTC", "WTLA", "WTRA", "WTLL", "WTRL")
IMU_DEVICE_TO_INDEX = {name: index for index, name in enumerate(IMU_DEVICE_NAMES)}
IMU_DEVICE_BODY_PARTS = (
    "torso",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
)
IMU_CHANNEL_NAMES = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "relative_quat_w",
    "relative_quat_x",
    "relative_quat_y",
    "relative_quat_z",
)
SKELETON_FEATURE_NAMES = (
    "relative_x",
    "relative_y",
    "relative_z",
    "bone_x",
    "bone_y",
    "bone_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "confidence",
)
SKELETON_RELATION_NAMES = (
    "left_wrist_head_distance",
    "right_wrist_head_distance",
    "hand_to_hand_distance",
    "left_wrist_pelvis_distance",
    "right_wrist_pelvis_distance",
    "left_elbow_angle_cosine",
    "right_elbow_angle_cosine",
    "left_knee_angle_cosine",
    "right_knee_angle_cosine",
    "shoulder_width",
    "hip_width",
    "pelvis_head_height",
    "torso_direction_x",
    "torso_direction_y",
    "torso_direction_z",
    "left_wrist_speed",
    "right_wrist_speed",
    "whole_body_motion_energy",
)


@dataclass(frozen=True)
class IMULoadAudit:
    csv_files: int
    data_rows: int
    accepted_rows: int
    rejected_rows: int
    unknown_device_rows: int


def safe_trial_path(sample_id: str) -> Path:
    parts = sample_id.split("/")
    if len(parts) != 3 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe sample_id: {sample_id}")
    return Path(*parts)


def parse_frame_timestamp(frame_id: str) -> float:
    """Parse a D/IR/Skeleton canonical frame id without local-time ambiguity."""

    timestamp_text = frame_id.rsplit("_", 1)[0]
    parsed = datetime.strptime(timestamp_text, "%Y-%m-%d_%H-%M-%S.%f")
    return parsed.replace(tzinfo=timezone.utc).timestamp()


def parse_imu_timestamp(value: str) -> float:
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None
        # Some otherwise valid lower-body CSVs use non-zero-padded month/day,
        # for example ``2025-5-7 16:42:19.85``.  strptime accepts that legacy
        # spelling while fromisoformat does not on every Python version.
        for timestamp_format in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S.%f",
            "%Y/%m/%d %H:%M:%S",
        ):
            try:
                parsed = datetime.strptime(text, timestamp_format)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError(f"unsupported IMU timestamp: {text}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def frame_times_from_ids(frame_ids: list[str] | np.ndarray) -> np.ndarray:
    texts = [str(value) for value in frame_ids]
    if texts and all(value.isdigit() for value in texts):
        # Four legacy user1 trials contain only a monotonically increasing frame
        # counter.  Those trials have no IMU files; 10 Hz is used solely for the
        # Skeleton derivatives and is surfaced as a fallback in the cache audit.
        times = np.asarray([int(value) * 0.1 for value in texts], dtype=np.float64)
    else:
        times = np.asarray([parse_frame_timestamp(value) for value in texts], dtype=np.float64)
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise ValueError("frame timestamps are not strictly increasing")
    return times


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def relative_quaternion(quaternion: np.ndarray) -> np.ndarray:
    if len(quaternion) == 0:
        return quaternion.astype(np.float32)
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-8)
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0:
            result[index] *= -1
    inverse_first = result[0].copy()
    inverse_first[1:] *= -1
    result = quaternion_multiply(
        np.repeat(inverse_first[None], len(result), axis=0), result
    )
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-8)
    result[result[:, 0] < 0] *= -1
    return result.astype(np.float32)


def load_full_imu_trial(
    path: Path,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], IMULoadAudit]:
    """Load every finite, known-device IMU row; no temporal resampling occurs."""

    samples: dict[str, list[tuple[float, int, np.ndarray]]] = {
        name: [] for name in IMU_DEVICE_NAMES
    }
    csv_files = 0
    data_rows = 0
    rejected_rows = 0
    unknown_device_rows = 0
    ordinal = 0
    if path.is_dir():
        for file_path in sorted(path.glob("*.csv")):
            csv_files += 1
            try:
                with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.reader(handle)
                    next(reader, None)
                    for fields in reader:
                        data_rows += 1
                        ordinal += 1
                        if len(fields) < 18:
                            rejected_rows += 1
                            continue
                        device = fields[1].split("(", 1)[0].strip()
                        if device not in IMU_DEVICE_TO_INDEX:
                            unknown_device_rows += 1
                            continue
                        try:
                            timestamp = parse_imu_timestamp(fields[0])
                            acc_gyro = np.asarray(
                                [float(fields[index]) for index in range(2, 8)],
                                dtype=np.float64,
                            )
                            quaternion = np.asarray(
                                [float(fields[index]) for index in range(14, 18)],
                                dtype=np.float64,
                            )
                        except (IndexError, ValueError, OverflowError):
                            rejected_rows += 1
                            continue
                        values = np.concatenate((acc_gyro, quaternion))
                        if not np.isfinite(timestamp) or not np.isfinite(values).all():
                            rejected_rows += 1
                            continue
                        samples[device].append((timestamp, ordinal, values))
            except (OSError, UnicodeError):
                continue

    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    accepted_rows = 0
    for device in IMU_DEVICE_NAMES:
        device_samples = sorted(samples[device], key=lambda item: (item[0], item[1]))
        if not device_samples:
            continue
        timestamps = np.asarray([item[0] for item in device_samples], dtype=np.float64)
        values = np.stack([item[2] for item in device_samples]).astype(np.float32)
        values[:, 6:10] = relative_quaternion(values[:, 6:10])
        output[device] = (timestamps, values)
        accepted_rows += len(values)
    audit = IMULoadAudit(
        csv_files=csv_files,
        data_rows=data_rows,
        accepted_rows=accepted_rows,
        rejected_rows=rejected_rows,
        unknown_device_rows=unknown_device_rows,
    )
    return output, audit


def assign_points_to_frame_intervals(
    point_times: np.ndarray, frame_times: np.ndarray
) -> np.ndarray:
    """Assign each point exactly once using midpoints between adjacent frames.

    The first and last intervals are open-ended, so valid points slightly before
    or after the camera recording are retained at the respective boundary frame.
    """

    point_times = np.asarray(point_times, dtype=np.float64)
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if len(frame_times) == 0:
        raise ValueError("cannot align IMU points to an empty frame sequence")
    if len(frame_times) == 1:
        return np.zeros(len(point_times), dtype=np.int32)
    midpoints = (frame_times[:-1] + frame_times[1:]) * 0.5
    return np.searchsorted(midpoints, point_times, side="right").astype(np.int32)


def _time_derivative(
    values: np.ndarray,
    valid: np.ndarray,
    frame_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    derivative = np.zeros_like(values, dtype=np.float32)
    derivative_valid = np.zeros_like(valid, dtype=bool)
    if len(values) < 2:
        return derivative, derivative_valid
    delta_t = np.diff(frame_times).astype(np.float32)
    delta_t = np.maximum(delta_t, 1e-4)
    pair_valid = valid[1:] & valid[:-1]
    candidate = (values[1:] - values[:-1]) / delta_t[:, None, None]
    derivative[1:] = np.where(pair_valid[:, :, None], candidate, 0.0)
    derivative_valid[1:] = pair_valid
    return derivative, derivative_valid


def _distance(
    xyz: np.ndarray, valid: np.ndarray, left: int, right: int
) -> tuple[np.ndarray, np.ndarray]:
    mask = valid[:, left] & valid[:, right]
    value = np.linalg.norm(xyz[:, left] - xyz[:, right], axis=1).astype(np.float32)
    return np.where(mask, value, 0.0), mask


def _angle_cosine(
    xyz: np.ndarray, valid: np.ndarray, a: int, b: int, c: int
) -> tuple[np.ndarray, np.ndarray]:
    first = xyz[:, a] - xyz[:, b]
    second = xyz[:, c] - xyz[:, b]
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    mask = valid[:, a] & valid[:, b] & valid[:, c] & (denominator > 1e-6)
    cosine = np.sum(first * second, axis=1) / np.maximum(denominator, 1e-6)
    return np.where(mask, np.clip(cosine, -1.0, 1.0), 0.0).astype(np.float32), mask


def build_skeleton_features(
    raw: np.ndarray,
    frame_times: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Create translation/scale-normalized all-frame Skeleton inputs."""

    raw = np.asarray(raw, dtype=np.float32)
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (17, 4):
        raise ValueError(f"expected Skeleton [T,17,4], got {raw.shape}")
    if len(raw) != len(frame_times):
        raise ValueError("Skeleton and frame time lengths differ")

    source_xyz = raw[..., :3]
    confidence = np.clip(np.nan_to_num(raw[..., 3], nan=0.0), 0.0, 1.0)
    valid = np.isfinite(source_xyz).all(axis=2) & (confidence > 0.0)
    root = source_xyz[:, 0].copy()
    root_valid = valid[:, 0]
    for index in np.flatnonzero(~root_valid):
        candidates = source_xyz[index, valid[index]]
        root[index] = np.median(candidates, axis=0) if len(candidates) else 0.0
    centered = source_xyz - root[:, None, :]
    centered[~valid] = 0.0

    radius = np.linalg.norm(centered, axis=2)
    radius[~valid] = np.nan
    with np.errstate(all="ignore"):
        frame_scale = np.nanmax(radius, axis=1)
    usable_scale = frame_scale[np.isfinite(frame_scale) & (frame_scale > 1e-6)]
    clip_scale = float(np.median(usable_scale)) if len(usable_scale) else 1.0
    centered = (centered / max(clip_scale, 1e-6)).astype(np.float32)

    parent_xyz = centered[:, H36M_PARENTS]
    bone = centered - parent_xyz
    bone_valid = valid & valid[:, H36M_PARENTS]
    bone[:, 0] = 0.0
    bone_valid[:, 0] = valid[:, 0]
    bone[~bone_valid] = 0.0

    velocity, velocity_valid = _time_derivative(centered, valid, frame_times)
    acceleration, acceleration_valid = _time_derivative(
        velocity, velocity_valid, frame_times
    )
    features = np.concatenate(
        (centered, bone, velocity, acceleration, confidence[..., None]), axis=2
    ).astype(np.float32)
    feature_mask = np.concatenate(
        (
            np.repeat(valid[..., None], 3, axis=2),
            np.repeat(bone_valid[..., None], 3, axis=2),
            np.repeat(velocity_valid[..., None], 3, axis=2),
            np.repeat(acceleration_valid[..., None], 3, axis=2),
            valid[..., None],
        ),
        axis=2,
    )
    features[~feature_mask] = 0.0

    relation_values: list[np.ndarray] = []
    relation_masks: list[np.ndarray] = []
    for value, mask in (
        _distance(centered, valid, 13, 10),
        _distance(centered, valid, 16, 10),
        _distance(centered, valid, 13, 16),
        _distance(centered, valid, 13, 0),
        _distance(centered, valid, 16, 0),
        _angle_cosine(centered, valid, 11, 12, 13),
        _angle_cosine(centered, valid, 14, 15, 16),
        _angle_cosine(centered, valid, 4, 5, 6),
        _angle_cosine(centered, valid, 1, 2, 3),
        _distance(centered, valid, 11, 14),
        _distance(centered, valid, 4, 1),
        _distance(centered, valid, 0, 10),
    ):
        relation_values.append(value)
        relation_masks.append(mask)

    torso_vector = centered[:, 10] - centered[:, 0]
    torso_norm = np.linalg.norm(torso_vector, axis=1, keepdims=True)
    torso_mask = valid[:, 10] & valid[:, 0] & (torso_norm[:, 0] > 1e-6)
    torso_direction = torso_vector / np.maximum(torso_norm, 1e-6)
    for axis in range(3):
        relation_values.append(np.where(torso_mask, torso_direction[:, axis], 0.0))
        relation_masks.append(torso_mask)

    left_wrist_speed = np.linalg.norm(velocity[:, 13], axis=1)
    right_wrist_speed = np.linalg.norm(velocity[:, 16], axis=1)
    valid_velocity_count = velocity_valid.sum(axis=1)
    whole_body_energy = np.linalg.norm(velocity, axis=2).sum(axis=1) / np.maximum(
        valid_velocity_count, 1
    )
    relation_values.extend((left_wrist_speed, right_wrist_speed, whole_body_energy))
    relation_masks.extend(
        (
            velocity_valid[:, 13],
            velocity_valid[:, 16],
            valid_velocity_count > 0,
        )
    )
    relations = np.stack(relation_values, axis=1).astype(np.float32)
    relation_mask = np.stack(relation_masks, axis=1).astype(bool)
    relations[~relation_mask] = 0.0
    if relations.shape[1] != len(SKELETON_RELATION_NAMES):
        raise AssertionError("Skeleton relation feature contract changed")

    return {
        "features": features,
        "feature_mask": feature_mask.astype(bool),
        "joint_mask": valid.astype(bool),
        "relations": relations,
        "relation_mask": relation_mask,
        "frame_quality": (confidence * valid).sum(axis=1)
        / np.maximum(valid.sum(axis=1), 1),
        "clip_scale": clip_scale,
    }
