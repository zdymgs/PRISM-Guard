from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from skeleton_imu_preprocessing import H36M_PARENTS


LOCAL_REGIONS = (
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "hand_workspace",
)

# roi stores sorted COCO arm joints [5, 6, 7, 8, 9, 10].
LEFT_SHOULDER, RIGHT_SHOULDER = 0, 1
LEFT_ELBOW, RIGHT_ELBOW = 2, 3
LEFT_WRIST, RIGHT_WRIST = 4, 5


def _normalise(vector: np.ndarray, epsilon: float = 1e-6) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= epsilon:
        return np.zeros(3, dtype=np.float32), False
    return (vector / norm).astype(np.float32), True


def _orthonormal_axes(x_axis: np.ndarray, y_seed: np.ndarray) -> tuple[np.ndarray, bool]:
    x_axis, x_valid = _normalise(x_axis)
    if not x_valid:
        return np.eye(3, dtype=np.float32), False
    z_axis, z_valid = _normalise(np.cross(x_axis, y_seed))
    if not z_valid:
        return np.eye(3, dtype=np.float32), False
    y_axis, y_valid = _normalise(np.cross(z_axis, x_axis))
    if not y_valid:
        return np.eye(3, dtype=np.float32), False
    return np.stack((x_axis, y_axis, z_axis), axis=1), True


def body_coordinate_axes(
    xyz: np.ndarray,
    joint_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stable camera-to-body axes without pretending camera/Skeleton calibration.

    Axis columns are left, up and forward in the raw Skeleton camera coordinate
    system. Missing frames are interpolated and re-orthonormalised. ``raw_valid``
    distinguishes measured axes from filled axes.
    """

    xyz = np.asarray(xyz, dtype=np.float32)
    joint_mask = np.asarray(joint_mask, dtype=bool)
    if xyz.ndim != 3 or xyz.shape[1:] != (17, 3):
        raise ValueError(f"xyz must be [T,17,3], got {xyz.shape}")
    if joint_mask.shape != xyz.shape[:2]:
        raise ValueError("joint_mask does not align with xyz")
    time_steps = len(xyz)
    axes = np.repeat(np.eye(3, dtype=np.float32)[None], time_steps, axis=0)
    raw_valid = np.zeros(time_steps, dtype=bool)
    for index in range(time_steps):
        shoulders_valid = joint_mask[index, 11] and joint_mask[index, 14]
        pelvis_valid = joint_mask[index, 0]
        upper_index = next(
            (joint for joint in (8, 9, 10) if joint_mask[index, joint]), None
        )
        if not shoulders_valid or not pelvis_valid or upper_index is None:
            continue
        left_axis = xyz[index, 11] - xyz[index, 14]
        up_seed = xyz[index, upper_index] - xyz[index, 0]
        candidate, valid = _orthonormal_axes(left_axis, up_seed)
        if valid:
            axes[index] = candidate
            raw_valid[index] = True

    if not raw_valid.any():
        return axes, raw_valid, np.ones(time_steps, dtype=bool)
    valid_indices = np.flatnonzero(raw_valid)
    positions = np.arange(time_steps)
    interpolated = np.empty_like(axes)
    for row in range(3):
        for column in range(3):
            interpolated[:, row, column] = np.interp(
                positions, valid_indices, axes[valid_indices, row, column]
            )
    previous: np.ndarray | None = None
    for index in range(time_steps):
        candidate, valid = _orthonormal_axes(
            interpolated[index, :, 0], interpolated[index, :, 1]
        )
        if not valid:
            candidate = previous.copy() if previous is not None else np.eye(3, dtype=np.float32)
        if previous is not None and float(np.dot(candidate[:, 2], previous[:, 2])) < 0:
            candidate[:, 2] *= -1
            candidate[:, 1] *= -1
        interpolated[index] = candidate
        previous = candidate
    filled = ~raw_valid
    return interpolated.astype(np.float32), raw_valid, filled


def _time_derivative(
    values: np.ndarray,
    valid: np.ndarray,
    frame_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    derivative = np.zeros_like(values, dtype=np.float32)
    derivative_valid = np.zeros_like(valid, dtype=bool)
    if len(values) < 2:
        return derivative, derivative_valid
    delta = np.maximum(np.diff(frame_times).astype(np.float32), 1e-4)
    pair_valid = valid[1:] & valid[:-1]
    candidate = (values[1:] - values[:-1]) / delta[:, None, None]
    derivative[1:] = np.where(pair_valid[..., None], candidate, 0.0)
    derivative_valid[1:] = pair_valid
    return derivative, derivative_valid


def rotate_skeleton_cache(
    skeleton_features: np.ndarray,
    skeleton_feature_mask: np.ndarray,
    skeleton_joint_mask: np.ndarray,
    skeleton_relations: np.ndarray,
    skeleton_relation_mask: np.ndarray,
    frame_times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Rotate cached motion_source root-centred Skeleton into an explicit body frame."""

    source = np.asarray(skeleton_features, dtype=np.float32)
    feature_mask = np.asarray(skeleton_feature_mask, dtype=bool)
    joint_mask = np.asarray(skeleton_joint_mask, dtype=bool)
    frame_times = np.asarray(frame_times, dtype=np.float64)
    xyz = source[..., :3]
    axes, axes_raw_valid, axes_filled = body_coordinate_axes(xyz, joint_mask)
    body_xyz = np.einsum("tjc,tck->tjk", xyz, axes).astype(np.float32)
    body_xyz[~joint_mask] = 0.0

    parent_xyz = body_xyz[:, H36M_PARENTS]
    bone = body_xyz - parent_xyz
    bone_valid = joint_mask & joint_mask[:, H36M_PARENTS]
    bone[:, 0] = 0.0
    bone_valid[:, 0] = joint_mask[:, 0]
    bone[~bone_valid] = 0.0
    velocity, velocity_valid = _time_derivative(body_xyz, joint_mask, frame_times)
    acceleration, acceleration_valid = _time_derivative(
        velocity, velocity_valid, frame_times
    )
    confidence = source[..., 12:13]
    output = np.concatenate((body_xyz, bone, velocity, acceleration, confidence), axis=2)
    output_mask = np.concatenate(
        (
            np.repeat(joint_mask[..., None], 3, axis=2),
            np.repeat(bone_valid[..., None], 3, axis=2),
            np.repeat(velocity_valid[..., None], 3, axis=2),
            np.repeat(acceleration_valid[..., None], 3, axis=2),
            feature_mask[..., 12:13],
        ),
        axis=2,
    )
    output[~output_mask] = 0.0

    relations = np.asarray(skeleton_relations, dtype=np.float32).copy()
    relation_mask = np.asarray(skeleton_relation_mask, dtype=bool).copy()
    torso = body_xyz[:, 10] - body_xyz[:, 0]
    torso_norm = np.linalg.norm(torso, axis=1, keepdims=True)
    torso_valid = joint_mask[:, 10] & joint_mask[:, 0] & (torso_norm[:, 0] > 1e-6)
    torso_direction = torso / np.maximum(torso_norm, 1e-6)
    relations[:, 12:15] = np.where(torso_valid[:, None], torso_direction, 0.0)
    relation_mask[:, 12:15] = torso_valid[:, None]
    left_speed = np.linalg.norm(velocity[:, 13], axis=1)
    right_speed = np.linalg.norm(velocity[:, 16], axis=1)
    velocity_count = velocity_valid.sum(axis=1)
    whole_energy = np.linalg.norm(velocity, axis=2).sum(axis=1) / np.maximum(
        velocity_count, 1
    )
    relations[:, 15] = np.where(velocity_valid[:, 13], left_speed, 0.0)
    relations[:, 16] = np.where(velocity_valid[:, 16], right_speed, 0.0)
    relations[:, 17] = np.where(velocity_count > 0, whole_energy, 0.0)
    relation_mask[:, 15] = velocity_valid[:, 13]
    relation_mask[:, 16] = velocity_valid[:, 16]
    relation_mask[:, 17] = velocity_count > 0
    return {
        "features": output.astype(np.float32),
        "feature_mask": output_mask,
        "relations": relations,
        "relation_mask": relation_mask,
        "body_axes_camera": axes,
        "body_axes_raw_valid": axes_raw_valid,
        "body_axes_filled": axes_filled,
    }


def quaternion_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate 3-vectors by unit quaternions in wxyz order."""

    quaternion = np.asarray(quaternion, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    q = quaternion / np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8)
    qv = q[..., 1:]
    qw = q[..., :1]
    twice_cross = 2.0 * np.cross(qv, vector)
    return (vector + qw * twice_cross + np.cross(qv, twice_cross)).astype(np.float32)


def device_relative_imu(imu_values: np.ndarray) -> dict[str, np.ndarray]:
    """Orientation-compensate each device without claiming body-frame calibration.

    motion_source quaternions are relative to the first sample of each device. The rotated
    vectors therefore live in a device-relative trial coordinate, not the
    Skeleton body coordinate. Raw and compensated vectors are both preserved.
    """

    source = np.asarray(imu_values, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 10:
        raise ValueError(f"imu_values must be [N,10], got {source.shape}")
    if len(source) == 0:
        return {"values": source.copy(), "raw_vectors": source[:, :6].copy()}
    quaternion = source[:, 6:10]
    acceleration = quaternion_rotate(quaternion, source[:, :3])
    angular_velocity = quaternion_rotate(quaternion, source[:, 3:6])
    compensated = np.concatenate((acceleration, angular_velocity, quaternion), axis=1)
    return {
        "values": compensated.astype(np.float32),
        "raw_vectors": source[:, :6].astype(np.float32),
    }


def oriented_roi_geometry(
    boxes: np.ndarray,
    valid: np.ndarray,
    arm_joints: np.ndarray,
    joint_quality: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build centre/half-size/angle geometry for two arms, two hands and workspace."""

    boxes = np.asarray(boxes, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    arm_joints = np.asarray(arm_joints, dtype=np.float32)
    joint_quality = np.asarray(joint_quality, dtype=np.float32)
    if boxes.shape[1:] != (len(LOCAL_REGIONS), 4):
        raise ValueError("boxes must follow LOCAL_REGIONS")
    result = np.zeros((len(boxes), len(LOCAL_REGIONS), 6), dtype=np.float32)
    angle_valid = np.zeros((len(boxes), len(LOCAL_REGIONS)), dtype=bool)

    def point(frame: int, joint: int) -> tuple[np.ndarray, bool]:
        value = arm_joints[frame, joint, :2]
        usable = bool(joint_quality[frame, joint] > 0 and np.isfinite(value).all())
        return value, usable

    for frame in range(len(boxes)):
        for region in range(len(LOCAL_REGIONS)):
            x1, y1, x2, y2 = boxes[frame, region]
            if not valid[frame, region] or not np.isfinite((x1, y1, x2, y2)).all():
                x1, y1, x2, y2 = 0.0, 0.0, float(width - 1), float(height - 1)
            center = np.asarray(((x1 + x2) * 0.5, (y1 + y2) * 0.5), dtype=np.float32)
            half_x = max(float(x2 - x1) * 0.5, 2.0)
            half_y = max(float(y2 - y1) * 0.5, 2.0)
            angle = 0.0

            side = 0 if region in (0, 2) else 1
            shoulder_index = LEFT_SHOULDER if side == 0 else RIGHT_SHOULDER
            elbow_index = LEFT_ELBOW if side == 0 else RIGHT_ELBOW
            wrist_index = LEFT_WRIST if side == 0 else RIGHT_WRIST
            shoulder, shoulder_ok = point(frame, shoulder_index)
            elbow, elbow_ok = point(frame, elbow_index)
            wrist, wrist_ok = point(frame, wrist_index)
            if region in (0, 1) and shoulder_ok and wrist_ok:
                direction = wrist - shoulder
                length = float(np.linalg.norm(direction))
                if length > 2.0:
                    center = (shoulder + wrist) * 0.5
                    angle = math.atan2(float(direction[1]), float(direction[0]))
                    half_x = max(0.65 * length, 4.0)
                    half_y = max(0.20 * length, 4.0)
                    angle_valid[frame, region] = True
            elif region in (2, 3) and elbow_ok and wrist_ok:
                direction = wrist - elbow
                if float(np.linalg.norm(direction)) > 2.0:
                    center = wrist
                    angle = math.atan2(float(direction[1]), float(direction[0]))
                    side_length = max(float(x2 - x1), float(y2 - y1), 6.0)
                    half_x = half_y = side_length * 0.5
                    angle_valid[frame, region] = True
            elif region == 4:
                left_wrist, left_ok = point(frame, LEFT_WRIST)
                right_wrist, right_ok = point(frame, RIGHT_WRIST)
                if left_ok and right_ok and float(np.linalg.norm(right_wrist - left_wrist)) > 2.0:
                    direction = right_wrist - left_wrist
                    angle = math.atan2(float(direction[1]), float(direction[0]))
                    angle_valid[frame, region] = True
                cosine, sine = math.cos(angle), math.sin(angle)
                axis_x = np.asarray((cosine, sine), dtype=np.float32)
                axis_y = np.asarray((-sine, cosine), dtype=np.float32)
                corners = np.asarray(((x1, y1), (x1, y2), (x2, y1), (x2, y2)))
                offsets = corners - center
                half_x = max(float(np.abs(offsets @ axis_x).max()), 4.0)
                half_y = max(float(np.abs(offsets @ axis_y).max()), 4.0)

            center[0] = np.clip(center[0], 0.0, width - 1.0)
            center[1] = np.clip(center[1], 0.0, height - 1.0)
            result[frame, region] = (
                center[0],
                center[1],
                half_x,
                half_y,
                math.sin(angle),
                math.cos(angle),
            )
    return result, angle_valid


def oriented_grid_crops(
    images: torch.Tensor,
    geometry: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Vectorised affine crop for images [M,N,C,H,W] and geometry [N,R,6]."""

    if images.ndim != 5:
        raise ValueError("images must be [M,N,C,H,W]")
    modalities, frames, channels, height, width = images.shape
    if geometry.ndim != 3 or geometry.shape[0] != frames or geometry.shape[2] != 6:
        raise ValueError("geometry must be [N,R,6]")
    regions = geometry.shape[1]
    repeated_images = images[:, :, None].expand(
        modalities, frames, regions, channels, height, width
    ).reshape(modalities * frames * regions, channels, height, width)
    repeated_geometry = geometry[None].expand(modalities, -1, -1, -1).reshape(-1, 6)
    cx, cy, half_x, half_y, sine, cosine = repeated_geometry.unbind(dim=1)
    coordinate = torch.linspace(
        -1.0, 1.0, output_size, device=images.device, dtype=images.dtype
    )
    vertical, horizontal = torch.meshgrid(coordinate, coordinate, indexing="ij")
    horizontal = horizontal[None]
    vertical = vertical[None]
    source_x = (
        cx[:, None, None]
        + horizontal * half_x[:, None, None] * cosine[:, None, None]
        - vertical * half_y[:, None, None] * sine[:, None, None]
    )
    source_y = (
        cy[:, None, None]
        + horizontal * half_x[:, None, None] * sine[:, None, None]
        + vertical * half_y[:, None, None] * cosine[:, None, None]
    )
    grid_x = source_x * (2.0 / max(width - 1, 1)) - 1.0
    grid_y = source_y * (2.0 / max(height - 1, 1)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    crops = F.grid_sample(
        repeated_images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return crops.reshape(
        modalities, frames, regions, channels, output_size, output_size
    )


def upper_body_human_prior(
    keypoints: np.ndarray,
    person_boxes: np.ndarray,
    width: int,
    height: int,
    confidence: float = 0.25,
) -> np.ndarray:
    """Create a conservative upper-body capsule mask; objects are not hard segmented."""

    keypoints = np.asarray(keypoints, dtype=np.float32)
    boxes = np.asarray(person_boxes, dtype=np.float32)
    output = np.zeros((len(keypoints), height, width), dtype=np.uint8)
    edges = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12))
    for frame in range(len(keypoints)):
        points = keypoints[frame]
        valid = np.isfinite(points[:, :2]).all(axis=1) & (points[:, 2] >= confidence)
        box_height = float(boxes[frame, 3] - boxes[frame, 1])
        if not np.isfinite(box_height) or box_height <= 0.0:
            upper_valid = valid[[5, 6, 7, 8, 9, 10]]
            upper_y = points[[5, 6, 7, 8, 9, 10], 1][upper_valid]
            box_height = float(np.ptp(upper_y)) if len(upper_y) >= 2 else 40.0
        thickness = max(4, int(round(max(box_height, 40.0) * 0.055)))
        for first, second in edges:
            if valid[first] and valid[second]:
                p1 = tuple(np.rint(points[first, :2]).astype(int))
                p2 = tuple(np.rint(points[second, :2]).astype(int))
                cv2.line(output[frame], p1, p2, 255, thickness, cv2.LINE_AA)
        for joint in (5, 6, 7, 8, 9, 10):
            if valid[joint]:
                centre = tuple(np.rint(points[joint, :2]).astype(int))
                cv2.circle(output[frame], centre, max(2, thickness // 2), 255, -1)
    return output
