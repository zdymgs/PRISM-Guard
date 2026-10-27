from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from audit_yolo11_pose_skeleton import atomic_json, atomic_npz, safe_name, write_csv


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "six_modality_audit" / "train_union_manifest.csv"
DEFAULT_POSE_RUN = PROJECT_DIR / "runs" / "adaptive_ir_pose_skeleton_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "dir_multiscale_roi_full"

REGION_NAMES = (
    "full_body",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "hand_workspace",
    "global_fallback",
)
REGION_INDEX = {name: index for index, name in enumerate(REGION_NAMES)}

SOURCE_INVALID = np.uint8(0)
SOURCE_PRIMARY = np.uint8(1)
SOURCE_FALLBACK = np.uint8(2)
SOURCE_TEMPORAL_INTERPOLATION = np.uint8(3)
SOURCE_EXTRAPOLATED_GEOMETRY = np.uint8(4)
SOURCE_BROAD_FALLBACK = np.uint8(5)
SOURCE_GLOBAL_FRAME = np.uint8(6)

LEFT_ARM = (5, 7, 9)
RIGHT_ARM = (6, 8, 10)

# pose retry-reason bit mask.  These flags describe why the primary 640/conf=.10
# pass was considered unreliable.  When the fallback pass did not replace that
# primary pose, high raw keypoint confidence alone must not make a local ROI look
# trustworthy (the common failure is a close/partially visible person).
RETRY_MISSING_PERSON = np.uint8(1)
RETRY_LOW_BOX_CONF = np.uint8(2)
RETRY_SMALL_PERSON = np.uint8(4)
RETRY_WEAK_UPPER_BODY = np.uint8(8)
RETRY_WEAK_WRISTS = np.uint8(16)
RETRY_TEMPORAL_JUMP = np.uint8(32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build seven full-frame D/IR ROI geometries from cached YOLO11-pose tracks."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pose-run", type=Path, default=DEFAULT_POSE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--keypoint-conf", type=float, default=0.25)
    parser.add_argument("--arm-margin", type=float, default=0.35)
    parser.add_argument("--hand-forearm-scale", type=float, default=1.60)
    parser.add_argument("--hand-min-person-height", type=float, default=0.12)
    parser.add_argument("--hand-max-person-height", type=float, default=0.35)
    parser.add_argument("--workspace-margin", type=float, default=0.15)
    parser.add_argument("--single-hand-workspace-margin", type=float, default=0.30)
    parser.add_argument("--max-joint-gap", type=int, default=3)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pose_cache_path(pose_run: Path, sample_id: str) -> Path:
    return pose_run / "trial_cache" / safe_name(sample_id).with_suffix(".npz")


def select_rows(rows: list[dict[str, str]], pose_run: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    selected = [row for row in rows if pose_cache_path(pose_run, row["sample_id"]).is_file()]
    selected.sort(key=lambda row: (int(row["class_id"]), row["user_id"], row["trial_id"]))
    if args.sample_id:
        requested = set(args.sample_id)
        selected = [row for row in selected if row["sample_id"] in requested]
        missing = sorted(requested - {row["sample_id"] for row in selected})
        if missing:
            raise KeyError(f"Requested pose caches unavailable: {missing}")
    if args.limit_per_class > 0:
        counts: dict[int, int] = defaultdict(int)
        limited: list[dict[str, str]] = []
        for row in selected:
            class_id = int(row["class_id"])
            if counts[class_id] >= args.limit_per_class:
                continue
            counts[class_id] += 1
            limited.append(row)
        selected = limited
    if args.max_trials > 0:
        selected = selected[: args.max_trials]
    return selected


def finite_box(box: np.ndarray) -> bool:
    return bool(np.isfinite(box[:4]).all() and box[2] > box[0] and box[3] > box[1])


def clip_box(
    raw_box: np.ndarray | list[float], width: int, height: int
) -> tuple[np.ndarray, float]:
    raw = np.asarray(raw_box, dtype=np.float32)
    raw_width = max(float(raw[2] - raw[0]), 1e-6)
    raw_height = max(float(raw[3] - raw[1]), 1e-6)
    raw_area = raw_width * raw_height
    clipped = raw.copy()
    clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0.0, float(width - 1))
    clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0.0, float(height - 1))
    clipped_area = max(float(clipped[2] - clipped[0]), 0.0) * max(
        float(clipped[3] - clipped[1]), 0.0
    )
    clipped_ratio = float(np.clip(1.0 - clipped_area / raw_area, 0.0, 1.0))
    return clipped.astype(np.float32), clipped_ratio


def expand_box(
    box: np.ndarray, margin: float, width: int, height: int
) -> tuple[np.ndarray, float]:
    box_width = max(float(box[2] - box[0]), 1.0)
    box_height = max(float(box[3] - box[1]), 1.0)
    raw = np.asarray(
        (
            float(box[0]) - margin * box_width,
            float(box[1]) - margin * box_height,
            float(box[2]) + margin * box_width,
            float(box[3]) + margin * box_height,
        ),
        dtype=np.float32,
    )
    return clip_box(raw, width, height)


def union_boxes(boxes: list[np.ndarray], width: int, height: int, margin: float) -> tuple[np.ndarray, float]:
    raw = np.asarray(
        (
            min(float(box[0]) for box in boxes),
            min(float(box[1]) for box in boxes),
            max(float(box[2]) for box in boxes),
            max(float(box[3]) for box in boxes),
        ),
        dtype=np.float32,
    )
    if margin > 0:
        return expand_box(raw, margin, width, height)
    return clip_box(raw, width, height)


def interpolate_arm_joints(
    keypoints: np.ndarray, keypoint_conf: float, max_gap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    working = keypoints.copy()
    joint_source = np.zeros(keypoints.shape[:2], dtype=np.uint8)
    joint_quality = np.zeros(keypoints.shape[:2], dtype=np.float32)
    raw_valid = np.isfinite(keypoints[:, :, 2]) & (keypoints[:, :, 2] >= keypoint_conf)
    joint_source[raw_valid] = SOURCE_PRIMARY
    joint_quality[raw_valid] = np.clip(keypoints[:, :, 2][raw_valid], 0.0, 1.0)
    for joint in sorted(set(LEFT_ARM + RIGHT_ARM)):
        valid = raw_valid[:, joint].copy()
        index = 0
        while index < len(working):
            if valid[index]:
                index += 1
                continue
            start = index
            while index < len(working) and not valid[index]:
                index += 1
            end = index
            gap = end - start
            if start == 0 or end == len(working) or gap > max_gap:
                continue
            left = working[start - 1, joint]
            right = working[end, joint]
            if not (np.isfinite(left[:2]).all() and np.isfinite(right[:2]).all()):
                continue
            endpoint_quality = 0.5 * min(float(left[2]), float(right[2]))
            for offset, frame_index in enumerate(range(start, end), 1):
                weight = float(offset) / float(gap + 1)
                working[frame_index, joint, :2] = (1.0 - weight) * left[:2] + weight * right[:2]
                working[frame_index, joint, 2] = endpoint_quality
                joint_source[frame_index, joint] = SOURCE_TEMPORAL_INTERPOLATION
                joint_quality[frame_index, joint] = endpoint_quality
                valid[frame_index] = True
    return working.astype(np.float32), joint_source, joint_quality


def detect_lr_ambiguity(
    joints: np.ndarray, joint_source: np.ndarray, person_boxes: np.ndarray
) -> np.ndarray:
    ambiguous = np.zeros(len(joints), dtype=bool)
    previous_index: int | None = None
    for index in range(len(joints)):
        if not (joint_source[index, 9] and joint_source[index, 10]):
            continue
        if previous_index is not None and index == previous_index + 1:
            current_left = joints[index, 9, :2]
            current_right = joints[index, 10, :2]
            previous_left = joints[previous_index, 9, :2]
            previous_right = joints[previous_index, 10, :2]
            same_cost = float(np.linalg.norm(current_left - previous_left)) + float(
                np.linalg.norm(current_right - previous_right)
            )
            swapped_cost = float(np.linalg.norm(current_left - previous_right)) + float(
                np.linalg.norm(current_right - previous_left)
            )
            if finite_box(person_boxes[index]):
                person_height = max(float(person_boxes[index, 3] - person_boxes[index, 1]), 1.0)
            else:
                person_height = 200.0
            if swapped_cost + 0.08 * person_height < same_cost:
                ambiguous[index] = True
                ambiguous[previous_index] = True
        previous_index = index
    # Include immediate neighbours because a one-frame swap often straddles the transition.
    expanded = ambiguous.copy()
    expanded[:-1] |= ambiguous[1:]
    expanded[1:] |= ambiguous[:-1]
    return expanded


def arm_geometry(
    joint_indices: tuple[int, int, int],
    joints: np.ndarray,
    joint_source: np.ndarray,
    joint_quality: np.ndarray,
    person_box: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, bool, float, np.uint8, float]:
    available = [index for index in joint_indices if joint_source[index] != SOURCE_INVALID]
    person_width = max(float(person_box[2] - person_box[0]), 1.0)
    person_height = max(float(person_box[3] - person_box[1]), 1.0)
    if len(available) >= 2:
        points = [joints[index, :2].copy() for index in available]
        source = SOURCE_PRIMARY
        source_factor = 1.0
        if len(available) < 3:
            shoulder, elbow, wrist = joint_indices
            if shoulder in available and elbow in available and wrist not in available:
                estimated_wrist = joints[elbow, :2] + 0.90 * (
                    joints[elbow, :2] - joints[shoulder, :2]
                )
                points.append(estimated_wrist)
            source = SOURCE_EXTRAPOLATED_GEOMETRY
            source_factor = 0.45
        elif any(joint_source[index] == SOURCE_TEMPORAL_INTERPOLATION for index in available):
            source = SOURCE_TEMPORAL_INTERPOLATION
            source_factor = 0.65
        point_array = np.stack(points)
        center = 0.5 * (point_array.min(axis=0) + point_array.max(axis=0))
        span = point_array.max(axis=0) - point_array.min(axis=0)
        span[0] = max(float(span[0]), 0.08 * person_width)
        span[1] = max(float(span[1]), 0.08 * person_height)
        half = 0.5 * span * (1.0 + 2.0 * args.arm_margin)
        box, clipped_ratio = clip_box(
            (center[0] - half[0], center[1] - half[1], center[0] + half[0], center[1] + half[1]),
            args.width,
            args.height,
        )
        quality = source_factor * float(np.mean([joint_quality[index] for index in available]))
        return box, True, float(np.clip(quality, 0.0, 1.0)), source, clipped_ratio
    if finite_box(person_box):
        # Both labelled arm fallbacks intentionally receive the same upper-body area.
        # This keeps pixels without pretending that the left/right identity is known.
        upper = np.asarray(
            (person_box[0], person_box[1], person_box[2], person_box[1] + 0.68 * person_height),
            dtype=np.float32,
        )
        box, clipped_ratio = clip_box(upper, args.width, args.height)
        return box, True, 0.10, SOURCE_BROAD_FALLBACK, clipped_ratio
    return np.full(4, np.nan, dtype=np.float32), False, 0.0, SOURCE_INVALID, 0.0


def hand_geometry(
    shoulder: int,
    elbow: int,
    wrist: int,
    joints: np.ndarray,
    joint_source: np.ndarray,
    joint_quality: np.ndarray,
    person_box: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, bool, float, np.uint8, float]:
    if not finite_box(person_box):
        return np.full(4, np.nan, dtype=np.float32), False, 0.0, SOURCE_INVALID, 0.0
    person_height = max(float(person_box[3] - person_box[1]), 1.0)
    source = SOURCE_INVALID
    quality = 0.0
    if joint_source[wrist] and joint_source[elbow]:
        wrist_point = joints[wrist, :2]
        elbow_point = joints[elbow, :2]
        vector = wrist_point - elbow_point
        forearm = max(float(np.linalg.norm(vector)), 1.0)
        center = wrist_point + 0.15 * vector
        source = SOURCE_PRIMARY
        factor = 1.0
        if (
            joint_source[wrist] == SOURCE_TEMPORAL_INTERPOLATION
            or joint_source[elbow] == SOURCE_TEMPORAL_INTERPOLATION
        ):
            source = SOURCE_TEMPORAL_INTERPOLATION
            factor = 0.65
        quality = factor * min(float(joint_quality[wrist]), float(joint_quality[elbow]))
    elif joint_source[wrist]:
        center = joints[wrist, :2]
        forearm = 0.14 * person_height
        source = SOURCE_EXTRAPOLATED_GEOMETRY
        quality = 0.40 * float(joint_quality[wrist])
    elif joint_source[shoulder] and joint_source[elbow]:
        shoulder_point = joints[shoulder, :2]
        elbow_point = joints[elbow, :2]
        vector = elbow_point - shoulder_point
        forearm = max(0.90 * float(np.linalg.norm(vector)), 1.0)
        estimated_wrist = elbow_point + 0.90 * vector
        center = estimated_wrist + 0.15 * vector
        source = SOURCE_EXTRAPOLATED_GEOMETRY
        quality = 0.25 * min(float(joint_quality[shoulder]), float(joint_quality[elbow]))
    else:
        return np.full(4, np.nan, dtype=np.float32), False, 0.0, SOURCE_INVALID, 0.0
    side = float(
        np.clip(
            args.hand_forearm_scale * forearm,
            args.hand_min_person_height * person_height,
            args.hand_max_person_height * person_height,
        )
    )
    box, clipped_ratio = clip_box(
        (center[0] - side / 2, center[1] - side / 2, center[0] + side / 2, center[1] + side / 2),
        args.width,
        args.height,
    )
    return box, True, float(np.clip(quality, 0.0, 1.0)), source, clipped_ratio


def validated_geometry(
    geometry: tuple[np.ndarray, bool, float, np.uint8, float]
) -> tuple[np.ndarray, bool, float, np.uint8, float]:
    """Invalidate a nominal ROI when clipping collapsed it at an image edge."""
    box, valid, quality, source, clipped_ratio = geometry
    if valid and finite_box(box):
        return box, True, quality, source, clipped_ratio
    return np.full(4, np.nan, dtype=np.float32), False, 0.0, SOURCE_INVALID, clipped_ratio


def assign_region(
    boxes: np.ndarray,
    valid: np.ndarray,
    quality: np.ndarray,
    source: np.ndarray,
    clipped: np.ndarray,
    frame_index: int,
    region_name: str,
    geometry: tuple[np.ndarray, bool, float, np.uint8, float],
) -> None:
    region = REGION_INDEX[region_name]
    region_box, region_valid, region_quality, region_source, clipped_ratio = validated_geometry(geometry)
    boxes[frame_index, region] = region_box
    valid[frame_index, region] = region_valid
    quality[frame_index, region] = region_quality
    source[frame_index, region] = region_source
    clipped[frame_index, region] = clipped_ratio


def pose_quality_factor(
    final_pose_source: np.ndarray, retry_reason_bits: np.ndarray
) -> np.ndarray:
    """Turn pose reliability diagnostics into a conservative local-ROI weight.

    A fallback pose (source=2) has already superseded the warned primary pose,
    so its geometry is judged from its final keypoint confidences.  A warned
    primary pose that survived fallback is retained as a candidate crop but is
    explicitly down-weighted for downstream soft fusion.
    """
    factor = np.ones(len(final_pose_source), dtype=np.float32)
    primary = final_pose_source == SOURCE_PRIMARY
    weak_pose = (retry_reason_bits & (RETRY_WEAK_UPPER_BODY | RETRY_WEAK_WRISTS)) != 0
    temporal_jump = (retry_reason_bits & RETRY_TEMPORAL_JUMP) != 0
    small_person = (retry_reason_bits & RETRY_SMALL_PERSON) != 0
    low_box_conf = (retry_reason_bits & RETRY_LOW_BOX_CONF) != 0

    factor[primary & low_box_conf] = np.minimum(factor[primary & low_box_conf], 0.80)
    factor[primary & small_person] = np.minimum(factor[primary & small_person], 0.80)
    factor[primary & temporal_jump] = np.minimum(factor[primary & temporal_jump], 0.60)
    factor[primary & weak_pose] = np.minimum(factor[primary & weak_pose], 0.25)
    factor[final_pose_source == SOURCE_TEMPORAL_INTERPOLATION] = 0.25
    factor[final_pose_source == SOURCE_INVALID] = 0.0
    return factor


def build_trial_rois(pose: np.lib.npyio.NpzFile, args: argparse.Namespace) -> dict[str, np.ndarray]:
    frame_ids = pose["frame_ids"]
    keypoints = pose["ir_keypoints_xy_conf"]
    final_pose_source = pose["final_pose_source"].astype(np.uint8)
    retry_reason_bits = pose["retry_reason_bits"].astype(np.uint8)
    pose_quality_factor = pose_quality_factor(final_pose_source, retry_reason_bits)
    person_roi = pose["ir_person_roi_boxes_xyxy_conf"]
    full_context = pose["depth_search_boxes_from_ir"]
    roi_box_estimated = pose["roi_box_estimated"].astype(bool)
    frame_count = len(frame_ids)

    joints, joint_source, joint_quality = interpolate_arm_joints(
        keypoints, args.keypoint_conf, args.max_joint_gap
    )
    lr_ambiguous = detect_lr_ambiguity(joints, joint_source, person_roi)
    boxes = np.full((frame_count, len(REGION_NAMES), 4), np.nan, dtype=np.float32)
    valid = np.zeros((frame_count, len(REGION_NAMES)), dtype=bool)
    quality = np.zeros((frame_count, len(REGION_NAMES)), dtype=np.float32)
    source = np.zeros((frame_count, len(REGION_NAMES)), dtype=np.uint8)
    clipped = np.zeros((frame_count, len(REGION_NAMES)), dtype=np.float32)

    for frame_index in range(frame_count):
        person_box = person_roi[frame_index, :4]
        full_box = full_context[frame_index, :4]
        if finite_box(full_box):
            clipped_full, clipped_ratio = clip_box(full_box, args.width, args.height)
            pose_source = final_pose_source[frame_index]
            if roi_box_estimated[frame_index]:
                region_source = SOURCE_BROAD_FALLBACK
                full_quality = 0.20
            elif pose_source == SOURCE_TEMPORAL_INTERPOLATION:
                region_source = SOURCE_TEMPORAL_INTERPOLATION
                full_quality = 0.35
            elif pose_source == SOURCE_FALLBACK:
                region_source = SOURCE_FALLBACK
                full_quality = float(np.clip(full_context[frame_index, 4], 0.0, 1.0))
            else:
                region_source = SOURCE_PRIMARY
                full_quality = float(np.clip(full_context[frame_index, 4], 0.0, 1.0))
            assign_region(
                boxes,
                valid,
                quality,
                source,
                clipped,
                frame_index,
                "full_body",
                (clipped_full, True, full_quality, region_source, clipped_ratio),
            )

        left_arm = arm_geometry(
            LEFT_ARM,
            joints[frame_index],
            joint_source[frame_index],
            joint_quality[frame_index],
            person_box,
            args,
        )
        right_arm = arm_geometry(
            RIGHT_ARM,
            joints[frame_index],
            joint_source[frame_index],
            joint_quality[frame_index],
            person_box,
            args,
        )
        left_hand = hand_geometry(
            5,
            7,
            9,
            joints[frame_index],
            joint_source[frame_index],
            joint_quality[frame_index],
            person_box,
            args,
        )
        right_hand = hand_geometry(
            6,
            8,
            10,
            joints[frame_index],
            joint_source[frame_index],
            joint_quality[frame_index],
            person_box,
            args,
        )
        left_arm = validated_geometry(left_arm)
        right_arm = validated_geometry(right_arm)
        left_hand = validated_geometry(left_hand)
        right_hand = validated_geometry(right_hand)
        frame_pose_factor = float(pose_quality_factor[frame_index])
        left_arm = (left_arm[0], left_arm[1], frame_pose_factor * left_arm[2], left_arm[3], left_arm[4])
        right_arm = (right_arm[0], right_arm[1], frame_pose_factor * right_arm[2], right_arm[3], right_arm[4])
        left_hand = (left_hand[0], left_hand[1], frame_pose_factor * left_hand[2], left_hand[3], left_hand[4])
        right_hand = (right_hand[0], right_hand[1], frame_pose_factor * right_hand[2], right_hand[3], right_hand[4])
        if lr_ambiguous[frame_index]:
            left_arm = (left_arm[0], left_arm[1], 0.5 * left_arm[2], left_arm[3], left_arm[4])
            right_arm = (right_arm[0], right_arm[1], 0.5 * right_arm[2], right_arm[3], right_arm[4])
            left_hand = (left_hand[0], left_hand[1], 0.5 * left_hand[2], left_hand[3], left_hand[4])
            right_hand = (right_hand[0], right_hand[1], 0.5 * right_hand[2], right_hand[3], right_hand[4])
        assign_region(boxes, valid, quality, source, clipped, frame_index, "left_arm", left_arm)
        assign_region(boxes, valid, quality, source, clipped, frame_index, "right_arm", right_arm)
        assign_region(boxes, valid, quality, source, clipped, frame_index, "left_hand", left_hand)
        assign_region(boxes, valid, quality, source, clipped, frame_index, "right_hand", right_hand)

        hand_boxes = [geometry[0] for geometry in (left_hand, right_hand) if geometry[1]]
        hand_qualities = [geometry[2] for geometry in (left_hand, right_hand) if geometry[1]]
        if len(hand_boxes) == 2:
            workspace_box, clipped_ratio = union_boxes(
                hand_boxes, args.width, args.height, args.workspace_margin
            )
            workspace_quality = float(np.mean(hand_qualities))
            workspace_source = max(left_hand[3], right_hand[3])
            workspace = (workspace_box, True, workspace_quality, workspace_source, clipped_ratio)
        elif len(hand_boxes) == 1:
            workspace_box, clipped_ratio = expand_box(
                hand_boxes[0], args.single_hand_workspace_margin, args.width, args.height
            )
            workspace = (
                workspace_box,
                True,
                0.70 * hand_qualities[0],
                SOURCE_EXTRAPOLATED_GEOMETRY,
                clipped_ratio,
            )
        elif finite_box(full_box):
            person_height = max(float(full_box[3] - full_box[1]), 1.0)
            upper = np.asarray(
                (full_box[0], full_box[1], full_box[2], full_box[1] + 0.68 * person_height),
                dtype=np.float32,
            )
            workspace_box, clipped_ratio = clip_box(upper, args.width, args.height)
            workspace = (workspace_box, True, 0.08, SOURCE_BROAD_FALLBACK, clipped_ratio)
        else:
            workspace = (
                np.full(4, np.nan, dtype=np.float32),
                False,
                0.0,
                SOURCE_INVALID,
                0.0,
            )
        assign_region(
            boxes, valid, quality, source, clipped, frame_index, "hand_workspace", workspace
        )

        global_box = np.asarray((0.0, 0.0, float(args.width - 1), float(args.height - 1)), dtype=np.float32)
        assign_region(
            boxes,
            valid,
            quality,
            source,
            clipped,
            frame_index,
            "global_fallback",
            (global_box, True, 1.0, SOURCE_GLOBAL_FRAME, 0.0),
        )

    return {
        "frame_ids": frame_ids,
        "region_names": np.asarray(REGION_NAMES),
        "roi_boxes_xyxy": boxes,
        "roi_valid": valid,
        "roi_quality": quality,
        "roi_source": source,
        "roi_clipped_ratio": clipped,
        "pose_quality_factor": pose_quality_factor,
        "left_right_ambiguous": lr_ambiguous,
        "arm_joint_xy_conf_for_roi": joints[:, sorted(set(LEFT_ARM + RIGHT_ARM))],
        "arm_joint_source_for_roi": joint_source[:, sorted(set(LEFT_ARM + RIGHT_ARM))],
        "arm_joint_quality_for_roi": joint_quality[:, sorted(set(LEFT_ARM + RIGHT_ARM))],
        "image_width": np.asarray(args.width, dtype=np.int32),
        "image_height": np.asarray(args.height, dtype=np.int32),
    }


def trial_summary(row: dict[str, str], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    valid = arrays["roi_valid"]
    quality = arrays["roi_quality"]
    clipped = arrays["roi_clipped_ratio"]
    boxes = arrays["roi_boxes_xyxy"]
    frame_count = len(valid)
    summary: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "class_id": int(row["class_id"]),
        "class_name": row["class_name"],
        "user_id": row["user_id"],
        "trial_id": row["trial_id"],
        "frames": frame_count,
        "left_right_ambiguous_frames": int(arrays["left_right_ambiguous"].sum()),
        "left_right_ambiguous_rate": float(arrays["left_right_ambiguous"].mean()),
    }
    image_area = float(int(arrays["image_width"]) * int(arrays["image_height"]))
    for region_index, region_name in enumerate(REGION_NAMES):
        region_valid = valid[:, region_index]
        region_quality = quality[:, region_index]
        region_boxes = boxes[:, region_index]
        areas = np.maximum(region_boxes[:, 2] - region_boxes[:, 0], 0.0) * np.maximum(
            region_boxes[:, 3] - region_boxes[:, 1], 0.0
        )
        summary[f"{region_name}_valid_rate"] = float(region_valid.mean())
        summary[f"{region_name}_quality_ge_025_rate"] = float(
            (region_valid & (region_quality >= 0.25)).mean()
        )
        summary[f"{region_name}_mean_quality"] = (
            float(region_quality[region_valid].mean()) if region_valid.any() else 0.0
        )
        summary[f"{region_name}_mean_area_ratio"] = (
            float((areas[region_valid] / image_area).mean()) if region_valid.any() else 0.0
        )
        summary[f"{region_name}_clipped_ge_025_rate"] = float(
            (region_valid & (clipped[:, region_index] >= 0.25)).mean()
        )
    summary["both_hands_valid_rate"] = float(
        (valid[:, REGION_INDEX["left_hand"]] & valid[:, REGION_INDEX["right_hand"]]).mean()
    )
    summary["any_hand_valid_rate"] = float(
        (valid[:, REGION_INDEX["left_hand"]] | valid[:, REGION_INDEX["right_hand"]]).mean()
    )
    return summary


def aggregate(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    metrics = (
        "full_body_valid_rate",
        "left_arm_quality_ge_025_rate",
        "right_arm_quality_ge_025_rate",
        "left_hand_quality_ge_025_rate",
        "right_hand_quality_ge_025_rate",
        "hand_workspace_quality_ge_025_rate",
        "both_hands_valid_rate",
        "left_right_ambiguous_rate",
    )
    output: list[dict[str, Any]] = []
    for group_name, group in sorted(groups.items()):
        result: dict[str, Any] = {key: group_name, "trials": len(group), "frames": sum(row["frames"] for row in group)}
        total_frames = max(int(result["frames"]), 1)
        for metric in metrics:
            result[metric] = float(
                sum(float(row[metric]) * int(row["frames"]) for row in group) / total_frames
            )
        output.append(result)
    return output


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    pose_run = args.pose_run.resolve()
    output_dir = args.output_dir.resolve()
    rows = load_manifest(manifest)
    selected = select_rows(rows, pose_run, args)
    if not selected:
        raise RuntimeError("No matching pose caches selected")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "version": 1,
        "stage": "36_step_07_dir_multiscale_roi_geometry",
        "manifest": str(manifest),
        "pose_run": str(pose_run),
        "selected_trials": len(selected),
        "selected_classes": len({row["class_id"] for row in selected}),
        "selected_subjects": len({row["user_id"] for row in selected}),
        "image_size": [args.width, args.height],
        "region_names": REGION_NAMES,
        "keypoint_conf": args.keypoint_conf,
        "arm_margin": args.arm_margin,
        "hand_forearm_scale": args.hand_forearm_scale,
        "hand_person_height_range": [args.hand_min_person_height, args.hand_max_person_height],
        "workspace_margin": args.workspace_margin,
        "single_hand_workspace_margin": args.single_hand_workspace_margin,
        "max_joint_gap": args.max_joint_gap,
        "source_codes": {
            "0": "invalid",
            "1": "primary_pose_geometry",
            "2": "fallback_pose_geometry",
            "3": "temporal_interpolation",
            "4": "extrapolated_geometry",
            "5": "broad_fallback",
            "6": "global_frame",
        },
        "coordinate_contract": "one IR-derived XYXY geometry is applied unchanged to same-frame IR and Depth",
        "pixel_policy": "no crop images are persisted; downstream reads original frames using this geometry",
    }
    atomic_json(output_dir / "config.json", config)

    summaries: list[dict[str, Any]] = []
    for trial_index, row in enumerate(selected, 1):
        relative = safe_name(row["sample_id"])
        output_cache = output_dir / "trial_roi_cache" / relative.with_suffix(".npz")
        output_summary = output_dir / "trial_summary" / relative.with_suffix(".json")
        if output_cache.is_file() and output_summary.is_file() and not args.overwrite:
            summaries.append(json.loads(output_summary.read_text(encoding="utf-8")))
            print(f"[{trial_index}/{len(selected)}] resume {row['sample_id']}", flush=True)
            continue
        input_cache = pose_cache_path(pose_run, row["sample_id"])
        with np.load(input_cache) as pose:
            arrays = build_trial_rois(pose, args)
        atomic_npz(output_cache, **arrays)
        summary = trial_summary(row, arrays)
        atomic_json(output_summary, summary)
        summaries.append(summary)
        print(
            f"[{trial_index}/{len(selected)}] {row['sample_id']} frames={summary['frames']} "
            f"hands={summary['both_hands_valid_rate']:.3f} "
            f"workspaceQ={summary['hand_workspace_quality_ge_025_rate']:.3f} "
            f"lrAmb={summary['left_right_ambiguous_rate']:.3f}",
            flush=True,
        )

    write_csv(output_dir / "trial_summary.csv", summaries)
    write_csv(output_dir / "per_class_summary.csv", aggregate(summaries, "class_id"))
    write_csv(output_dir / "per_subject_summary.csv", aggregate(summaries, "user_id"))
    total_frames = sum(int(row["frames"]) for row in summaries)
    final: dict[str, Any] = {
        **config,
        "completed_trials": len(summaries),
        "completed_frames": total_frames,
    }
    for metric in (
        "full_body_valid_rate",
        "left_arm_quality_ge_025_rate",
        "right_arm_quality_ge_025_rate",
        "left_hand_quality_ge_025_rate",
        "right_hand_quality_ge_025_rate",
        "hand_workspace_quality_ge_025_rate",
        "both_hands_valid_rate",
        "any_hand_valid_rate",
        "left_right_ambiguous_rate",
    ):
        final[metric] = float(
            sum(float(row[metric]) * int(row["frames"]) for row in summaries) / max(total_frames, 1)
        )
    atomic_json(output_dir / "summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
