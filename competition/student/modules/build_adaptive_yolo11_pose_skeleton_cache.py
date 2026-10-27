from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO, __version__ as ultralytics_version

from audit_yolo11_pose_skeleton import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    SEMANTIC_PAIRS,
    atomic_json,
    atomic_npz,
    bbox_local_keypoints,
    box_iou,
    expanded_search_boxes,
    frame_map,
    load_raw_skeleton,
    load_rows,
    modality_summary,
    normalise_skeleton,
    result_candidates,
    safe_name,
    select_rows,
    select_track,
    semantic_pair_features,
    write_csv,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "adaptive_ir_pose_skeleton_full"
UPPER_BODY_INDICES = np.asarray((5, 6, 7, 8, 9, 10), dtype=np.int64)
WRIST_INDICES = np.asarray((9, 10), dtype=np.int64)

# Bit flags are saved per frame so downstream code can audit why a retry happened.
RETRY_MISSING = np.uint8(1)
RETRY_LOW_BOX_CONF = np.uint8(2)
RETRY_SMALL_PERSON = np.uint8(4)
RETRY_WEAK_UPPER_BODY = np.uint8(8)
RETRY_WEAK_WRISTS = np.uint8(16)
RETRY_TEMPORAL_JUMP = np.uint8(32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an automatic two-pass IR YOLO11-pose cache, copy the final IR geometry "
            "to Depth, and pair YOLO joints semantically with the dataset Skeleton."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--primary-imgsz", type=int, default=640)
    parser.add_argument("--primary-conf", type=float, default=0.10)
    parser.add_argument("--fallback-imgsz", type=int, default=1280)
    parser.add_argument("--fallback-conf", type=float, default=0.03)
    parser.add_argument("--keypoint-conf", type=float, default=0.25)
    parser.add_argument("--retry-box-conf", type=float, default=0.15)
    parser.add_argument("--retry-min-height-ratio", type=float, default=0.20)
    parser.add_argument("--retry-min-upper-joints", type=int, default=4)
    parser.add_argument("--retry-min-wrists", type=int, default=2)
    parser.add_argument("--retry-center-jump-ratio", type=float, default=0.15)
    parser.add_argument("--retry-area-ratio", type=float, default=2.5)
    parser.add_argument("--interpolate-box-gap", type=int, default=3)
    parser.add_argument("--transfer-margin", type=float, default=0.20)
    parser.add_argument("--primary-batch", type=int, default=32)
    parser.add_argument("--fallback-batch", type=int, default=16)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def predict_candidates(
    model: YOLO,
    paths: list[Path],
    *,
    imgsz: int,
    conf: float,
    batch: int,
    device: str,
) -> list[dict[str, np.ndarray]]:
    if not paths:
        return []
    results = model.predict(
        source=[str(path) for path in paths],
        stream=True,
        imgsz=imgsz,
        conf=conf,
        max_det=5,
        device=device,
        half=str(device).lower() != "cpu",
        batch=batch,
        verbose=False,
    )
    candidates = [result_candidates(result) for result in results]
    if len(candidates) != len(paths):
        raise RuntimeError(f"YOLO result count mismatch: {len(candidates)} != {len(paths)}")
    return candidates


def pose_counts(keypoints: np.ndarray, keypoint_conf: float) -> tuple[np.ndarray, np.ndarray]:
    confidence = keypoints[:, :, 2]
    valid = np.isfinite(confidence) & (confidence >= keypoint_conf)
    return valid[:, UPPER_BODY_INDICES].sum(axis=1), valid[:, WRIST_INDICES].sum(axis=1)


def temporal_jump_mask(
    boxes: np.ndarray,
    *,
    image_width: float,
    image_height: float,
    center_ratio: float,
    area_ratio: float,
) -> np.ndarray:
    jump = np.zeros(len(boxes), dtype=bool)
    diagonal = max(math.hypot(image_width, image_height), 1.0)
    valid = np.flatnonzero(np.isfinite(boxes[:, :4]).all(axis=1))
    for previous, current in zip(valid[:-1], valid[1:]):
        # A long missing run is handled by RETRY_MISSING, not by comparing distant poses.
        if current != previous + 1:
            continue
        previous_box = boxes[previous]
        current_box = boxes[current]
        previous_center = np.asarray(
            ((previous_box[0] + previous_box[2]) * 0.5, (previous_box[1] + previous_box[3]) * 0.5)
        )
        current_center = np.asarray(
            ((current_box[0] + current_box[2]) * 0.5, (current_box[1] + current_box[3]) * 0.5)
        )
        center_jump = float(np.linalg.norm(current_center - previous_center)) / diagonal
        previous_area = max(
            float(previous_box[2] - previous_box[0]) * float(previous_box[3] - previous_box[1]), 1.0
        )
        current_area = max(
            float(current_box[2] - current_box[0]) * float(current_box[3] - current_box[1]), 1.0
        )
        ratio = max(previous_area / current_area, current_area / previous_area)
        if center_jump > center_ratio or ratio > area_ratio:
            # Retry both sides: this lets the fallback resolve which side of the transition is stable.
            jump[previous] = True
            jump[current] = True
    return jump


def retry_reasons(
    boxes: np.ndarray,
    keypoints: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    keypoint_conf: float,
    box_conf: float,
    min_height_ratio: float,
    min_upper_joints: int,
    min_wrists: int,
    center_jump_ratio: float,
    area_ratio: float,
) -> np.ndarray:
    detected = np.isfinite(boxes[:, :4]).all(axis=1)
    reasons = np.zeros(len(boxes), dtype=np.uint8)
    reasons[~detected] |= RETRY_MISSING
    reasons[detected & (boxes[:, 4] < box_conf)] |= RETRY_LOW_BOX_CONF
    height_ratio = (boxes[:, 3] - boxes[:, 1]) / max(float(image_height), 1.0)
    reasons[detected & (height_ratio < min_height_ratio)] |= RETRY_SMALL_PERSON
    upper_count, wrist_count = pose_counts(keypoints, keypoint_conf)
    reasons[detected & (upper_count < min_upper_joints)] |= RETRY_WEAK_UPPER_BODY
    reasons[detected & (wrist_count < min_wrists)] |= RETRY_WEAK_WRISTS
    jump = temporal_jump_mask(
        boxes,
        image_width=image_width,
        image_height=image_height,
        center_ratio=center_jump_ratio,
        area_ratio=area_ratio,
    )
    reasons[jump] |= RETRY_TEMPORAL_JUMP
    return reasons


def nearest_reference_box(boxes: np.ndarray, frame_index: int) -> np.ndarray | None:
    if np.isfinite(boxes[frame_index, :4]).all():
        return boxes[frame_index]
    valid = np.flatnonzero(np.isfinite(boxes[:, :4]).all(axis=1))
    if not len(valid):
        return None
    before = valid[valid < frame_index]
    after = valid[valid > frame_index]
    if len(before) and len(after):
        left = int(before[-1])
        right = int(after[0])
        weight = float(frame_index - left) / float(right - left)
        return ((1.0 - weight) * boxes[left] + weight * boxes[right]).astype(np.float32)
    nearest = int(valid[np.argmin(np.abs(valid - frame_index))])
    return boxes[nearest]


def candidate_quality(keypoints: np.ndarray, box_conf: float, keypoint_conf: float) -> tuple[int, int, float]:
    confidence = keypoints[:, 2]
    valid = np.isfinite(confidence) & (confidence >= keypoint_conf)
    wrists = int(valid[WRIST_INDICES].sum())
    upper = int(valid[UPPER_BODY_INDICES].sum())
    return wrists, upper, float(box_conf)


def select_fallback_candidate(
    candidate: dict[str, np.ndarray],
    reference: np.ndarray | None,
    keypoint_conf: float,
    strict_reference_scale: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    frame_boxes = candidate["boxes"]
    frame_keypoints = candidate["keypoints"]
    if not len(frame_boxes):
        return None
    height, width = map(float, candidate["shape"])
    diagonal = max(math.hypot(width, height), 1.0)
    scores: list[float] = []
    for box, keypoints in zip(frame_boxes, frame_keypoints):
        wrists, upper, confidence = candidate_quality(keypoints, float(box[4]), keypoint_conf)
        score = confidence + 0.12 * wrists + 0.08 * upper
        if reference is not None and np.isfinite(reference[:4]).all():
            reference_center = np.asarray(
                ((reference[0] + reference[2]) * 0.5, (reference[1] + reference[3]) * 0.5)
            )
            center = np.asarray(((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5))
            center_distance = float(np.linalg.norm(center - reference_center)) / diagonal
            overlap = box_iou(reference, box)
            x0 = max(float(reference[0]), float(box[0]))
            y0 = max(float(reference[1]), float(box[1]))
            x1 = min(float(reference[2]), float(box[2]))
            y1 = min(float(reference[3]), float(box[3]))
            intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            candidate_area = max(
                float(box[2] - box[0]) * float(box[3] - box[1]), 1e-6
            )
            reference_area = max(
                float(reference[2] - reference[0]) * float(reference[3] - reference[1]), 1e-6
            )
            scale_ratio = candidate_area / reference_area
            containment = intersection / candidate_area
            # Low conf=0.03 can hallucinate a pose on furniture. A fallback must remain
            # spatially compatible with the nearest reliable person track. A partial-body
            # box inside a large close-up box is allowed through `containment`.
            if center_distance > 0.25 and overlap < 0.01 and containment < 0.50:
                scores.append(float("-inf"))
                continue
            # If the primary frame itself is missing, the nearest temporal box is our only
            # identity anchor. Reject a one-frame collapse/expansion that is characteristic
            # of low-threshold pose hallucinations on furniture or clothing texture.
            if strict_reference_scale and not (0.35 <= scale_ratio <= 3.00):
                scores.append(float("-inf"))
                continue
            center_similarity = max(0.0, 1.0 - center_distance)
            score += 0.75 * overlap + 0.25 * center_similarity
        scores.append(score)
    if not np.isfinite(scores).any():
        return None
    selected = int(np.argmax(scores))
    return frame_boxes[selected].copy(), frame_keypoints[selected].copy()


def union_boxes(a: np.ndarray | None, b: np.ndarray) -> np.ndarray:
    if a is None or not np.isfinite(a[:4]).all():
        return b.copy()
    output = b.copy()
    output[0] = min(float(a[0]), float(b[0]))
    output[1] = min(float(a[1]), float(b[1]))
    output[2] = max(float(a[2]), float(b[2]))
    output[3] = max(float(a[3]), float(b[3]))
    output[4] = max(float(a[4]), float(b[4]))
    return output


def fallback_is_better(
    primary_box: np.ndarray,
    primary_keypoints: np.ndarray,
    fallback_box: np.ndarray,
    fallback_keypoints: np.ndarray,
    keypoint_conf: float,
) -> bool:
    if not np.isfinite(primary_box[:4]).all():
        return True
    primary_quality = candidate_quality(primary_keypoints, float(primary_box[4]), keypoint_conf)
    fallback_quality = candidate_quality(fallback_keypoints, float(fallback_box[4]), keypoint_conf)
    # Lexicographic order deliberately prioritises wrists, then the whole upper body.
    if fallback_quality[:2] != primary_quality[:2]:
        return fallback_quality[:2] > primary_quality[:2]
    return fallback_quality[2] > primary_quality[2] + 0.02


def interpolate_short_box_gaps(
    boxes: np.ndarray, source: np.ndarray, max_gap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = boxes.copy()
    output_source = source.copy()
    interpolated = np.zeros(len(boxes), dtype=bool)
    valid = np.isfinite(output[:, :4]).all(axis=1)
    index = 0
    while index < len(output):
        if valid[index]:
            index += 1
            continue
        start = index
        while index < len(output) and not valid[index]:
            index += 1
        end = index
        gap = end - start
        if start == 0 or end == len(output) or gap > max_gap:
            continue
        left = output[start - 1]
        right = output[end]
        for offset, frame_index in enumerate(range(start, end), 1):
            weight = float(offset) / float(gap + 1)
            output[frame_index, :4] = (1.0 - weight) * left[:4] + weight * right[:4]
            # This is only an ROI confidence, never a claim that YOLO detected a person here.
            output[frame_index, 4] = 0.5 * min(float(left[4]), float(right[4]))
            output_source[frame_index] = 3
            interpolated[frame_index] = True
    return output, output_source, interpolated


def fill_short_roi_gaps(boxes: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill ROI-only gaps, including sequence edges; never used as measured pose."""
    output = boxes.copy()
    estimated = np.zeros(len(boxes), dtype=bool)
    valid = np.isfinite(output[:, :4]).all(axis=1)
    index = 0
    while index < len(output):
        if valid[index]:
            index += 1
            continue
        start = index
        while index < len(output) and not valid[index]:
            index += 1
        end = index
        gap = end - start
        if gap > max_gap:
            continue
        if start > 0 and end < len(output):
            left = output[start - 1]
            right = output[end]
            for offset, frame_index in enumerate(range(start, end), 1):
                weight = float(offset) / float(gap + 1)
                output[frame_index] = (1.0 - weight) * left + weight * right
                estimated[frame_index] = True
        elif start > 0:
            output[start:end] = output[start - 1]
            estimated[start:end] = True
        elif end < len(output):
            output[start:end] = output[end]
            estimated[start:end] = True
    return output, estimated


def reason_counts(reasons: np.ndarray) -> dict[str, int]:
    return {
        "retry_missing_frames": int(((reasons & RETRY_MISSING) != 0).sum()),
        "retry_low_box_conf_frames": int(((reasons & RETRY_LOW_BOX_CONF) != 0).sum()),
        "retry_small_person_frames": int(((reasons & RETRY_SMALL_PERSON) != 0).sum()),
        "retry_weak_upper_body_frames": int(((reasons & RETRY_WEAK_UPPER_BODY) != 0).sum()),
        "retry_weak_wrists_frames": int(((reasons & RETRY_WEAK_WRISTS) != 0).sum()),
        "retry_temporal_jump_frames": int(((reasons & RETRY_TEMPORAL_JUMP) != 0).sum()),
    }


def aggregate(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    metrics = (
        "primary_detection_rate",
        "primary_both_wrist_rate",
        "retry_rate",
        "fallback_replacement_rate",
        "final_detection_rate",
        "final_both_wrist_rate",
        "final_measured_pose_rate",
    )
    output: list[dict[str, Any]] = []
    for group_name, group in sorted(groups.items()):
        item: dict[str, Any] = {key: group_name, "trials": len(group)}
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            item[metric] = float(np.mean(values)) if values else float("nan")
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    model_path = args.model.resolve()
    output_dir = args.output_dir.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    union_rows, usable_rows = load_rows(manifest)
    selected_rows = select_rows(usable_rows, args)
    if not selected_rows:
        raise RuntimeError("No usable trials selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "version": 1,
        "policy": "automatic_two_pass_ir_pose_then_exact_geometry_transfer_to_depth",
        "manifest": str(manifest),
        "model": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "ultralytics": ultralytics_version,
        "device": str(args.device),
        "primary": {"imgsz": args.primary_imgsz, "conf": args.primary_conf, "batch": args.primary_batch},
        "fallback": {
            "imgsz": args.fallback_imgsz,
            "conf": args.fallback_conf,
            "batch": args.fallback_batch,
        },
        "retry": {
            "keypoint_conf": args.keypoint_conf,
            "box_conf_below": args.retry_box_conf,
            "person_height_ratio_below": args.retry_min_height_ratio,
            "valid_upper_joints_below": args.retry_min_upper_joints,
            "valid_wrists_below": args.retry_min_wrists,
            "center_jump_ratio_above": args.retry_center_jump_ratio,
            "area_change_ratio_above": args.retry_area_ratio,
        },
        "fallback_selection": (
            "hard_temporal_spatial_gate_then_wrists_then_upper_body_then_box_confidence"
        ),
        "fallback_max_unmatched_center_distance_ratio": 0.25,
        "fallback_missing_primary_area_ratio_range": [0.35, 3.0],
        "fallback_roi": "union_of_fallback_pose_box_and_nearest_primary_track_box",
        "short_box_gap_interpolation": args.interpolate_box_gap,
        "transfer_margin": args.transfer_margin,
        "selected_trials": len(selected_rows),
        "selected_classes": len({row["class_id"] for row in selected_rows}),
        "selected_subjects": len({row["user_id"] for row in selected_rows}),
        "usable_depth_ir_skeleton_trials": len(usable_rows),
        "union_trials": len(union_rows),
        "human_decisions_at_inference": False,
        "coordinate_contract": {
            "ir_yolo": "2D IR image pixels plus bbox-local coordinates",
            "depth": "exact final IR box/keypoint geometry copied because D/IR share frame ids and view",
            "dataset_skeleton": "relative 3D H36M coordinates",
            "fusion": "same frame id plus semantic joint; coordinate values are never forced equal",
        },
        "semantic_pairs": [
            {"name": name, "coco_index": coco, "h36m_index": h36m}
            for name, coco, h36m in SEMANTIC_PAIRS
        ],
        "source_codes": {"0": "unresolved", "1": "primary_640", "2": "fallback_1280", "3": "box_interpolation"},
    }
    atomic_json(output_dir / "config.json", config)
    model = YOLO(str(model_path))

    summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    start = time.time()
    processed_frames = 0
    for trial_index, row in enumerate(selected_rows, 1):
        relative = safe_name(row["sample_id"])
        cache_path = output_dir / "trial_cache" / relative.with_suffix(".npz")
        summary_path = output_dir / "trial_summary" / relative.with_suffix(".json")
        if cache_path.is_file() and summary_path.is_file() and not args.overwrite:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summaries.append(summary)
            processed_frames += int(summary["common_frames"])
            print(f"[{trial_index}/{len(selected_rows)}] resume {row['sample_id']}", flush=True)
            continue

        maps = {
            "depth": frame_map(Path(row["depth_color_path"]), "depth"),
            "ir": frame_map(Path(row["ir_path"]), "ir"),
            "skeleton": frame_map(Path(row["skeleton_path"]), "skeleton"),
        }
        frame_ids = sorted(set(maps["depth"]) & set(maps["ir"]) & set(maps["skeleton"]))
        if not frame_ids:
            skipped_item = {
                "sample_id": row["sample_id"],
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "user_id": row["user_id"],
                "trial_id": row["trial_id"],
                "reason": "zero_common_canonical_frame_ids_across_depth_ir_skeleton",
                "depth_frames": len(maps["depth"]),
                "ir_frames": len(maps["ir"]),
                "skeleton_frames": len(maps["skeleton"]),
            }
            skipped.append(skipped_item)
            atomic_json(
                output_dir / "skipped_trials" / relative.with_suffix(".json"), skipped_item
            )
            print(
                f"[{trial_index}/{len(selected_rows)}] skip-unsynchronised {row['sample_id']} "
                f"D/I/S={len(maps['depth'])}/{len(maps['ir'])}/{len(maps['skeleton'])}",
                flush=True,
            )
            continue
        ir_paths = [maps["ir"][frame_id] for frame_id in frame_ids]

        primary_candidates = predict_candidates(
            model,
            ir_paths,
            imgsz=args.primary_imgsz,
            conf=args.primary_conf,
            batch=args.primary_batch,
            device=args.device,
        )
        primary_boxes, primary_keypoints, primary_people = select_track(primary_candidates)
        image_height, image_width = map(int, primary_candidates[0]["shape"])
        reasons = retry_reasons(
            primary_boxes,
            primary_keypoints,
            image_width=image_width,
            image_height=image_height,
            keypoint_conf=args.keypoint_conf,
            box_conf=args.retry_box_conf,
            min_height_ratio=args.retry_min_height_ratio,
            min_upper_joints=args.retry_min_upper_joints,
            min_wrists=args.retry_min_wrists,
            center_jump_ratio=args.retry_center_jump_ratio,
            area_ratio=args.retry_area_ratio,
        )
        retry_indices = np.flatnonzero(reasons != 0)
        fallback_candidates = predict_candidates(
            model,
            [ir_paths[int(index)] for index in retry_indices],
            imgsz=args.fallback_imgsz,
            conf=args.fallback_conf,
            batch=args.fallback_batch,
            device=args.device,
        )

        final_boxes = primary_boxes.copy()
        final_keypoints = primary_keypoints.copy()
        final_people = primary_people.copy()
        source = np.where(np.isfinite(primary_boxes[:, :4]).all(axis=1), 1, 0).astype(np.uint8)
        fallback_box = np.full_like(primary_boxes, np.nan)
        fallback_keypoints = np.full_like(primary_keypoints, np.nan)
        fallback_people = np.zeros(len(frame_ids), dtype=np.uint8)
        fallback_selected = np.zeros(len(frame_ids), dtype=bool)
        fallback_reference = np.full_like(primary_boxes, np.nan)
        for retry_index, candidate in zip(retry_indices, fallback_candidates):
            frame_index = int(retry_index)
            fallback_people[frame_index] = min(len(candidate["boxes"]), 255)
            reference = nearest_reference_box(primary_boxes, frame_index)
            if reference is not None:
                fallback_reference[frame_index] = reference
            selected = select_fallback_candidate(
                candidate,
                reference,
                args.keypoint_conf,
                strict_reference_scale=not np.isfinite(primary_boxes[frame_index, :4]).all(),
            )
            if selected is None:
                continue
            selected_box, selected_keypoints = selected
            fallback_box[frame_index] = selected_box
            fallback_keypoints[frame_index] = selected_keypoints
            if fallback_is_better(
                primary_boxes[frame_index],
                primary_keypoints[frame_index],
                selected_box,
                selected_keypoints,
                args.keypoint_conf,
            ):
                final_boxes[frame_index] = selected_box
                final_keypoints[frame_index] = selected_keypoints
                final_people[frame_index] = fallback_people[frame_index]
                source[frame_index] = 2
                fallback_selected[frame_index] = True

        final_boxes, source, interpolated_box = interpolate_short_box_gaps(
            final_boxes, source, args.interpolate_box_gap
        )
        # No keypoint interpolation: an interpolated crop is useful, but invented joints would pollute motion.
        final_keypoints[interpolated_box] = np.nan
        person_roi_boxes = final_boxes.copy()
        for frame_index in np.flatnonzero(fallback_selected):
            reference = fallback_reference[frame_index]
            person_roi_boxes[frame_index] = union_boxes(reference, final_boxes[frame_index])
        person_roi_boxes, roi_box_estimated = fill_short_roi_gaps(
            person_roi_boxes, args.interpolate_box_gap
        )
        depth_boxes_from_ir = final_boxes.copy()
        depth_keypoints_from_ir = final_keypoints.copy()
        depth_search_from_ir = expanded_search_boxes(
            person_roi_boxes, args.transfer_margin, width=image_width, height=image_height
        )

        skeleton_items = [load_raw_skeleton(maps["skeleton"][frame_id]) for frame_id in frame_ids]
        skeleton_raw = np.stack([item[0] for item in skeleton_items]).astype(np.float32)
        skeleton_people = np.asarray([item[1] for item in skeleton_items], dtype=np.uint8)
        skeleton_normalised = normalise_skeleton(skeleton_raw)
        final_local = bbox_local_keypoints(final_keypoints, final_boxes)
        semantic_pairs = semantic_pair_features(final_local, skeleton_normalised)

        atomic_npz(
            cache_path,
            frame_ids=np.asarray(frame_ids),
            primary_ir_boxes_xyxy_conf=primary_boxes,
            primary_ir_keypoints_xy_conf=primary_keypoints,
            primary_ir_person_count=primary_people,
            retry_reason_bits=reasons,
            fallback_attempted=(reasons != 0),
            fallback_ir_boxes_xyxy_conf=fallback_box,
            fallback_ir_keypoints_xy_conf=fallback_keypoints,
            fallback_ir_person_count=fallback_people,
            fallback_reference_boxes_xyxy_conf=fallback_reference,
            fallback_selected=fallback_selected,
            final_pose_source=source,
            box_interpolated=interpolated_box,
            ir_boxes_xyxy_conf=final_boxes,
            ir_keypoints_xy_conf=final_keypoints,
            ir_keypoints_bbox_local=final_local,
            ir_person_count=final_people,
            ir_person_roi_boxes_xyxy_conf=person_roi_boxes,
            roi_box_estimated=roi_box_estimated,
            depth_boxes_from_ir_xyxy_conf=depth_boxes_from_ir,
            depth_keypoints_from_ir_xy_conf=depth_keypoints_from_ir,
            depth_search_boxes_from_ir=depth_search_from_ir,
            skeleton_h36m_xyz_conf_raw=skeleton_raw,
            skeleton_h36m_xyz_conf_normalised=skeleton_normalised,
            skeleton_person_count=skeleton_people,
            ir_skeleton_semantic_pairs=semantic_pairs,
            depth_skeleton_semantic_pairs=semantic_pairs,
        )

        primary_metrics = modality_summary(
            "primary", primary_boxes, primary_keypoints, primary_people, args.keypoint_conf
        )
        final_metrics = modality_summary(
            "final", final_boxes, final_keypoints, final_people, args.keypoint_conf
        )
        final_measured = np.isin(source, (1, 2))
        summary: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "class_id": int(row["class_id"]),
            "class_name": row["class_name"],
            "user_id": row["user_id"],
            "trial_id": row["trial_id"],
            "common_frames": len(frame_ids),
            "retry_frames": int(len(retry_indices)),
            "retry_rate": float(len(retry_indices) / len(frame_ids)),
            "fallback_detected_frames": int(np.isfinite(fallback_box[:, :4]).all(axis=1).sum()),
            "fallback_selected_frames": int(fallback_selected.sum()),
            "fallback_replacement_rate": float(fallback_selected.mean()),
            "interpolated_box_frames": int(interpolated_box.sum()),
            "estimated_roi_only_frames": int(roi_box_estimated.sum()),
            "unresolved_frames": int((source == 0).sum()),
            "final_measured_pose_rate": float(final_measured.mean()),
            "skeleton_single_person_rate": float((skeleton_people == 1).mean()),
            "skeleton_multi_person_frames": int((skeleton_people > 1).sum()),
        }
        summary.update(reason_counts(reasons))
        summary.update(primary_metrics)
        summary.update(final_metrics)
        atomic_json(summary_path, summary)
        summaries.append(summary)
        processed_frames += len(frame_ids)

        elapsed = time.time() - start
        frame_rate = processed_frames / max(elapsed, 1e-6)
        remaining_frames = max(0, 85879 - processed_frames) if len(selected_rows) > 40 else 0
        eta = remaining_frames / max(frame_rate, 1e-6) if remaining_frames else 0.0
        print(
            f"[{trial_index}/{len(selected_rows)}] {row['sample_id']} frames={len(frame_ids)} "
            f"retry={summary['retry_rate']:.3f} replace={summary['fallback_replacement_rate']:.3f} "
            f"final={summary['final_detection_rate']:.3f} wrists={summary['final_both_wrist_rate']:.3f} "
            f"elapsed={elapsed:.1f}s eta~={eta:.1f}s",
            flush=True,
        )

    write_csv(output_dir / "trial_summary.csv", summaries)
    write_csv(output_dir / "skipped_trials.csv", skipped)
    write_csv(output_dir / "per_class_summary.csv", aggregate(summaries, "class_id"))
    write_csv(output_dir / "per_subject_summary.csv", aggregate(summaries, "user_id"))
    total_frames = int(sum(int(row["common_frames"]) for row in summaries))
    total_retries = int(sum(int(row["retry_frames"]) for row in summaries))
    total_replacements = int(sum(int(row["fallback_selected_frames"]) for row in summaries))
    total_interpolated = int(sum(int(row["interpolated_box_frames"]) for row in summaries))
    total_estimated_roi = int(sum(int(row["estimated_roi_only_frames"]) for row in summaries))
    total_unresolved = int(sum(int(row["unresolved_frames"]) for row in summaries))
    final = {
        **config,
        "completed_trials": len(summaries),
        "skipped_unsynchronised_trials": len(skipped),
        "skipped_unsynchronised_frames_per_modality": int(
            sum(int(row["ir_frames"]) for row in skipped)
        ),
        "completed_frames": total_frames,
        "retry_frames": total_retries,
        "retry_rate": float(total_retries / max(total_frames, 1)),
        "fallback_selected_frames": total_replacements,
        "fallback_replacement_rate": float(total_replacements / max(total_frames, 1)),
        "interpolated_box_frames": total_interpolated,
        "estimated_roi_only_frames": total_estimated_roi,
        "unresolved_frames": total_unresolved,
        "mean_primary_detection_rate": float(
            np.mean([float(row["primary_detection_rate"]) for row in summaries])
        ),
        "mean_primary_both_wrist_rate": float(
            np.mean([float(row["primary_both_wrist_rate"]) for row in summaries])
        ),
        "mean_final_detection_rate": float(
            np.mean([float(row["final_detection_rate"]) for row in summaries])
        ),
        "mean_final_both_wrist_rate": float(
            np.mean([float(row["final_both_wrist_rate"]) for row in summaries])
        ),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    atomic_json(output_dir / "summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
