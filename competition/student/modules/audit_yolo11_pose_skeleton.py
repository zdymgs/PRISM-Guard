from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO, __version__ as ultralytics_version


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "six_modality_audit" / "train_union_manifest.csv"
DEFAULT_MODEL = REPO_DIR / "assets" / "models" / "yolo11n-pose.pt"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "yolo11_pose_skeleton_audit"

H36M_NAMES = (
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
H36M_PARENTS = (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)

COCO_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
COCO_EDGES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)

# Only true semantic counterparts are paired. These are not treated as equal coordinates.
SEMANTIC_PAIRS = (
    ("left_shoulder", 5, 11),
    ("left_elbow", 7, 12),
    ("left_wrist", 9, 13),
    ("right_shoulder", 6, 14),
    ("right_elbow", 8, 15),
    ("right_wrist", 10, 16),
    ("left_hip", 11, 4),
    ("left_knee", 13, 5),
    ("left_ankle", 15, 6),
    ("right_hip", 12, 1),
    ("right_knee", 14, 2),
    ("right_ankle", 16, 3),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO11-pose on every aligned D/IR frame and pair it semantically with raw Skeleton."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--keypoint-conf", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--detect-modality", choices=("both", "ir", "depth"), default="both")
    parser.add_argument("--transfer-margin", type=float, default=0.20)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize-fps", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_frame_id(path: Path, modality: str) -> str:
    stem = path.stem
    if modality == "depth":
        if not stem.startswith("Depth_") or not stem.endswith("_Color"):
            raise ValueError(f"Unexpected Depth filename: {path.name}")
        return stem[len("Depth_") : -len("_Color")]
    if modality == "ir":
        if not stem.startswith("IR_"):
            raise ValueError(f"Unexpected IR filename: {path.name}")
        return stem[len("IR_") :]
    if modality == "skeleton":
        if not stem.startswith("Color_"):
            raise ValueError(f"Unexpected Skeleton filename: {path.name}")
        value = stem[len("Color_") :]
        # A small set of otherwise usable recordings stores Skeleton frames as
        # ``Color_<timestamp>_<8-digit-frame-id>.json`` while Depth/IR use only
        # the final frame id.  The suffix is the shared acquisition counter, so
        # normalising it here restores the same synchronization contract used by
        # the standard ``Color_<frame-id>.json`` files.
        suffix = value.rsplit("_", 1)[-1]
        if len(suffix) == 8 and suffix.isdigit():
            return suffix
        return value
    raise ValueError(modality)


def frame_map(trial_dir: Path, modality: str) -> dict[str, Path]:
    if modality == "skeleton":
        files = (trial_dir / "predictions").glob("*.json")
    else:
        files = (
            path
            for path in trial_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    return {canonical_frame_id(path, modality): path for path in files if path.is_file()}


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        union = list(csv.DictReader(handle))
    usable = [
        row
        for row in union
        if row.get("depth_color_usable") == "1"
        and row.get("ir_usable") == "1"
        and row.get("skeleton_usable") == "1"
    ]
    return union, usable


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected = sorted(rows, key=lambda row: (int(row["class_id"]), row["user_id"], row["trial_id"]))
    if args.sample_id:
        requested = set(args.sample_id)
        selected = [row for row in selected if row["sample_id"] in requested]
        missing = sorted(requested - {row["sample_id"] for row in selected})
        if missing:
            raise KeyError(f"Requested sample_id not usable or absent: {missing}")
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


def load_raw_skeleton(path: Path) -> tuple[np.ndarray, int]:
    try:
        people = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        people = []
    person_count = len(people) if isinstance(people, list) else 0
    if not isinstance(people, list) or not people or not isinstance(people[0], dict):
        return np.full((17, 4), np.nan, dtype=np.float32), person_count
    keypoints = np.asarray(people[0].get("keypoints", []), dtype=np.float32)
    scores = np.asarray(people[0].get("keypoint_scores", []), dtype=np.float32)
    if keypoints.shape != (17, 3):
        return np.full((17, 4), np.nan, dtype=np.float32), person_count
    if scores.shape != (17,):
        scores = np.ones(17, dtype=np.float32)
    return np.concatenate([keypoints, scores[:, None]], axis=1).astype(np.float32), person_count


def normalise_skeleton(raw: np.ndarray) -> np.ndarray:
    result = raw.copy()
    xyz = result[:, :, :3]
    scores = result[:, :, 3]
    valid = np.isfinite(xyz).all(axis=2)
    roots = xyz[:, 0:1, :]
    centered = xyz - roots
    radii = np.linalg.norm(centered, axis=2)
    radii[~valid] = np.nan
    scales = np.nanmax(radii, axis=1)
    scales[~np.isfinite(scales) | (scales < 1e-6)] = 1.0
    centered = centered / scales[:, None, None]
    result[:, :, :3] = centered
    result[:, :, 3] = scores
    return result.astype(np.float32)


def result_candidates(result: Any) -> dict[str, np.ndarray]:
    if result.boxes is None or len(result.boxes) == 0:
        return {
            "boxes": np.empty((0, 5), dtype=np.float32),
            "keypoints": np.empty((0, 17, 3), dtype=np.float32),
            "shape": np.asarray(result.orig_shape, dtype=np.int32),
        }
    boxes = np.concatenate(
        [
            result.boxes.xyxy.detach().cpu().numpy().astype(np.float32),
            result.boxes.conf.detach().cpu().numpy().astype(np.float32)[:, None],
        ],
        axis=1,
    )
    if result.keypoints is None:
        keypoints = np.zeros((len(boxes), 17, 3), dtype=np.float32)
    else:
        keypoints = result.keypoints.data.detach().cpu().numpy().astype(np.float32)
    return {
        "boxes": boxes,
        "keypoints": keypoints,
        "shape": np.asarray(result.orig_shape, dtype=np.int32),
    }


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-6)


def select_track(candidates: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = len(candidates)
    boxes = np.full((frame_count, 5), np.nan, dtype=np.float32)
    keypoints = np.full((frame_count, 17, 3), np.nan, dtype=np.float32)
    person_counts = np.zeros(frame_count, dtype=np.uint8)
    previous: np.ndarray | None = None
    gap = 0
    for index, candidate in enumerate(candidates):
        frame_boxes = candidate["boxes"]
        frame_keypoints = candidate["keypoints"]
        person_counts[index] = min(len(frame_boxes), 255)
        if len(frame_boxes) == 0:
            gap += 1
            if gap > 5:
                previous = None
            continue
        if previous is None:
            selected = int(np.argmax(frame_boxes[:, 4]))
        else:
            height, width = map(float, candidate["shape"])
            diagonal = max(math.hypot(width, height), 1.0)
            scores: list[float] = []
            previous_center = np.asarray(
                [(previous[0] + previous[2]) * 0.5, (previous[1] + previous[3]) * 0.5]
            )
            for box in frame_boxes:
                center = np.asarray([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
                center_similarity = max(0.0, 1.0 - float(np.linalg.norm(center - previous_center)) / diagonal)
                scores.append(float(box[4]) + 0.75 * box_iou(previous, box) + 0.25 * center_similarity)
            selected = int(np.argmax(scores))
        boxes[index] = frame_boxes[selected]
        keypoints[index] = frame_keypoints[selected]
        previous = frame_boxes[selected]
        gap = 0
    return boxes, keypoints, person_counts


def bbox_local_keypoints(keypoints: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    local = keypoints.copy()
    for frame_index, box in enumerate(boxes):
        if not np.isfinite(box[:4]).all():
            local[frame_index] = np.nan
            continue
        width = max(float(box[2] - box[0]), 1e-6)
        height = max(float(box[3] - box[1]), 1e-6)
        local[frame_index, :, 0] = (local[frame_index, :, 0] - box[0]) / width
        local[frame_index, :, 1] = (local[frame_index, :, 1] - box[1]) / height
    return local.astype(np.float32)


def expanded_search_boxes(boxes: np.ndarray, margin: float, width: int = 640, height: int = 480) -> np.ndarray:
    expanded = boxes.copy()
    for index, box in enumerate(boxes):
        if not np.isfinite(box[:4]).all():
            continue
        box_width = float(box[2] - box[0])
        box_height = float(box[3] - box[1])
        expanded[index, 0] = max(0.0, float(box[0]) - margin * box_width)
        expanded[index, 1] = max(0.0, float(box[1]) - margin * box_height)
        expanded[index, 2] = min(float(width - 1), float(box[2]) + margin * box_width)
        expanded[index, 3] = min(float(height - 1), float(box[3]) + margin * box_height)
    return expanded.astype(np.float32)


def empty_pose(frame_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.full((frame_count, 5), np.nan, dtype=np.float32),
        np.full((frame_count, 17, 3), np.nan, dtype=np.float32),
        np.zeros(frame_count, dtype=np.uint8),
    )


def semantic_pair_features(
    yolo_local: np.ndarray, skeleton_normalised: np.ndarray
) -> np.ndarray:
    # [YOLO bbox-local x/y/conf, Skeleton root-relative x/y/z/conf].
    paired = np.full((len(yolo_local), len(SEMANTIC_PAIRS), 7), np.nan, dtype=np.float32)
    for pair_index, (_, coco_index, h36m_index) in enumerate(SEMANTIC_PAIRS):
        paired[:, pair_index, :3] = yolo_local[:, coco_index]
        paired[:, pair_index, 3:] = skeleton_normalised[:, h36m_index]
    return paired


def longest_false_run(valid: np.ndarray) -> int:
    maximum = current = 0
    for value in valid:
        if bool(value):
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def modality_summary(
    prefix: str,
    boxes: np.ndarray,
    keypoints: np.ndarray,
    person_counts: np.ndarray,
    keypoint_conf: float,
) -> dict[str, float | int]:
    detected = np.isfinite(boxes[:, 4])
    confidences = boxes[detected, 4]
    left_wrist = np.isfinite(keypoints[:, 9, 2]) & (keypoints[:, 9, 2] >= keypoint_conf)
    right_wrist = np.isfinite(keypoints[:, 10, 2]) & (keypoints[:, 10, 2] >= keypoint_conf)
    arm_indices = np.asarray((5, 6, 7, 8, 9, 10))
    arm_conf = keypoints[:, arm_indices, 2]
    valid_arm_values = np.isfinite(arm_conf)
    arm_valid_rate = float((arm_conf[valid_arm_values] >= keypoint_conf).mean()) if valid_arm_values.any() else 0.0
    centers: list[np.ndarray] = []
    for box in boxes[detected]:
        centers.append(np.asarray(((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)))
    if len(centers) > 1:
        center_array = np.stack(centers)
        center_jitter = float(np.median(np.linalg.norm(np.diff(center_array, axis=0), axis=1)))
    else:
        center_jitter = float("nan")
    return {
        f"{prefix}_detected_frames": int(detected.sum()),
        f"{prefix}_detection_rate": float(detected.mean()),
        f"{prefix}_mean_box_conf": float(confidences.mean()) if len(confidences) else 0.0,
        f"{prefix}_multi_person_frames": int((person_counts > 1).sum()),
        f"{prefix}_multi_person_rate": float((person_counts > 1).mean()),
        f"{prefix}_longest_missing_run": longest_false_run(detected),
        f"{prefix}_left_wrist_rate": float(left_wrist.mean()),
        f"{prefix}_right_wrist_rate": float(right_wrist.mean()),
        f"{prefix}_both_wrist_rate": float((left_wrist & right_wrist).mean()),
        f"{prefix}_arm_keypoint_value_rate": arm_valid_rate,
        f"{prefix}_median_center_jitter_px": center_jitter,
    }


def safe_name(sample_id: str) -> Path:
    parts = sample_id.split("/")
    if len(parts) != 3 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe sample_id: {sample_id}")
    return Path(*parts)


def draw_yolo(image: np.ndarray, box: np.ndarray, keypoints: np.ndarray, label: str) -> np.ndarray:
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    if np.isfinite(box[:4]).all():
        x0, y0, x1, y1 = np.rint(box[:4]).astype(int)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{label} conf={box[4]:.2f}",
            (max(4, x0), max(20, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        for a, b in COCO_EDGES:
            if keypoints[a, 2] >= 0.25 and keypoints[b, 2] >= 0.25:
                pa = tuple(np.rint(keypoints[a, :2]).astype(int))
                pb = tuple(np.rint(keypoints[b, :2]).astype(int))
                cv2.line(canvas, pa, pb, (0, 215, 255), 2, cv2.LINE_AA)
        for point in keypoints:
            if point[2] >= 0.25:
                cv2.circle(canvas, tuple(np.rint(point[:2]).astype(int)), 3, (0, 0, 255), -1)
    else:
        cv2.putText(canvas, f"{label}: NO DETECTION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    return canvas


def draw_h36m(raw: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    xyz = raw[:, :3]
    if not np.isfinite(xyz).all():
        cv2.putText(canvas, "Raw Skeleton unavailable", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return canvas
    centered = xyz - xyz[0:1]
    points = centered[:, [0, 2]]
    scale = float(np.max(np.linalg.norm(points, axis=1)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    points = points / scale
    pixels = np.empty((17, 2), dtype=np.int32)
    pixels[:, 0] = np.rint(width * 0.5 + points[:, 0] * width * 0.36).astype(np.int32)
    pixels[:, 1] = np.rint(height * 0.88 - points[:, 1] * height * 0.72).astype(np.int32)
    for child, parent in enumerate(H36M_PARENTS):
        if child == parent:
            continue
        cv2.line(canvas, tuple(pixels[parent]), tuple(pixels[child]), (0, 215, 255), 3, cv2.LINE_AA)
    for index, point in enumerate(pixels):
        color = (0, 0, 255) if index in {13, 16} else (0, 255, 0)
        cv2.circle(canvas, tuple(point), 4, color, -1)
    cv2.putText(canvas, "Dataset Skeleton: relative 3D X-Z", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2)
    cv2.putText(canvas, "red = wrists; not image pixels", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    return canvas


def write_video(
    output_path: Path,
    frame_ids: list[str],
    ir_paths: list[Path],
    depth_paths: list[Path],
    ir_boxes: np.ndarray,
    ir_keypoints: np.ndarray,
    depth_boxes: np.ndarray,
    depth_keypoints: np.ndarray,
    skeleton_raw: np.ndarray,
    fps: float,
    ir_label: str = "IR YOLO",
    depth_label: str = "Depth YOLO",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640 * 3, 480)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    try:
        for index, frame_id in enumerate(frame_ids):
            ir = cv2.imread(str(ir_paths[index]), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(depth_paths[index]), cv2.IMREAD_COLOR)
            if ir is None or depth is None:
                continue
            ir = cv2.resize(ir, (640, 480), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_AREA)
            ir_canvas = draw_yolo(ir, ir_boxes[index], ir_keypoints[index], ir_label)
            depth_canvas = draw_yolo(depth, depth_boxes[index], depth_keypoints[index], depth_label)
            skeleton_canvas = draw_h36m(skeleton_raw[index])
            cv2.putText(
                skeleton_canvas,
                f"frame {index + 1}/{len(frame_ids)} {frame_id[-8:]}",
                (12, 475),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )
            writer.write(np.concatenate([depth_canvas, ir_canvas, skeleton_canvas], axis=1))
    finally:
        writer.release()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    output: list[dict[str, Any]] = []
    metric_names = (
        "ir_detection_rate",
        "ir_both_wrist_rate",
        "ir_arm_keypoint_value_rate",
        "depth_detection_rate",
        "depth_both_wrist_rate",
        "depth_arm_keypoint_value_rate",
    )
    for group_name, group_rows in sorted(groups.items()):
        result: dict[str, Any] = {key: group_name, "trials": len(group_rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in group_rows if metric in row and np.isfinite(float(row[metric]))]
            result[metric] = float(np.mean(values)) if values else float("nan")
        output.append(result)
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
    config = {
        "version": 1,
        "manifest": str(manifest),
        "model": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "ultralytics": ultralytics_version,
        "device": str(args.device),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "keypoint_conf": args.keypoint_conf,
        "batch": args.batch,
        "detect_modality": args.detect_modality,
        "transfer_margin": args.transfer_margin,
        "union_trials": len(union_rows),
        "usable_depth_ir_skeleton_trials": len(usable_rows),
        "selected_trials": len(selected_rows),
        "selected_classes": len({row["class_id"] for row in selected_rows}),
        "selected_subjects": len({row["user_id"] for row in selected_rows}),
        "semantic_pairs": [
            {"name": name, "coco_index": coco, "h36m_index": h36m}
            for name, coco, h36m in SEMANTIC_PAIRS
        ],
        "coordinate_contract": {
            "yolo": "2D image pixels and bbox-local coordinates",
            "dataset_skeleton": "relative 3D H36M coordinates",
            "fusion": "same frame id + semantic joint, never coordinate equality",
        },
    }
    atomic_json(output_dir / "config.json", config)
    model = YOLO(str(model_path))

    summaries: list[dict[str, Any]] = []
    start = time.time()
    for trial_index, row in enumerate(selected_rows, 1):
        relative = safe_name(row["sample_id"])
        cache_path = output_dir / "trial_cache" / relative.with_suffix(".npz")
        summary_path = output_dir / "trial_summary" / relative.with_suffix(".json")
        video_path = output_dir / "videos" / relative.with_suffix(".mp4")
        if cache_path.is_file() and summary_path.is_file() and not args.overwrite:
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            print(f"[{trial_index}/{len(selected_rows)}] resume {row['sample_id']}", flush=True)
            continue

        maps = {
            "depth": frame_map(Path(row["depth_color_path"]), "depth"),
            "ir": frame_map(Path(row["ir_path"]), "ir"),
            "skeleton": frame_map(Path(row["skeleton_path"]), "skeleton"),
        }
        frame_ids = sorted(set(maps["depth"]) & set(maps["ir"]) & set(maps["skeleton"]))
        if not frame_ids:
            raise RuntimeError(f"No common frames: {row['sample_id']}")
        ir_paths = [maps["ir"][frame_id] for frame_id in frame_ids]
        depth_paths = [maps["depth"][frame_id] for frame_id in frame_ids]
        if args.detect_modality == "both":
            sources = [str(path) for path in ir_paths + depth_paths]
        elif args.detect_modality == "ir":
            sources = [str(path) for path in ir_paths]
        else:
            sources = [str(path) for path in depth_paths]
        raw_results = model.predict(
            source=sources,
            stream=True,
            imgsz=args.imgsz,
            conf=args.conf,
            max_det=5,
            device=args.device,
            half=str(args.device).lower() != "cpu",
            batch=args.batch,
            verbose=False,
        )
        candidates = [result_candidates(result) for result in raw_results]
        if len(candidates) != len(sources):
            raise RuntimeError(f"YOLO result count mismatch: {len(candidates)} != {len(sources)}")
        split = len(frame_ids)
        if args.detect_modality == "both":
            ir_boxes, ir_keypoints, ir_people = select_track(candidates[:split])
            depth_boxes, depth_keypoints, depth_people = select_track(candidates[split:])
        elif args.detect_modality == "ir":
            ir_boxes, ir_keypoints, ir_people = select_track(candidates)
            depth_boxes, depth_keypoints, depth_people = empty_pose(split)
        else:
            depth_boxes, depth_keypoints, depth_people = select_track(candidates)
            ir_boxes, ir_keypoints, ir_people = empty_pose(split)
        depth_search_from_ir = expanded_search_boxes(ir_boxes, args.transfer_margin)
        ir_search_from_depth = expanded_search_boxes(depth_boxes, args.transfer_margin)

        skeleton_items = [load_raw_skeleton(maps["skeleton"][frame_id]) for frame_id in frame_ids]
        skeleton_raw = np.stack([item[0] for item in skeleton_items]).astype(np.float32)
        skeleton_people = np.asarray([item[1] for item in skeleton_items], dtype=np.uint8)
        skeleton_normalised = normalise_skeleton(skeleton_raw)
        ir_local = bbox_local_keypoints(ir_keypoints, ir_boxes)
        depth_local = bbox_local_keypoints(depth_keypoints, depth_boxes)
        ir_semantic = semantic_pair_features(ir_local, skeleton_normalised)
        depth_semantic = semantic_pair_features(depth_local, skeleton_normalised)

        atomic_npz(
            cache_path,
            frame_ids=np.asarray(frame_ids),
            ir_boxes_xyxy_conf=ir_boxes,
            ir_keypoints_xy_conf=ir_keypoints,
            ir_keypoints_bbox_local=ir_local,
            ir_person_count=ir_people,
            depth_boxes_xyxy_conf=depth_boxes,
            depth_keypoints_xy_conf=depth_keypoints,
            depth_keypoints_bbox_local=depth_local,
            depth_person_count=depth_people,
            depth_search_boxes_from_ir=depth_search_from_ir,
            ir_search_boxes_from_depth=ir_search_from_depth,
            skeleton_h36m_xyz_conf_raw=skeleton_raw,
            skeleton_h36m_xyz_conf_normalised=skeleton_normalised,
            skeleton_person_count=skeleton_people,
            ir_skeleton_semantic_pairs=ir_semantic,
            depth_skeleton_semantic_pairs=depth_semantic,
        )

        summary: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "class_id": int(row["class_id"]),
            "class_name": row["class_name"],
            "user_id": row["user_id"],
            "trial_id": row["trial_id"],
            "common_frames": len(frame_ids),
            "skeleton_single_person_rate": float((skeleton_people == 1).mean()),
            "skeleton_multi_person_frames": int((skeleton_people > 1).sum()),
            "ir_detection_source": "yolo11-pose" if args.detect_modality in {"both", "ir"} else "not_run",
            "depth_detection_source": "yolo11-pose"
            if args.detect_modality in {"both", "depth"}
            else "not_run",
            "depth_search_box_source": "expanded_ir_yolo"
            if args.detect_modality == "ir"
            else "not_used",
            "ir_search_box_source": "expanded_depth_yolo"
            if args.detect_modality == "depth"
            else "not_used",
        }
        summary.update(modality_summary("ir", ir_boxes, ir_keypoints, ir_people, args.keypoint_conf))
        summary.update(
            modality_summary("depth", depth_boxes, depth_keypoints, depth_people, args.keypoint_conf)
        )
        atomic_json(summary_path, summary)
        summaries.append(summary)

        if args.visualize:
            video_ir_boxes = ir_boxes if args.detect_modality != "depth" else ir_search_from_depth
            video_depth_boxes = depth_boxes if args.detect_modality != "ir" else depth_search_from_ir
            write_video(
                video_path,
                frame_ids,
                ir_paths,
                depth_paths,
                video_ir_boxes,
                ir_keypoints,
                video_depth_boxes,
                depth_keypoints,
                skeleton_raw,
                args.visualize_fps,
                ir_label="IR YOLO" if args.detect_modality != "depth" else "IR wide ROI from Depth",
                depth_label="Depth YOLO"
                if args.detect_modality != "ir"
                else "Depth wide ROI from IR",
            )
        elapsed = time.time() - start
        rate = trial_index / max(elapsed, 1e-6)
        eta = (len(selected_rows) - trial_index) / max(rate, 1e-6)
        print(
            f"[{trial_index}/{len(selected_rows)}] {row['sample_id']} frames={len(frame_ids)} "
            f"IR={summary['ir_detection_rate']:.3f} D={summary['depth_detection_rate']:.3f} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )

    write_csv(output_dir / "trial_summary.csv", summaries)
    write_csv(output_dir / "per_class_summary.csv", aggregate(summaries, "class_id"))
    write_csv(output_dir / "per_subject_summary.csv", aggregate(summaries, "user_id"))
    final = {
        **config,
        "completed_trials": len(summaries),
        "completed_frames": int(sum(int(row["common_frames"]) for row in summaries)),
        "mean_ir_detection_rate": float(np.mean([float(row["ir_detection_rate"]) for row in summaries])),
        "mean_depth_detection_rate": float(
            np.mean([float(row["depth_detection_rate"]) for row in summaries])
        ),
        "mean_ir_both_wrist_rate": float(np.mean([float(row["ir_both_wrist_rate"]) for row in summaries])),
        "mean_depth_both_wrist_rate": float(
            np.mean([float(row["depth_both_wrist_rate"]) for row in summaries])
        ),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    atomic_json(output_dir / "summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
