from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from audit_yolo11_pose_skeleton import frame_map
from build_shared_dir_roi_feature_cache import read_ir
from protocol import DEFAULT_MANIFEST


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ROI = PROJECT_DIR / "runs" / "dir_multiscale_roi_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "videomae_foundation"
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
VIEW_NAMES = ("scene", "person", "workspace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen Kinetics-VideoMAE features for synchronized video IR clips "
            "at scene, person, and hand-workspace scales."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roi-run", type=Path, default=DEFAULT_ROI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trial-batch", type=int, default=4)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is not None and "p46_ir_readable" in reader.fieldnames:
            master = list(reader)
            if len(master) != 405:
                raise RuntimeError(f"Official Test count changed: {len(master)}")
            rows = []
            for source in master:
                if source["p46_ir_readable"] != "1":
                    continue
                row = dict(source)
                # Present the anonymous Test manifest through the same interface
                # as the frozen Detail21 training manifest.  The proxy source_id
                # preserves the roi cache layout; sample_id remains Kaggle's ID.
                row.update(
                    sample_id=source["official_sample_id"],
                    source_id=source["sample_id"],
                    user_id="anonymous",
                    class_id="-1",
                    ir_dir=source["ir_path"],
                    depth_dir=source["depth_color_path"],
                )
                rows.append(row)
            if len(rows) != 401:
                raise RuntimeError(f"Expected 401 readable Test IR trials, got {len(rows)}")
        else:
            rows = [row for row in reader if row["detail_selected"] == "1"]
            if len(rows) != 1384:
                raise RuntimeError(f"Frozen Detail21 count changed: {len(rows)}")
    rows.sort(key=lambda row: row["sample_id"])
    return rows


def safe_relative(source_id: str) -> Path:
    parts = source_id.split("/")
    if len(parts) != 3 or any(value in {"", ".", ".."} for value in parts):
        raise ValueError(f"unsafe source ID: {source_id}")
    return Path(*parts)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def restore_attention_biases(
    model: VideoMAEForVideoClassification, snapshot: Path
) -> dict[str, Any]:
    """Map the original split q/v bias tensors into Transformers 5.x modules."""
    safetensors_path = snapshot / "model.safetensors"
    pytorch_path = snapshot / "pytorch_model.bin"
    if safetensors_path.is_file():
        state = load_file(str(safetensors_path), device="cpu")
        checkpoint_format = "safetensors"
    elif pytorch_path.is_file():
        state = torch.load(pytorch_path, map_location="cpu", weights_only=True)
        checkpoint_format = "pytorch_bin"
    else:
        raise FileNotFoundError(f"no supported VideoMAE checkpoint in {snapshot}")
    restored = 0
    maximum_difference = 0.0
    with torch.no_grad():
        for layer_index, layer in enumerate(model.videomae.encoder.layer):
            attention = layer.attention.attention
            q_bias = state[
                f"videomae.encoder.layer.{layer_index}.attention.attention.q_bias"
            ]
            v_bias = state[
                f"videomae.encoder.layer.{layer_index}.attention.attention.v_bias"
            ]
            attention.query.bias.copy_(q_bias)
            attention.key.bias.zero_()
            attention.value.bias.copy_(v_bias)
            maximum_difference = max(
                maximum_difference,
                float((attention.query.bias.cpu() - q_bias).abs().max()),
                float((attention.value.bias.cpu() - v_bias).abs().max()),
                float(attention.key.bias.detach().cpu().abs().max()),
            )
            restored += 2
    expected_restored = 2 * len(model.videomae.encoder.layer)
    if restored != expected_restored or maximum_difference != 0.0:
        raise RuntimeError(
            f"VideoMAE legacy attention bias restoration failed: "
            f"restored={restored}, max_diff={maximum_difference}"
        )
    return {
        "reason": (
            "The checkpoint was authored with Transformers 4.21 split q_bias/v_bias; "
            "Transformers 5.x exposes query/key/value bias parameters."
        ),
        "restored_tensors": restored,
        "key_bias_policy": "exact zero, matching the original qkv implementation",
        "maximum_verification_difference": maximum_difference,
        "checkpoint_format": checkpoint_format,
    }


def uniform_indices(frame_count: int, count: int = 16) -> np.ndarray:
    if frame_count < 1:
        raise ValueError("video has no aligned frames")
    return np.rint(np.linspace(0, frame_count - 1, count)).astype(np.int64)


def square_crop(image: np.ndarray, box: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    if box.shape != (4,) or not np.isfinite(box).all():
        return image
    x1, y1, x2, y2 = [float(value) for value in box]
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return image
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1) * float(scale)
    left = max(0, int(round(center_x - 0.5 * side)))
    right = min(width, int(round(center_x + 0.5 * side)))
    top = max(0, int(round(center_y - 0.5 * side)))
    bottom = min(height, int(round(center_y + 0.5 * side)))
    if right <= left + 2 or bottom <= top + 2:
        return image
    return image[top:bottom, left:right]


def prepare_trial(
    row: dict[str, str], roi_run: Path, frame_count: int = 16
) -> tuple[list[list[np.ndarray]], dict[str, np.ndarray]]:
    roi_path = roi_run / "trial_roi_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
    with np.load(roi_path, allow_pickle=False) as data:
        frame_ids = np.asarray(data["frame_ids"]).astype(str)
        region_names = tuple(np.asarray(data["region_names"]).astype(str))
        boxes = np.asarray(data["roi_boxes_xyxy"], dtype=np.float32)
        valid = np.asarray(data["roi_valid"], dtype=bool)
        quality = np.asarray(data["roi_quality"], dtype=np.float32)
    if "full_body" not in region_names or "hand_workspace" not in region_names:
        raise RuntimeError(f"roi view regions changed: {region_names}")
    person_index = region_names.index("full_body")
    workspace_index = region_names.index("hand_workspace")
    chosen = uniform_indices(len(frame_ids), frame_count)
    ir_paths = frame_map(Path(row["ir_dir"]), "ir")
    missing = [frame_ids[index] for index in chosen if frame_ids[index] not in ir_paths]
    if missing:
        raise RuntimeError(f"missing IR frame {missing[0]}: {row['source_id']}")
    scene: list[np.ndarray] = []
    person: list[np.ndarray] = []
    workspace: list[np.ndarray] = []
    for index in chosen:
        image = read_ir(ir_paths[frame_ids[index]])
        scene.append(image)
        person_box = boxes[index, person_index] if valid[index, person_index] else np.asarray([np.nan] * 4)
        workspace_box = (
            boxes[index, workspace_index]
            if valid[index, workspace_index]
            else person_box
        )
        person.append(square_crop(image, person_box, scale=1.15))
        workspace.append(square_crop(image, workspace_box, scale=1.40))
    metadata = {
        "frame_indices": chosen.astype(np.int16),
        "person_valid": valid[chosen, person_index].astype(np.uint8),
        "workspace_valid": valid[chosen, workspace_index].astype(np.uint8),
        "person_quality": quality[chosen, person_index].astype(np.float32),
        "workspace_quality": quality[chosen, workspace_index].astype(np.float32),
    }
    return [scene, person, workspace], metadata


@torch.inference_mode()
def encode(
    model: VideoMAEForVideoClassification,
    processor: VideoMAEImageProcessor,
    videos: list[list[np.ndarray]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    pixel_values = processor(videos, return_tensors="pt").pixel_values.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        output = model.videomae(pixel_values).last_hidden_state
        feature = model.fc_norm(output.mean(dim=1)) if model.fc_norm is not None else output[:, 0]
        logits = model.classifier(feature)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    return (
        feature.float().cpu().numpy(),
        logits.float().cpu().numpy(),
        peak,
    )


def valid_cache(
    path: Path,
    row: dict[str, str],
    classifier_classes: int = 400,
    hidden_size: int = 768,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["sample_id"].item()) == row["sample_id"]
                and int(data["label"].item()) == int(row["class_id"])
                and data["features"].shape == (len(VIEW_NAMES), hidden_size)
                and data["kinetics_logits"].shape == (len(VIEW_NAMES), classifier_classes)
                and np.isfinite(data["features"]).all()
                and np.isfinite(data["kinetics_logits"]).all()
            )
    except (OSError, ValueError, KeyError, EOFError):
        return False


def aggregate(
    rows: list[dict[str, str]], output: Path, classifier_classes: int = 400
    , hidden_size: int = 768
) -> Path:
    sample_ids: list[str] = []
    source_ids: list[str] = []
    users: list[str] = []
    labels: list[int] = []
    features: list[np.ndarray] = []
    kinetics_logits: list[np.ndarray] = []
    person_valid_rate: list[float] = []
    workspace_valid_rate: list[float] = []
    for row in rows:
        path = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
        if not valid_cache(path, row, classifier_classes, hidden_size):
            raise RuntimeError(f"missing or invalid VideoMAE cache: {row['source_id']}")
        with np.load(path, allow_pickle=False) as data:
            sample_ids.append(row["sample_id"])
            source_ids.append(row["source_id"])
            users.append(row["user_id"])
            labels.append(int(row["class_id"]))
            features.append(np.asarray(data["features"], dtype=np.float16))
            kinetics_logits.append(np.asarray(data["kinetics_logits"], dtype=np.float16))
            person_valid_rate.append(float(np.asarray(data["person_valid"]).mean()))
            workspace_valid_rate.append(float(np.asarray(data["workspace_valid"]).mean()))
    path = output / "complete_features.npz"
    atomic_npz(
        path,
        sample_ids=np.asarray(sample_ids),
        source_ids=np.asarray(source_ids),
        users=np.asarray(users),
        labels=np.asarray(labels, dtype=np.int64),
        view_names=np.asarray(VIEW_NAMES),
        features=np.stack(features),
        kinetics_logits=np.stack(kinetics_logits),
        person_valid_rate=np.asarray(person_valid_rate, dtype=np.float32),
        workspace_valid_rate=np.asarray(workspace_valid_rate, dtype=np.float32),
    )
    return path


def main() -> None:
    args = parse_args()
    if args.trial_batch < 1:
        raise ValueError("--trial-batch must be positive")
    rows = read_rows(args.manifest.resolve())
    if args.max_trials > 0:
        rows = rows[: args.max_trials]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            args.model,
            allow_patterns=("*.json", "*.safetensors", "*.txt"),
            local_files_only=True,
        )
    )
    processor = VideoMAEImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = VideoMAEForVideoClassification.from_pretrained(
        snapshot, local_files_only=True
    )
    bias_report = restore_attention_biases(model, snapshot)
    classifier_classes = int(model.config.num_labels)
    hidden_size = int(model.config.hidden_size)
    device = torch.device(args.device)
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    completed = 0
    skipped = 0
    peak_cuda_gib = 0.0
    for batch_start in range(0, len(rows), args.trial_batch):
        batch_rows = rows[batch_start : batch_start + args.trial_batch]
        work_rows: list[dict[str, str]] = []
        all_videos: list[list[np.ndarray]] = []
        metadata: list[dict[str, np.ndarray]] = []
        for row in batch_rows:
            cache = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
            if not args.overwrite and valid_cache(
                cache, row, classifier_classes, hidden_size
            ):
                skipped += 1
                continue
            videos, trial_metadata = prepare_trial(row, args.roi_run.resolve())
            work_rows.append(row)
            all_videos.extend(videos)
            metadata.append(trial_metadata)
        if work_rows:
            features, logits, peak = encode(model, processor, all_videos, device)
            peak_cuda_gib = max(peak_cuda_gib, peak)
            features = features.reshape(len(work_rows), len(VIEW_NAMES), -1)
            logits = logits.reshape(len(work_rows), len(VIEW_NAMES), -1)
            for index, row in enumerate(work_rows):
                cache = output / "trial_feature_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
                atomic_npz(
                    cache,
                    sample_id=np.asarray(row["sample_id"]),
                    source_id=np.asarray(row["source_id"]),
                    user=np.asarray(row["user_id"]),
                    label=np.asarray(int(row["class_id"]), dtype=np.int64),
                    view_names=np.asarray(VIEW_NAMES),
                    features=features[index].astype(np.float16),
                    kinetics_logits=logits[index].astype(np.float16),
                    **metadata[index],
                )
                completed += 1
        processed = min(batch_start + len(batch_rows), len(rows))
        if processed == len(rows) or processed % max(20, args.trial_batch) == 0:
            print(
                json.dumps(
                    {
                        "stage": "cache_progress",
                        "processed": processed,
                        "total": len(rows),
                        "built": completed,
                        "skipped": skipped,
                        "elapsed_seconds": time.perf_counter() - started,
                        "peak_cuda_gib": peak_cuda_gib,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    complete = aggregate(rows, output, classifier_classes, hidden_size)
    summary = {
        "protocol": "frozen Kinetics VideoMAE scene/person/workspace IR features",
        "model": args.model,
        "model_snapshot": str(snapshot),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "attention_bias_compatibility": bias_report,
        "views": list(VIEW_NAMES),
        "frames_per_view": 16,
        "pretraining_classifier_classes": classifier_classes,
        "hidden_size": hidden_size,
        "trials": len(rows),
        "built": completed,
        "skipped": skipped,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_gib": peak_cuda_gib,
        "complete_features": str(complete),
    }
    (output / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
