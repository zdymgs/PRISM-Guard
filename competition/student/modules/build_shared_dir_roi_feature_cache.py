from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from torchvision.ops import roi_align

from audit_yolo11_pose_skeleton import atomic_json, frame_map, safe_name, write_csv
from shared_dir_roi_model import (
    MODALITY_NAMES,
    PYRAMID_FEATURE_DIM,
    REGION_NAMES,
    SharedResNet18Pyramid,
    imagenet_normalize,
    model_size_mib,
    parameter_count,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "six_modality_audit" / "train_union_manifest.csv"
DEFAULT_ROI_RUN = PROJECT_DIR / "runs" / "dir_multiscale_roi_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "shared_dir_roi_features_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all-frame multi-scale D/IR ROI features with one shared ResNet18."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roi-run", type=Path, default=DEFAULT_ROI_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--frame-batch", type=int, default=8)
    parser.add_argument("--backbone-batch", type=int, default=128)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--uncompressed", action="store_true")
    return parser.parse_args()


def roi_cache_path(roi_run: Path, sample_id: str) -> Path:
    return roi_run / "trial_roi_cache" / safe_name(sample_id).with_suffix(".npz")


def feature_cache_path(output_dir: Path, sample_id: str) -> Path:
    return output_dir / "trial_feature_cache" / safe_name(sample_id).with_suffix(".npz")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_rows(
    rows: list[dict[str, str]], roi_run: Path, args: argparse.Namespace
) -> list[dict[str, str]]:
    selected = [row for row in rows if roi_cache_path(roi_run, row["sample_id"]).is_file()]
    selected.sort(key=lambda row: (int(row["class_id"]), row["user_id"], row["trial_id"]))
    if args.sample_id:
        requested = set(args.sample_id)
        selected = [row for row in selected if row["sample_id"] in requested]
        missing = sorted(requested - {row["sample_id"] for row in selected})
        if missing:
            raise KeyError(f"requested ROI caches unavailable: {missing}")
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


def read_depth(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read Depth image: {path}")
    if image.shape[:2] != (480, 640):
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_ir(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"could not read IR image: {path}")
    if image.shape != (480, 640):
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    return np.repeat(image[:, :, None], 3, axis=2)


def pixel_roi_crops(
    images: torch.Tensor,
    boxes: torch.Tensor,
    valid: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Return crops in [2,N,7,3,S,S] order from full-resolution pixels."""
    frame_count, regions = boxes.shape[:2]
    if regions != len(REGION_NAMES):
        raise ValueError(f"expected {len(REGION_NAMES)} regions, got {regions}")
    safe_boxes = boxes.clone()
    global_box = boxes[:, REGION_NAMES.index("global_fallback")]
    safe_boxes[~valid] = global_box[:, None, :].expand_as(safe_boxes)[~valid]
    repeated_boxes = safe_boxes.unsqueeze(0).expand(len(MODALITY_NAMES), -1, -1, -1)
    image_indices = torch.arange(
        len(MODALITY_NAMES) * frame_count,
        device=images.device,
        dtype=boxes.dtype,
    ).reshape(len(MODALITY_NAMES), frame_count, 1)
    image_indices = image_indices.expand(-1, -1, regions).reshape(-1, 1)
    rois = torch.cat((image_indices, repeated_boxes.reshape(-1, 4)), dim=1)
    crops = roi_align(
        images,
        rois,
        output_size=(output_size, output_size),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )
    return crops.reshape(
        len(MODALITY_NAMES), frame_count, regions, 3, output_size, output_size
    )


@torch.inference_mode()
def extract_trial(
    row: dict[str, str],
    roi_path: Path,
    backbone: SharedResNet18Pyramid,
    device: torch.device,
    crop_size: int,
    frame_batch: int,
    backbone_batch: int,
) -> dict[str, np.ndarray]:
    with np.load(roi_path, allow_pickle=False) as roi:
        arrays = {key: np.asarray(roi[key]) for key in roi.files}
    frame_ids = [str(value) for value in arrays["frame_ids"]]
    if tuple(str(value) for value in arrays["region_names"]) != REGION_NAMES:
        raise RuntimeError(f"region order mismatch: {row['sample_id']}")
    ir_paths = frame_map(Path(row["ir_path"]), "ir")
    depth_paths = frame_map(Path(row["depth_color_path"]), "depth")
    missing = [frame_id for frame_id in frame_ids if frame_id not in ir_paths or frame_id not in depth_paths]
    if missing:
        raise RuntimeError(f"missing synchronized frame {missing[0]} in {row['sample_id']}")

    output = np.zeros(
        (len(frame_ids), len(MODALITY_NAMES), len(REGION_NAMES), PYRAMID_FEATURE_DIM),
        dtype=np.float16,
    )
    use_amp = device.type == "cuda"
    for start in range(0, len(frame_ids), frame_batch):
        stop = min(start + frame_batch, len(frame_ids))
        selected = frame_ids[start:stop]
        depth = np.stack([read_depth(depth_paths[frame_id]) for frame_id in selected])
        ir = np.stack([read_ir(ir_paths[frame_id]) for frame_id in selected])
        pixels = torch.from_numpy(np.concatenate((depth, ir), axis=0)).to(
            device=device, dtype=torch.float32
        )
        pixels = pixels.permute(0, 3, 1, 2).div_(255.0)
        boxes = torch.from_numpy(arrays["roi_boxes_xyxy"][start:stop]).to(
            device=device, dtype=torch.float32
        )
        valid = torch.from_numpy(arrays["roi_valid"][start:stop]).to(
            device=device, dtype=torch.bool
        )
        crops = pixel_roi_crops(pixels, boxes, valid, crop_size)
        flat = crops.reshape(-1, 3, crop_size, crop_size)
        features: list[torch.Tensor] = []
        for crop_start in range(0, len(flat), backbone_batch):
            crop_stop = min(crop_start + backbone_batch, len(flat))
            normalized = imagenet_normalize(flat[crop_start:crop_stop])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                features.append(backbone(normalized).float().cpu())
        chunk = torch.cat(features, dim=0).reshape(
            len(MODALITY_NAMES), stop - start, len(REGION_NAMES), PYRAMID_FEATURE_DIM
        )
        chunk = chunk.permute(1, 0, 2, 3).numpy().astype(np.float16)
        chunk *= arrays["roi_valid"][start:stop, None, :, None]
        output[start:stop] = chunk

    return {
        "frame_ids": arrays["frame_ids"],
        "modality_names": np.asarray(MODALITY_NAMES),
        "region_names": arrays["region_names"],
        "features": output,
        "roi_valid": arrays["roi_valid"],
        "roi_quality": arrays["roi_quality"],
        "roi_source": arrays["roi_source"],
        "roi_clipped_ratio": arrays["roi_clipped_ratio"],
        "left_right_ambiguous": arrays["left_right_ambiguous"],
        "pose_quality_factor": arrays["pose_quality_factor"],
    }


def atomic_feature_npz(path: Path, arrays: dict[str, np.ndarray], compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("wb") as handle:
        if compressed:
            np.savez_compressed(handle, **arrays)
        else:
            np.savez(handle, **arrays)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    roi_run = args.roi_run.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = select_rows(load_manifest(manifest), roi_run, args)
    if not rows:
        raise RuntimeError("no trials selected")

    device = torch.device(args.device)
    backbone = SharedResNet18Pyramid(imagenet_pretrained=True).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    config = {
        "stage": "36_step_12_shared_dir_multiscale_visual_features",
        "manifest": str(manifest),
        "roi_run": str(roi_run),
        "selected_trials": len(rows),
        "device": str(device),
        "crop_size": args.crop_size,
        "frame_batch": args.frame_batch,
        "backbone_batch": args.backbone_batch,
        "modalities": list(MODALITY_NAMES),
        "regions": list(REGION_NAMES),
        "feature_dim": PYRAMID_FEATURE_DIM,
        "backbone": "torchvision ResNet18 IMAGENET1K_V1 shared by all D/IR ROI crops",
        "pyramid": "global-average pooled layer2(128)+layer3(256)+layer4(512)",
        "pixel_crop_policy": "ROIAlign on original 640x480 pixels, then resize each ROI to square crop",
        "all_frame_policy": "every synchronized roi frame is encoded; no uniform or motion-only selection",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "backbone_parameters": parameter_count(backbone),
        "backbone_fp32_mib": model_size_mib(backbone, 4),
        "backbone_fp16_mib": model_size_mib(backbone, 2),
        "compressed_npz": not args.uncompressed,
    }
    atomic_json(output_dir / "config.json", config)

    started = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        output_path = feature_cache_path(output_dir, sample_id)
        trial_started = time.perf_counter()
        if output_path.is_file() and not args.overwrite:
            with np.load(output_path, allow_pickle=False) as cached:
                frame_count = len(cached["frame_ids"])
            status = "cached"
        else:
            arrays = extract_trial(
                row,
                roi_cache_path(roi_run, sample_id),
                backbone,
                device,
                args.crop_size,
                args.frame_batch,
                args.backbone_batch,
            )
            atomic_feature_npz(output_path, arrays, compressed=not args.uncompressed)
            frame_count = len(arrays["frame_ids"])
            status = "built"
        seconds = time.perf_counter() - trial_started
        summaries.append(
            {
                "sample_id": sample_id,
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "user_id": row["user_id"],
                "trial_id": row["trial_id"],
                "frames": frame_count,
                "status": status,
                "seconds": seconds,
                "bytes": output_path.stat().st_size,
            }
        )
        print(
            f"[{index}/{len(rows)}] {sample_id} frames={frame_count} "
            f"{status} {seconds:.2f}s size={output_path.stat().st_size / 1024**2:.2f}MiB",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    write_csv(output_dir / "trial_summary.csv", summaries)
    built_rows = [row for row in summaries if row["status"] == "built"]
    cached_rows = [row for row in summaries if row["status"] == "cached"]
    built_frames = sum(int(row["frames"]) for row in built_rows)
    built_seconds = sum(float(row["seconds"]) for row in built_rows)
    summary = {
        **config,
        "completed_trials": len(summaries),
        "completed_frames": sum(int(row["frames"]) for row in summaries),
        "total_cache_bytes": sum(int(row["bytes"]) for row in summaries),
        "built_trials_this_invocation": len(built_rows),
        "cached_trials_this_invocation": len(cached_rows),
        "built_frames_this_invocation": built_frames,
        "last_invocation_elapsed_seconds": elapsed,
        "new_feature_frames_per_second": built_frames / max(built_seconds, 1e-6),
        "timing_note": "throughput counts only newly built trials; cached trials are excluded",
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
