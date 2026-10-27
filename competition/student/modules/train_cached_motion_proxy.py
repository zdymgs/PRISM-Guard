from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from cached_motion_data import (
    MOTION_FIELDS,
    CachedSequenceMotionDataset,
    collate_cached_motion,
)
from mc3_visual_model import MC3VisualStudent
from motion_fusion_model import UnifiedVisualMotionStudent
from train_visual_student_oof import (
    class_weights,
    metric_dict,
    relation_loss,
    split_universe,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PIXELS = PROJECT_DIR / "runs/visual_pixel_cache_t16_r160_v12"
DEFAULT_MOTION = PROJECT_DIR / "runs/motion_window_cache_t16"
DEFAULT_TEACHER_FEATURES = (
    PROJECT_DIR / "runs/videomae_large_multiclip_full40/complete_features.npz"
)
DEFAULT_TEACHER_LOGITS = (
    PROJECT_DIR
    / "runs/videomae_large_multiclip_full40_head/candidate_oof_logits.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast training-only student aligned-motion mechanism screen. A winning mechanism "
            "must subsequently be trained from raw pixels for formal confirmation."
        )
    )
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument("--motion-cache", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--pixel-cache", type=Path, default=DEFAULT_PIXELS)
    parser.add_argument("--teacher-features", type=Path, default=DEFAULT_TEACHER_FEATURES)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER_LOGITS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modality", choices=("imu", "skeleton", "both"), default="imu")
    parser.add_argument("--stage-a-epochs", type=int, default=6)
    parser.add_argument("--stage-b-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--motion-learning-rate", type=float, default=3e-4)
    parser.add_argument("--visual-learning-rate", type=float, default=3e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.35)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--stage-distillation-weight", type=float, default=0.0)
    parser.add_argument("--relation-weight", type=float, default=0.2)
    parser.add_argument("--anchor-weight", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_visual(path: Path) -> tuple[MC3VisualStudent, dict[str, Any]]:
    checkpoint = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    config = dict(checkpoint["model_config"])
    if config.get("backbone") != "mc3_18_temporal":
        raise ValueError("aligned motion requires a temporal MC3 visual checkpoint")
    model = MC3VisualStudent(
        classes=int(config.get("classes", 40)),
        width=int(config.get("width", 512)),
        dropout=float(config.get("dropout", 0.18)),
        fusion_mode=str(config.get("fusion_mode", "gated")),
        enable_distillation_projection=bool(
            config.get("enable_distillation_projection", False)
        ),
        frames=int(config["frames"]),
        kinetics_pretrained=False,
        temporal_modeling=True,
        exact_time_modeling=bool(config.get("exact_time_modeling", False)),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, config


def make_dataset(
    args: argparse.Namespace,
    full: CachedSequenceMotionDataset,
    sample_ids: np.ndarray,
    temporal_augment: bool,
) -> CachedSequenceMotionDataset:
    missing = set(sample_ids.tolist()) - set(full.index_lookup)
    if missing:
        raise RuntimeError(f"cached motion is missing samples: {sorted(missing)[:3]}")
    indices = np.asarray([full.index_lookup[value] for value in sample_ids], dtype=np.int64)
    return CachedSequenceMotionDataset(
        args.sequence_cache,
        args.motion_cache,
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
        indices=indices,
        temporal_augment=temporal_augment,
    )


def loader(
    dataset: CachedSequenceMotionDataset, args: argparse.Namespace, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        collate_fn=collate_cached_motion,
        drop_last=shuffle and len(dataset) >= args.batch_size,
    )


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_forward(
    model: UnifiedVisualMotionStudent, batch: dict[str, Any]
) -> dict[str, torch.Tensor]:
    motion = {field: batch[field] for field in MOTION_FIELDS}
    return model.forward_from_backbone_sequence(
        batch["backbone_sequence"],
        batch["view_valid"],
        batch["view_quality"],
        batch["global_time_position"],
        motion,
    )


def visual_head_parameters(visual: MC3VisualStudent) -> list[nn.Parameter]:
    modules: list[nn.Module] = [
        visual.temporal_encoder,
        visual.temporal_fusion,
        visual.token_encoder,
        visual.quality_gate,
        visual.stage_fusion,
        visual.classifier,
    ]
    result = [visual.time_position, visual.view_embedding, visual.window_embedding]
    result.extend(parameter for module in modules for parameter in module.parameters())
    return [parameter for parameter in result if parameter is not None]


def set_stage(model: UnifiedVisualMotionStudent, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.motion_residual.parameters():
        parameter.requires_grad_(True)
    if stage == "B":
        for parameter in visual_head_parameters(model.visual):
            parameter.requires_grad_(True)
    elif stage != "A":
        raise ValueError(stage)


def losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    weights: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    temperature = args.distillation_temperature
    ce = F.cross_entropy(
        output["logits"],
        batch["label"],
        weight=weights,
        label_smoothing=args.label_smoothing,
    )
    kd = F.kl_div(
        F.log_softmax(output["logits"] / temperature, dim=1),
        F.softmax(batch["teacher_logits"] / temperature, dim=1),
        reduction="batchmean",
    ) * temperature**2
    teacher_stage = torch.stack(
        (
            batch["teacher_early_logits"],
            batch["teacher_late_logits"],
            batch["teacher_temporal_delta_logits"],
        ),
        dim=1,
    )
    stage_kd = F.kl_div(
        F.log_softmax(output["stage_logits"] / temperature, dim=-1),
        F.softmax(teacher_stage / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2 / 3.0
    relation = relation_loss(
        output["clip_embeddings"], batch["teacher_features"], output["clip_mask"]
    )
    anchor = F.kl_div(
        F.log_softmax(output["logits"] / temperature, dim=1),
        F.softmax(batch["anchor_logits"] / temperature, dim=1),
        reduction="batchmean",
    ) * temperature**2
    total = (
        ce
        + args.distillation_weight * kd
        + args.stage_distillation_weight * stage_kd
        + args.relation_weight * relation
        + args.anchor_weight * anchor
    )
    return {
        "loss": total,
        "ce": ce,
        "kd": kd,
        "stage_kd": stage_kd,
        "relation": relation,
        "anchor": anchor,
    }


def train_stage(
    model: UnifiedVisualMotionStudent,
    train_loader: DataLoader,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    stage: str,
    epochs: int,
) -> list[dict[str, Any]]:
    set_stage(model, stage)
    groups: list[dict[str, Any]] = [
        {
            "params": [
                parameter
                for parameter in model.motion_residual.parameters()
                if parameter.requires_grad
            ],
            "lr": args.motion_learning_rate,
            "initial_lr": args.motion_learning_rate,
        }
    ]
    if stage == "B":
        groups.append(
            {
                "params": [
                    parameter
                    for parameter in visual_head_parameters(model.visual)
                    if parameter.requires_grad
                ],
                "lr": args.visual_learning_rate,
                "initial_lr": args.visual_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    class_weight = class_weights(labels, args.class_weight_power, device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        if stage == "A":
            model.visual.eval()
            model.motion_residual.train()
        ratio = args.minimum_learning_rate / args.motion_learning_rate + 0.5 * (
            1.0 - args.minimum_learning_rate / args.motion_learning_rate
        ) * (1.0 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1)))
        for group in optimizer.param_groups:
            group["lr"] = max(
                args.minimum_learning_rate, float(group["initial_lr"]) * ratio
            )
        sums = {key: 0.0 for key in ("loss", "ce", "kd", "stage_kd", "relation", "anchor")}
        samples = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model_forward(model, batch)
                values = losses(output, batch, class_weight, args)
            scaler.scale(values["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 2.0
            )
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["label"])
            samples += count
            for key in sums:
                sums[key] += float(values[key].detach()) * count
        record = {
            "stage": stage,
            "epoch": epoch,
            "motion_learning_rate": optimizer.param_groups[0]["lr"],
            "visual_learning_rate": (
                optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0
            ),
            **{f"train_{key}": value / max(samples, 1) for key, value in sums.items()},
            "train_samples": samples,
            "seconds": time.perf_counter() - started,
            "residual_strength": float(
                (0.25 * torch.sigmoid(model.motion_residual.residual_logit)).detach()
            ),
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return history


def evaluate(
    model: UnifiedVisualMotionStudent,
    eval_loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(eval_loader):
            if max_batches and batch_index >= max_batches:
                break
            batch = to_device(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model_forward(model, batch)["logits"]
            probability = torch.softmax(logits.float(), dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            for index, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "user_id": batch["user_id"][index],
                        "label": int(labels[index]),
                        "prediction": int(probability[index].argmax()),
                        "confidence": float(probability[index].max()),
                    }
                )
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    predictions = np.asarray([row["prediction"] for row in rows], dtype=np.int64)
    return metric_dict(labels, predictions, [row["user_id"] for row in rows]), rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.stage_a_epochs = min(args.stage_a_epochs, 1)
        args.stage_b_epochs = min(args.stage_b_epochs, 1)
        args.max_train_batches = args.max_train_batches or 2
        args.max_eval_batches = args.max_eval_batches or 2
    seed_all(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher = load_npz(args.teacher_logits)
    split = split_universe(teacher, outer_fold=0, seed=args.seed)
    train_indices = np.asarray(split["inner_train"], dtype=np.int64)
    proxy_indices = np.asarray(split["outer_held"], dtype=np.int64)
    forbidden_indices = np.asarray(split["inner_dev"], dtype=np.int64)
    if len(train_indices) != 1497 or len(proxy_indices) != 973 or len(forbidden_indices) != 444:
        raise RuntimeError("student fixed proxy counts changed")
    full = CachedSequenceMotionDataset(
        args.sequence_cache,
        args.motion_cache,
        args.pixel_cache,
        args.teacher_features,
        args.teacher_logits,
    )
    training = make_dataset(
        args, full, split["sample_ids"][train_indices], temporal_augment=True
    )
    proxy = make_dataset(
        args, full, split["sample_ids"][proxy_indices], temporal_augment=False
    )
    visual, visual_config = load_visual(args.visual_checkpoint)
    model = UnifiedVisualMotionStudent(
        visual,
        use_skeleton=args.modality in {"skeleton", "both"},
        use_imu=args.modality in {"imu", "both"},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # The zero-initialized motion projection makes this an exact cached-sequence
    # replay of the visual anchor under the same evaluation batch geometry as the
    # candidate. This is the fair speed-screen baseline.
    baseline_metrics, baseline_predictions = evaluate(
        model, loader(proxy, args, False), device, args.max_eval_batches
    )
    history = []
    history.extend(
        train_stage(
            model,
            loader(training, args, True),
            split["labels"][train_indices],
            args,
            device,
            "A",
            args.stage_a_epochs,
        )
    )
    history.extend(
        train_stage(
            model,
            loader(training, args, True),
            split["labels"][train_indices],
            args,
            device,
            "B",
            args.stage_b_epochs,
        )
    )
    metrics, predictions = evaluate(
        model, loader(proxy, args, False), device, args.max_eval_batches
    )
    write_rows(output / "training_history.csv", history)
    write_rows(output / "cached_visual_baseline_predictions.csv", baseline_predictions)
    write_rows(output / "proxy_validation_predictions.csv", predictions)
    checkpoint = {
        "stage": "cached_aligned_motion_proxy",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "visual_config": visual_config,
        "modality": args.modality,
        "proxy_metrics": metrics,
        "cached_visual_baseline_metrics": baseline_metrics,
        "delta_vs_cached_visual": {
            "correct": int(metrics["correct"] - baseline_metrics["correct"]),
            "accuracy_pp": 100.0 * (metrics["accuracy"] - baseline_metrics["accuracy"]),
            "macro_f1_pp": 100.0 * (metrics["macro_f1"] - baseline_metrics["macro_f1"]),
            "worst_subject_accuracy_pp": 100.0
            * (
                metrics["worst_subject_accuracy"]
                - baseline_metrics["worst_subject_accuracy"]
            ),
        },
    }
    torch.save(checkpoint, output / "unified_student.pt")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "stage": "cached_aligned_motion_proxy",
        "status": "smoke" if args.smoke else "formal_proxy",
        "modality": args.modality,
        "protocol": (
            "Train nine candidate-train subjects and evaluate six disjoint proxy subjects "
            "once. Permanent user1/user2/user21 validation remains untouched."
        ),
        "counts": {"train": len(training), "proxy": len(proxy), "permanent_untouched": 444},
        "proxy_metrics": metrics,
        "cached_visual_baseline_metrics": baseline_metrics,
        "delta_vs_cached_visual": {
            "correct": int(metrics["correct"] - baseline_metrics["correct"]),
            "accuracy_pp": 100.0 * (metrics["accuracy"] - baseline_metrics["accuracy"]),
            "macro_f1_pp": 100.0 * (metrics["macro_f1"] - baseline_metrics["macro_f1"]),
            "worst_subject_accuracy_pp": 100.0
            * (
                metrics["worst_subject_accuracy"]
                - baseline_metrics["worst_subject_accuracy"]
            ),
        },
        "selection_score": (
            metrics["accuracy"]
            + 0.5 * metrics["macro_f1"]
            + 0.25 * metrics["worst_subject_accuracy"]
        ),
        "parameters": parameters,
        "fp32_mib": parameters * 4 / 1024**2,
        "residual_strength": float(
            (0.25 * torch.sigmoid(model.motion_residual.residual_logit)).detach()
        ),
        "large_videomae_required_at_inference": False,
        "cached_sequence_is_final_model": False,
        "accuracy_contract": (
            "Cache is only a mechanism screen. Any winner must be retrained from raw full "
            "16x160 inputs for 16 epochs before formal validation/test deployment."
        ),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
