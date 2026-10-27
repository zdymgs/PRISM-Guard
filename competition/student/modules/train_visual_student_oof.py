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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from shared_dir_roi_model import SharedResNet18Pyramid, model_size_mib as feature_size_mib
from visual_student_data import (
    CompactVisualStudentDataset,
    VisualStudentDataset,
    collate_visual,
    load_npz,
)
from visual_student_model import VisualStudent, model_size_mib, parameter_count


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ROI_FEATURES = PROJECT_DIR / "runs/shared_dir_roi_features_full"
DEFAULT_TEACHER_FEATURES = (
    PROJECT_DIR / "runs/videomae_large_multiclip_full40/complete_features.npz"
)
DEFAULT_TEACHER_LOGITS = (
    PROJECT_DIR
    / "runs/videomae_large_multiclip_full40_head/candidate_oof_logits.npz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs/visual_student_fold0_mechanism"
DEFAULT_COMPACT_CACHE = PROJECT_DIR / "runs/visual_student_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one nested subject-disjoint student visual student fold.")
    parser.add_argument("--mode", choices=("mechanism", "kd", "hybrid"), default="mechanism")
    parser.add_argument("--outer-fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--run", type=Path, default=DEFAULT_ROI_FEATURES)
    parser.add_argument("--compact-cache", type=Path, default=DEFAULT_COMPACT_CACHE)
    parser.add_argument("--teacher-features", type=Path, default=DEFAULT_TEACHER_FEATURES)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER_LOGITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=0.6)
    parser.add_argument("--relation-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260810)
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


def split_universe(
    teacher_logits: dict[str, np.ndarray], outer_fold: int, seed: int
) -> dict[str, np.ndarray]:
    sample_ids = np.asarray(teacher_logits["sample_ids"]).astype(str)
    labels = np.asarray(teacher_logits["labels"], dtype=np.int64)
    users = np.asarray(teacher_logits["users"]).astype(str)
    folds = np.asarray(teacher_logits["folds"], dtype=np.int64)
    if len(sample_ids) != 2914 or set(labels.tolist()) != set(range(40)):
        raise RuntimeError("student universe changed")
    outer_train = np.flatnonzero(folds != outer_fold)
    outer_held = np.flatnonzero(folds == outer_fold)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    inner_train_local, inner_dev_local = next(
        splitter.split(labels[outer_train], labels[outer_train], groups=users[outer_train])
    )
    inner_train = outer_train[inner_train_local]
    inner_dev = outer_train[inner_dev_local]
    groups = {
        "inner_train": inner_train,
        "inner_dev": inner_dev,
        "outer_train": outer_train,
        "outer_held": outer_held,
    }
    for left, right in (("inner_train", "inner_dev"), ("outer_train", "outer_held")):
        if set(users[groups[left]].tolist()) & set(users[groups[right]].tolist()):
            raise RuntimeError(f"subject leakage between {left} and {right}")
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "users": users,
        "folds": folds,
        **groups,
    }


def class_weights(labels: np.ndarray, power: float, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=40).astype(np.float64)
    reference = counts[counts > 0].mean()
    weights = np.zeros(40, dtype=np.float32)
    present = counts > 0
    weights[present] = np.power(reference / counts[present], power).astype(np.float32)
    return torch.from_numpy(weights).to(device)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=False) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def relation_loss(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    student_norm = F.normalize(student, dim=-1)
    teacher_norm = F.normalize(teacher, dim=-1)
    student_gram = torch.einsum("bwd,bvd->bwv", student_norm.flatten(1, 2), student_norm.flatten(1, 2))
    teacher_gram = torch.einsum("bwd,bvd->bwv", teacher_norm.flatten(1, 2), teacher_norm.flatten(1, 2))
    flat_mask = mask.flatten(1)
    pair_mask = flat_mask.unsqueeze(1) & flat_mask.unsqueeze(2)
    difference = (student_gram - teacher_gram).square() * pair_mask
    return difference.sum() / pair_mask.sum().clamp_min(1)


def metric_dict(labels: np.ndarray, prediction: np.ndarray, users: list[str]) -> dict[str, Any]:
    per_user = {}
    user_array = np.asarray(users)
    for user in sorted(set(users)):
        target = user_array == user
        per_user[user] = float(accuracy_score(labels[target], prediction[target]))
    return {
        "correct": int(np.sum(labels == prediction)),
        "total": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "worst_subject_accuracy": float(min(per_user.values())),
        "per_user_accuracy": per_user,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            batch = batch_to_device(batch, device)
            output = model(
                batch["features"], batch["view_mask"], batch["view_quality"], batch["time_position"]
            )
            probability = torch.softmax(output["logits"], dim=1).cpu().numpy()
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
    prediction = np.asarray([row["prediction"] for row in rows], dtype=np.int64)
    metrics = metric_dict(labels, prediction, [str(row["user_id"]) for row in rows])
    return metrics, rows


def train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader | None,
    labels_for_weights: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
    early_stop: bool,
) -> tuple[nn.Module, list[dict[str, Any]], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, foreach=False
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights = class_weights(labels_for_weights, args.class_weight_power, device)
    history: list[dict[str, Any]] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 1
    best_score = -math.inf
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        sums = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "relation": 0.0, "samples": 0}
        learning_rate = args.minimum_learning_rate + 0.5 * (
            args.learning_rate - args.minimum_learning_rate
        ) * (1.0 + math.cos(math.pi * (epoch - 1) / max(args.max_epochs - 1, 1)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(
                    batch["features"], batch["view_mask"], batch["view_quality"], batch["time_position"]
                )
                ce = F.cross_entropy(
                    output["logits"], batch["label"], weight=weights, label_smoothing=0.05
                )
                temperature = args.distillation_temperature
                kd = F.kl_div(
                    F.log_softmax(output["logits"] / temperature, dim=1),
                    F.softmax(batch["teacher_logits"] / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature**2
                relation = relation_loss(
                    output["clip_embeddings"], batch["teacher_features"], output["clip_mask"]
                )
                loss = ce
                if args.mode in {"kd", "hybrid"}:
                    loss = loss + args.distillation_weight * kd
                if args.mode == "hybrid":
                    loss = loss + args.relation_weight * relation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["label"])
            sums["loss"] += float(loss.detach()) * count
            sums["ce"] += float(ce.detach()) * count
            sums["kd"] += float(kd.detach()) * count
            sums["relation"] += float(relation.detach()) * count
            sums["samples"] += count
        record: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": sums["loss"] / max(sums["samples"], 1),
            "train_ce": sums["ce"] / max(sums["samples"], 1),
            "train_kd": sums["kd"] / max(sums["samples"], 1),
            "train_relation": sums["relation"] / max(sums["samples"], 1),
            "train_samples": sums["samples"],
            "seconds": time.perf_counter() - start,
        }
        if eval_loader is not None:
            metrics, _ = evaluate(model, eval_loader, device, args.max_eval_batches)
            record.update({f"val_{key}": value for key, value in metrics.items() if key != "per_user_accuracy"})
            score = float(metrics["accuracy"]) + 0.5 * float(metrics["macro_f1"]) + 0.25 * float(
                metrics["worst_subject_accuracy"]
            )
            if score > best_score + 1e-4:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
        else:
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if early_stop and epoch >= 5 and stale >= args.patience:
            break
    model.load_state_dict(best_state, strict=True)
    return model, history, best_epoch


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=False,
        collate_fn=collate_visual,
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_epochs = min(args.max_epochs, 2)
        args.max_train_batches = args.max_train_batches or 2
        args.max_eval_batches = args.max_eval_batches or 2
    seed_all(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher = load_npz(args.teacher_logits)
    split = split_universe(teacher, args.outer_fold, args.seed)
    sample_ids = split["sample_ids"]
    labels = split["labels"]

    compact_rows = args.compact_cache.resolve() / "rows.csv"
    if compact_rows.exists():
        full_dataset: Dataset = CompactVisualStudentDataset(
            args.compact_cache, args.teacher_features, args.teacher_logits
        )
        index_lookup = full_dataset.index_lookup
    else:
        full_dataset = VisualStudentDataset(
            args.feature_run, args.teacher_features, args.teacher_logits
        )
        index_lookup = {
            full_dataset.teacher_ids[full_dataset.source_lookup[row["sample_id"]]]: index
            for index, row in enumerate(full_dataset.base.rows)
        }

    def subset(key: str) -> Subset:
        canonical = sample_ids[split[key]].tolist()
        missing = set(canonical) - set(index_lookup)
        if missing:
            raise RuntimeError(f"student dataset missing split rows: {sorted(missing)[:3]}")
        return Subset(full_dataset, [index_lookup[sample_id] for sample_id in canonical])

    inner_train_dataset = subset("inner_train")
    inner_dev_dataset = subset("inner_dev")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisualStudent().to(device)
    model, inner_history, best_epoch = train_epochs(
        model,
        make_loader(inner_train_dataset, args, True),
        make_loader(inner_dev_dataset, args, False),
        labels[split["inner_train"]],
        args,
        device,
        args.max_epochs,
        early_stop=True,
    )

    seed_all(args.seed + 1000 + args.outer_fold)
    refit_model = VisualStudent().to(device)
    outer_train_dataset = subset("outer_train")
    refit_model, refit_history, _ = train_epochs(
        refit_model,
        make_loader(outer_train_dataset, args, True),
        None,
        labels[split["outer_train"]],
        args,
        device,
        best_epoch,
        early_stop=False,
    )
    outer_held_dataset = subset("outer_held")
    outer_metrics, prediction_rows = evaluate(
        refit_model, make_loader(outer_held_dataset, args, False), device, args.max_eval_batches
    )
    write_rows(output / "outer_predictions.csv", prediction_rows)
    write_rows(output / "inner_history.csv", inner_history)
    write_rows(output / "refit_history.csv", refit_history)

    backbone = SharedResNet18Pyramid(imagenet_pretrained=False)
    checkpoint = {
        "stage": "visual_student",
        "mode": args.mode,
        "outer_fold": args.outer_fold,
        "model_state": {key: value.detach().cpu() for key, value in refit_model.state_dict().items()},
        "model_config": {"input_width": 896, "width": 192, "embedding_width": 256, "classes": 40},
        "best_inner_epoch": best_epoch,
        "outer_metrics": outer_metrics,
    }
    torch.save(checkpoint, output / "visual_student.pt")
    summary = {
        "stage": "visual_student_nested_outer_fold",
        "status": "smoke" if args.smoke else "formal",
        "mode": args.mode,
        "outer_fold": args.outer_fold,
        "protocol": (
            "Inner subject split selects epoch; model is reinitialized and refit for that fixed "
            "epoch count on all outer-train subjects; outer-held subjects are evaluated once."
        ),
        "counts": {key: int(len(split[key])) for key in ("inner_train", "inner_dev", "outer_train", "outer_held")},
        "best_inner_epoch": best_epoch,
        "outer_metrics": outer_metrics,
        "student_parameters": parameter_count(refit_model),
        "student_fp32_mib": model_size_mib(refit_model, 4),
        "student_fp16_mib": model_size_mib(refit_model, 2),
        "shared_resnet18_fp32_mib": feature_size_mib(backbone, 4),
        "shared_resnet18_fp16_mib": feature_size_mib(backbone, 2),
        "estimated_student_plus_resnet_fp32_mib": model_size_mib(refit_model, 4)
        + feature_size_mib(backbone, 4),
        "large_videomae_required_at_inference": False,
        "test_used_for_selection": False,
        "compact_cache_used": compact_rows.exists(),
        "config": vars(args),
    }
    # Convert Paths before JSON serialization.
    summary["config"] = {
        key: str(value) if isinstance(value, Path) else value for key, value in summary["config"].items()
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
