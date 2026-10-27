"""Original MMAct visual source epoch and schedule."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .augmentation import augment_rgb
def relation_gram_loss(embedding: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
    left = nn.functional.normalize(embedding.float(), dim=1)
    right = nn.functional.normalize(geometry.float(), dim=1)
    return nn.functional.mse_loss(left @ left.T, right @ right.T)

def should_optimizer_step(batch_index: int, total_batches: int, accumulation: int) -> bool:
    if accumulation <= 0 or total_batches <= 0 or not 0 <= batch_index < total_batches:
        raise ValueError("invalid accumulation position")
    return (batch_index + 1) % accumulation == 0 or batch_index + 1 == total_batches

def accumulation_divisor(batch_index: int, total_batches: int, accumulation: int) -> int:
    if accumulation <= 0 or total_batches <= 0 or not 0 <= batch_index < total_batches:
        raise ValueError("invalid accumulation position")
    group_start = (batch_index // accumulation) * accumulation
    return min(accumulation, total_batches - group_start)

def head_lr_factor(epoch: int, *, total_epochs: int = 16, warmup_epochs: int = 2, minimum_ratio: float = 0.05) -> float:
    if not 0 <= epoch < total_epochs:
        raise ValueError("epoch outside frozen schedule")
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

def configure_visual_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    model.freeze_low_level("layer2")
    backbone = [parameter for module in (model.layer3, model.layer4) for parameter in module.parameters() if parameter.requires_grad]
    backbone_ids = {id(parameter) for parameter in model.backbone_parameters()}
    head = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in backbone_ids]
    if not backbone or not head:
        raise ValueError("visual parameter partition is empty")
    return backbone, head

def freeze_batch_norm_statistics(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.eval()

def fixed_macro_f1(labels: np.ndarray, predictions: np.ndarray, classes: int = 37) -> float:
    values = []
    for label in range(classes):
        tp = np.sum((labels == label) & (predictions == label))
        fp = np.sum((labels != label) & (predictions == label))
        fn = np.sum((labels == label) & (predictions != label))
        values.append(0.0 if 2 * tp + fp + fn == 0 else float(2 * tp / (2 * tp + fp + fn)))
    return float(np.mean(values))
def _soft_teacher_at_temperature(probability: torch.Tensor, temperature: float) -> torch.Tensor:
    value = probability.clamp_min(1e-12).pow(1.0 / temperature)
    return value / value.sum(1, keepdim=True)

def _optimizer(model: nn.Module):
    backbone, head = configure_visual_parameters(model)
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": 1e-5, "name": "backbone"},
        {"params": head, "lr": 2e-4, "name": "head"},
    ], weight_decay=0.08)
    lambdas = [lambda epoch: 0.5 if epoch == 0 else 1.0,
               lambda epoch: head_lr_factor(min(epoch, 15))]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    return optimizer, scheduler

def _run_epoch(model, loader, device, class_weight, optimizer, scaler, train, accumulation, max_batches):
    model.train(train)
    if train:
        freeze_batch_norm_statistics(model)
        optimizer.zero_grad(set_to_none=True)
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    labels_all, predictions_all = [], []
    sums = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "relation": 0.0}
    seen = 0
    updates = {"attempted": 0, "successful": 0, "skipped_amp": 0, "finite_gradient": 0}
    for batch_index, batch in enumerate(loader):
        if batch_index >= total_batches:
            break
        images = batch["images"].to(device, non_blocking=True)
        valid = batch["view_valid"].to(device, non_blocking=True)
        quality = batch["view_quality"].to(device, non_blocking=True)
        time_position = batch["time_position"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        teacher = batch["teacher"].to(device, non_blocking=True)
        geometry = batch["geometry"].to(device, non_blocking=True)
        if train:
            images = augment_rgb(images)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(images, valid, quality, time_position)
                logits = output["logits"]
                ce = nn.functional.cross_entropy(logits, labels, weight=class_weight, label_smoothing=0.1)
                target_t = _soft_teacher_at_temperature(teacher, 2.0)
                kd = nn.functional.kl_div(nn.functional.log_softmax(logits / 2.0, dim=1), target_t, reduction="batchmean") * 4.0
                relation = relation_gram_loss(output["visual_embedding"], geometry)
                loss = ce + kd + 0.2 * relation
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite visual training loss")
        if train:
            divisor = accumulation_divisor(batch_index, total_batches, accumulation)
            scaler.scale(loss / divisor).backward()
            if should_optimizer_step(batch_index, total_batches, accumulation):
                scaler.unscale_(optimizer)
                trainable = [p for p in model.parameters() if p.requires_grad]
                finite_gradient = all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
                if finite_gradient:
                    updates["finite_gradient"] += 1
                nn.utils.clip_grad_norm_(trainable, 5.0)
                before = scaler.get_scale(); updates["attempted"] += 1
                scaler.step(optimizer); scaler.update(); after = scaler.get_scale()
                if finite_gradient and after >= before:
                    updates["successful"] += 1
                else:
                    updates["skipped_amp"] += 1
                optimizer.zero_grad(set_to_none=True)
        count = len(labels); seen += count
        for key, value in (("loss", loss), ("ce", ce), ("kd", kd), ("relation", relation)):
            sums[key] += float(value.detach()) * count
        labels_all.append(labels.detach().cpu().numpy())
        predictions_all.append(logits.detach().argmax(1).cpu().numpy())
    labels_np = np.concatenate(labels_all); predictions_np = np.concatenate(predictions_all)
    if train and updates["successful"] == 0:
        raise RuntimeError("epoch has no successful optimizer update")
    return {**{key: value / seen for key, value in sums.items()}, "accuracy": float(np.mean(labels_np == predictions_np)),
            "macro_f1_fixed37": fixed_macro_f1(labels_np, predictions_np), "rows": seen, "batches": total_batches,
            "optimizer_updates": updates}
