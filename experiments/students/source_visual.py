"""Original fixed-budget CUHK-X visual source training loop."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from .source_losses import class_weights, relation_loss

def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}

def forward_model(model: nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    if bool(getattr(model, 'exact_time_modeling', False)):
        return model(batch['images'], batch['view_valid'], batch['view_quality'], batch['global_time_position'])
    return model(batch['images'], batch['view_valid'], batch['view_quality'])

def optimizer_for(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone = [parameter for parameter in model.backbone_parameters() if parameter.requires_grad]
    head = [parameter for parameter in model.head_parameters() if parameter.requires_grad]
    if not backbone or not head:
        raise RuntimeError('student V2 optimizer groups are empty')
    return torch.optim.AdamW([{'params': backbone, 'lr': args.backbone_learning_rate, 'initial_lr': args.backbone_learning_rate}, {'params': head, 'lr': args.head_learning_rate, 'initial_lr': args.head_learning_rate}], weight_decay=args.weight_decay, foreach=False)

def learning_rate_scale(epoch: int, maximum_epochs: int, minimum_ratio: float) -> float:
    if epoch <= 2:
        return epoch / 2.0
    progress = (epoch - 2) / max(maximum_epochs - 2, 1)
    return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * progress))

def train_epochs(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader | None, labels_for_weights: np.ndarray, args: argparse.Namespace, device: torch.device, epochs: int, early_stop: bool) -> tuple[nn.Module, list[dict[str, Any]], int]:
    optimizer = optimizer_for(model, args)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    weights = class_weights(labels_for_weights, args.class_weight_power, device)
    history: list[dict[str, Any]] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 1
    best_score = -math.inf
    stale = 0
    accumulation = max(int(args.gradient_accumulation), 1)
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        ratio = learning_rate_scale(epoch, max(epochs, 2), args.minimum_learning_rate / args.head_learning_rate)
        for group in optimizer.param_groups:
            group['lr'] = float(group['initial_lr']) * ratio
        sums = {'loss': 0.0, 'ce': 0.0, 'kd': 0.0, 'stage_kd': 0.0, 'relation': 0.0, 'feature': 0.0, 'samples': 0}
        optimizer.zero_grad(set_to_none=True)
        completed_batches = 0
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                output = forward_model(model, batch)
                ce = F.cross_entropy(output['logits'], batch['label'], weight=weights, label_smoothing=args.label_smoothing)
                temperature = args.distillation_temperature
                kd = F.kl_div(F.log_softmax(output['logits'] / temperature, dim=1), F.softmax(batch['teacher_logits'] / temperature, dim=1), reduction='batchmean') * temperature ** 2
                if args.stage_distillation_weight > 0:
                    teacher_stage = torch.stack((batch['teacher_early_logits'], batch['teacher_late_logits'], batch['teacher_temporal_delta_logits']), dim=1)
                    stage_kd = F.kl_div(F.log_softmax(output['stage_logits'] / temperature, dim=-1), F.softmax(teacher_stage / temperature, dim=-1), reduction='batchmean') * temperature ** 2 / teacher_stage.shape[1]
                else:
                    stage_kd = output['logits'].sum() * 0.0
                relation = relation_loss(output['clip_embeddings'], batch['teacher_features'], output['clip_mask'])
                if args.mode == 'feature':
                    feature_distance = 1.0 - F.cosine_similarity(output['projected_clip_embeddings'].float(), batch['teacher_features'].float(), dim=-1)
                    feature = (feature_distance * output['clip_mask'].to(feature_distance.dtype)).sum() / output['clip_mask'].sum().clamp_min(1)
                else:
                    feature = output['clip_embeddings'].sum() * 0.0
                loss = ce
                if args.mode in {'kd', 'hybrid', 'feature'}:
                    loss = loss + args.distillation_weight * kd
                    loss = loss + args.stage_distillation_weight * stage_kd
                if args.mode in {'hybrid', 'feature'}:
                    loss = loss + args.relation_weight * relation
                if args.mode == 'feature':
                    loss = loss + args.feature_weight * feature
            scaler.scale(loss / accumulation).backward()
            completed_batches += 1
            if completed_batches % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            count = len(batch['label'])
            sums['loss'] += float(loss.detach()) * count
            sums['ce'] += float(ce.detach()) * count
            sums['kd'] += float(kd.detach()) * count
            sums['stage_kd'] += float(stage_kd.detach()) * count
            sums['relation'] += float(relation.detach()) * count
            sums['feature'] += float(feature.detach()) * count
            sums['samples'] += count
        if completed_batches and completed_batches % accumulation:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        record: dict[str, Any] = {'epoch': epoch, 'backbone_learning_rate': optimizer.param_groups[0]['lr'], 'head_learning_rate': optimizer.param_groups[1]['lr'], 'train_loss': sums['loss'] / max(sums['samples'], 1), 'train_ce': sums['ce'] / max(sums['samples'], 1), 'train_kd': sums['kd'] / max(sums['samples'], 1), 'train_stage_kd': sums['stage_kd'] / max(sums['samples'], 1), 'train_relation': sums['relation'] / max(sums['samples'], 1), 'train_feature': sums['feature'] / max(sums['samples'], 1), 'train_samples': sums['samples'], 'seconds': time.perf_counter() - started}
        best_epoch = epoch
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    model.load_state_dict(best_state, strict=True)
    return (model, history, best_epoch)
