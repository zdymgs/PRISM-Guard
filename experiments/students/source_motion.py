"""Original CUHK-X source motion objectives and training loop."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from .cache import MOTION_FIELDS
from .source_losses import class_weights

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}

def motion_fields(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    motion = {field: batch[field] for field in MOTION_FIELDS}
    for field in ('imu_event_features', 'imu_event_valid'):
        if field in batch:
            motion[field] = batch[field]
    return motion

def diagonal_contrastive(first: torch.Tensor, second: torch.Tensor, first_mask: torch.Tensor, second_mask: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bidirectionally retrieve the matching index within each independent group."""
    logits = torch.matmul(first, second.transpose(-1, -2)) / temperature
    size = logits.shape[-1]
    target = torch.arange(size, device=logits.device)

    def direction(scores: torch.Tensor, query_mask: torch.Tensor, candidate_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_positive = query_mask & candidate_mask
        scores = scores.masked_fill(~candidate_mask.unsqueeze(-2), -10000.0)
        flat_scores = scores.reshape(-1, size)
        flat_valid = valid_positive.reshape(-1)
        flat_target = target.view(*[1] * (scores.ndim - 2), size).expand(*scores.shape[:-1]).reshape(-1)
        if not flat_valid.any():
            zero = scores.sum() * 0.0
            return (zero, zero.detach(), zero.detach())
        selected_scores = flat_scores[flat_valid]
        selected_target = flat_target[flat_valid]
        loss = F.cross_entropy(selected_scores, selected_target)
        correct = (selected_scores.argmax(dim=-1) == selected_target).sum()
        count = flat_valid.sum()
        return (loss, correct, count)
    forward_loss, forward_correct, forward_count = direction(logits, first_mask, second_mask)
    reverse_loss, reverse_correct, reverse_count = direction(logits.transpose(-1, -2), second_mask, first_mask)
    return (0.5 * (forward_loss + reverse_loss), forward_correct + reverse_correct, forward_count + reverse_count)

def supervised_cross_modal_contrastive(first: torch.Tensor, second: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = torch.matmul(first, second.transpose(0, 1)) / temperature
    candidate_valid = valid.unsqueeze(0).expand_as(logits)
    positive = labels[:, None].eq(labels[None, :]) & candidate_valid
    query_valid = valid & positive.any(dim=1)
    logits = logits.masked_fill(~candidate_valid, -10000.0)
    positive_logits = logits.masked_fill(~positive, -10000.0)
    loss = -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1))
    if not query_valid.any():
        return logits.sum() * 0.0
    forward = loss[query_valid].mean()
    reverse = supervised_cross_modal_contrastive_one_direction(logits.transpose(0, 1), positive.transpose(0, 1), valid)
    return 0.5 * (forward + reverse)

def supervised_cross_modal_contrastive_one_direction(logits: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    candidate_valid = valid.unsqueeze(0).expand_as(logits)
    positive = positive & candidate_valid
    query_valid = valid & positive.any(dim=1)
    logits = logits.masked_fill(~candidate_valid, -10000.0)
    positive_logits = logits.masked_fill(~positive, -10000.0)
    loss = -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1))
    return loss[query_valid].mean() if query_valid.any() else logits.sum() * 0.0

def pooled_alignment(tokens: torch.Tensor, mask: torch.Tensor, dimensions: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    weight = mask.to(tokens.dtype).unsqueeze(-1)
    pooled = (tokens * weight).sum(dim=dimensions) / weight.sum(dim=dimensions).clamp_min(1.0)
    valid = mask.any(dim=dimensions)
    return (F.normalize(pooled, dim=-1), valid)

def reconstruction_loss(output: dict[str, torch.Tensor], modality: str) -> torch.Tensor:
    predicted = output[f'{modality}_reconstruction']
    target = output[f'{modality}_reconstruction_target']
    mask = output[f'{modality}_reconstruction_mask']
    if not mask.any():
        return predicted.sum() * 0.0
    return F.smooth_l1_loss(predicted[mask], target[mask])

def distillation_loss(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = float(temperature)
    return F.kl_div(F.log_softmax(student / temperature, dim=-1), F.softmax(teacher / temperature, dim=-1), reduction='batchmean') * temperature ** 2

def teacher_feature_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()

def asymmetric_token_distillation(student: torch.Tensor, teacher: torch.Tensor, student_mask: torch.Tensor, teacher_mask: torch.Tensor) -> torch.Tensor:
    """Match exact part/time pairs without moving the privileged teacher."""
    valid = student_mask & teacher_mask
    if not valid.any():
        return student.sum() * 0.0
    similarity = F.cosine_similarity(student, teacher.detach(), dim=-1)
    return (1.0 - similarity[valid]).mean()

def losses(output: dict[str, torch.Tensor], batch: dict[str, Any], weights: torch.Tensor, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    labels = batch['label']
    skeleton_ce = F.cross_entropy(output['skeleton_logits'], labels, weight=weights, label_smoothing=args.label_smoothing)
    imu_ce = F.cross_entropy(output['imu_logits'], labels, weight=weights, label_smoothing=args.label_smoothing)
    skeleton_alignment = output['skeleton_alignment']
    imu_alignment = output['imu_alignment']
    skeleton_mask = output['skeleton_mask']
    imu_mask = output['imu_mask']
    skeleton_time = skeleton_alignment.permute(0, 1, 3, 2, 4)
    imu_time = imu_alignment.permute(0, 1, 3, 2, 4)
    skeleton_time_mask = skeleton_mask.permute(0, 1, 3, 2)
    imu_time_mask = imu_mask.permute(0, 1, 3, 2)
    token, token_correct, token_count = diagonal_contrastive(skeleton_time, imu_time, skeleton_time_mask, imu_time_mask, args.temperature)
    skeleton_local, skeleton_local_mask = pooled_alignment(skeleton_alignment, skeleton_mask, (2,))
    imu_local, imu_local_mask = pooled_alignment(imu_alignment, imu_mask, (2,))
    local, local_correct, local_count = diagonal_contrastive(skeleton_local, imu_local, skeleton_local_mask, imu_local_mask, args.temperature)
    skeleton_global, skeleton_global_mask = pooled_alignment(skeleton_alignment, skeleton_mask, (1, 2, 3))
    imu_global, imu_global_mask = pooled_alignment(imu_alignment, imu_mask, (1, 2, 3))
    global_valid = skeleton_global_mask & imu_global_mask
    global_alignment = supervised_cross_modal_contrastive(skeleton_global, imu_global, labels, global_valid, args.temperature)
    reconstruction = 0.5 * (reconstruction_loss(output, 'skeleton') + reconstruction_loss(output, 'imu'))
    teacher_logits = batch['teacher_logits']
    distillation = 0.5 * (distillation_loss(output['skeleton_logits'], teacher_logits, args.distillation_temperature) + distillation_loss(output['imu_logits'], teacher_logits, args.distillation_temperature))
    teacher_features = batch['teacher_features']
    feature_distillation = 0.5 * (teacher_feature_loss(output['skeleton_teacher_features'], teacher_features) + teacher_feature_loss(output['imu_teacher_features'], teacher_features))
    imu_teacher_distillation = output['imu_logits'].sum() * 0.0
    if args.imu_teacher_weight > 0 and 'imu_teacher_logits' in batch:
        teacher_valid = batch['imu_teacher_valid']
        if teacher_valid.any():
            imu_teacher_distillation = distillation_loss(output['imu_logits'][teacher_valid], batch['imu_teacher_logits'][teacher_valid], args.imu_teacher_temperature)
    skeleton_token_distillation = asymmetric_token_distillation(output['imu_alignment'], output['skeleton_alignment'], imu_mask, skeleton_mask)
    skeleton_semantic_distillation = teacher_feature_loss(output['imu_compact'], output['skeleton_compact'].detach())
    skeleton_logit_distillation = distillation_loss(output['imu_logits'], output['skeleton_logits'].detach(), args.distillation_temperature)
    semantic = 0.5 * (skeleton_ce + imu_ce)
    domain = semantic * 0.0
    total = args.semantic_weight * semantic + args.token_weight * token + args.local_weight * local + args.global_weight * global_alignment + args.reconstruction_weight * reconstruction + args.distillation_weight * distillation + args.teacher_feature_weight * feature_distillation + args.imu_teacher_weight * imu_teacher_distillation + args.imu_to_skeleton_token_weight * skeleton_token_distillation + args.imu_to_skeleton_semantic_weight * skeleton_semantic_distillation + args.imu_to_skeleton_logit_weight * skeleton_logit_distillation + args.domain_adversarial_weight * domain
    return {'loss': total, 'semantic': semantic, 'skeleton_ce': skeleton_ce, 'imu_ce': imu_ce, 'token': token, 'local': local, 'global': global_alignment, 'reconstruction': reconstruction, 'distillation': distillation, 'feature_distillation': feature_distillation, 'imu_teacher_distillation': imu_teacher_distillation, 'skeleton_token_distillation': skeleton_token_distillation, 'skeleton_semantic_distillation': skeleton_semantic_distillation, 'skeleton_logit_distillation': skeleton_logit_distillation, 'domain': domain, 'token_correct': token_correct.to(total.dtype), 'token_count': token_count.to(total.dtype), 'local_correct': local_correct.to(total.dtype), 'local_count': local_count.to(total.dtype)}

def train(model: MoBindLite, data: DataLoader, labels: np.ndarray, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    weights = class_weights(labels, args.class_weight_power, device)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_skeleton_teacher:
            for module in (model.skeleton_encoder, model.skeleton_head, model.skeleton_projection, model.skeleton_teacher_projection, model.skeleton_context, model.skeleton_reconstruction):
                module.eval()
        ratio = args.minimum_learning_rate / args.learning_rate + 0.5 * (1.0 - args.minimum_learning_rate / args.learning_rate) * (1.0 + math.cos(math.pi * (epoch - 1) / max(args.epochs - 1, 1)))
        learning_rate = args.learning_rate * ratio
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        sums = {key: 0.0 for key in ('loss', 'semantic', 'skeleton_ce', 'imu_ce', 'token', 'local', 'global', 'reconstruction', 'distillation', 'feature_distillation', 'imu_teacher_distillation', 'skeleton_token_distillation', 'skeleton_semantic_distillation', 'skeleton_logit_distillation', 'domain')}
        token_correct = token_count = local_correct = local_count = 0.0
        samples = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(data):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                output = model(motion_fields(batch), mask_ratio=args.mask_ratio)
                values = losses(output, batch, weights, args)
            scaler.scale(values['loss']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch['label'])
            samples += count
            for key in sums:
                sums[key] += float(values[key].detach()) * count
            token_correct += float(values['token_correct'].detach())
            token_count += float(values['token_count'].detach())
            local_correct += float(values['local_correct'].detach())
            local_count += float(values['local_count'].detach())
        if args.all_label_refit and (not args.max_train_batches) and (samples != len(data.dataset)):
            raise RuntimeError(f'all-label MoBind refit did not consume all 2914 rows in this epoch: observed={samples}, expected={len(data.dataset)}')
        record = {'epoch': epoch, 'learning_rate': learning_rate, **{f'train_{key}': value / max(samples, 1) for key, value in sums.items()}, 'train_token_retrieval': token_correct / max(token_count, 1.0), 'train_part_retrieval': local_correct / max(local_count, 1.0), 'train_samples': samples, 'seconds': time.perf_counter() - started}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return history
