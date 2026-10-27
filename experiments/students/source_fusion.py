"""Original clip-level CUHK-X source fusion loop and losses."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from prism.student.native.mobind import NativeSkeletonPartEncoder as SkeletonPartEncoder, NativeSeparateMotionEncoder as SeparateMotionEncoder, NativeJointMotionEncoder as JointMotionEncoder
from .cache import MOTION_FIELDS
from .source_losses import class_weights, relation_loss

def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}

def corrupt_visual_sequence(sequence: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    batch = sequence.shape[0]
    selected = torch.rand(batch, 1, 1, 1, 1, device=sequence.device)
    selected = selected < args.visual_corruption_probability
    feature_keep = torch.rand(batch, 1, 1, 1, sequence.shape[-1], device=sequence.device) >= args.visual_feature_dropout
    view_keep = torch.rand(batch, 2, 3, 1, 1, device=sequence.device) >= args.visual_view_dropout
    empty = ~view_keep.any(dim=2, keepdim=True)
    fallback = torch.zeros_like(view_keep)
    fallback[:, :, 0] = True
    view_keep = torch.where(empty, fallback, view_keep)
    corrupted = sequence * feature_keep.to(sequence.dtype) * view_keep.to(sequence.dtype)
    return torch.where(selected, corrupted, sequence)

def model_forward(model, batch, args=None, augment_visual=False):
    motion = {field: batch[field] for field in MOTION_FIELDS}
    sequence = batch['backbone_sequence']
    if augment_visual:
        if args is None:
            raise ValueError('visual corruption requires training arguments')
        sequence = corrupt_visual_sequence(sequence, args)
    return model.forward_from_backbone_sequence(sequence, batch['view_valid'], batch['view_quality'], batch['global_time_position'], motion)

def visual_head_parameters(visual: nn.Module) -> list[nn.Parameter]:
    modules = [visual.temporal_encoder, visual.temporal_fusion, visual.token_encoder, visual.quality_gate, visual.stage_fusion, visual.classifier]
    result = [visual.time_position, visual.view_embedding, visual.window_embedding]
    result.extend((parameter for module in modules for parameter in module.parameters()))
    return [parameter for parameter in result if parameter is not None]

def set_stage(model: nn.Module, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.motion_residual.parameters():
        parameter.requires_grad_(True)
    if stage == 'B':
        for parameter in visual_head_parameters(model.visual):
            parameter.requires_grad_(True)
    elif stage != 'A':
        raise ValueError(stage)

def selective_anchor_loss(logits: torch.Tensor, anchor_logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    anchor_correct = anchor_logits.argmax(dim=1).eq(labels)
    if not anchor_correct.any():
        return logits.sum() * 0.0
    per_sample = F.kl_div(F.log_softmax(logits / temperature, dim=1), F.softmax(anchor_logits / temperature, dim=1), reduction='none').sum(dim=1) * temperature ** 2
    return per_sample[anchor_correct].mean()

def subject_robust_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, users: list[str], weights: torch.Tensor, label_smoothing: float, worst_group_weight: float) -> torch.Tensor:
    mean_loss = F.cross_entropy(logits, labels, weight=weights, label_smoothing=label_smoothing)
    if worst_group_weight <= 0:
        return mean_loss
    per_sample = F.cross_entropy(logits, labels, weight=weights, label_smoothing=label_smoothing, reduction='none')
    group_losses = []
    for user in sorted(set(users)):
        selected = torch.tensor([value == user for value in users], device=logits.device, dtype=torch.bool)
        if selected.any():
            group_losses.append(per_sample[selected].mean())
    worst = torch.stack(group_losses).amax() if group_losses else mean_loss
    amount = float(worst_group_weight)
    return (1.0 - amount) * mean_loss + amount * worst

def losses(output: dict[str, torch.Tensor], batch: dict[str, Any], weights: torch.Tensor, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    temperature = args.distillation_temperature
    classification = subject_robust_cross_entropy(output['logits'], batch['label'], batch['user_id'], weights, args.label_smoothing, args.worst_group_weight)
    distillation = F.kl_div(F.log_softmax(output['logits'] / temperature, dim=1), F.softmax(batch['teacher_logits'] / temperature, dim=1), reduction='batchmean') * temperature ** 2
    relation = relation_loss(output['clip_embeddings'], batch['teacher_features'], output['clip_mask'])
    motion_aux = subject_robust_cross_entropy(output['motion_logits'], batch['label'], batch['user_id'], weights, args.label_smoothing, args.worst_group_weight)
    imu_teacher_distillation = output['motion_logits'].sum() * 0.0
    if args.imu_teacher_weight > 0 and 'imu_teacher_logits' in batch:
        teacher_valid = batch['imu_teacher_valid']
        if teacher_valid.any():
            imu_temperature = args.imu_teacher_temperature
            imu_teacher_distillation = F.kl_div(F.log_softmax(output['motion_logits'][teacher_valid] / imu_temperature, dim=1), F.softmax(batch['imu_teacher_logits'][teacher_valid] / imu_temperature, dim=1), reduction='batchmean') * imu_temperature ** 2
    anchor_logits = output.get('anchor_fusion_logits', output['unfused_visual_logits'].detach() if args.live_visual_anchor else batch['anchor_logits'])
    anchor = selective_anchor_loss(output['logits'], anchor_logits, batch['label'], temperature)
    joint_gate = output['logits'].sum() * 0.0
    if args.joint_gate_weight > 0 and 'joint_raw_imu_gate' in output:
        with torch.no_grad():
            skeleton_loss = F.cross_entropy(output['joint_skeleton_logits'], batch['label'], reduction='none')
            imu_loss = F.cross_entropy(output['joint_imu_logits'], batch['label'], reduction='none')
            gate_temperature = max(float(args.joint_gate_temperature), 0.001)
            joint_gate_target = torch.sigmoid((skeleton_loss - imu_loss) / gate_temperature)
        predicted_gate = output['joint_raw_imu_gate'].flatten(1).mean(dim=1)
        predicted_gate = predicted_gate.float().clamp(1e-06, 1.0 - 1e-06)
        joint_gate_target = joint_gate_target.float()
        joint_gate = -(joint_gate_target * predicted_gate.log() + (1.0 - joint_gate_target) * torch.log1p(-predicted_gate)).mean()
    with torch.no_grad():
        if args.reliability_target == 'loss_advantage':
            visual_loss = F.cross_entropy(output['unfused_visual_logits'], batch['label'], reduction='none')
            motion_loss = F.cross_entropy(output['motion_logits'], batch['label'], reduction='none')
            advantage_temperature = max(float(args.reliability_advantage_temperature), 0.001)
            reliability_target = torch.sigmoid((visual_loss - motion_loss) / advantage_temperature)
            reliability_valid = output['motion_available'].any(dim=1)
        elif args.reliability_target == 'oof_complementarity':
            if 'imu_teacher_logits' not in batch:
                raise RuntimeError('OOF complementarity requires --imu-teacher-logits')
            visual_correct = batch['teacher_logits'].argmax(dim=1).eq(batch['label'])
            motion_correct = batch['imu_teacher_logits'].argmax(dim=1).eq(batch['label'])
            reliability_valid = batch['imu_teacher_valid'] & (visual_correct ^ motion_correct)
        elif args.reliability_target == 'oof_visual_error':
            visual_correct = batch['teacher_logits'].argmax(dim=1).eq(batch['label'])
            motion_correct = ~visual_correct
            reliability_valid = output['motion_available'].any(dim=1)
        else:
            visual_correct = output['unfused_visual_logits'].argmax(dim=1).eq(batch['label'])
            motion_correct = output['motion_logits'].argmax(dim=1).eq(batch['label'])
            reliability_valid = visual_correct ^ motion_correct
        if args.reliability_target != 'loss_advantage':
            reliability_target = motion_correct.to(output['logits'].dtype)
    global_reliability = output['motion_global_reliability'].mean(dim=-1)
    local_reliability = output['motion_reliability'].flatten(1).mean(dim=1)
    if reliability_valid.any():
        target = reliability_target[reliability_valid].float()

        def probability_bce(probability: torch.Tensor) -> torch.Tensor:
            probability = probability[reliability_valid].float().clamp(1e-06, 1.0 - 1e-06)
            sample_weight = torch.where(target > 0.5, torch.full_like(target, args.reliability_positive_weight), torch.ones_like(target))
            per_sample = -(target * probability.log() + (1.0 - target) * torch.log1p(-probability))
            return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
        reliability = 0.5 * (probability_bce(global_reliability) + probability_bce(local_reliability))
    else:
        reliability = output['logits'].sum() * 0.0
    total = classification + args.distillation_weight * distillation + args.relation_weight * relation + args.motion_aux_weight * motion_aux + args.selective_anchor_weight * anchor + args.reliability_weight * reliability + args.imu_teacher_weight * imu_teacher_distillation + args.joint_gate_weight * joint_gate
    return {'loss': total, 'classification': classification, 'distillation': distillation, 'relation': relation, 'motion_aux': motion_aux, 'anchor': anchor, 'reliability': reliability, 'imu_teacher_distillation': imu_teacher_distillation, 'joint_gate': joint_gate}

def train_stage(model: nn.Module, data: DataLoader, labels: np.ndarray, args: argparse.Namespace, device: torch.device, stage: str, epochs: int) -> list[dict[str, Any]]:
    if epochs <= 0:
        return []
    set_stage(model, stage)
    is_temporal_candidate = False
    fresh_encoder_parameters: list[nn.Parameter] = []
    encoder = model.motion_residual.encoder
    if isinstance(encoder, SeparateMotionEncoder) and args.joint_freeze_pretrained_encoders:
        for module in (encoder.skeleton_encoder, encoder.imu_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    if isinstance(encoder, JointMotionEncoder) and args.joint_freeze_pretrained_encoders:
        frozen_modules: tuple[nn.Module, ...] = (encoder.skeleton_encoder, encoder.imu_encoder, encoder.imu_semantic_head)
        if not args.joint_train_alignment:
            frozen_modules += (encoder.skeleton_alignment, encoder.imu_alignment)
        for module in frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    if isinstance(encoder, SkeletonPartEncoder) and encoder.multistream_input:
        assert encoder.stream_stems is not None
        assert encoder.stream_fusion is not None
        fresh_encoder_parameters.extend(encoder.stream_stems.parameters())
        fresh_encoder_parameters.extend(encoder.stream_fusion.parameters())
        assert encoder.multistream_residual_logit is not None
        fresh_encoder_parameters.append(encoder.multistream_residual_logit)
    if isinstance(encoder, SkeletonPartEncoder) and encoder.adaptive_graph:
        fresh_encoder_parameters.extend((block.adaptive_adjacency for block in encoder.blocks if block.adaptive_adjacency is not None))
    if isinstance(encoder, (SeparateMotionEncoder, JointMotionEncoder)):
        fresh_encoder_parameters.extend(encoder.fresh_parameters())
        if stage == 'A' and args.joint_stage_a_adapter_only:
            for parameter in model.motion_residual.parameters():
                parameter.requires_grad_(False)
            for parameter in fresh_encoder_parameters:
                parameter.requires_grad_(True)
            if args.joint_train_alignment:
                for module in (encoder.skeleton_alignment, encoder.imu_alignment):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
    fusion_learning_rate = args.stage_b_fusion_learning_rate if stage == 'B' and args.stage_b_fusion_learning_rate is not None else args.fusion_learning_rate
    fresh_encoder_ids = {id(parameter) for parameter in fresh_encoder_parameters}
    core_parameters = [parameter for parameter in encoder.parameters() if id(parameter) not in fresh_encoder_ids and parameter.requires_grad] + [parameter for parameter in model.motion_residual.semantic_head.parameters() if parameter.requires_grad]
    core_ids = {id(parameter) for parameter in core_parameters}
    fusion_parameters = [parameter for parameter in model.motion_residual.parameters() if id(parameter) not in core_ids and parameter.requires_grad]
    groups = [{'params': core_parameters, 'lr': args.encoder_learning_rate, 'initial_lr': args.encoder_learning_rate}, {'params': fusion_parameters, 'lr': fusion_learning_rate, 'initial_lr': fusion_learning_rate}]
    if stage == 'B':
        groups.append({'params': visual_head_parameters(model.visual), 'lr': args.visual_learning_rate, 'initial_lr': args.visual_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    weights = class_weights(labels, args.class_weight_power, device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        if stage == 'A':
            model.visual.eval()
            model.motion_residual.train()
        cosine = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1)))
        for group in optimizer.param_groups:
            group['lr'] = args.minimum_learning_rate + (float(group['initial_lr']) - args.minimum_learning_rate) * cosine
        sums = {key: 0.0 for key in ('loss', 'classification', 'distillation', 'relation', 'motion_aux', 'anchor', 'reliability', 'imu_teacher_distillation', 'joint_gate')}
        samples = 0
        reliability_sum = reliability_count = 0.0
        temporal_ratio_sum = temporal_ratio_count = 0.0
        spatial_ratio_sum = spatial_ratio_count = 0.0
        started = time.perf_counter()
        for batch_index, batch in enumerate(data):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                output = model_forward(model, batch, args=args, augment_visual=True)
                values = losses(output, batch, weights, args)
            scaler.scale(values['loss']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 2.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch['label'])
            samples += count
            for key in sums:
                sums[key] += float(values[key].detach()) * count
            reliability_sum += float(output['motion_reliability'].detach().sum())
            reliability_count += output['motion_reliability'].numel()
            temporal_effect = output.get('temporal_effect_rms_ratio', output.get('temporal_residual_rms_ratio'))
            if temporal_effect is not None:
                temporal_ratio_sum += float(temporal_effect.detach().sum())
                temporal_ratio_count += temporal_effect.numel()
            spatial_effect = output.get('spatial_clip_effect_rms_ratio')
            if spatial_effect is not None:
                spatial_ratio_sum += float(spatial_effect.detach().sum())
                spatial_ratio_count += spatial_effect.numel()
        if args.all_label_refit and (not args.max_train_batches) and (samples != len(data.dataset)):
            raise RuntimeError(f'all-label fusion refit did not consume all 2914 rows in this epoch: observed={samples}, expected={len(data.dataset)}')
        record = {'stage': stage, 'epoch': epoch, 'encoder_learning_rate': 0.0 if False else optimizer.param_groups[0]['lr'], 'fusion_learning_rate': optimizer.param_groups[0]['lr'] if False else optimizer.param_groups[1]['lr'], 'visual_learning_rate': optimizer.param_groups[2]['lr'] if not False and len(optimizer.param_groups) > 2 else 0.0, **{f'train_{key}': value / max(samples, 1) for key, value in sums.items()}, 'mean_motion_reliability': reliability_sum / max(reliability_count, 1.0), 'residual_strength': float(model.motion_residual.strength().detach()), 'mean_temporal_effect_rms_ratio': temporal_ratio_sum / max(temporal_ratio_count, 1.0), 'train_samples': samples, 'seconds': time.perf_counter() - started}
        if spatial_ratio_count:
            record['mean_spatial_clip_effect_rms_ratio'] = spatial_ratio_sum / spatial_ratio_count
        if isinstance(encoder, SkeletonPartEncoder) and encoder.multistream_residual_logit is not None:
            record['skeleton_multistream_strength'] = float(torch.sigmoid(encoder.multistream_residual_logit).detach())
        if isinstance(encoder, JointMotionEncoder) and encoder.last_token_reliability is not None:
            record['joint_mean_imu_gate'] = float(encoder.last_token_reliability)
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return history
