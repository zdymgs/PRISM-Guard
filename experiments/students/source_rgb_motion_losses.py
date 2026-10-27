"""Native six-channel source motion objectives."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
def diagonal_contrastive(first, second, first_mask, second_mask, temperature=0.08):
    logits = torch.matmul(first, second.transpose(-1, -2)) / float(temperature)
    size = logits.shape[-1]
    target = torch.arange(size, device=logits.device)

    def direction(scores, query_mask, candidate_mask):
        valid = query_mask & candidate_mask
        scores = scores.masked_fill(~candidate_mask.unsqueeze(-2), -10000.0)
        flat_valid = valid.reshape(-1)
        if not flat_valid.any():
            zero = scores.sum() * 0.0
            return (zero, zero.detach(), zero.detach())
        flat = scores.reshape(-1, size)[flat_valid]
        target_flat = target.view(*[1] * (scores.ndim - 2), size).expand(*scores.shape[:-1]).reshape(-1)[flat_valid]
        loss = F.cross_entropy(flat, target_flat)
        correct = (flat.argmax(-1) == target_flat).sum()
        count = flat_valid.sum()
        return (loss, correct, count)
    a = direction(logits, first_mask, second_mask)
    b = direction(logits.transpose(-1, -2), second_mask, first_mask)
    return (0.5 * (a[0] + b[0]), a[1] + b[1], a[2] + b[2])

def pooled_alignment(tokens, mask, dimensions):
    weight = mask.to(tokens.dtype).unsqueeze(-1)
    pooled = (tokens * weight).sum(dim=dimensions) / weight.sum(dim=dimensions).clamp_min(1.0)
    return (F.normalize(pooled, dim=-1), mask.any(dim=dimensions))

def supervised_cross_modal_contrastive(first, second, labels, valid, temperature=0.08):
    logits = torch.matmul(first, second.transpose(0, 1)) / float(temperature)
    candidate = valid.unsqueeze(0).expand_as(logits)
    positive = labels[:, None].eq(labels[None, :]) & candidate
    query = valid & positive.any(1)
    logits = logits.masked_fill(~candidate, -10000.0)
    positive_logits = logits.masked_fill(~positive, -10000.0)
    loss = -(torch.logsumexp(positive_logits, 1) - torch.logsumexp(logits, 1))
    if not query.any():
        return logits.sum() * 0.0
    forward = loss[query].mean()
    reverse = supervised_cross_modal_contrastive_one_direction(logits.t(), positive.t(), valid)
    return 0.5 * (forward + reverse)

def supervised_cross_modal_contrastive_one_direction(logits, positive, valid):
    candidate = valid.unsqueeze(0).expand_as(logits)
    positive = positive & candidate
    query = valid & positive.any(1)
    logits = logits.masked_fill(~candidate, -10000.0)
    positive_logits = logits.masked_fill(~positive, -10000.0)
    loss = -(torch.logsumexp(positive_logits, 1) - torch.logsumexp(logits, 1))
    return loss[query].mean() if query.any() else logits.sum() * 0.0

def reconstruction_loss(output, modality):
    predicted, target, mask = (output[f'{modality}_reconstruction'], output[f'{modality}_reconstruction_target'], output[f'{modality}_reconstruction_mask'])
    return F.smooth_l1_loss(predicted[mask], target[mask]) if mask.any() else predicted.sum() * 0.0

def distillation_loss(student, teacher, temperature=2.0):
    temperature = float(temperature)
    return F.kl_div(F.log_softmax(student / temperature, -1), F.softmax(teacher / temperature, -1), reduction='batchmean') * temperature ** 2

def teacher_feature_loss(student, teacher):
    return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean() if student.numel() else student.sum() * 0.0

def asymmetric_token_distillation(student, teacher, student_mask, teacher_mask):
    valid = student_mask & teacher_mask
    if not valid.any():
        return student.sum() * 0.0
    return (1.0 - F.cosine_similarity(student, teacher.detach(), dim=-1)[valid]).mean()

def _masked_ce(logits, labels, valid, weights, smoothing):
    return F.cross_entropy(logits[valid], labels[valid], weight=weights, label_smoothing=smoothing) if valid.any() else logits.sum() * 0.0

def losses(output, batch, *, class_weights=None, semantic_weight=2.0, token_weight=0.25, local_weight=0.05, global_weight=0.15, reconstruction_weight=0.1, distillation_weight=0.75, teacher_feature_weight=0.25, temperature=0.08, distillation_temperature=2.0, imu_teacher_weight=1.0, imu_teacher_temperature=1.0, imu_to_skeleton_token_weight=0.0, imu_to_skeleton_semantic_weight=0.0, imu_to_skeleton_logit_weight=0.0, label_smoothing=0.08):
    """Return original loss components with availability-aware branch masking."""
    labels = batch['label']
    skeleton_valid = output['skeleton_mask'].any(dim=(1, 2, 3))
    imu_valid = output['imu_mask'].any(dim=(1, 2, 3))
    semantic = 0.5 * (_masked_ce(output['skeleton_logits'], labels, skeleton_valid, class_weights, label_smoothing) + _masked_ce(output['imu_logits'], labels, imu_valid, class_weights, label_smoothing))
    sa, ia = (output['skeleton_alignment'], output['imu_alignment'])
    sm, im = (output['skeleton_mask'], output['imu_mask'])
    token, token_correct, token_count = diagonal_contrastive(sa.permute(0, 1, 3, 2, 4), ia.permute(0, 1, 3, 2, 4), sm.permute(0, 1, 3, 2), im.permute(0, 1, 3, 2), temperature)
    sl, slm = pooled_alignment(sa, sm, (2,))
    il, ilm = pooled_alignment(ia, im, (2,))
    local, local_correct, local_count = diagonal_contrastive(sl, il, slm, ilm, temperature)
    sg, sgm = pooled_alignment(sa, sm, (1, 2, 3))
    ig, igm = pooled_alignment(ia, im, (1, 2, 3))
    global_loss = supervised_cross_modal_contrastive(sg, ig, labels, sgm & igm, temperature)
    reconstruction = 0.5 * (reconstruction_loss(output, 'skeleton') + reconstruction_loss(output, 'imu'))
    teacher_logits = batch['teacher_logits']
    distill = 0.5 * (_masked_distill(output['skeleton_logits'], teacher_logits, skeleton_valid, distillation_temperature) + _masked_distill(output['imu_logits'], teacher_logits, imu_valid, distillation_temperature))
    teacher_features = batch['teacher_features']
    if teacher_features.ndim == 2:
        teacher_features = teacher_features[:, None].expand(-1, 2, -1)
    feature_valid = torch.isfinite(teacher_features).all(-1)
    sf = output['skeleton_teacher_features']
    inf = output['imu_teacher_features']
    skeleton_window_valid = sm.any(dim=(2, 3))
    imu_window_valid = im.any(dim=(2, 3))
    feature_distill = 0.5 * (_masked_feature(sf, teacher_features, feature_valid & skeleton_window_valid) + _masked_feature(inf, teacher_features, feature_valid & imu_window_valid))
    imu_teacher = output['imu_logits'].sum() * 0.0
    if imu_teacher_weight > 0 and 'imu_teacher_logits' in batch:
        valid = batch.get('imu_teacher_valid', torch.ones_like(imu_valid)) & imu_valid
        if valid.any():
            imu_teacher = distillation_loss(output['imu_logits'][valid], batch['imu_teacher_logits'][valid], imu_teacher_temperature)
    token_sd = asymmetric_token_distillation(ia, sa, im, sm)
    semantic_sd = _masked_feature(output['imu_compact'], output['skeleton_compact'].detach(), imu_valid & skeleton_valid)
    logit_sd = _masked_distill(output['imu_logits'], output['skeleton_logits'].detach(), imu_valid & skeleton_valid, distillation_temperature)
    domain = semantic * 0.0
    total = semantic_weight * semantic + token_weight * token + local_weight * local + global_weight * global_loss + reconstruction_weight * reconstruction + distillation_weight * distill + teacher_feature_weight * feature_distill + imu_teacher_weight * imu_teacher + imu_to_skeleton_token_weight * token_sd + imu_to_skeleton_semantic_weight * semantic_sd + imu_to_skeleton_logit_weight * logit_sd
    return {'loss': total, 'semantic': semantic, 'skeleton_ce': _masked_ce(output['skeleton_logits'], labels, skeleton_valid, class_weights, label_smoothing), 'imu_ce': _masked_ce(output['imu_logits'], labels, imu_valid, class_weights, label_smoothing), 'token': token, 'local': local, 'global': global_loss, 'reconstruction': reconstruction, 'distillation': distill, 'feature_distillation': feature_distill, 'imu_teacher_distillation': imu_teacher, 'skeleton_token_distillation': token_sd, 'skeleton_semantic_distillation': semantic_sd, 'skeleton_logit_distillation': logit_sd, 'domain': domain, 'token_correct': token_correct.to(total.dtype), 'token_count': token_count.to(total.dtype), 'local_correct': local_correct.to(total.dtype), 'local_count': local_count.to(total.dtype)}

def _masked_distill(student, teacher, valid, temperature):
    return distillation_loss(student[valid], teacher[valid], temperature) if valid.any() else student.sum() * 0.0

def _masked_feature(student, teacher, valid):
    return teacher_feature_loss(student[valid], teacher[valid]) if valid.any() else student.sum() * 0.0
