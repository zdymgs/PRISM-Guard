from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
SEED = 20260913
CAMERAS = ('cam1', 'cam2', 'cam3', 'cam4')
MODELS = ('videomaev2_large', 'vjepa2_vitl_ssv2')

def subject_fold(subject: str) -> str:
    match = re.fullmatch('subject(\\d+)', subject.lower())
    if not match:
        raise ValueError(f'invalid MMACT subject: {subject}')
    subject_id = int(match.group(1))
    return 'F0' if 17 <= subject_id <= 20 else f'F{subject_id % 4 + 1}'

def fixed_macro_f1(y: np.ndarray, pred: np.ndarray, classes: int) -> float:
    scores = []
    for label in range(classes):
        tp = int(np.sum((y == label) & (pred == label)))
        fp = int(np.sum((y != label) & (pred == label)))
        fn = int(np.sum((y == label) & (pred != label)))
        scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))

def metrics(y: np.ndarray, logits: np.ndarray, classes: int) -> dict[str, float]:
    pred = np.asarray(logits).argmax(1)
    return {'macro_f1_fixed37': fixed_macro_f1(y, pred, classes), 'accuracy': float(np.mean(y == pred))}

def train_normalizer(x: np.ndarray, mask: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chosen = np.asarray(x[train][mask[train]], np.float64)
    if len(chosen) == 0:
        raise ValueError('no present training rows for normalizer')
    mean = chosen.mean(0)
    std = np.maximum(chosen.std(0), 1e-06)
    return (mean.astype(np.float32), std.astype(np.float32))

def masked_view_mean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = mask.astype(np.float32)[..., None]
    return (np.nan_to_num(x, nan=0.0).astype(np.float32) * weights).sum(1) / np.maximum(weights.sum(1), 1.0)

class RegularizedHead(nn.Module):

    def __init__(self, width: int, classes: int):
        super().__init__()
        hidden = min(192, max(64, width // 4))
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@torch.inference_mode()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch: int=512) -> np.ndarray:
    model.eval()
    result = []
    loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(x, np.float32))), batch_size=batch)
    for value, in loader:
        result.append(model(value.to(device)).float().cpu().numpy())
    return np.concatenate(result)

def train_head(x: np.ndarray, y: np.ndarray, train: np.ndarray, validation: np.ndarray, classes: int, device: torch.device, checkpoint: Path, log: Path, seed: int, epochs: int=60) -> tuple[RegularizedHead, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = RegularizedHead(x.shape[1], classes).to(device)
    counts = np.bincount(y[train], minlength=classes).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    weights /= weights.mean()
    loss_weights = torch.from_numpy(weights).to(device)
    data = TensorDataset(torch.from_numpy(np.asarray(x[train], np.float32)), torch.from_numpy(np.asarray(y[train], np.int64)))
    loader = DataLoader(data, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.0003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best = (-1.0, -1.0)
    started = time.perf_counter()
    best_epoch = 0
    with log.open('w', encoding='utf-8') as handle:
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = (xb.to(device), yb.to(device))
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb, weight=loss_weights, label_smoothing=0.02)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach()) * len(yb)
            scheduler.step()
            score = metrics(y[validation], predict(model, x[validation], device), classes)
            record = {'epoch': epoch, 'train_loss': total / len(train), **score, 'elapsed_seconds': time.perf_counter() - started}
            handle.write(json.dumps(record) + '\n')
            handle.flush()
            key = (score['macro_f1_fixed37'], score['accuracy'])
            if key > best:
                best, best_epoch = (key, epoch)
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'validation_metrics': score, 'input_width': x.shape[1], 'classes': classes, 'seed': seed}, checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved['state_dict'])
    return (model, {'best_epoch': best_epoch, 'macro_f1_fixed37': best[0], 'accuracy': best[1], 'elapsed_seconds': time.perf_counter() - started})

def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, np.float64)
    x -= x.max(1, keepdims=True)
    out = np.exp(x)
    out /= out.sum(1, keepdims=True)
    return out.astype(np.float32)

def masked_equal_mean(probabilities: np.ndarray, availability: np.ndarray) -> np.ndarray:
    weights = availability.astype(np.float32)[..., None]
    return (probabilities * weights).sum(0) / np.maximum(weights.sum(0), 1.0)

def sparse_forward_select(val_prob: np.ndarray, val_available: np.ndarray, y: np.ndarray, names: Sequence[str], classes: int) -> dict:
    candidates = []
    singles = []
    for i, name in enumerate(names):
        fused = weighted_available_mix(val_prob, val_available, [i], [1.0])
        score = metrics(y, fused, classes)
        singles.append((score, i))
        candidates.append({'step': 0, 'expert': name, **score})
    chosen = [max(singles, key=lambda z: (z[0]['macro_f1_fixed37'], z[0]['accuracy'], -z[1]))[1]]
    weights = [1.0]
    current = metrics(y, masked_equal_mean(val_prob[chosen], val_available[chosen]), classes)
    while len(chosen) < min(4, len(names)):
        best = None
        for i, name in enumerate(names):
            if i in chosen:
                continue
            for alpha in (0.25, 0.5, 0.75):
                proposed_weights = [w * (1 - alpha) for w in weights] + [alpha]
                fused = weighted_available_mix(val_prob, val_available, chosen + [i], proposed_weights)
                score = metrics(y, fused, classes)
                record = {'step': len(chosen), 'expert': name, 'new_weight': alpha, **score}
                candidates.append(record)
                key = (score['macro_f1_fixed37'], score['accuracy'], -i, -alpha)
                if best is None or key > best[0]:
                    best = (key, i, alpha, score)
        if best is None or best[3]['macro_f1_fixed37'] <= current['macro_f1_fixed37']:
            break
        weights = [w * (1 - best[2]) for w in weights] + [best[2]]
        chosen.append(best[1])
        current = best[3]
    recomputed = metrics(y, weighted_available_mix(val_prob, val_available, chosen, weights), classes)
    if recomputed != current:
        raise AssertionError('selector score does not match frozen weighted inference')
    return {'indices': chosen, 'names': [names[i] for i in chosen], 'weights': weights, 'validation_metrics': recomputed, 'candidates': candidates}

def weighted_available_mix(prob: np.ndarray, available: np.ndarray, indices: Sequence[int], weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, np.float32)[:, None] * available[np.asarray(indices)].astype(np.float32)
    numerator = (prob[np.asarray(indices)] * w[..., None]).sum(0)
    denominator = w.sum(0)[:, None]
    fallback = np.full_like(numerator, 1.0 / numerator.shape[1])
    return np.where(denominator > 0, numerator / np.maximum(denominator, 1e-12), fallback)

def ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(('\n'.join(values) + '\n').encode()).hexdigest()
