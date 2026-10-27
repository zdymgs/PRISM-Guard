from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from session_kd.mmfit_rgb_benchmark import CLASSES, CompactStudent, metrics
SEED = 20260912
TEACHER_ORDER = ('rgb', 'pose', 'imu', 'videomaev2_large', 'vjepa2', 'joint')
EXPECTED_SPLITS = {'train': 23612, 'validation': 5195, 'test': 11668}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def ordered_id_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest()

def validate_split_firewall(rows: Sequence[dict[str, str]], labels: np.ndarray) -> dict[str, np.ndarray]:
    if len(rows) != 40475 or labels.shape != (40475,):
        raise ValueError('V2 requires the frozen 40,475-row V1 index and label sentinel')
    split = np.asarray([row['split'] for row in rows])
    indices = {name: np.flatnonzero(split == name) for name in EXPECTED_SPLITS}
    if {name: len(value) for name, value in indices.items()} != EXPECTED_SPLITS:
        raise ValueError('frozen split counts changed')
    if not np.all(labels[indices['test']] == -1):
        raise ValueError('TEST labels must be sentinel -1 before final evaluation')
    source = np.r_[indices['train'], indices['validation']]
    if np.any(labels[source] < 0) or np.any(labels[source] >= CLASSES):
        raise ValueError('source labels are missing or invalid')
    return indices

def validate_strong_cache(array: np.ndarray, rows: int=40475, width: int=1024) -> None:
    if array.shape != (rows, width):
        raise ValueError(f'strong feature cache shape must be [{rows},{width}], got {array.shape}')
    for start in range(0, rows, 2048):
        if not np.isfinite(np.asarray(array[start:start + 2048], dtype=np.float32)).all():
            raise ValueError('strong feature cache contains non-finite values')

def fit_feature_normalizer(array: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(array[train_indices], dtype=np.float64)
    mean = x.mean(0)
    std = np.maximum(x.std(0), 1e-06)
    return (mean.astype(np.float32), std.astype(np.float32))

def standardize(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(array, dtype=np.float32) - mean) / std

def normalized_pooled_statistics(pose: np.ndarray, imu: np.ndarray, presence: np.ndarray, normalizer: dict[str, np.ndarray], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = (np.asarray(pose[indices], np.float32) - normalizer['pose_mean']) / normalizer['pose_std']
    m = (np.asarray(imu[indices], np.float32) - normalizer['imu_mean']) / normalizer['imu_std']
    present = np.asarray(presence[indices], np.float32)
    m = m.reshape(len(indices), 8, 3, m.shape[-1]) * present[:, :, None, None]
    m = m.reshape(len(indices), 24, m.shape[-1])
    return (np.concatenate((p.mean(2), p.std(2)), axis=1), np.concatenate((m.mean(2), m.std(2)), axis=1), present)

class StrongMLP(nn.Module):

    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.inputs = inputs
        self.net = nn.Sequential(nn.Linear(inputs, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ArrayDataset(Dataset):

    def __init__(self, arrays: Sequence[np.ndarray], indices: np.ndarray, labels: np.ndarray | None=None):
        self.arrays, self.indices, self.labels = (arrays, np.asarray(indices, dtype=np.int64), labels)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        row = int(self.indices[position])
        parts = [np.asarray(value[row], np.float32).reshape(-1) for value in self.arrays]
        x = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return x if self.labels is None else (x, int(self.labels[row]))

def make_loader(dataset: Dataset, shuffle: bool, seed: int, batch: int=256) -> DataLoader:
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0, generator=torch.Generator().manual_seed(seed), drop_last=False, pin_memory=torch.cuda.is_available())

@torch.inference_mode()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        output.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(output)

def train_strong_head(model: StrongMLP, train_loader: DataLoader, val_loader: DataLoader, val_labels: np.ndarray, class_weights: np.ndarray, device: torch.device, checkpoint: Path, log_path: Path, *, epochs: int=60) -> dict[str, float | int]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best_f1, best_accuracy, best_epoch = (-1.0, -1.0, 0)
    started = time.perf_counter()
    with log_path.open('w', encoding='utf-8') as log:
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            seen = 0
            for x, y in train_loader:
                x, y = (x.to(device, non_blocking=True), y.to(device, non_blocking=True))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    loss = F.cross_entropy(model(x), y, weight=weights)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach()) * len(y)
                seen += len(y)
            scheduler.step()
            logits = predict_logits(model, val_loader, device)
            score = metrics(val_labels, logits.argmax(1))
            record = {'epoch': epoch, 'train_loss': total / seen, 'macro_f1_fixed11': score['macro_f1_fixed11'], 'accuracy': score['accuracy'], 'elapsed_seconds': time.perf_counter() - started}
            log.write(json.dumps(record) + '\n')
            log.flush()
            if (score['macro_f1_fixed11'], score['accuracy']) > (best_f1, best_accuracy):
                best_f1, best_accuracy, best_epoch = (score['macro_f1_fixed11'], score['accuracy'], epoch)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'val_metrics': score}, checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved['state_dict'])
    return {'best_epoch': best_epoch, 'best_macro_f1': best_f1, 'best_accuracy': best_accuracy, 'elapsed_seconds': time.perf_counter() - started}

def probabilities_from_logits(logits: np.ndarray, temperature: float=2.0) -> np.ndarray:
    x = np.asarray(logits, np.float64) / temperature
    x -= x.max(1, keepdims=True)
    result = np.exp(x)
    result /= result.sum(1, keepdims=True)
    return result.astype(np.float32)
