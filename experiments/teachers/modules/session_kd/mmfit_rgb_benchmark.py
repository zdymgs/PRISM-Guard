from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
CLASSES = 11
SEED = 20260912

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    np.add.at(matrix, (y.astype(int), pred.astype(int)), 1)
    return matrix

def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    cm = confusion(y, pred)
    denom = cm.sum(1) + cm.sum(0)
    f1 = np.divide(2 * np.diag(cm), denom, out=np.zeros(CLASSES), where=denom > 0)
    return {'accuracy': float(np.mean(y == pred)), 'macro_f1_fixed11': float(f1.mean()), 'per_class_f1': f1.tolist(), 'confusion': cm.tolist()}

class TemporalBranch(nn.Module):

    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(inputs, hidden, 5, stride=2, padding=2), nn.GELU(), nn.Conv1d(hidden, hidden, 5, stride=2, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class ModalityTeacher(nn.Module):

    def __init__(self, modality: str) -> None:
        super().__init__()
        inputs = {'rgb': 512, 'pose': 51, 'imu': 24}[modality]
        self.modality = modality
        self.branch = TemporalBranch(inputs, 128)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(136 if modality == 'imu' else 128, CLASSES))

    def forward(self, rgb, pose, imu, presence):
        x = {'rgb': rgb.transpose(1, 2), 'pose': pose, 'imu': imu}[self.modality]
        z = self.branch(x)
        if self.modality == 'imu':
            z = torch.cat([z, presence], 1)
        return self.head(z)

class CompactStudent(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rgb = TemporalBranch(512, 16)
        self.pose = TemporalBranch(51, 16)
        self.imu = TemporalBranch(24, 16)
        self.head = nn.Sequential(nn.Linear(56, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, CLASSES))

    def forward(self, rgb, pose, imu, presence):
        z = torch.cat([self.rgb(rgb.transpose(1, 2)), self.pose(pose), self.imu(imu), presence], 1)
        return self.head(z)

class MMFitDataset(Dataset):

    def __init__(self, rgb, pose, imu, presence, indices: np.ndarray, labels: np.ndarray | None, normalizer: dict[str, np.ndarray]) -> None:
        self.rgb, self.pose, self.imu, self.presence = (rgb, pose, imu, presence)
        self.indices, self.labels, self.n = (indices, labels, normalizer)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = int(self.indices[item])
        rgb = (np.asarray(self.rgb[i], np.float32) - self.n['rgb_mean']) / self.n['rgb_std']
        pose = (np.asarray(self.pose[i], np.float32) - self.n['pose_mean']) / self.n['pose_std']
        imu = (np.asarray(self.imu[i], np.float32) - self.n['imu_mean']) / self.n['imu_std']
        present = np.asarray(self.presence[i], np.float32).copy()
        imu = imu.reshape(8, 3, 250) * present[:, None, None]
        values = (rgb, pose, imu.reshape(24, 250), present)
        return values if self.labels is None else (*values, int(self.labels[i]))

def _moments(array, indices: np.ndarray, axes: tuple[int, ...], chunk: int=512):
    total = total2 = None
    count = 0
    for start in range(0, len(indices), chunk):
        x = np.asarray(array[indices[start:start + chunk]], dtype=np.float64)
        s, s2 = (x.sum(axis=axes), np.square(x).sum(axis=axes))
        total = s if total is None else total + s
        total2 = s2 if total2 is None else total2 + s2
        count += math.prod((x.shape[a] for a in axes))
    mean = total / count
    std = np.sqrt(np.maximum(total2 / count - mean * mean, 1e-12))
    return (mean.astype(np.float32), np.maximum(std, 1e-06).astype(np.float32))

def fit_normalizer(rgb, pose, imu, presence, train_indices: np.ndarray) -> dict[str, np.ndarray]:
    rgb_mean, rgb_std = _moments(rgb, train_indices, (0, 1))
    pose_mean, pose_std = _moments(pose, train_indices, (0, 2))
    if not np.all(np.asarray(presence[train_indices]) == 1):
        raise ValueError('V1 requires presence-aware IMU fitting implementation for missing streams')
    imu_mean, imu_std = _moments(imu, train_indices, (0, 2))
    return {'rgb_mean': rgb_mean[None, :], 'rgb_std': rgb_std[None, :], 'pose_mean': pose_mean[:, None], 'pose_std': pose_std[:, None], 'imu_mean': imu_mean[:, None], 'imu_std': imu_std[:, None]}

def make_loader(dataset, batch: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, generator=generator, num_workers=0, pin_memory=True, drop_last=False)

def _move(batch, device, labeled=True):
    xs = [x.to(device, non_blocking=True) for x in batch[:4]]
    return (*xs, batch[4].to(device, non_blocking=True)) if labeled else tuple(xs)

@torch.inference_mode()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    outputs = []
    for batch in loader:
        xs = _move(batch, device, labeled=len(batch) == 5)[:4]
        outputs.append(model(*xs).float().cpu().numpy())
    return np.concatenate(outputs)

def train_source(model, train_loader, val_loader, val_labels: np.ndarray, class_weights: np.ndarray, device, checkpoint: Path, log_path: Path, epochs: int=30) -> dict:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    weights = torch.as_tensor(class_weights, device=device, dtype=torch.float32)
    best = (-1.0, -1.0)
    best_epoch = 0
    started = time.perf_counter()
    with log_path.open('w', encoding='utf-8') as log:
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            seen = 0
            for batch in train_loader:
                *xs, y = _move(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    loss = F.cross_entropy(model(*xs), y, weight=weights)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * len(y)
                seen += len(y)
            scheduler.step()
            logits = predict(model, val_loader, device)
            score = metrics(val_labels, logits.argmax(1))
            record = {'epoch': epoch, 'train_loss': loss_sum / seen, **score, 'elapsed_seconds': time.perf_counter() - started}
            record.pop('confusion')
            record.pop('per_class_f1')
            log.write(json.dumps(record) + '\n')
            log.flush()
            print(json.dumps(record), flush=True)
            key = (score['macro_f1_fixed11'], score['accuracy'])
            if score['macro_f1_fixed11'] > best[0]:
                best = key
                best_epoch = epoch
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'val_metrics': score}, checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved['state_dict'])
    return {'best_epoch': best_epoch, 'best_macro_f1': best[0], 'best_accuracy': best[1], 'elapsed_seconds': time.perf_counter() - started}
