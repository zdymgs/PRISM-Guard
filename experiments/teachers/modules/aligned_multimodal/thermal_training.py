from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import copy
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
try:
    from thermal_baseline.thermal_tsm_model import ThermalResNetTSM
except (ImportError, ValueError):
    from thermal_baseline.thermal_tsm_model import ThermalResNetTSM
_WEIGHTS = str(PRETRAINED_ROOT / 'torch/hub/checkpoints/resnet18-f37072fd.pth')
_WEIGHTS_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'

class _LabelOverride(Dataset):

    def __init__(self, dataset: Dataset, labels: Sequence[int]) -> None:
        if len(dataset) != len(labels):
            raise ValueError('source_labels must align with train_dataset')
        self.dataset, self.labels = (dataset, tuple((int(x) for x in labels)))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        if not isinstance(item, dict) or 'clip' not in item:
            raise TypeError('thermal dataset items must be mappings containing clip')
        result = dict(item)
        result['label'] = self.labels[index]
        return result

def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def _loader(dataset: Dataset, labels: Sequence[int], *, seed: int, train: bool) -> DataLoader:
    sampler = None
    if train:
        counts = Counter((int(x) for x in labels))
        weights = torch.tensor([1.0 / counts[int(x)] for x in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(seed))
    return DataLoader(dataset, batch_size=32, shuffle=False, sampler=sampler, num_workers=0, pin_memory=False, drop_last=train)

def _imagenet_backbone_state(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'local ImageNet weights not found: {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != _WEIGHTS_SHA256:
        raise ValueError(f'unexpected ResNet18 weights SHA256: {digest}')
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        state = torch.load(path, map_location='cpu')
    if not isinstance(state, dict):
        raise TypeError('ImageNet checkpoint is not a state dictionary')
    return {k: v for k, v in state.items() if k.startswith(('conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.'))}

def _make_model(device: torch.device, pretrained_path: str | Path) -> ThermalResNetTSM:
    model = ThermalResNetTSM(num_classes=40, dropout=0.3, imagenet_pretrained=False)
    source = _imagenet_backbone_state(pretrained_path)
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        destination = 'stem.0.' + key[6:] if key.startswith('conv1.') else 'stem.1.' + key[4:] if key.startswith('bn1.') else key
        if destination in target:
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f'non-finite tensor in ImageNet checkpoint: {key}')
            mapped[destination] = value
    expected = {k for k in target if k.startswith(('stem.', 'layer1.', 'layer2.', 'layer3.', 'layer4.')) and (not k.endswith('num_batches_tracked'))}
    if set(mapped) != expected:
        missing = sorted(expected - set(mapped))
        extra = sorted(set(mapped) - expected)
        raise RuntimeError(f'incomplete ResNet stem/layer checkpoint mapping; missing={missing}, extra={extra}')
    target.update(mapped)
    model.load_state_dict(target, strict=True)
    return model.to(device)

def train_trajectory(train_dataset: Dataset, prediction_dataset: Dataset, source_labels: Sequence[int], *, epochs: int=15, seed: int=20260720, device: str | torch.device='cuda', deadline: float | None=None, collect_each_epoch: bool=True, pretrained_path: str | Path=_WEIGHTS) -> tuple[np.ndarray, dict[str, torch.Tensor], dict[str, Any]]:
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or (not 1 <= int(epochs) <= 15):
        raise ValueError('epochs must be an integer in [1, 15]')
    labels = tuple(source_labels)
    if len(train_dataset) != len(labels):
        raise ValueError('source_labels must align with train_dataset')
    if any((isinstance(x, bool) or not isinstance(x, (int, np.integer)) or (not 0 <= int(x) < 40) for x in labels)):
        raise ValueError('source_labels must be integer class IDs in [0, 39]')
    prediction_labels = getattr(prediction_dataset, 'labels', None)
    if prediction_labels is not None:
        raise AssertionError('prediction dataset must have labels=None')
    _seed(int(seed))
    requested = torch.device(device)
    if requested.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested for thermal thermal training but is unavailable')
    actual = requested
    train_dataset = _LabelOverride(train_dataset, labels)
    model = _make_model(actual, pretrained_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = actual.type == 'cuda'
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    diagnostics: dict[str, Any] = {'losses': [], 'successful_amp_steps': [], 'skipped_steps': [], 'learning_rates': [], 'epochs_completed': 0, 'device': str(actual), 'requested_device': str(requested), 'batch_size': 32, 'drop_last': True, 'num_workers': 0, 'sampler': 'inverse_class_weighted_replacement'}
    trajectory: list[np.ndarray] = []
    loader = _loader(train_dataset, labels, seed=int(seed), train=True)
    started = time.monotonic()
    print(json.dumps({'event': 'thermal_fit_start', 'source_rows': len(train_dataset), 'prediction_rows': len(prediction_dataset), 'epochs': int(epochs), 'device': str(actual)}), flush=True)
    for epoch in range(int(epochs)):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('thermal thermal training deadline exceeded')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        good = skipped = 0
        for batch in loader:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('thermal thermal training deadline exceeded')
            clips = batch['clip'].to(actual, non_blocking=False)
            targets = batch['label'].to(actual, dtype=torch.long, non_blocking=False)
            with torch.autocast(device_type=actual.type, enabled=use_amp):
                loss = criterion(model(clips), targets)
            if not bool(torch.isfinite(loss).item()):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not bool(torch.isfinite(norm).item()):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            good += 1
            losses.append(float(loss.detach().cpu()))
        total = good + skipped
        if not good or (total and good / total < 0.9):
            raise RuntimeError(f'thermal epoch {epoch + 1} had insufficient successful steps: {good}/{total}')
        scheduler.step()
        diagnostics['epochs_completed'] += 1
        diagnostics['losses'].append(float(np.mean(losses)))
        diagnostics['successful_amp_steps'].append(good)
        diagnostics['skipped_steps'].append(skipped)
        diagnostics['learning_rates'].append(float(optimizer.param_groups[0]['lr']))
        logits = _predict(model, prediction_dataset, actual, use_amp, deadline)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('thermal thermal prediction exceeded deadline')
        if not np.isfinite(logits).all():
            raise FloatingPointError(f'non-finite prediction logits in epoch {epoch + 1}')
        trajectory.append(logits)
        print(json.dumps({'event': 'thermal_epoch_complete', 'epoch': epoch + 1, 'seconds': time.monotonic() - started, 'successful_steps': good}), flush=True)
    if not collect_each_epoch:
        trajectory = [trajectory[-1]]
    diagnostics['elapsed_seconds'] = time.monotonic() - started
    diagnostics['finite_training'] = all((np.isfinite(x) for x in diagnostics['losses']))
    state = {k: copy.deepcopy(v.detach().cpu()) for k, v in model.state_dict().items()}
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('thermal thermal state capture exceeded deadline')
    return (np.stack(trajectory, axis=0).astype(np.float32, copy=False), state, diagnostics)

@torch.no_grad()
def _predict(model: nn.Module, dataset: Dataset, device: torch.device, use_amp: bool, deadline: float | None) -> np.ndarray:
    loader = _loader(dataset, [], seed=0, train=False)
    model.eval()
    outputs: list[np.ndarray] = []
    devices = [device.index] if device.type == 'cuda' and device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        for batch in loader:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('thermal thermal prediction deadline exceeded')
            if 'label' in batch:
                raise AssertionError('prediction dataset yielded a label key')
            clips = batch['clip'].to(device, non_blocking=False)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs.append(model(clips).detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 40), dtype=np.float32)
