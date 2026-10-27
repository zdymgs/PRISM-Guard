from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import copy
import json
import random
import time
from collections import Counter
from typing import Any, Sequence
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
try:
    from aligned_multimodal.aligned_model import AlignedMultimodalModel
except ImportError:
    from aligned_multimodal.aligned_model import AlignedMultimodalModel

class _LabelOverride(Dataset):
    """Attach caller-supplied labels without inspecting dataset metadata."""

    def __init__(self, dataset: Dataset, labels: Sequence[int]) -> None:
        if len(dataset) != len(labels):
            raise ValueError('source_labels must align with train_dataset')
        self.dataset = dataset
        self.labels = tuple((int(x) for x in labels))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        if not isinstance(item, dict) or 'skeleton' not in item:
            raise TypeError('skeleton dataset items must be mappings containing skeleton')
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
    shuffle = False
    if train:
        counts = Counter((int(x) for x in labels))
        if not counts:
            raise ValueError('cannot train on an empty source dataset')
        weights = torch.tensor([1.0 / counts[int(x)] for x in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(seed))
    return DataLoader(dataset, batch_size=32, shuffle=shuffle, sampler=sampler, num_workers=0, pin_memory=False, drop_last=train)

def _make_datasets(train_dataset: Dataset | None, prediction_dataset: Dataset | None, source_labels: Sequence[int], *, cache_dir: str | None, sample_ids: Sequence[str] | None, train_indices: Sequence[int] | None, predict_indices: Sequence[int] | None) -> tuple[Dataset, Dataset]:
    if train_dataset is None or prediction_dataset is None:
        if cache_dir is None or sample_ids is None or train_indices is None or (predict_indices is None):
            raise ValueError('provide both datasets or cache_dir, sample_ids, train_indices and predict_indices')
        try:
            from aligned_multimodal.skeleton_data import SkeletonDataset
        except ImportError:
            from aligned_multimodal.skeleton_data import SkeletonDataset
        if train_dataset is None:
            train_dataset = SkeletonDataset(cache_dir, sample_ids, train_indices, labels=source_labels, augment=True)
        if prediction_dataset is None:
            prediction_dataset = SkeletonDataset(cache_dir, sample_ids, predict_indices, labels=None, augment=False)
    if len(train_dataset) != len(source_labels):
        raise ValueError('source_labels must align with train_dataset')
    train_dataset = _LabelOverride(train_dataset, source_labels)
    return (train_dataset, prediction_dataset)

def train_trajectory(train_dataset: Dataset | None, prediction_dataset: Dataset | None, source_labels: Sequence[int], *, epochs: int=15, seed: int=20260720, device: str | torch.device='cuda', deadline: float | None=None, collect_each_epoch: bool=True, cache_dir: str | None=None, sample_ids: Sequence[str] | None=None, train_indices: Sequence[int] | None=None, predict_indices: Sequence[int] | None=None) -> tuple[np.ndarray, dict[str, torch.Tensor], dict[str, Any]]:
    """Fit P0 skeleton-only model and return prediction trajectory and CPU weights."""
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or (not 1 <= int(epochs) <= 15):
        raise ValueError('epochs must be an integer in [1, 15]')
    _seed(int(seed))
    train_dataset, prediction_dataset = _make_datasets(train_dataset, prediction_dataset, source_labels, cache_dir=cache_dir, sample_ids=sample_ids, train_indices=train_indices, predict_indices=predict_indices)
    requested = torch.device(device)
    if requested.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested for skeleton skeleton training but is unavailable')
    actual = requested
    model = AlignedMultimodalModel(['skeleton'], num_classes=40, dropout=0.3).to(actual)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = actual.type == 'cuda'
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    diagnostics: dict[str, Any] = {'losses': [], 'successful_amp_steps': [], 'skipped_steps': [], 'learning_rates': [], 'epochs_completed': 0, 'device': str(actual), 'requested_device': str(requested), 'batch_size': 32, 'drop_last': True, 'num_workers': 0}
    trajectory: list[np.ndarray] = []
    train_loader = _loader(train_dataset, source_labels, seed=int(seed), train=True)
    started = time.monotonic()
    print(json.dumps({'event': 'skeleton_fit_start', 'source_rows': len(train_dataset), 'prediction_rows': len(prediction_dataset), 'epochs': int(epochs), 'device': str(actual)}), flush=True)
    for epoch in range(int(epochs)):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('skeleton skeleton training deadline exceeded')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        good_steps = skipped = 0
        for batch in train_loader:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('skeleton skeleton training deadline exceeded')
            skeleton = batch['skeleton'].to(actual, non_blocking=False)
            labels = batch['label'].to(actual, dtype=torch.long, non_blocking=False)
            with torch.autocast(device_type=actual.type, enabled=use_amp):
                loss = criterion(model({'skeleton': skeleton}), labels)
            if not bool(torch.isfinite(loss).item()):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not bool(torch.isfinite(grad_norm).item()):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            good_steps += 1
            losses.append(float(loss.detach().cpu()))
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('skeleton skeleton training deadline exceeded')
        total_steps = good_steps + skipped
        if not good_steps or (total_steps and good_steps / total_steps < 0.9):
            raise RuntimeError(f'skeleton epoch {epoch + 1} had insufficient successful steps: {good_steps}/{total_steps}')
        scheduler.step()
        diagnostics['epochs_completed'] += 1
        diagnostics['losses'].append(float(np.mean(losses)))
        diagnostics['successful_amp_steps'].append(good_steps)
        diagnostics['skipped_steps'].append(skipped)
        diagnostics['learning_rates'].append(float(optimizer.param_groups[0]['lr']))
        epoch_logits = _predict(model, prediction_dataset, actual, deadline=deadline)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('skeleton final prediction exceeded deadline')
        if not np.isfinite(epoch_logits).all():
            raise FloatingPointError(f'non-finite prediction logits in epoch {epoch + 1}')
        trajectory.append(epoch_logits)
        print(json.dumps({'event': 'skeleton_epoch_complete', 'epoch': epoch + 1, 'seconds': time.monotonic() - started, 'successful_steps': good_steps}), flush=True)
    if diagnostics['epochs_completed'] != int(epochs):
        raise RuntimeError('skeleton did not complete the requested number of epochs')
    if not collect_each_epoch:
        trajectory = [trajectory[-1]]
    diagnostics['elapsed_seconds'] = time.monotonic() - started
    diagnostics['finite_training'] = all((np.isfinite(x) for x in diagnostics['losses']))
    state = {key: copy.deepcopy(value.detach().cpu()) for key, value in model.state_dict().items()}
    return (np.stack(trajectory, axis=0).astype(np.float32, copy=False), state, diagnostics)

@torch.no_grad()
def _predict(model: nn.Module, dataset: Dataset, device: torch.device, deadline: float | None=None) -> np.ndarray:
    if hasattr(dataset, 'labels') and getattr(dataset, 'labels') is not None:
        raise AssertionError('prediction dataset must have labels=None')
    loader = _loader(dataset, [], seed=0, train=False)
    model.eval()
    outputs: list[np.ndarray] = []
    devices = [device.index] if device.type == 'cuda' and device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        for batch in loader:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('skeleton skeleton training deadline exceeded')
            if 'label' in batch:
                raise AssertionError('prediction dataset yielded a label key')
            skeleton = batch['skeleton'].to(device, non_blocking=False)
            outputs.append(model({'skeleton': skeleton}).detach().float().cpu().numpy())
    if not outputs:
        return np.empty((0, 40), dtype=np.float32)
    return np.concatenate(outputs, axis=0)
