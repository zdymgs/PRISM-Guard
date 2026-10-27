from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import json
import math
import random
import time
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from aligned_multimodal.vjepa_token_transformer_oof import TokenHead, class_weights, soft_cross_entropy
FIXED_RECIPE: dict[str, Any] = {'input_dim': 768, 'token_count': 18, 'hidden_dim': 192, 'heads': 6, 'layers': 2, 'dropout': 0.2, 'view_dropout': 0.15, 'mixup_alpha': 0.2, 'learning_rate': 0.0003, 'weight_decay': 0.05, 'epochs': 35, 'batch_size': 128, 'num_classes': 40, 'grad_clip': 2.0, 'scheduler_eta_fraction': 0.05, 'scheduler_step_policy': 'every_planned_batch'}

def _expired(deadline: Any) -> bool:
    if deadline is None:
        return False
    return bool(deadline()) if callable(deadline) else time.monotonic() >= float(deadline)

def _check_deadline(deadline: Any, where: str) -> None:
    if _expired(deadline):
        raise TimeoutError(f'physical training deadline exceeded {where}')

def _seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

def _validate_tokens(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 3 or tuple(arr.shape[1:]) != (18, 768):
        raise ValueError(f'{name} must have shape [N,18,768], got {arr.shape}')
    if arr.dtype.kind not in 'fiu' or not np.isfinite(arr).all():
        raise ValueError(f'{name} must contain finite numeric values')
    result = arr.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} overflows float32')
    return result

def _validate_labels(labels: Any, n: int) -> np.ndarray:
    raw = np.asarray(labels)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError('source_labels must have an integer (non-boolean) dtype')
    if len(raw) != n or np.any((raw < 0) | (raw >= 40)):
        raise ValueError('source_labels must be integer class IDs in [0,39]')
    return raw.astype(np.int64, copy=False)

def train_member(source_tokens, source_labels, target_tokens, *, seed, deadline=None, device='cuda'):
    """Fit the fixed original physical_tokens/token head and return target logits/state."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    if seed < 0 or seed > 2 ** 32 - 1:
        raise ValueError('seed must be in [0,2**32-1]')
    src = _validate_tokens(source_tokens, 'source_tokens')
    tgt = _validate_tokens(target_tokens, 'target_tokens')
    y = _validate_labels(source_labels, len(src))
    if len(src) == 0:
        raise ValueError('source_tokens must be nonempty')
    try:
        dev = torch.device(device)
    except Exception as exc:
        raise ValueError(f'invalid device: {device}') from exc
    if dev.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError("physical requires available CUDA; pass device='cpu' explicitly for tests")
    if dev.type == 'cuda':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    _seed(seed, dev)
    _check_deadline(deadline, 'before model creation')
    model = TokenHead(input_dim=768, hidden_dim=192, heads=6, layers=2, dropout=0.2, view_dropout=0.15, num_tokens=18).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.05)
    batches = math.ceil(len(src) / 128)
    total_steps = 35 * batches
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=0.0003 * 0.05)
    loader = DataLoader(TensorDataset(torch.arange(len(src), dtype=torch.long), torch.from_numpy(y)), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0, drop_last=False)
    weights = torch.from_numpy(class_weights(y)).to(dev)
    amp_enabled = dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    successful = skipped = scheduler_steps = 0
    telemetry: list[dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(35):
        _check_deadline(deadline, f'before epoch {epoch + 1}')
        model.train()
        loss_sum = 0.0
        rows = 0
        for rows_idx, batch_y in loader:
            _check_deadline(deadline, f'during epoch {epoch + 1}')
            values = torch.from_numpy(src[rows_idx.numpy()]).to(dev)
            target = F.one_hot(batch_y.to(dev), num_classes=40).float()
            if len(values) > 1:
                lam = float(np.random.beta(0.2, 0.2))
                order = torch.randperm(len(values), device=dev)
                values = lam * values + (1.0 - lam) * values[order]
                target = lam * target + (1.0 - lam) * target[order]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=torch.float16):
                loss = soft_cross_entropy(model(values), target, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite physical training loss')
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            if not all((torch.isfinite(p).all().item() for p in model.parameters())):
                raise FloatingPointError('nonfinite physical model parameter')
            if scaler.get_scale() >= old_scale:
                successful += 1
            else:
                skipped += 1
            scheduler.step()
            scheduler_steps += 1
            loss_sum += float(loss.detach()) * len(values)
            rows += len(values)
        if epoch in (0, 9, 19, 29, 34):
            event = {'seed': seed, 'epoch': epoch + 1, 'epochs': 35, 'train_loss': loss_sum / max(rows, 1), 'optimizer_steps': successful, 'amp_skips': skipped, 'scheduler_steps': scheduler_steps, 'seconds': time.monotonic() - started}
            telemetry.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
    if successful < 0.9 * total_steps:
        raise FloatingPointError('fewer than 90% of physical optimizer updates succeeded')
    _check_deadline(deadline, 'after training')
    model.eval()
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(tgt), 256):
            _check_deadline(deadline, 'before target prediction batch')
            x = torch.from_numpy(tgt[start:start + 256]).to(dev)
            with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=torch.float16):
                outputs.append(model(x).float().cpu())
    logits = torch.cat(outputs).numpy().astype(np.float32) if outputs else np.empty((0, 40), np.float32)
    if not np.isfinite(logits).all():
        raise FloatingPointError('nonfinite physical target logits')
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if not all((torch.isfinite(v).all().item() for v in state.values())):
        raise FloatingPointError('nonfinite physical model state')
    _check_deadline(deadline, 'after target prediction')
    diagnostics = {'seed': seed, 'epochs': 35, 'planned_steps': total_steps, 'optimizer_steps': successful, 'amp_skips': skipped, 'scheduler_steps': scheduler_steps, 'final_lr': optimizer.param_groups[0]['lr'], 'telemetry': telemetry, 'token_count': 18, 'input_dim': 768, 'device': str(dev), 'recipe': dict(FIXED_RECIPE)}
    return (logits, state, diagnostics)
