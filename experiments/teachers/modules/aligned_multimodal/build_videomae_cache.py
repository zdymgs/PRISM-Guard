from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import csv
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
PROJECT_DIR = WORK_ROOT / 'aligned_multimodal'
DEFAULT_ROI = PROJECT_DIR / 'runs' / 'dir_multiscale_roi_full'
MODEL_NAME = 'MCG-NJU/videomae-base-finetuned-kinetics'
VIEW_NAMES = ('scene', 'person', 'workspace')

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is not None and 'p46_ir_readable' in reader.fieldnames:
            master = list(reader)
            if len(master) != 405:
                raise RuntimeError(f'Official Test count changed: {len(master)}')
            rows = []
            for source in master:
                if source['p46_ir_readable'] != '1':
                    continue
                row = dict(source)
                row.update(sample_id=source['official_sample_id'], source_id=source['sample_id'], user_id='anonymous', class_id='-1', ir_dir=source['ir_path'], depth_dir=source['depth_color_path'])
                rows.append(row)
            if len(rows) != 401:
                raise RuntimeError(f'Expected 401 readable Test IR trials, got {len(rows)}')
        else:
            rows = [row for row in reader if row['detail_selected'] == '1']
            if len(rows) != 1384:
                raise RuntimeError(f'Frozen Detail21 count changed: {len(rows)}')
    rows.sort(key=lambda row: row['sample_id'])
    return rows

def safe_relative(source_id: str) -> Path:
    parts = source_id.split('/')
    if len(parts) != 3 or any((value in {'', '.', '..'} for value in parts)):
        raise ValueError(f'unsafe source ID: {source_id}')
    return Path(*parts)

def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.building')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)

def restorecore_attention_biases(model: VideoMAEForVideoClassification, snapshot: Path) -> dict[str, Any]:
    """Map the original split q/v bias tensors into Transformers 5.x modules."""
    safetensors_path = snapshot / 'model.safetensors'
    pytorch_path = snapshot / 'pytorch_model.bin'
    if safetensors_path.is_file():
        state = load_file(str(safetensors_path), device='cpu')
        checkpoint_format = 'safetensors'
    elif pytorch_path.is_file():
        state = torch.load(pytorch_path, map_location='cpu', weights_only=True)
        checkpoint_format = 'pytorch_bin'
    else:
        raise FileNotFoundError(f'no supported VideoMAE checkpoint in {snapshot}')
    restored = 0
    maximum_difference = 0.0
    with torch.no_grad():
        for layer_index, layer in enumerate(model.videomae.encoder.layer):
            attention = layer.attention.attention
            q_bias = state[f'videomae.encoder.layer.{layer_index}.attention.attention.q_bias']
            v_bias = state[f'videomae.encoder.layer.{layer_index}.attention.attention.v_bias']
            attention.query.bias.copy_(q_bias)
            attention.key.bias.zero_()
            attention.value.bias.copy_(v_bias)
            maximum_difference = max(maximum_difference, float((attention.query.bias.cpu() - q_bias).abs().max()), float((attention.value.bias.cpu() - v_bias).abs().max()), float(attention.key.bias.detach().cpu().abs().max()))
            restored += 2
    expected_restored = 2 * len(model.videomae.encoder.layer)
    if restored != expected_restored or maximum_difference != 0.0:
        raise RuntimeError(f'VideoMAE legacy attention bias restoration failed: restored={restored}, max_diff={maximum_difference}')
    return {'reason': 'The checkpoint was authored with Transformers 4.21 split q_bias/v_bias; Transformers 5.x exposes query/key/value bias parameters.', 'restored_tensors': restored, 'key_bias_policy': 'exact zero, matching the original qkv implementation', 'maximum_verification_difference': maximum_difference, 'checkpoint_format': checkpoint_format}

def uniform_indices(frame_count: int, count: int=16) -> np.ndarray:
    if frame_count < 1:
        raise ValueError('video has no aligned frames')
    return np.rint(np.linspace(0, frame_count - 1, count)).astype(np.int64)

def square_crop(image: np.ndarray, box: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    if box.shape != (4,) or not np.isfinite(box).all():
        return image
    x1, y1, x2, y2 = [float(value) for value in box]
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return image
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1) * float(scale)
    left = max(0, int(round(center_x - 0.5 * side)))
    right = min(width, int(round(center_x + 0.5 * side)))
    top = max(0, int(round(center_y - 0.5 * side)))
    bottom = min(height, int(round(center_y + 0.5 * side)))
    if right <= left + 2 or bottom <= top + 2:
        return image
    return image[top:bottom, left:right]

@torch.inference_mode()
def encode(model: VideoMAEForVideoClassification, processor: VideoMAEImageProcessor, videos: list[list[np.ndarray]], device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    pixel_values = processor(videos, return_tensors='pt').pixel_values.to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        output = model.videomae(pixel_values).last_hidden_state
        feature = model.fc_norm(output.mean(dim=1)) if model.fc_norm is not None else output[:, 0]
        logits = model.classifier(feature)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else 0.0
    return (feature.float().cpu().numpy(), logits.float().cpu().numpy(), peak)
