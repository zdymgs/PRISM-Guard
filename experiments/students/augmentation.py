"""Original coherent pixel augmentation and native temporal sampling."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
def augment_ir(images: np.ndarray, allow_horizontal_flip: bool=True) -> tuple[np.ndarray, np.ndarray]:
    """Apply label-preserving transforms coherently across all six clips."""
    if allow_horizontal_flip and np.random.random() < 0.5:
        images = images[..., ::-1].copy()
    height, width = images.shape[-2:]
    pad = max(2, int(round(min(height, width) * 0.055)))
    padded = np.pad(images, ((0, 0), (0, 0), (0, 0), (pad, pad), (pad, pad)), mode='reflect')
    top = int(np.random.randint(0, 2 * pad + 1))
    left = int(np.random.randint(0, 2 * pad + 1))
    images = padded[..., top:top + height, left:left + width]
    steps = images.shape[1]
    temporal_indices = np.arange(steps, dtype=np.int64)
    if np.random.random() < 0.75:
        scale = float(np.random.uniform(0.86, 1.14))
        shift = float(np.random.uniform(-0.07, 0.07) * max(steps - 1, 1))
        base = np.arange(steps, dtype=np.float32)
        positions = np.clip((base - 0.5 * (steps - 1)) * scale + 0.5 * (steps - 1) + shift, 0.0, float(steps - 1))
        temporal_indices = np.rint(positions).astype(np.int64)
        images = images[:, temporal_indices]
    gain = float(np.random.uniform(0.82, 1.18))
    bias = float(np.random.uniform(-14.0, 14.0))
    lookup = np.clip(np.arange(256, dtype=np.float32) * gain + bias, 0.0, 255.0)
    return (lookup.astype(np.uint8)[images], temporal_indices)

def augment_rgb(images: torch.Tensor) -> torch.Tensor:
    """Batch-consistent RGB brightness/contrast only; ROI geometry is untouched."""
    value = images.float().div(255.0)
    batch = len(value)
    brightness = torch.empty((batch, 1, 1, 1, 1, 1, 1), device=value.device).uniform_(0.9, 1.1)
    contrast = torch.empty((batch, 1, 1, 1, 1, 1, 1), device=value.device).uniform_(0.9, 1.1)
    mean = value.mean(dim=(-1, -2), keepdim=True)
    return ((value - mean) * contrast + mean).mul(brightness).clamp(0, 1)
