from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
import os
import pathlib
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE = WORK_ROOT / 'aligned_multimodal'
PROJECT = WORK_ROOT
PIXELS = HERE / 'runs/visual_pixel_cache_t16_r160_v12'
EXTERNAL = EXTERNAL_ROOT / 'LaViLa'
CHECKPOINT = PRETRAINED_ROOT / 'LaViLa/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth'
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while (block := handle.read(4 * 1024 * 1024)):
            digest.update(block)
    return digest.hexdigest()

def load_checkpoint(path: Path):
    original = pathlib.PosixPath
    if os.name == 'nt':
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    finally:
        pathlib.PosixPath = original
    state = checkpoint.get('state_dict', checkpoint)
    cleaned = {}
    for key, value in state.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        cleaned[key] = value
    return (checkpoint, cleaned)

def resize_temporal_embedding(value: torch.Tensor, frames: int) -> torch.Tensor:
    if value.shape[1] == frames:
        return value
    return F.interpolate(value.permute(0, 2, 1), size=frames, mode='linear', align_corners=False).permute(0, 2, 1)

def load_visual(path: Path, device: torch.device, frames: int):
    sys.path.insert(0, str(EXTERNAL))
    from lavila.models.openai_model import QuickGELU
    from lavila.models.timesformer import SpaceTimeTransformer
    checkpoint, state = load_checkpoint(path)
    temporal_key = 'visual.temporal_embed'
    if temporal_key not in state:
        matching = [key for key in state if key.endswith('visual.temporal_embed')]
        raise RuntimeError(f'checkpoint lacks visual temporal embedding: {matching}')
    checkpoint_frames = int(state[temporal_key].shape[1])
    visual = SpaceTimeTransformer(num_frames=frames, time_init='zeros', attention_style='frozen-in-time', ln_pre=True, act_layer=QuickGELU)
    visual.head = nn.Identity()
    visual.pre_logits = nn.Identity()
    visual_state = {key[len('visual.'):]: value for key, value in state.items() if key.startswith('visual.')}
    visual_state['temporal_embed'] = resize_temporal_embedding(visual_state['temporal_embed'], frames)
    missing, unexpected = visual.load_state_dict(visual_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'LaViLa visual mismatch missing={missing[:8]} unexpected={unexpected[:8]}')
    projection = None
    if 'image_projection' in state:
        projection = nn.Linear(768, int(state['image_projection'].shape[1]), bias=False)
        projection.weight.data.copy_(state['image_projection'].T)
    visual.to(device).eval()
    if projection is not None:
        projection.to(device).eval()
    for parameter in visual.parameters():
        parameter.requires_grad_(False)
    if projection is not None:
        for parameter in projection.parameters():
            parameter.requires_grad_(False)
    return (visual, projection, checkpoint, checkpoint_frames)

def prepare_video(values: np.ndarray, device: torch.device, frames: int) -> torch.Tensor:
    values = values.reshape(len(values), 32, values.shape[-2], values.shape[-1])
    indices = np.rint(np.linspace(0, 31, frames)).astype(np.int64)
    values = values[:, indices]
    video = torch.from_numpy(values).to(device=device, dtype=torch.float32)
    video = video[:, None].repeat(1, 3, 1, 1, 1) / 255.0
    batch, channels, frame_count, height, width = video.shape
    video = F.interpolate(video.permute(0, 2, 1, 3, 4).reshape(batch * frame_count, channels, height, width), size=(224, 224), mode='bilinear', align_corners=False).reshape(batch, frame_count, channels, 224, 224).permute(0, 2, 1, 3, 4)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1, 1)
    return (video - mean) / std
