from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_local_resnet18_encoder(weights_path: Path, device: torch.device) -> nn.Module:
    """Strictly load the existing torchvision weights without network access."""
    if not weights_path.is_file():
        raise FileNotFoundError(f'Pretrained visual weights are absent; downloads are forbidden: {weights_path}')
    model = resnet18(weights=None)
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    encoder = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder

@torch.inference_mode()
def encode_rgb_frames(encoder: nn.Module, rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    frames = torch.from_numpy(np.ascontiguousarray(rgb)).to(device=device, dtype=torch.float32)
    frames = frames.permute(0, 3, 1, 2).div_(255.0)
    frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)
    mean = frames.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = frames.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return encoder((frames - mean) / std).flatten(1)
