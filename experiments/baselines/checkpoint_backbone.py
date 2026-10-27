"""Recompute MC3 blocks during backward; preserve the exact microbatch BN math."""
from types import MethodType
import torch
from torch.utils.checkpoint import checkpoint
from prism.student.native.constants import KINETICS_MEAN, KINETICS_STD


def enable_checkpointing(visual):
    def encode(self, images):
        batch, windows, steps, views = images.shape[:4]
        if images.ndim == 6:
            assert tuple(images.shape[1:4]) == (2, self.frames, 3)
            height, width = images.shape[-2:]
            clips = images.permute(0, 1, 3, 2, 4, 5).reshape(batch*windows*views, steps, 1, height, width)
            unit = clips.float().div_(255.0)
            value = unit.repeat(1, 1, 3, 1, 1).permute(0, 2, 1, 3, 4)
        else:
            assert images.ndim == 7 and tuple(images.shape[1:5]) == (2, self.frames, self.views, 3)
            height, width = images.shape[-2:]
            clips = images.permute(0, 1, 3, 4, 2, 5, 6).reshape(batch*windows*views, 3, steps, height, width)
            value = clips.float().div(255.0) if images.dtype == torch.uint8 else clips.float()
        mean = value.new_tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1)
        std = value.new_tensor(KINETICS_STD).view(1, 3, 1, 1, 1)
        value = (value-mean)/std
        modules = [self.stem, *self.layer1, *self.layer2, *self.layer3, *self.layer4]
        for module in modules:
            value = checkpoint(module, value, use_reentrant=False) if torch.is_grad_enabled() else module(value)
        return value.reshape(batch, windows, views, *value.shape[1:])
    visual._encode_backbone_map = MethodType(encode, visual)
