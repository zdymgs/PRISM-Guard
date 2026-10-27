from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
from torch import nn
from session_kd.mmact_backbones.mobind import IMUPartEncoder
MMACT_PARTS = 5
WATCH_RIGHT_ARM_SLOT = 2
PHONE_RIGHT_POCKET_SLOT = 4
BASE_CHANNELS = ('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z')
ORIENTATION_CHANNELS = ('azimuth', 'pitch', 'roll')

@dataclass(frozen=True)
class NativeIMUSchema:
    channel_names: tuple[str, ...]
    phone_slot: int = PHONE_RIGHT_POCKET_SLOT
    watch_slot: int = WATCH_RIGHT_ARM_SLOT
    orientation_semantics: str = 'absent'

    @classmethod
    def create(cls, include_orientation: bool=False) -> 'NativeIMUSchema':
        names = BASE_CHANNELS + ORIENTATION_CHANNELS if include_orientation else BASE_CHANNELS
        semantics = 'measured phone azimuth/pitch/roll in source units' if include_orientation else 'absent'
        return cls(names, orientation_semantics=semantics)

@dataclass(frozen=True)
class ChannelNormalizer:
    mean: np.ndarray
    std: np.ndarray
    training_ids: tuple[str, ...]
    channel_names: tuple[str, ...]

def assemblemodules_part_grid(phone_acc: np.ndarray, phone_gyro: np.ndarray, watch_acc: np.ndarray, phone_acc_mask: np.ndarray, phone_gyro_mask: np.ndarray, watch_acc_mask: np.ndarray, *, phone_orientation: np.ndarray | None=None, phone_orientation_mask: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray, NativeIMUSchema]:
    """Place real sensors in a `[B,2,16,5,4,C]` body-part grid."""
    arrays = [np.asarray(value, np.float32) for value in (phone_acc, phone_gyro, watch_acc)]
    masks = [np.asarray(value, bool) for value in (phone_acc_mask, phone_gyro_mask, watch_acc_mask)]
    shape = arrays[0].shape
    if len(shape) != 5 or shape[-1] != 3 or any((value.shape != shape for value in arrays)):
        raise ValueError('native streams must share shape [B,2,16,4,3]')
    if any((value.shape != shape for value in masks)):
        raise ValueError('native masks must be per-channel [B,2,16,4,3]')
    include_orientation = phone_orientation is not None
    if include_orientation != (phone_orientation_mask is not None):
        raise ValueError('orientation values and mask must be supplied together')
    if include_orientation:
        orientation = np.asarray(phone_orientation, np.float32)
        orientation_mask = np.asarray(phone_orientation_mask, bool)
        if orientation.shape != shape or orientation_mask.shape != shape:
            raise ValueError('orientation and mask must share native stream shape')
    schema = NativeIMUSchema.create(include_orientation)
    channels = len(schema.channel_names)
    output = np.zeros((*shape[:3], MMACT_PARTS, shape[3], channels), np.float32)
    valid = np.zeros_like(output, dtype=bool)
    output[..., PHONE_RIGHT_POCKET_SLOT, :, 0:3] = arrays[0]
    output[..., PHONE_RIGHT_POCKET_SLOT, :, 3:6] = arrays[1]
    valid[..., PHONE_RIGHT_POCKET_SLOT, :, 0:3] = masks[0]
    valid[..., PHONE_RIGHT_POCKET_SLOT, :, 3:6] = masks[1]
    output[..., WATCH_RIGHT_ARM_SLOT, :, 0:3] = arrays[2]
    valid[..., WATCH_RIGHT_ARM_SLOT, :, 0:3] = masks[2]
    if include_orientation:
        output[..., PHONE_RIGHT_POCKET_SLOT, :, 6:9] = orientation
        valid[..., PHONE_RIGHT_POCKET_SLOT, :, 6:9] = orientation_mask
    output[~valid] = 0.0
    return (output, valid, schema)

def fit_channel_normalizer(values: np.ndarray, mask: np.ndarray, sample_ids: Sequence[str], fit_indices: Sequence[int], schema: NativeIMUSchema) -> ChannelNormalizer:
    x = np.asarray(values, np.float64)
    present = np.asarray(mask, bool)
    ids = np.asarray(sample_ids, str)
    fit = np.asarray(fit_indices, np.int64)
    if x.shape != present.shape or x.ndim != 6 or x.shape[-1] != len(schema.channel_names):
        raise ValueError('normalizer inputs do not match native schema')
    if len(ids) != len(x) or not len(fit) or np.any(fit < 0) or np.any(fit >= len(x)):
        raise ValueError('explicit valid source fit indices are required')
    axes = tuple(range(5))
    weight = present[fit].astype(np.float64)
    count = weight.sum(axes)
    if np.any(count == 0):
        raise ValueError('a native channel has no valid source-fit observation')
    mean = (x[fit] * weight).sum(axes) / count
    variance = ((x[fit] - mean) ** 2 * weight).sum(axes) / count
    return ChannelNormalizer(mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-06)).astype(np.float32), tuple(ids[fit]), schema.channel_names)

def apply_channel_normalizer(values: np.ndarray, mask: np.ndarray, normalizer: ChannelNormalizer) -> np.ndarray:
    x = np.asarray(values, np.float32)
    present = np.asarray(mask, bool)
    if x.shape != present.shape or x.shape[-1] != len(normalizer.channel_names):
        raise ValueError('normalizer application shape mismatch')
    output = (x - normalizer.mean) / normalizer.std
    return np.where(present, output, 0.0).astype(np.float32)

def _masked_statistics(values: torch.Tensor, mask: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
    weight = mask.to(values.dtype)
    count = weight.sum(dim=dimensions).clamp_min(1)
    mean = (values * weight).sum(dim=dimensions) / count
    expanded = mean
    for dimension in sorted(dimensions):
        expanded = expanded.unsqueeze(dimension)
    variance = ((values - expanded).square() * weight).sum(dim=dimensions) / count
    minimum = values.masked_fill(~mask, 10000.0).amin(dim=dimensions)
    maximum = values.masked_fill(~mask, -10000.0).amax(dim=dimensions)
    any_valid = mask.any(dim=dimensions)
    minimum = torch.where(any_valid, minimum, 0)
    maximum = torch.where(any_valid, maximum, 0)
    fraction = weight.mean(dim=dimensions)
    return torch.cat((mean, variance.clamp_min(1e-06).sqrt(), minimum, maximum, fraction), dim=-1)

class MMACTIMUEncoder(IMUPartEncoder):
    """student hierarchy with native MMACT channel stems and statistic widths."""

    def __init__(self, schema: NativeIMUSchema, width: int=96, dropout: float=0.12) -> None:
        super().__init__(width=width, dropout=dropout, instance_normalization=False, event_feature_width=0)
        self.schema = schema
        self.native_channels = len(schema.channel_names)
        self.native_stem = nn.Sequential(nn.LayerNorm(self.native_channels * 2), nn.Linear(self.native_channels * 2, width), nn.GELU())
        stat_width = self.native_channels * 5
        self.bin_statistics_projection = nn.Sequential(nn.LayerNorm(stat_width), nn.Linear(stat_width, width), nn.GELU())
        self.global_projection = nn.Sequential(nn.LayerNorm(stat_width), nn.Linear(stat_width, width), nn.GELU())
        self.statistics_input_width = stat_width * 4 * 2 * MMACT_PARTS
        self.statistics_projection = nn.Sequential(nn.LayerNorm(self.statistics_input_width), nn.Linear(self.statistics_input_width, width * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 4, width * 2), nn.GELU())

    def _stats(self, sequences: torch.Tensor, channel_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bins = _masked_statistics(sequences, channel_mask, (4,))
        global_stats = _masked_statistics(sequences, channel_mask, (2, 4))
        return (bins, global_stats)

    def statistics_embedding(self, sequences: torch.Tensor, channel_mask: torch.Tensor, bin_statistics: torch.Tensor | None=None, bin_mask: torch.Tensor | None=None, global_statistics: torch.Tensor | None=None, global_mask: torch.Tensor | None=None) -> torch.Tensor:
        full = _masked_statistics(sequences, channel_mask, (2, 4)).flatten(1)
        midpoint = max(1, sequences.shape[2] // 2)
        early = _masked_statistics(sequences[:, :, :midpoint], channel_mask[:, :, :midpoint], (2, 4)).flatten(1)
        if sequences.shape[2] == 1:
            late = early
            delta = torch.zeros_like(early)
        else:
            late = _masked_statistics(sequences[:, :, midpoint:], channel_mask[:, :, midpoint:], (2, 4)).flatten(1)
            delta = late - early
        flat = torch.cat((full, early, late, delta), 1)
        if flat.shape[1] != self.statistics_input_width:
            raise RuntimeError('native statistics width mismatch')
        return self.statistics_projection(flat)

    def forward(self, sequences: torch.Tensor, channel_mask: torch.Tensor, bin_statistics: torch.Tensor | None=None, bin_mask: torch.Tensor | None=None, global_statistics: torch.Tensor | None=None, global_mask: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if sequences.ndim != 6 or sequences.shape[3] != MMACT_PARTS or sequences.shape[-1] != self.native_channels:
            raise ValueError('sequences must be [B,2,16,5,4,C] for the configured schema')
        if channel_mask.shape != sequences.shape:
            raise ValueError('channel_mask must be per native channel')
        values = torch.where(channel_mask, sequences, torch.zeros_like(sequences))
        point_mask = channel_mask.any(-1)
        encoded = self.native_stem(torch.cat((values, channel_mask.to(values.dtype)), -1))
        b, w, t, d, p, _ = encoded.shape
        encoded = encoded.permute(0, 1, 3, 2, 4, 5).reshape(b * w * d, t * p, self.width)
        flat_mask = point_mask.permute(0, 1, 3, 2, 4).reshape(b * w * d, t * p)
        position = torch.linspace(0, 1, t * p, device=values.device, dtype=values.dtype).view(1, t * p, 1)
        encoded = (encoded + self.point_time_encoder(position)) * flat_mask.unsqueeze(-1)
        for block in self.point_blocks:
            encoded = block(encoded) * flat_mask.unsqueeze(-1)
        encoded = encoded.reshape(b, w, d, t, p, self.width).permute(0, 1, 3, 2, 4, 5)
        weight = point_mask.to(values.dtype).unsqueeze(-1)
        mean = (encoded * weight).sum(4) / weight.sum(4).clamp_min(1)
        maximum = encoded.masked_fill(~point_mask.unsqueeze(-1), -10000.0).amax(4)
        maximum = torch.where(point_mask.any(4).unsqueeze(-1), maximum, 0)
        bin_stats, global_stats = self._stats(values, channel_mask)
        tokens = self.bin_projection(torch.cat((mean, maximum), -1)) + self.bin_statistics_projection(bin_stats)
        device_mask = point_mask.any(4)
        tokens = tokens + self.global_projection(global_stats)[:, :, None] + self.device_embedding[None, None, None]
        temporal = tokens.permute(0, 1, 3, 2, 4).reshape(b * w * d, t, self.width)
        temporal_mask = device_mask.permute(0, 1, 3, 2).reshape(b * w * d, t)
        safe = temporal_mask.clone()
        safe[~safe.any(1), 0] = True
        temporal = self.temporal_encoder(temporal, src_key_padding_mask=~safe) * temporal_mask.unsqueeze(-1)
        tokens = temporal.reshape(b, w, d, t, self.width).permute(0, 1, 3, 2, 4)
        device_tokens = tokens.reshape(b * w * t, d, self.width)
        flat_device_mask = device_mask.reshape(b * w * t, d)
        safe = flat_device_mask.clone()
        safe[~safe.any(1), 0] = True
        device_tokens = self.device_encoder(device_tokens, src_key_padding_mask=~safe) * flat_device_mask.unsqueeze(-1)
        return (device_tokens.reshape(b, w, t, d, self.width), device_mask)
