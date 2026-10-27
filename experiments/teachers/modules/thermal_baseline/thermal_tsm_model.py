from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

def temporal_shift(features: torch.Tensor, batch_size: int, time_steps: int, fold_divisor: int=8) -> torch.Tensor:
    """在相邻帧间移动部分通道，不增加可训练参数。"""
    _, channels, height, width = features.shape
    sequence = features.reshape(batch_size, time_steps, channels, height, width)
    fold = channels // fold_divisor
    if fold == 0 or time_steps == 1:
        return features
    shifted = torch.zeros_like(sequence)
    shifted[:, :-1, :fold] = sequence[:, 1:, :fold]
    shifted[:, 1:, fold:2 * fold] = sequence[:, :-1, fold:2 * fold]
    shifted[:, :, 2 * fold:] = sequence[:, :, 2 * fold:]
    return shifted.reshape(batch_size * time_steps, channels, height, width)

class TemporalResidualBlock(nn.Module):

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels, bias=False), nn.BatchNorm1d(channels), nn.GELU(), nn.Conv1d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm1d(channels), nn.Dropout(dropout))
        self.activation = nn.GELU()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.activation(sequence + self.block(sequence))

class TemporalAttentionPooling(nn.Module):

    def __init__(self, feature_dim: int, attention_dim: int=128) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(feature_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1))

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        pooled = torch.sum(sequence * attention.unsqueeze(-1), dim=1)
        return (pooled, attention)

class ThermalResNetTSM(nn.Module):
    """ResNet18 + stage-level TSM + multi-scale TCN + attention pooling。"""

    def __init__(self, num_classes: int=40, dropout: float=0.3, tsm_fold_divisor: int=8, imagenet_pretrained: bool=False) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.feature_dim = 512
        self.tsm_fold_divisor = tsm_fold_divisor
        self.temporal = nn.Sequential(TemporalResidualBlock(self.feature_dim, dilation=1, dropout=dropout * 0.5), TemporalResidualBlock(self.feature_dim, dilation=2, dropout=dropout * 0.5))
        self.attention_pool = TemporalAttentionPooling(self.feature_dim)
        self.classifier = nn.Sequential(nn.LayerNorm(self.feature_dim * 2), nn.Dropout(dropout), nn.Linear(self.feature_dim * 2, num_classes))

    def _shift(self, features: torch.Tensor, batch_size: int, time_steps: int) -> torch.Tensor:
        return temporal_shift(features, batch_size, time_steps, self.tsm_fold_divisor)

    def encode_pooled(self, clip: torch.Tensor, return_attention: bool=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps, channels, height, width = clip.shape
        frames = clip.reshape(batch_size * time_steps, channels, height, width)
        features = self.stem(frames)
        features = self.layer1(self._shift(features, batch_size, time_steps))
        features = self.layer2(self._shift(features, batch_size, time_steps))
        features = self.layer3(self._shift(features, batch_size, time_steps))
        features = self.layer4(self._shift(features, batch_size, time_steps))
        features = features.mean(dim=(-2, -1))
        sequence = features.reshape(batch_size, time_steps, self.feature_dim)
        sequence = self.temporal(sequence.transpose(1, 2)).transpose(1, 2)
        attended, attention = self.attention_pool(sequence)
        temporal_max = sequence.amax(dim=1)
        pooled = torch.cat([attended, temporal_max], dim=1)
        if return_attention:
            return (pooled, attention)
        return pooled

    def forward(self, clip: torch.Tensor, return_attention: bool=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_pooled(clip, return_attention=return_attention)
        if return_attention:
            pooled, attention = encoded
            return (self.classifier(pooled), attention)
        return self.classifier(encoded)
