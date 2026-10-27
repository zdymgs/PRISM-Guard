from __future__ import annotations

import math

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


REGION_NAMES = (
    "full_body",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "hand_workspace",
    "global_fallback",
)
MODALITY_NAMES = ("depth", "ir")
PYRAMID_FEATURE_DIM = 128 + 256 + 512


class SharedResNet18Pyramid(nn.Module):
    """One ImageNet ResNet18 shared exactly by Depth and IR pixel crops.

    IR crops are repeated to RGB before this module.  Keeping one literal trunk
    makes the deployment weight count independent of the number of modalities
    and ROI types.  Layer2/3/4 pooled features preserve both fine local detail
    and high-level semantics.
    """

    def __init__(self, imagenet_pretrained: bool = True) -> None:
        super().__init__()
        backbone = resnet18(
            weights=(ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None)
        )
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.output_dim = PYRAMID_FEATURE_DIM

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        layer2, layer3, layer4 = self.forward_feature_maps(crops)
        pooled = [
            nn.functional.adaptive_avg_pool2d(value, 1).flatten(1)
            for value in (layer2, layer3, layer4)
        ]
        return torch.cat(pooled, dim=1)

    def forward_feature_maps(
        self, crops: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return layer2/3/4 maps before global pooling.

        roi_features intentionally used only ``forward``.  C reuses the exact same
        shared D/IR backbone but keeps a small spatial grid from layer2 so that
        hand/object position is not destroyed by global average pooling.
        """
        source = self.stem(crops)
        source = self.layer1(source)
        layer2 = self.layer2(source)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer2, layer3, layer4


def imagenet_normalize(crops: torch.Tensor) -> torch.Tensor:
    mean = crops.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = crops.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (crops - mean) / std


class ContinuousTimeEncoding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        if width % 2:
            raise ValueError("continuous time width must be even")
        frequencies = torch.pow(2.0, torch.linspace(0.0, 7.0, width // 2)) * math.pi
        self.register_buffer("frequencies", frequencies)

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        angle = position.unsqueeze(-1) * self.frequencies
        return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)


class MaskedTemporalPool(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.Tanh(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.score(sequence).squeeze(-1).masked_fill(~mask, -1e4)
        weight = torch.softmax(score, dim=1)
        attended = torch.sum(sequence * weight.unsqueeze(-1), dim=1)
        maximum = sequence.masked_fill(~mask.unsqueeze(-1), -1e4).amax(dim=1)
        mean = (sequence * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return torch.cat((attended, maximum, mean), dim=-1)


class SharedDIRTemporalEncoder(nn.Module):
    """Fuse all D/IR ROI features, then model the complete variable sequence.

    Input feature order is [Depth, IR] x seven fixed regions.  The classifier is
    deliberately part of this reusable module, but downstream Skeleton/IMU
    fusion can consume ``frame_sequence``, ``region_sequence`` or
    ``visual_embedding`` before this head.
    """

    def __init__(
        self,
        input_dim: int = PYRAMID_FEATURE_DIM,
        width: int = 256,
        frame_layers: int = 2,
        temporal_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.20,
        num_classes: int = 40,
    ) -> None:
        super().__init__()
        self.width = width
        self.input_project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
        )
        self.modality_embedding = nn.Parameter(
            torch.randn(1, 1, len(MODALITY_NAMES), 1, width) / math.sqrt(width)
        )
        self.region_embedding = nn.Parameter(
            torch.randn(1, 1, 1, len(REGION_NAMES), width) / math.sqrt(width)
        )
        self.source_embedding = nn.Embedding(7, width)
        self.quality_embedding = nn.Sequential(
            nn.Linear(4, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, width),
        )
        self.frame_token = nn.Parameter(torch.zeros(1, 1, width))
        frame_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.frame_encoder = nn.TransformerEncoder(frame_layer, num_layers=frame_layers)
        self.modality_gate = nn.Sequential(
            nn.Linear(width * 2 + 2, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )
        self.region_refine = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.time_encoding = ContinuousTimeEncoding(width)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_layers
        )
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal_pool = MaskedTemporalPool(width)
        self.embedding = nn.Sequential(
            nn.Linear(width * 3, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(384),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        roi_quality: torch.Tensor,
        roi_valid: torch.Tensor,
        roi_source: torch.Tensor,
        roi_clipped_ratio: torch.Tensor,
        pose_quality_factor: torch.Tensor,
        frame_mask: torch.Tensor,
        time_position: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 5:
            raise ValueError("features must have shape [B,T,2,7,F]")
        batch_size, time_steps, modalities, regions, _ = features.shape
        if (modalities, regions) != (len(MODALITY_NAMES), len(REGION_NAMES)):
            raise ValueError(f"expected modality/region layout 2x7, got {modalities}x{regions}")

        quality_inputs = torch.stack(
            (
                roi_quality,
                roi_valid.to(roi_quality.dtype),
                1.0 - roi_clipped_ratio,
                pose_quality_factor.unsqueeze(-1).expand_as(roi_quality),
            ),
            dim=-1,
        )
        projected = self.input_project(features)
        projected = projected + self.modality_embedding + self.region_embedding
        projected = projected + self.source_embedding(roi_source.clamp(0, 6)).unsqueeze(2)
        projected = projected + self.quality_embedding(quality_inputs).unsqueeze(2)

        token_valid = roi_valid.unsqueeze(2).expand(batch_size, time_steps, modalities, regions)
        token_valid = token_valid & frame_mask[:, :, None, None]
        flat_tokens = projected.reshape(batch_size * time_steps, modalities * regions, self.width)
        flat_valid = token_valid.reshape(batch_size * time_steps, modalities * regions)
        cls = self.frame_token.expand(batch_size * time_steps, -1, -1)
        encoded = self.frame_encoder(
            torch.cat((cls, flat_tokens), dim=1),
            src_key_padding_mask=torch.cat(
                (
                    torch.zeros(batch_size * time_steps, 1, dtype=torch.bool, device=features.device),
                    ~flat_valid,
                ),
                dim=1,
            ),
        )
        frame_sequence = encoded[:, 0].reshape(batch_size, time_steps, self.width)
        tokens = encoded[:, 1:].reshape(
            batch_size, time_steps, modalities, regions, self.width
        )

        depth_tokens = tokens[:, :, 0]
        ir_tokens = tokens[:, :, 1]
        gate_inputs = torch.cat(
            (depth_tokens, ir_tokens, roi_quality.unsqueeze(-1), roi_valid.unsqueeze(-1)),
            dim=-1,
        )
        gate = torch.softmax(self.modality_gate(gate_inputs), dim=-1)
        region_sequence = gate[..., :1] * depth_tokens + gate[..., 1:] * ir_tokens
        region_sequence = self.region_refine(region_sequence)
        region_sequence = region_sequence * roi_valid.unsqueeze(-1)

        temporal = frame_sequence + self.time_encoding(time_position)
        temporal = self.temporal_encoder(temporal, src_key_padding_mask=~frame_mask)
        temporal = self.temporal_norm(temporal) * frame_mask.unsqueeze(-1)
        visual_embedding = self.embedding(self.temporal_pool(temporal, frame_mask))
        logits = self.classifier(visual_embedding)
        return {
            "logits": logits,
            "visual_embedding": visual_embedding,
            "frame_sequence": temporal,
            "region_sequence": region_sequence,
            "modality_gate": gate,
        }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def model_size_mib(module: nn.Module, bytes_per_parameter: int = 4) -> float:
    return parameter_count(module) * bytes_per_parameter / (1024**2)
