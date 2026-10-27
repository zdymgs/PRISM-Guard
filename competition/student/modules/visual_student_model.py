from __future__ import annotations

import torch
from torch import nn

from shared_dir_roi_model import ContinuousTimeEncoding, PYRAMID_FEATURE_DIM


class TemporalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=width,
            bias=False,
        )
        self.pointwise = nn.Conv1d(width, width * 2, kernel_size=1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value, gate = self.pointwise(self.depthwise(source.transpose(1, 2))).chunk(2, dim=1)
        update = (value * torch.sigmoid(gate)).transpose(1, 2)
        return self.norm(source + self.dropout(update)) * mask.unsqueeze(-1)


class VisualStudent(nn.Module):
    """One small full-40 visual model over matched small-backbone tokens."""

    def __init__(
        self,
        input_width: int = PYRAMID_FEATURE_DIM,
        width: int = 192,
        embedding_width: int = 256,
        classes: int = 40,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_width)
        self.input_projection = nn.Linear(input_width, width)
        self.view_embedding = nn.Parameter(torch.zeros(3, width))
        self.window_embedding = nn.Parameter(torch.zeros(2, width))
        nn.init.trunc_normal_(self.view_embedding, std=0.02)
        nn.init.trunc_normal_(self.window_embedding, std=0.02)
        self.time_encoding = ContinuousTimeEncoding(width)
        self.quality_projection = nn.Sequential(
            nn.Linear(2, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        view_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.view_encoder = nn.TransformerEncoder(view_layer, num_layers=1)
        self.view_score = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.Tanh(),
            nn.Linear(width // 2, 1),
        )
        self.temporal_blocks = nn.ModuleList(
            [TemporalResidualBlock(width, dilation, dropout) for dilation in (1, 2, 4, 8)]
        )
        self.temporal_score = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.Tanh(),
            nn.Linear(width // 2, 1),
        )
        self.window_projection = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, embedding_width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stage_fusion = nn.Sequential(
            nn.LayerNorm(embedding_width * 4),
            nn.Linear(embedding_width * 4, embedding_width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_width * 2, embedding_width),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_width),
            nn.Dropout(dropout),
            nn.Linear(embedding_width, classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        view_mask: torch.Tensor,
        view_quality: torch.Tensor,
        time_position: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 5 or features.shape[1:4] != (2, 16, 3):
            raise ValueError(f"expected [B,2,16,3,D] features, got {tuple(features.shape)}")
        if view_mask.shape != features.shape[:-1]:
            raise ValueError("view mask shape mismatch")
        if view_quality.shape != view_mask.shape or time_position.shape != features.shape[:3]:
            raise ValueError("student quality/time shape mismatch")
        batch, windows, frames, views, width_in = features.shape
        source = self.input_projection(self.input_norm(features))
        source = source + self.view_embedding.view(1, 1, 1, views, -1)
        source = source + self.window_embedding.view(1, windows, 1, 1, -1)
        source = source + self.time_encoding(time_position).unsqueeze(3)
        quality_input = torch.stack((view_quality, view_mask.to(view_quality.dtype)), dim=-1)
        source = source + self.quality_projection(quality_input)

        flat_source = source.reshape(batch * windows * frames, views, -1)
        flat_mask = view_mask.reshape(batch * windows * frames, views)
        safe_mask = flat_mask.clone()
        safe_mask[:, 0] = True
        flat_source = self.view_encoder(flat_source, src_key_padding_mask=~safe_mask)
        flat_source = flat_source * flat_mask.unsqueeze(-1)
        source = flat_source.reshape(batch, windows, frames, views, -1)

        view_logits = self.view_score(source).squeeze(-1)
        view_logits = view_logits + torch.log(view_quality.clamp_min(1e-3))
        view_logits = view_logits.masked_fill(~view_mask, -1e4)
        view_weight = torch.softmax(view_logits, dim=3) * view_mask
        view_weight = view_weight / view_weight.sum(dim=3, keepdim=True).clamp_min(1e-6)
        frame_sequence = torch.sum(source * view_weight.unsqueeze(-1), dim=3)
        frame_mask = view_mask.any(dim=3)

        temporal = frame_sequence.reshape(batch * windows, frames, -1)
        temporal_mask = frame_mask.reshape(batch * windows, frames)
        for block in self.temporal_blocks:
            temporal = block(temporal, temporal_mask)
        score = self.temporal_score(temporal).squeeze(-1).masked_fill(~temporal_mask, -1e4)
        weight = torch.softmax(score, dim=1) * temporal_mask
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attended = torch.sum(temporal * weight.unsqueeze(-1), dim=1)
        maximum = temporal.masked_fill(~temporal_mask.unsqueeze(-1), -1e4).amax(dim=1)
        mean = (temporal * temporal_mask.unsqueeze(-1)).sum(dim=1) / temporal_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        window_embedding = self.window_projection(torch.cat((attended, maximum, mean), dim=1))
        window_embedding = window_embedding.reshape(batch, windows, -1)
        early = window_embedding[:, 0]
        late = window_embedding[:, 1]
        visual_embedding = self.stage_fusion(
            torch.cat((early, late, 0.5 * (early + late), late - early), dim=1)
        )

        clip_mask = view_mask.any(dim=2)
        clip_embeddings = (source * view_mask.unsqueeze(-1)).sum(dim=2) / view_mask.sum(
            dim=2, keepdim=False
        ).clamp_min(1).unsqueeze(-1)
        clip_embeddings = clip_embeddings * clip_mask.unsqueeze(-1)
        return {
            "logits": self.classifier(visual_embedding),
            "visual_embedding": visual_embedding,
            "window_embeddings": window_embedding,
            "clip_embeddings": clip_embeddings,
            "clip_mask": clip_mask,
            "frame_sequence": temporal.reshape(batch, windows, frames, -1),
            "frame_mask": frame_mask,
            "view_weight": view_weight,
        }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def model_size_mib(module: nn.Module, bytes_per_parameter: int = 4) -> float:
    return parameter_count(module) * bytes_per_parameter / (1024**2)
