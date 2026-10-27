from __future__ import annotations

import torch
from torch import nn
from torchvision.models.video import S3D_Weights, s3d

from aligned_model import AttentionPool, SkeletonEncoder, TemporalBlock


IMAGENET_GRAY_MEAN = (0.485 + 0.456 + 0.406) / 3.0
IMAGENET_GRAY_STD = (0.229 + 0.224 + 0.225) / 3.0
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


class IRS3DSkeletonModel(nn.Module):
    """Kinetics-pretrained IR video encoder plus the established Skeleton encoder."""

    def __init__(
        self,
        *,
        dropout: float = 0.3,
        skeleton_input_dim: int = 10,
        kinetics_pretrained: bool = True,
    ) -> None:
        super().__init__()
        video = s3d(
            weights=(
                S3D_Weights.KINETICS400_V1 if kinetics_pretrained else None
            )
        )
        self.video_features = video.features
        self.skeleton = SkeletonEncoder(dropout, skeleton_input_dim)
        self.video_project = nn.Sequential(
            nn.Linear(1024, 384),
            nn.LayerNorm(384),
            nn.GELU(),
        )
        self.skeleton_project = nn.Sequential(
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
        )
        self.gate = nn.Linear(768, 384)
        self.classifier = nn.Sequential(
            nn.LayerNorm(384 * 3),
            nn.Dropout(dropout),
            nn.Linear(384 * 3, 40),
        )

    @staticmethod
    def _prepare_ir(ir: torch.Tensor) -> torch.Tensor:
        # Dataset ImageNet-gray normalization -> unit range -> Kinetics RGB
        # normalization. Shape B,T,1,H,W -> B,3,T,H,W.
        unit = ir * IMAGENET_GRAY_STD + IMAGENET_GRAY_MEAN
        rgb = unit.repeat(1, 1, 3, 1, 1).permute(0, 2, 1, 3, 4)
        mean = rgb.new_tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1)
        std = rgb.new_tensor(KINETICS_STD).view(1, 3, 1, 1, 1)
        return (rgb - mean) / std

    def encode_video(self, ir: torch.Tensor) -> torch.Tensor:
        video = self.video_features(self._prepare_ir(ir))
        return nn.functional.adaptive_avg_pool3d(video, 1).flatten(1)

    def forward_with_sequences(
        self, ir: torch.Tensor, skeleton: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the established logits plus unpooled phase sequences."""
        video_map = self.video_features(self._prepare_ir(ir))
        video_pooled = nn.functional.adaptive_avg_pool3d(
            video_map, 1
        ).flatten(1)
        video_sequence = video_map.mean(dim=(-1, -2)).transpose(1, 2)
        skeleton_sequence = self.skeleton.encode_sequence(skeleton)
        skeleton_pooled = self.skeleton.pool(skeleton_sequence)
        video = self.video_project(video_pooled)
        body = self.skeleton_project(skeleton_pooled)
        gate = torch.sigmoid(self.gate(torch.cat([video, body], dim=1)))
        fused = gate * video + (1.0 - gate) * body
        logits = self.classifier(torch.cat([video, body, fused], dim=1))
        return logits, video_sequence, skeleton_sequence

    def forward(
        self, ir: torch.Tensor, skeleton: torch.Tensor
    ) -> torch.Tensor:
        logits, _, _ = self.forward_with_sequences(ir, skeleton)
        return logits


class S3DPhaseResidualModel(nn.Module):
    """Zero-initialised phase correction over aligned IR/Skeleton tokens."""

    def __init__(
        self,
        *,
        dropout: float = 0.25,
        skeleton_input_dim: int = 10,
        kinetics_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.base = IRS3DSkeletonModel(
            dropout=dropout,
            skeleton_input_dim=skeleton_input_dim,
            kinetics_pretrained=kinetics_pretrained,
        )
        self.video_token_project = nn.Sequential(
            nn.Linear(1024, 256), nn.LayerNorm(256), nn.GELU()
        )
        self.skeleton_token_norm = nn.LayerNorm(256)
        self.phase_input = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.phase_temporal = nn.Sequential(
            TemporalBlock(256, 1, dropout * 0.5),
            TemporalBlock(256, 2, dropout * 0.5),
        )
        self.phase_pool = AttentionPool(256, attention_dim=128)
        self.delta_classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 40),
        )
        final = self.delta_classifier[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        ir: torch.Tensor,
        skeleton: torch.Tensor,
        *,
        ablation: str | None = None,
    ) -> dict[str, torch.Tensor]:
        base_logits, video_sequence, skeleton_sequence = (
            self.base.forward_with_sequences(ir, skeleton)
        )
        time_steps = int(video_sequence.shape[1])
        skeleton_sequence = nn.functional.adaptive_avg_pool1d(
            skeleton_sequence.transpose(1, 2), time_steps
        ).transpose(1, 2)
        video = self.video_token_project(video_sequence)
        body = self.skeleton_token_norm(skeleton_sequence)
        phase = self.phase_input(
            torch.cat(
                [video, body, video * body, (video - body).abs()], dim=-1
            )
        )
        reverse_time = ablation == "phase_time_reverse"
        if reverse_time:
            phase = phase.flip(1)
        phase = self.phase_temporal(phase.transpose(1, 2)).transpose(1, 2)
        pooled = self.phase_pool(phase)
        if ablation == "phase_zero":
            pooled = torch.zeros_like(pooled)
        elif ablation == "phase_shuffle":
            pooled = pooled.roll(1, dims=0)
        elif reverse_time:
            pass
        elif ablation is not None:
            raise ValueError(f"unknown phase ablation: {ablation}")
        delta_logits = self.delta_classifier(pooled)
        return {
            "logits": base_logits + delta_logits,
            "base_logits": base_logits,
            "delta_logits": delta_logits,
            "phase_sequence": phase,
            "phase_pooled": pooled,
        }


def freeze_video_batch_norm_statistics(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.eval()
