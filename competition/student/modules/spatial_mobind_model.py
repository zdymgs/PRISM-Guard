from __future__ import annotations

import math

import torch
from torch import nn

from mc3_visual_model import MC3VisualStudent
from mobind_lite_model import (
    MoBindMotionResidual,
    SeparateMotionEncoder,
)


class _MotionConditionedSpatialScorer(nn.Module):
    """Score visual regions from one private motion modality at the same time."""

    def __init__(
        self,
        motion_width: int,
        spatial_regions: int,
        attention_heads: int,
        maximum_spatial_logit: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if attention_heads <= 0 or 128 % attention_heads:
            raise ValueError("attention_heads must divide 128")
        if spatial_regions <= 1:
            raise ValueError("spatial scorer requires more than one region")
        if maximum_spatial_logit <= 0.0:
            raise ValueError("maximum spatial logit must be positive")
        self.attention_heads = int(attention_heads)
        self.head_width = 128 // self.attention_heads
        self.maximum_spatial_logit = float(maximum_spatial_logit)
        self.region_key = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 128))
        self.motion_query = nn.Sequential(
            nn.LayerNorm(motion_width), nn.Linear(motion_width, 128)
        )
        self.region_position = nn.Parameter(torch.zeros(spatial_regions, 128))
        nn.init.trunc_normal_(self.region_position, std=0.02)
        self.compatibility = nn.Sequential(
            nn.LayerNorm(self.attention_heads),
            nn.Linear(self.attention_heads, self.attention_heads * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.attention_heads * 2, 1),
        )
        # Uniform spatial pooling is the exact paired-student boundary.
        nn.init.zeros_(self.compatibility[4].weight)
        nn.init.zeros_(self.compatibility[4].bias)

    def forward(
        self,
        regions: torch.Tensor,
        visual_mask: torch.Tensor,
        motion_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if regions.ndim != 6 or regions.shape[-1] != 512:
            raise ValueError("regions must be [B,2,3,T,R,512]")
        if visual_mask.shape != regions.shape[:-2]:
            raise ValueError("visual mask must match the region time grid")
        if motion_tokens.ndim != 5 or motion_mask.shape != motion_tokens.shape[:-1]:
            raise ValueError("motion tokens/mask must be [B,2,T,P,D]")
        if motion_tokens.shape[:3] != (
            regions.shape[0],
            regions.shape[1],
            regions.shape[3],
        ):
            raise ValueError("visual and motion time grids differ")
        region_count = regions.shape[-2]
        if region_count != self.region_position.shape[0]:
            raise ValueError("unexpected spatial region count")

        keys = self.region_key(regions) + self.region_position.view(
            1, 1, 1, 1, region_count, 128
        )
        queries = self.motion_query(motion_tokens)
        keys = keys.reshape(*keys.shape[:-1], self.attention_heads, self.head_width)
        queries = queries.reshape(
            *queries.shape[:-1], self.attention_heads, self.head_width
        )
        affinity = torch.einsum(
            "bwvtrhd,bwtphd->bwvtrph", keys, queries
        ) / math.sqrt(self.head_width)
        part_mask = motion_mask[:, :, None, :, None, :, None]
        safe_mask = part_mask.expand_as(affinity).clone()
        empty = ~safe_mask.any(dim=-2, keepdim=True)
        safe_mask[..., 0, :] = safe_mask[..., 0, :] | empty[..., 0, :]
        part_attention = torch.softmax(
            affinity.masked_fill(~safe_mask, -1e4), dim=-2
        )
        part_attention = part_attention * part_mask.to(part_attention.dtype)
        part_attention = part_attention / part_attention.sum(
            dim=-2, keepdim=True
        ).clamp_min(1e-6)
        head_compatibility = (part_attention * affinity).sum(dim=-2)
        logit = self.maximum_spatial_logit * torch.tanh(
            self.compatibility(head_compatibility).squeeze(-1)
        )
        available = (
            motion_mask.any(dim=-1)[:, :, None, :].expand(-1, -1, regions.shape[2], -1)
            & visual_mask
        )
        logit = logit * available.to(logit.dtype).unsqueeze(-1)
        return logit, {
            "part_attention": part_attention.mean(dim=-1),
            "available": available,
        }


class SpatialCrossAttentionPoolStudent(nn.Module):
    """Condition layer4 spatial pooling on aligned Skeleton/IMU tokens.

    The complete trained student clip-local/global path remains active and frozen.
    Only a modality-private scorer is added before spatial pooling.  It changes
    which layer4 regions form each visual time token; it never projects a motion
    feature vector into the 512-dimensional visual stream.
    """

    VALID_ABLATIONS = {"none", "zero", "reverse_time", "sample_roll"}

    def __init__(
        self,
        visual: MC3VisualStudent,
        motion_residual: MoBindMotionResidual,
        spatial_grid: int = 5,
        attention_heads: int = 4,
        maximum_spatial_logit: float = 1.0,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        if not visual.temporal_modeling:
            raise ValueError("spatial fusion requires temporal visual modeling")
        if spatial_grid <= 1:
            raise ValueError("spatial grid must be greater than one")
        if motion_residual.modality != "separate" or not isinstance(
            motion_residual.encoder, SeparateMotionEncoder
        ):
            raise ValueError("v4 requires the separate Skeleton/IMU student anchor")
        self.visual = visual
        self.motion_residual = motion_residual
        self.spatial_grid = int(spatial_grid)
        self.spatial_regions = self.spatial_grid**2
        self.spatial_ablation = "none"
        motion_width = int(motion_residual.encoder.width)
        scorer = lambda: _MotionConditionedSpatialScorer(
            motion_width,
            self.spatial_regions,
            attention_heads,
            maximum_spatial_logit,
            dropout,
        )
        self.skeleton_spatial_scorer = scorer()
        self.imu_spatial_scorer = scorer()

    def spatial_parameters(self) -> list[nn.Parameter]:
        return [
            *self.skeleton_spatial_scorer.parameters(),
            *self.imu_spatial_scorer.parameters(),
        ]

    def set_spatial_ablation(self, value: str) -> None:
        if value not in self.VALID_ABLATIONS:
            raise ValueError(
                f"spatial ablation must be one of {sorted(self.VALID_ABLATIONS)}"
            )
        self.spatial_ablation = value

    def _apply_counterfactual(
        self, tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spatial_ablation == "reverse_time":
            return tokens.flip(2), token_mask.flip(2)
        if self.spatial_ablation == "sample_roll" and len(tokens) > 1:
            return tokens.roll(1, dims=0), token_mask.roll(1, dims=0)
        return tokens, token_mask

    def _condition_spatial_pool(
        self,
        compact_sequence: torch.Tensor,
        region_sequence: torch.Tensor,
        visual_mask: torch.Tensor,
        motion_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if compact_sequence.ndim != 5 or compact_sequence.shape[-1] != 512:
            raise ValueError("compact sequence must be [B,2,3,T,512]")
        if region_sequence.shape[:-2] != compact_sequence.shape[:-1]:
            raise ValueError("compact and spatial sequence grids differ")
        if region_sequence.shape[-2:] != (self.spatial_regions, 512):
            raise ValueError("unexpected v4 spatial grid")
        if motion_tokens.shape[3] != 10:
            raise ValueError("v4 expects five Skeleton plus five IMU parts")
        motion_tokens, motion_mask = self._apply_counterfactual(
            motion_tokens, motion_mask
        )
        skeleton_logit, skeleton_audit = self.skeleton_spatial_scorer(
            region_sequence,
            visual_mask,
            motion_tokens[:, :, :, :5],
            motion_mask[:, :, :, :5],
        )
        imu_logit, imu_audit = self.imu_spatial_scorer(
            region_sequence,
            visual_mask,
            motion_tokens[:, :, :, 5:],
            motion_mask[:, :, :, 5:],
        )
        available = torch.stack(
            (skeleton_audit["available"], imu_audit["available"]), dim=-1
        )
        modality_weight = available.to(region_sequence.dtype)
        modality_weight = modality_weight / modality_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        spatial_logit = (
            skeleton_logit * modality_weight[..., 0].unsqueeze(-1)
            + imu_logit * modality_weight[..., 1].unsqueeze(-1)
        )
        if self.spatial_ablation == "zero":
            spatial_logit = torch.zeros_like(spatial_logit)
        spatial_attention = torch.softmax(spatial_logit, dim=-1)
        uniform = torch.full_like(spatial_attention, 1.0 / self.spatial_regions)
        correction = torch.einsum(
            "bwvtr,bwvtrd->bwvtd", spatial_attention - uniform, region_sequence
        )
        corrected = compact_sequence + correction
        compact_rms = compact_sequence.square().mean(dim=-1).sqrt()
        correction_rms = correction.square().mean(dim=-1).sqrt()
        return corrected, {
            "spatial_skeleton_part_attention": skeleton_audit["part_attention"],
            "spatial_imu_part_attention": imu_audit["part_attention"],
            "spatial_motion_available": available,
            "spatial_modality_weight": modality_weight,
            "spatial_pool_logit": spatial_logit,
            "spatial_pool_attention": spatial_attention,
            "spatial_pool_weight_l1_from_uniform": (
                spatial_attention - uniform
            ).abs().sum(dim=-1),
            "spatial_correction_rms_ratio": correction_rms
            / compact_rms.clamp_min(1e-6),
        }

    def forward_from_backbone_region_sequence(
        self,
        compact_sequence: torch.Tensor,
        region_sequence: torch.Tensor,
        view_valid: torch.Tensor,
        view_quality: torch.Tensor,
        global_time_position: torch.Tensor,
        motion: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        anchor_visual_clips = self.visual.encode_clips_from_backbone_sequence(
            compact_sequence, view_valid, global_time_position
        )
        visual_only = self.visual._fuse_clips(
            anchor_visual_clips, view_valid, view_quality
        )
        anchor_clips, audit = self.motion_residual(
            anchor_visual_clips, global_time_position, motion
        )
        motion_tokens = audit.pop("motion_tokens")
        motion_mask = audit.pop("motion_token_mask")
        visual_mask = view_valid.permute(0, 1, 3, 2)
        corrected_sequence, spatial_audit = self._condition_spatial_pool(
            compact_sequence,
            region_sequence,
            visual_mask,
            motion_tokens,
            motion_mask,
        )
        corrected_visual_clips = self.visual.encode_clips_from_backbone_sequence(
            corrected_sequence, view_valid, global_time_position
        )
        clip_effect = corrected_visual_clips - anchor_visual_clips
        fused_clips = anchor_clips + clip_effect

        anchor_output = self.visual._fuse_clips(
            anchor_clips, view_valid, view_quality
        )
        anchor_embedding, _ = self.motion_residual.fuse_global(
            anchor_output["visual_embedding"],
            audit["motion_semantic_embedding"],
            audit["motion_available"],
            audit["motion_reliability_context"],
        )
        anchor_logits = self.visual.classifier(anchor_embedding)

        output = self.visual._fuse_clips(fused_clips, view_valid, view_quality)
        fused_embedding, global_reliability = self.motion_residual.fuse_global(
            output["visual_embedding"],
            audit["motion_semantic_embedding"],
            audit["motion_available"],
            audit["motion_reliability_context"],
        )
        output["visual_embedding"] = fused_embedding
        output["logits"] = self.visual.classifier(fused_embedding)
        output["anchor_fusion_logits"] = anchor_logits
        output["unfused_visual_logits"] = visual_only["logits"]
        spatial_audit["spatial_clip_effect_rms_ratio"] = (
            clip_effect.square().mean(dim=-1).sqrt()
            / anchor_visual_clips.square().mean(dim=-1).sqrt().clamp_min(1e-6)
        )
        audit.update(spatial_audit)
        audit["motion_global_reliability"] = global_reliability
        output.update(audit)
        return output

    def forward(
        self,
        images: torch.Tensor,
        view_valid: torch.Tensor,
        view_quality: torch.Tensor,
        global_time_position: torch.Tensor,
        motion: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        compact, regions = self.visual.encode_backbone_spatial_sequences(
            images, self.spatial_grid
        )
        return self.forward_from_backbone_region_sequence(
            compact,
            regions,
            view_valid,
            view_quality,
            global_time_position,
            motion,
        )
