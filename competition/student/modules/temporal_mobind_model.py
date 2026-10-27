from __future__ import annotations

import math

import torch
from torch import nn

from mc3_visual_model import MC3VisualStudent
from mobind_lite_model import (
    MoBindMotionResidual,
    SeparateMotionEncoder,
)


class TemporalMoBindStudent(nn.Module):
    """Fuse motion after visual temporal encoding but before temporal pooling.

    student asks one pooled clip token to retrieve a complete time x part motion
    window.  fusion keeps the visual time axis and lets every visual time token
    retrieve only a local motion neighbourhood.  It reuses the exact student
    motion encoder, projections, gates, global residual and classifier, so the
    paired experiment changes the interaction boundary rather than capacity.
    """

    def __init__(
        self,
        visual: MC3VisualStudent,
        motion_residual: MoBindMotionResidual,
        temporal_radius: int = 1,
    ) -> None:
        super().__init__()
        if not visual.temporal_modeling:
            raise ValueError("temporal MoBind requires temporal visual tokens")
        if temporal_radius < 0:
            raise ValueError("temporal_radius must be non-negative")
        self.visual = visual
        self.motion_residual = motion_residual
        self.temporal_radius = int(temporal_radius)

    @staticmethod
    def _masked_part_mean(
        tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = token_mask.to(tokens.dtype).unsqueeze(-1)
        mean = (tokens * weight).sum(dim=3) / weight.sum(dim=3).clamp_min(1.0)
        return mean, token_mask.any(dim=3)

    @staticmethod
    def _phase_delta(
        time_tokens: torch.Tensor, time_valid: torch.Tensor
    ) -> torch.Tensor:
        """Centered motion phase with one-sided fallback at valid boundaries."""
        previous = torch.cat((time_tokens[:, :, :1], time_tokens[:, :, :-1]), dim=2)
        following = torch.cat((time_tokens[:, :, 1:], time_tokens[:, :, -1:]), dim=2)
        previous_valid = torch.cat(
            (time_valid[:, :, :1], time_valid[:, :, :-1]), dim=2
        )
        following_valid = torch.cat(
            (time_valid[:, :, 1:], time_valid[:, :, -1:]), dim=2
        )
        previous = torch.where(previous_valid.unsqueeze(-1), previous, time_tokens)
        following = torch.where(following_valid.unsqueeze(-1), following, time_tokens)
        return following - previous

    def _fuse_temporal_tokens(
        self,
        visual_sequence: torch.Tensor,
        visual_mask: torch.Tensor,
        motion_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
        reliability_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, windows, views, visual_steps, visual_width = visual_sequence.shape
        if (windows, views, visual_width) != (2, 3, 512):
            raise ValueError("visual_sequence must be [B,2,3,T,512]")
        if visual_mask.shape != visual_sequence.shape[:-1]:
            raise ValueError("visual_mask must match visual_sequence")
        if motion_tokens.ndim != 5 or motion_mask.shape != motion_tokens.shape[:-1]:
            raise ValueError("motion tokens/mask must be [B,2,T,P,D]")
        if motion_tokens.shape[:2] != (batch, windows):
            raise ValueError("motion and visual batch/window grids differ")
        motion_steps, parts, motion_width = motion_tokens.shape[2:]
        if visual_steps != motion_steps:
            raise ValueError("fusion requires aligned visual and motion time bins")

        query = self.motion_residual.visual_query(visual_sequence)
        score = torch.einsum("bwvtd,bwspd->bwvtsp", query, motion_tokens)
        if score.shape != (batch, windows, views, visual_steps, motion_steps, parts):
            raise RuntimeError("unexpected fusion attention grid")
        score = score / math.sqrt(motion_width)

        visual_index = torch.arange(visual_steps, device=score.device)
        motion_index = torch.arange(motion_steps, device=score.device)
        neighbourhood = (
            visual_index[:, None] - motion_index[None, :]
        ).abs() <= self.temporal_radius
        local_mask = motion_mask[:, :, None, None, :, :] & neighbourhood[
            None, None, None, :, :, None
        ]
        local_mask = local_mask.expand(-1, -1, views, -1, -1, -1)
        flat_score = score.flatten(-2)
        flat_mask = local_mask.flatten(-2)
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=-1)
        safe_mask[..., 0] = safe_mask[..., 0] | empty
        flat_score = flat_score.masked_fill(~safe_mask, -1e4)
        attention = torch.softmax(flat_score, dim=-1)
        attention = attention * flat_mask.to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attention = attention.reshape(
            batch, windows, views, visual_steps, motion_steps, parts
        )
        attended = torch.einsum("bwvtsp,bwspd->bwvtd", attention, motion_tokens)

        local_available = local_mask.any(dim=(-1, -2))
        attended = attended * local_available.to(attended.dtype).unsqueeze(-1)
        local_weight = (
            motion_mask[:, :, None, :, :]
            & neighbourhood[None, None, :, :, None]
        ).to(motion_tokens.dtype)
        local_mean = torch.einsum("bwtsp,bwspd->bwtd", local_weight, motion_tokens)
        local_mean = local_mean / local_weight.sum(dim=(-1, -2)).clamp_min(1.0).unsqueeze(-1)
        local_mean = local_mean.unsqueeze(2).expand(-1, -1, views, -1, -1)

        motion_by_time, motion_time_valid = self._masked_part_mean(
            motion_tokens, motion_mask
        )
        phase = self._phase_delta(motion_by_time, motion_time_valid)
        phase = phase.unsqueeze(2).expand(-1, -1, views, -1, -1)

        reliability_values = [
            query,
            attended,
            query * attended,
            (query - attended).abs(),
        ]
        if reliability_context is not None:
            reliability_values.append(
                reliability_context[:, None, None, None].expand(
                    -1, windows, views, visual_steps, -1
                )
            )
        local_group_gate = torch.sigmoid(
            self.motion_residual.reliability(torch.cat(reliability_values, dim=-1))
        )
        local_gate = local_group_gate.repeat_interleave(
            visual_width // self.motion_residual.reliability_groups, dim=-1
        )
        adapter_input = torch.cat(
            (attended, local_mean, phase, query * attended), dim=-1
        )
        residual = (
            self.motion_residual.motion_projection(adapter_input)
            * local_gate
            * self.motion_residual.strength()
        )
        usable = local_available & visual_mask
        residual = residual * usable.to(residual.dtype).unsqueeze(-1)
        fused = visual_sequence + residual
        return fused, {
            "motion_part_attention": attention.sum(dim=-2),
            "motion_temporal_attention": attention.sum(dim=-1),
            "motion_reliability": local_group_gate,
            "motion_time_available": local_available,
        }

    def forward_from_backbone_sequence(
        self,
        backbone_sequence: torch.Tensor,
        view_valid: torch.Tensor,
        view_quality: torch.Tensor,
        global_time_position: torch.Tensor,
        motion: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        visual_sequence, visual_mask = (
            self.visual.encode_temporal_sequence_from_backbone(
                backbone_sequence, view_valid, global_time_position
            )
        )
        visual_clips = self.visual.pool_temporal_sequence(
            visual_sequence, visual_mask
        )
        visual_only = self.visual._fuse_clips(
            visual_clips, view_valid, view_quality
        )

        # This call performs the single motion-encoder pass and preserves the
        # proven student semantic/global auxiliary path.  Its clip residual is not
        # used; fusion instead reuses the same adapter weights on temporal tokens.
        _, audit = self.motion_residual(
            visual_clips, global_time_position, motion
        )
        fused_sequence, temporal_audit = self._fuse_temporal_tokens(
            visual_sequence,
            visual_mask,
            audit.pop("motion_tokens"),
            audit.pop("motion_token_mask"),
            audit["motion_reliability_context"],
        )
        fused_clips = self.visual.pool_temporal_sequence(
            fused_sequence, visual_mask
        )
        output = self.visual._fuse_clips(fused_clips, view_valid, view_quality)
        fused_embedding, global_reliability = self.motion_residual.fuse_global(
            output["visual_embedding"],
            audit["motion_semantic_embedding"],
            audit["motion_available"],
            audit["motion_reliability_context"],
        )
        output["visual_embedding"] = fused_embedding
        output["logits"] = self.visual.classifier(fused_embedding)
        output["unfused_visual_logits"] = visual_only["logits"]
        audit.update(temporal_audit)
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
        return self.forward_from_backbone_sequence(
            self.visual.encode_backbone_sequence(images),
            view_valid,
            view_quality,
            global_time_position,
            motion,
        )


class TemporalRefinedMoBindStudent(nn.Module):
    """Add a bounded temporal residual without replacing the proven student path.

    The original fusion candidate silently changed two things at once: it replaced
    student's clip-local residual with a pre-pooling residual and trained the whole
    fusion stack again from the motion pretrain.  This revision instead treats a
    trained student V+Skeleton+IMU model as an immutable anchor:

    * the complete student clip-local and global residual path remains active;
    * Skeleton and IMU retain separate five-part temporal retrieval branches;
    * the new branch has a fixed, small residual budget rather than an
      unsupervised sigmoid gate that can saturate;
    * zero-initialised output projections make the initial function exactly student;
    * missing motion and the explicit ``zero`` ablation are exact fallbacks.

    Only parameters returned by :meth:`temporal_parameters` should be trained in
    the controlled v2 experiment.
    """

    VALID_ABLATIONS = {"none", "zero", "reverse_time", "sample_roll"}

    def __init__(
        self,
        visual: MC3VisualStudent,
        motion_residual: MoBindMotionResidual,
        temporal_radius: int = 1,
        temporal_budget: float = 0.10,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        if not visual.temporal_modeling:
            raise ValueError("temporal MoBind requires temporal visual tokens")
        if temporal_radius < 0:
            raise ValueError("temporal_radius must be non-negative")
        if not 0.0 < temporal_budget <= 0.25:
            raise ValueError("temporal_budget must be in (0, 0.25]")
        if motion_residual.modality != "separate" or not isinstance(
            motion_residual.encoder, SeparateMotionEncoder
        ):
            raise ValueError("v2 requires the separate Skeleton/IMU student anchor")

        self.visual = visual
        self.motion_residual = motion_residual
        self.temporal_radius = int(temporal_radius)
        self.temporal_budget = float(temporal_budget)
        self.temporal_ablation = "none"

        motion_width = int(motion_residual.encoder.width)
        self.motion_width = motion_width
        self.skeleton_temporal_query = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, motion_width)
        )
        self.imu_temporal_query = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, motion_width)
        )

        def projection() -> nn.Sequential:
            module = nn.Sequential(
                nn.LayerNorm(motion_width * 4),
                nn.Linear(motion_width * 4, motion_width * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(motion_width * 2, 512),
                nn.Dropout(dropout),
            )
            # This is the exact-student fallback boundary.  A non-zero initial
            # adapter would make the anchor comparison ambiguous.
            nn.init.zeros_(module[4].weight)
            nn.init.zeros_(module[4].bias)
            return module

        self.skeleton_temporal_projection = projection()
        self.imu_temporal_projection = projection()

    def temporal_parameters(self) -> list[nn.Parameter]:
        modules = (
            self.skeleton_temporal_query,
            self.imu_temporal_query,
            self.skeleton_temporal_projection,
            self.imu_temporal_projection,
        )
        return [parameter for module in modules for parameter in module.parameters()]

    def set_temporal_ablation(self, value: str) -> None:
        if value not in self.VALID_ABLATIONS:
            raise ValueError(
                f"temporal ablation must be one of {sorted(self.VALID_ABLATIONS)}"
            )
        self.temporal_ablation = value

    @staticmethod
    def _masked_part_mean(
        tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = token_mask.to(tokens.dtype).unsqueeze(-1)
        mean = (tokens * weight).sum(dim=3) / weight.sum(dim=3).clamp_min(1.0)
        return mean, token_mask.any(dim=3)

    @staticmethod
    def _phase_delta(
        time_tokens: torch.Tensor, time_valid: torch.Tensor
    ) -> torch.Tensor:
        previous = torch.cat((time_tokens[:, :, :1], time_tokens[:, :, :-1]), dim=2)
        following = torch.cat((time_tokens[:, :, 1:], time_tokens[:, :, -1:]), dim=2)
        previous_valid = torch.cat(
            (time_valid[:, :, :1], time_valid[:, :, :-1]), dim=2
        )
        following_valid = torch.cat(
            (time_valid[:, :, 1:], time_valid[:, :, -1:]), dim=2
        )
        previous = torch.where(previous_valid.unsqueeze(-1), previous, time_tokens)
        following = torch.where(following_valid.unsqueeze(-1), following, time_tokens)
        return following - previous

    def _apply_counterfactual(
        self, tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.temporal_ablation == "reverse_time":
            return tokens.flip(2), token_mask.flip(2)
        if self.temporal_ablation == "sample_roll" and len(tokens) > 1:
            return tokens.roll(1, dims=0), token_mask.roll(1, dims=0)
        return tokens, token_mask

    def _retrieve_modality(
        self,
        visual_sequence: torch.Tensor,
        visual_mask: torch.Tensor,
        motion_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
        query_module: nn.Module,
        projection_module: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, windows, views, visual_steps, _ = visual_sequence.shape
        motion_steps, parts, motion_width = motion_tokens.shape[2:]
        if motion_width != self.motion_width:
            raise ValueError("unexpected motion width")
        motion_tokens, motion_mask = self._apply_counterfactual(
            motion_tokens, motion_mask
        )
        query = query_module(visual_sequence)
        score = torch.einsum("bwvtd,bwspd->bwvtsp", query, motion_tokens)
        score = score / math.sqrt(motion_width)

        visual_index = torch.arange(visual_steps, device=score.device)
        motion_index = torch.arange(motion_steps, device=score.device)
        neighbourhood = (
            visual_index[:, None] - motion_index[None, :]
        ).abs() <= self.temporal_radius
        local_mask = motion_mask[:, :, None, None, :, :] & neighbourhood[
            None, None, None, :, :, None
        ]
        local_mask = local_mask.expand(-1, -1, views, -1, -1, -1)
        flat_score = score.flatten(-2)
        flat_mask = local_mask.flatten(-2)
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=-1)
        safe_mask[..., 0] = safe_mask[..., 0] | empty
        attention = torch.softmax(
            flat_score.masked_fill(~safe_mask, -1e4), dim=-1
        )
        attention = attention * flat_mask.to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attention = attention.reshape(
            batch, windows, views, visual_steps, motion_steps, parts
        )
        attended = torch.einsum("bwvtsp,bwspd->bwvtd", attention, motion_tokens)
        available = local_mask.any(dim=(-1, -2)) & visual_mask
        attended = attended * available.to(attended.dtype).unsqueeze(-1)

        local_weight = (
            motion_mask[:, :, None, :, :]
            & neighbourhood[None, None, :, :, None]
        ).to(motion_tokens.dtype)
        local_mean = torch.einsum("bwtsp,bwspd->bwtd", local_weight, motion_tokens)
        local_mean = local_mean / local_weight.sum(dim=(-1, -2)).clamp_min(1.0).unsqueeze(-1)
        local_mean = local_mean.unsqueeze(2).expand(-1, -1, views, -1, -1)
        time_tokens, time_valid = self._masked_part_mean(motion_tokens, motion_mask)
        phase = self._phase_delta(time_tokens, time_valid)
        phase = phase.unsqueeze(2).expand(-1, -1, views, -1, -1)
        adapter_input = torch.cat(
            (attended, local_mean, phase, query * attended), dim=-1
        )
        # tanh bounds each residual coordinate before the fixed total budget is
        # applied.  The projection cannot silently defeat the small-gate audit by
        # increasing its raw output norm.
        residual = torch.tanh(projection_module(adapter_input))
        residual = residual * available.to(residual.dtype).unsqueeze(-1)
        return residual, {
            "part_attention": attention.sum(dim=-2),
            "time_attention": attention.sum(dim=-1),
            "available": available,
            "raw_residual_rms": residual.square().mean(dim=-1).sqrt(),
        }

    def _fuse_temporal_tokens(
        self,
        visual_sequence: torch.Tensor,
        visual_mask: torch.Tensor,
        motion_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if visual_sequence.ndim != 5 or visual_sequence.shape[1:3] != (2, 3):
            raise ValueError("visual_sequence must be [B,2,3,T,512]")
        if visual_mask.shape != visual_sequence.shape[:-1]:
            raise ValueError("visual_mask must match visual_sequence")
        if motion_tokens.ndim != 5 or motion_mask.shape != motion_tokens.shape[:-1]:
            raise ValueError("motion tokens/mask must be [B,2,T,10,D]")
        if motion_tokens.shape[3] != 10:
            raise ValueError("separate v2 expects five Skeleton plus five IMU parts")
        if motion_tokens.shape[:3] != (
            visual_sequence.shape[0],
            visual_sequence.shape[1],
            visual_sequence.shape[3],
        ):
            raise ValueError("visual and motion time grids differ")

        skeleton_residual, skeleton_audit = self._retrieve_modality(
            visual_sequence,
            visual_mask,
            motion_tokens[:, :, :, :5],
            motion_mask[:, :, :, :5],
            self.skeleton_temporal_query,
            self.skeleton_temporal_projection,
        )
        imu_residual, imu_audit = self._retrieve_modality(
            visual_sequence,
            visual_mask,
            motion_tokens[:, :, :, 5:],
            motion_mask[:, :, :, 5:],
            self.imu_temporal_query,
            self.imu_temporal_projection,
        )
        available = torch.stack(
            (skeleton_audit["available"], imu_audit["available"]), dim=-1
        )
        modality_weight = available.to(visual_sequence.dtype)
        modality_weight = modality_weight / modality_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        combined = (
            skeleton_residual * modality_weight[..., 0].unsqueeze(-1)
            + imu_residual * modality_weight[..., 1].unsqueeze(-1)
        )
        temporal_residual = self.temporal_budget * combined
        if self.temporal_ablation == "zero":
            temporal_residual = torch.zeros_like(temporal_residual)
        fused = visual_sequence + temporal_residual
        visual_rms = visual_sequence.square().mean(dim=-1).sqrt()
        residual_rms = temporal_residual.square().mean(dim=-1).sqrt()
        return fused, {
            "temporal_skeleton_part_attention": skeleton_audit["part_attention"],
            "temporal_skeleton_time_attention": skeleton_audit["time_attention"],
            "temporal_imu_part_attention": imu_audit["part_attention"],
            "temporal_imu_time_attention": imu_audit["time_attention"],
            "temporal_modality_weight": modality_weight,
            "temporal_motion_available": available,
            "temporal_residual_rms": residual_rms,
            "temporal_residual_rms_ratio": residual_rms
            / visual_rms.clamp_min(1e-6),
            "temporal_residual_budget": visual_sequence.new_tensor(
                self.temporal_budget
            ),
        }

    def forward_from_backbone_sequence(
        self,
        backbone_sequence: torch.Tensor,
        view_valid: torch.Tensor,
        view_quality: torch.Tensor,
        global_time_position: torch.Tensor,
        motion: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        visual_sequence, visual_mask = (
            self.visual.encode_temporal_sequence_from_backbone(
                backbone_sequence, view_valid, global_time_position
            )
        )
        visual_clips = self.visual.pool_temporal_sequence(
            visual_sequence, visual_mask
        )
        visual_only = self.visual._fuse_clips(
            visual_clips, view_valid, view_quality
        )

        # Run student once.  Its local residual is preserved, while its encoded
        # motion tokens are also reused by the new temporal branch.
        anchor_clips, audit = self.motion_residual(
            visual_clips, global_time_position, motion
        )
        motion_tokens = audit.pop("motion_tokens")
        motion_mask = audit.pop("motion_token_mask")
        fused_sequence, temporal_audit = self._fuse_temporal_tokens(
            visual_sequence, visual_mask, motion_tokens, motion_mask
        )
        temporal_clips = self.visual.pool_temporal_sequence(
            fused_sequence, visual_mask
        )
        # At zero temporal residual, both pool calls are identical and this delta
        # is exactly zero; therefore anchor_clips passes through bit-for-bit.
        fused_clips = anchor_clips + (temporal_clips - visual_clips)

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
        audit.update(temporal_audit)
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
        return self.forward_from_backbone_sequence(
            self.visual.encode_backbone_sequence(images),
            view_valid,
            view_quality,
            global_time_position,
            motion,
        )


class _LocalCrossAttentionScorer(nn.Module):
    """Score visual time tokens through genuine local motion cross-attention.

    Unlike v2, the attended motion vector is never projected into the
    visual feature space.  It is used only to score whether an existing visual
    token should receive more or less weight during the unchanged student temporal
    pooling operation.
    """

    def __init__(
        self,
        motion_width: int,
        heads: int,
        maximum_logit: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if motion_width % heads:
            raise ValueError("motion width must be divisible by attention heads")
        if maximum_logit <= 0.0:
            raise ValueError("maximum cross-attention pooling logit must be positive")
        self.motion_width = int(motion_width)
        self.heads = int(heads)
        self.head_width = self.motion_width // self.heads
        self.maximum_logit = float(maximum_logit)
        self.query_projection = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, self.motion_width)
        )
        self.key_projection = nn.Sequential(
            nn.LayerNorm(self.motion_width),
            nn.Linear(self.motion_width, self.motion_width),
        )
        self.value_projection = nn.Sequential(
            nn.LayerNorm(self.motion_width),
            nn.Linear(self.motion_width, self.motion_width),
        )
        self.attended_projection = nn.Sequential(
            nn.Linear(self.motion_width, self.motion_width),
            nn.Dropout(dropout),
        )
        self.compatibility = nn.Sequential(
            nn.LayerNorm(self.motion_width * 4),
            nn.Linear(self.motion_width * 4, self.motion_width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.motion_width * 2, 1),
        )
        # The exact student boundary is the scalar pooling logit, not a feature
        # residual.  A zero logit gives uniform student temporal pooling.
        nn.init.zeros_(self.compatibility[4].weight)
        nn.init.zeros_(self.compatibility[4].bias)

    def forward(
        self,
        visual_sequence: torch.Tensor,
        visual_mask: torch.Tensor,
        motion_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
        temporal_radius: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, windows, views, visual_steps, _ = visual_sequence.shape
        motion_steps, parts, width = motion_tokens.shape[2:]
        if width != self.motion_width:
            raise ValueError("unexpected motion width")

        query_flat = self.query_projection(visual_sequence)
        key_flat = self.key_projection(motion_tokens)
        value_flat = self.value_projection(motion_tokens)
        query = query_flat.reshape(
            batch, windows, views, visual_steps, self.heads, self.head_width
        )
        key = key_flat.reshape(
            batch, windows, motion_steps, parts, self.heads, self.head_width
        )
        value = value_flat.reshape(
            batch, windows, motion_steps, parts, self.heads, self.head_width
        )
        score = torch.einsum("bwvthd,bwsphd->bwvthsp", query, key)
        score = score / math.sqrt(self.head_width)

        visual_index = torch.arange(visual_steps, device=score.device)
        motion_index = torch.arange(motion_steps, device=score.device)
        neighbourhood = (
            visual_index[:, None] - motion_index[None, :]
        ).abs() <= temporal_radius
        local_mask = motion_mask[:, :, None, None, None, :, :] & neighbourhood[
            None, None, None, :, None, :, None
        ]
        local_mask = local_mask.expand(-1, -1, views, -1, self.heads, -1, -1)
        flat_score = score.flatten(-2)
        flat_mask = local_mask.flatten(-2)
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=-1)
        safe_mask[..., 0] = safe_mask[..., 0] | empty
        attention = torch.softmax(
            flat_score.masked_fill(~safe_mask, -1e4), dim=-1
        )
        attention = attention * flat_mask.to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attention = attention.reshape(
            batch,
            windows,
            views,
            visual_steps,
            self.heads,
            motion_steps,
            parts,
        )
        attended = torch.einsum("bwvthsp,bwsphd->bwvthd", attention, value)
        attended = attended.reshape(
            batch, windows, views, visual_steps, self.motion_width
        )
        attended = self.attended_projection(attended)
        available = local_mask.any(dim=(-1, -2, -3)) & visual_mask
        attended = attended * available.to(attended.dtype).unsqueeze(-1)
        interaction = torch.cat(
            (
                query_flat,
                attended,
                query_flat * attended,
                (query_flat - attended).abs(),
            ),
            dim=-1,
        )
        pooling_logit = self.maximum_logit * torch.tanh(
            self.compatibility(interaction).squeeze(-1)
        )
        pooling_logit = pooling_logit * available.to(pooling_logit.dtype)

        mean_attention = attention.mean(dim=4)
        time_attention = mean_attention.sum(dim=-1)
        absolute_offset = (
            visual_index[:, None] - motion_index[None, :]
        ).abs().to(time_attention.dtype)
        mean_absolute_offset = torch.einsum(
            "bwvts,ts->bwvt", time_attention, absolute_offset
        )
        return pooling_logit, {
            "cross_attention": mean_attention,
            "available": available,
            "mean_absolute_offset": mean_absolute_offset,
        }


class TemporalCrossAttentionPoolStudent(nn.Module):
    """Condition student temporal pooling on local Visual/Skeleton/IMU agreement.

    This is the structural v3 test.  It keeps the complete trained student
    clip-local/global path frozen and changes only how the already encoded
    visual time tokens are pooled.  Two modality-private local multi-head
    cross-attention blocks produce bounded scalar compatibility logits.  These
    logits reweight the original student whole/early/late visual pooling; no motion
    feature vector is injected into visual features, no reliability gate is
    learned, and the original visual classifier remains the only classifier.
    """

    VALID_ABLATIONS = {"none", "zero", "reverse_time", "sample_roll"}

    def __init__(
        self,
        visual: MC3VisualStudent,
        motion_residual: MoBindMotionResidual,
        temporal_radius: int = 1,
        attention_heads: int = 4,
        maximum_pooling_logit: float = 1.0,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        if not visual.temporal_modeling:
            raise ValueError("temporal cross-attention requires visual time tokens")
        if temporal_radius < 0:
            raise ValueError("temporal radius must be non-negative")
        if motion_residual.modality != "separate" or not isinstance(
            motion_residual.encoder, SeparateMotionEncoder
        ):
            raise ValueError("v3 requires the separate Skeleton/IMU student anchor")
        self.visual = visual
        self.motion_residual = motion_residual
        self.temporal_radius = int(temporal_radius)
        self.maximum_pooling_logit = float(maximum_pooling_logit)
        self.temporal_ablation = "none"
        motion_width = int(motion_residual.encoder.width)
        self.skeleton_cross_attention = _LocalCrossAttentionScorer(
            motion_width, attention_heads, maximum_pooling_logit, dropout
        )
        self.imu_cross_attention = _LocalCrossAttentionScorer(
            motion_width, attention_heads, maximum_pooling_logit, dropout
        )

    def temporal_parameters(self) -> list[nn.Parameter]:
        return [
            *self.skeleton_cross_attention.parameters(),
            *self.imu_cross_attention.parameters(),
        ]

    def set_temporal_ablation(self, value: str) -> None:
        if value not in self.VALID_ABLATIONS:
            raise ValueError(
                f"temporal ablation must be one of {sorted(self.VALID_ABLATIONS)}"
            )
        self.temporal_ablation = value

    def _apply_counterfactual(
        self, tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.temporal_ablation == "reverse_time":
            return tokens.flip(2), token_mask.flip(2)
        if self.temporal_ablation == "sample_roll" and len(tokens) > 1:
            return tokens.roll(1, dims=0), token_mask.roll(1, dims=0)
        return tokens, token_mask

    def _pool_segment(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        pooling_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.visual._masked_mean(sequence, mask)
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        attention = torch.softmax(
            pooling_logit.masked_fill(~safe_mask, -1e4), dim=1
        )
        attention = attention * mask.to(attention.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
        uniform = mask.to(sequence.dtype)
        uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
        # Express the weighted pool as a delta around the exact student mean.  When
        # compatibility logits are zero, attention and uniform coincide.
        delta = torch.einsum("bt,btd->bd", attention - uniform, sequence)
        return base + delta, attention, (attention - uniform).abs().sum(dim=1)

    def _pool_temporal_sequence(
        self,
        sequence: torch.Tensor,
        time_mask: torch.Tensor,
        pooling_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if sequence.ndim != 5 or sequence.shape[1:3] != (2, 3):
            raise ValueError("visual sequence must be [B,2,3,T,512]")
        if time_mask.shape != sequence.shape[:-1]:
            raise ValueError("visual time mask differs")
        if pooling_logit.shape != time_mask.shape:
            raise ValueError("pooling logits must match the visual time grid")
        batch, windows, views, steps, width = sequence.shape
        flat_sequence = sequence.reshape(batch * windows * views, steps, width)
        flat_mask = time_mask.reshape(batch * windows * views, steps)
        flat_logit = pooling_logit.reshape(batch * windows * views, steps)
        whole, whole_attention, l1 = self._pool_segment(
            flat_sequence, flat_mask, flat_logit
        )
        midpoint = max(steps // 2, 1)
        early, _, _ = self._pool_segment(
            flat_sequence[:, :midpoint],
            flat_mask[:, :midpoint],
            flat_logit[:, :midpoint],
        )
        late, _, _ = self._pool_segment(
            flat_sequence[:, midpoint:],
            flat_mask[:, midpoint:],
            flat_logit[:, midpoint:],
        )
        assert self.visual.temporal_fusion is not None
        ordered = self.visual.temporal_fusion(
            torch.cat((whole, early, late, late - early), dim=1)
        )
        clips = (whole + ordered).reshape(batch, windows, views, width)
        return clips, {
            "temporal_pool_attention": whole_attention.reshape(
                batch, windows, views, steps
            ),
            "temporal_pool_weight_l1_from_uniform": l1.reshape(
                batch, windows, views
            ),
        }

    def _cross_modal_pool(
        self,
        visual_sequence: torch.Tensor,
        visual_mask: torch.Tensor,
        motion_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if motion_tokens.ndim != 5 or motion_mask.shape != motion_tokens.shape[:-1]:
            raise ValueError("motion tokens/mask must be [B,2,T,10,D]")
        if motion_tokens.shape[3] != 10:
            raise ValueError("v3 expects five Skeleton plus five IMU parts")
        if motion_tokens.shape[:3] != (
            visual_sequence.shape[0],
            visual_sequence.shape[1],
            visual_sequence.shape[3],
        ):
            raise ValueError("visual and motion time grids differ")
        motion_tokens, motion_mask = self._apply_counterfactual(
            motion_tokens, motion_mask
        )
        skeleton_logit, skeleton_audit = self.skeleton_cross_attention(
            visual_sequence,
            visual_mask,
            motion_tokens[:, :, :, :5],
            motion_mask[:, :, :, :5],
            self.temporal_radius,
        )
        imu_logit, imu_audit = self.imu_cross_attention(
            visual_sequence,
            visual_mask,
            motion_tokens[:, :, :, 5:],
            motion_mask[:, :, :, 5:],
            self.temporal_radius,
        )
        available = torch.stack(
            (skeleton_audit["available"], imu_audit["available"]), dim=-1
        )
        modality_weight = available.to(visual_sequence.dtype)
        modality_weight = modality_weight / modality_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        pooling_logit = (
            skeleton_logit * modality_weight[..., 0]
            + imu_logit * modality_weight[..., 1]
        )
        if self.temporal_ablation == "zero":
            pooling_logit = torch.zeros_like(pooling_logit)
        clips, pool_audit = self._pool_temporal_sequence(
            visual_sequence, visual_mask, pooling_logit
        )
        return clips, {
            "temporal_skeleton_cross_attention": skeleton_audit[
                "cross_attention"
            ],
            "temporal_imu_cross_attention": imu_audit["cross_attention"],
            "temporal_skeleton_mean_abs_offset": skeleton_audit[
                "mean_absolute_offset"
            ],
            "temporal_imu_mean_abs_offset": imu_audit["mean_absolute_offset"],
            "temporal_motion_available": available,
            "temporal_modality_weight": modality_weight,
            "temporal_pool_logit": pooling_logit,
            **pool_audit,
        }

    def forward_from_backbone_sequence(
        self,
        backbone_sequence: torch.Tensor,
        view_valid: torch.Tensor,
        view_quality: torch.Tensor,
        global_time_position: torch.Tensor,
        motion: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        visual_sequence, visual_mask = (
            self.visual.encode_temporal_sequence_from_backbone(
                backbone_sequence, view_valid, global_time_position
            )
        )
        visual_clips = self.visual.pool_temporal_sequence(
            visual_sequence, visual_mask
        )
        visual_only = self.visual._fuse_clips(
            visual_clips, view_valid, view_quality
        )

        anchor_clips, audit = self.motion_residual(
            visual_clips, global_time_position, motion
        )
        motion_tokens = audit.pop("motion_tokens")
        motion_mask = audit.pop("motion_token_mask")
        cross_clips, temporal_audit = self._cross_modal_pool(
            visual_sequence, visual_mask, motion_tokens, motion_mask
        )
        clip_effect = cross_clips - visual_clips
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
        temporal_audit["temporal_effect_rms_ratio"] = (
            clip_effect.square().mean(dim=-1).sqrt()
            / visual_clips.square().mean(dim=-1).sqrt().clamp_min(1e-6)
        )
        audit.update(temporal_audit)
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
        return self.forward_from_backbone_sequence(
            self.visual.encode_backbone_sequence(images),
            view_valid,
            view_quality,
            global_time_position,
            motion,
        )
