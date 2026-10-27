from __future__ import annotations

import math

import torch
from torch import nn

from skeleton_imu_preprocessing import PART_JOINTS
from mc3_visual_model import MC3VisualStudent


class AlignedMotionResidual(nn.Module):
    """Inject aligned Skeleton/IMU evidence before MC3 temporal pooling.

    Motion is attended at the same window and time position as each visual token.
    Missing motion produces an exact zero residual. The residual starts at 1% and
    is capped at 25%, which lets motion learn without replacing the visual anchor.
    """

    def __init__(
        self,
        visual_width: int = 512,
        motion_width: int = 128,
        dropout: float = 0.12,
        use_skeleton: bool = True,
        use_imu: bool = True,
    ) -> None:
        super().__init__()
        if not use_skeleton and not use_imu:
            raise ValueError("at least one motion modality must be enabled")
        self.visual_width = int(visual_width)
        self.motion_width = int(motion_width)
        self.use_skeleton = bool(use_skeleton)
        self.use_imu = bool(use_imu)
        self.visual_query = nn.Sequential(
            nn.LayerNorm(visual_width), nn.Linear(visual_width, motion_width)
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, motion_width), nn.GELU(), nn.Linear(motion_width, motion_width)
        )
        if self.use_skeleton:
            self.skeleton_joint_encoder = nn.Sequential(
                nn.LayerNorm(27),
                nn.Linear(27, 96),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.skeleton_part_projection = nn.Sequential(
                nn.LayerNorm(192), nn.Linear(192, motion_width), nn.GELU()
            )
            self.skeleton_relation_encoder = nn.Sequential(
                nn.LayerNorm(37),
                nn.Linear(37, motion_width),
                nn.GELU(),
            )
            self.skeleton_part_embedding = nn.Parameter(
                torch.zeros(len(PART_JOINTS), motion_width)
            )
            nn.init.trunc_normal_(self.skeleton_part_embedding, std=0.02)
        if self.use_imu:
            self.imu_point_encoder = nn.Sequential(
                nn.LayerNorm(17),
                nn.Linear(17, 96),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.imu_bin_encoder = nn.Sequential(
                nn.LayerNorm(245),
                nn.Linear(245, motion_width),
                nn.GELU(),
            )
            self.imu_global_encoder = nn.Sequential(
                nn.LayerNorm(50), nn.Linear(50, motion_width), nn.GELU()
            )
            self.imu_device_embedding = nn.Parameter(torch.zeros(5, motion_width))
            nn.init.trunc_normal_(self.imu_device_embedding, std=0.02)

        self.cross_attention = nn.MultiheadAttention(
            motion_width, num_heads=4, dropout=dropout, batch_first=True
        )
        self.motion_projection = nn.Sequential(
            nn.LayerNorm(motion_width),
            nn.Linear(motion_width, visual_width),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.motion_projection[1].weight)
        nn.init.zeros_(self.motion_projection[1].bias)
        # 0.25 * sigmoid(logit(0.04)) = 0.01 initial residual strength.
        self.residual_logit = nn.Parameter(torch.tensor(math.log(0.04 / 0.96)))

    @staticmethod
    def _masked_mean_max(
        values: torch.Tensor, mask: torch.Tensor, dimension: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = mask.to(values.dtype).unsqueeze(-1)
        mean = (values * weight).sum(dim=dimension) / weight.sum(dim=dimension).clamp_min(
            1.0
        )
        maximum = values.masked_fill(~mask.unsqueeze(-1), -1e4).amax(dim=dimension)
        mask_dimension = dimension if dimension >= 0 else dimension + 1
        maximum = torch.where(
            mask.any(dim=mask_dimension).unsqueeze(-1), maximum, 0.0
        )
        return mean, maximum

    def encode_skeleton(
        self,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        joint_mask: torch.Tensor,
        relations: torch.Tensor,
        relation_mask: torch.Tensor,
        frame_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-2:] != (17, 13):
            raise ValueError("skeleton_features must end with [17,13]")
        quality = frame_quality[..., None, None].expand(*features.shape[:-1], 1)
        joint_input = torch.cat(
            (features, feature_mask.to(features.dtype), quality), dim=-1
        )
        encoded_joints = self.skeleton_joint_encoder(joint_input)
        part_tokens = []
        part_masks = []
        for part_index, indices in enumerate(PART_JOINTS):
            selected = encoded_joints[..., list(indices), :]
            selected_mask = joint_mask[..., list(indices)]
            mean, maximum = self._masked_mean_max(selected, selected_mask, -2)
            token = self.skeleton_part_projection(torch.cat((mean, maximum), dim=-1))
            token = token + self.skeleton_part_embedding[part_index]
            part_tokens.append(token)
            part_masks.append(selected_mask.any(dim=-1))
        tokens = torch.stack(part_tokens, dim=-2)
        masks = torch.stack(part_masks, dim=-1)
        relation_input = torch.cat(
            (
                relations,
                relation_mask.to(relations.dtype),
                frame_quality.unsqueeze(-1),
            ),
            dim=-1,
        )
        # The global part is the only token that receives trial-level relations.
        relation_token = self.skeleton_relation_encoder(relation_input)
        tokens = torch.cat(
            (tokens[..., :1, :] + relation_token.unsqueeze(-2), tokens[..., 1:, :]),
            dim=-2,
        )
        return tokens, masks

    def encode_imu(
        self,
        sequences: torch.Tensor,
        sequence_mask: torch.Tensor,
        bin_statistics: torch.Tensor,
        bin_mask: torch.Tensor,
        global_statistics: torch.Tensor,
        global_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequences.shape[-3] != 5 or sequences.shape[-1] != 16:
            raise ValueError("imu_sequences must end with [5,P,16]")
        points = sequences.shape[-2]
        position = torch.linspace(
            0.0, 1.0, points, device=sequences.device, dtype=sequences.dtype
        )
        position = position.view(*([1] * (sequences.ndim - 2)), points, 1)
        position = position.expand(*sequences.shape[:-1], 1)
        point_encoded = self.imu_point_encoder(torch.cat((sequences, position), dim=-1))
        mean, maximum = self._masked_mean_max(point_encoded, sequence_mask, -2)
        token_input = torch.cat(
            (
                mean,
                maximum,
                bin_statistics,
                bin_mask.to(sequences.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        tokens = self.imu_bin_encoder(token_input)
        global_input = torch.cat((global_statistics, global_mask), dim=-1)
        global_token = self.imu_global_encoder(global_input)
        # Broadcast full-trial device identity/statistics over both visual windows.
        while global_token.ndim < tokens.ndim:
            global_token = global_token.unsqueeze(1)
        tokens = tokens + global_token + self.imu_device_embedding
        return tokens, bin_mask

    def forward(
        self,
        visual_sequence: torch.Tensor,
        global_time_position: torch.Tensor,
        motion: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if visual_sequence.ndim != 5 or visual_sequence.shape[1:3] != (2, 3):
            raise ValueError("visual_sequence must be [B,2,3,T,C]")
        batch, windows, views, steps, width = visual_sequence.shape
        if width != self.visual_width:
            raise ValueError("visual width differs")
        if global_time_position.shape != (batch, windows, steps):
            raise ValueError("global_time_position must be [B,2,T]")
        time = self.time_encoder(global_time_position.unsqueeze(-1))
        token_groups = []
        mask_groups = []
        if self.use_skeleton:
            tokens, masks = self.encode_skeleton(
                motion["skeleton_features"],
                motion["skeleton_feature_mask"],
                motion["skeleton_joint_mask"],
                motion["skeleton_relations"],
                motion["skeleton_relation_mask"],
                motion["skeleton_frame_quality"],
            )
            token_groups.append(tokens)
            mask_groups.append(masks)
        if self.use_imu:
            tokens, masks = self.encode_imu(
                motion["imu_sequences"],
                motion["imu_sequence_mask"],
                motion["imu_bin_statistics"],
                motion["imu_bin_mask"],
                motion["imu_global_statistics"],
                motion["imu_global_mask"],
            )
            token_groups.append(tokens)
            mask_groups.append(masks)
        motion_tokens = torch.cat(token_groups, dim=-2) + time.unsqueeze(-2)
        motion_mask = torch.cat(mask_groups, dim=-1)
        query = self.visual_query(visual_sequence.permute(0, 1, 3, 2, 4))
        query = query + time.unsqueeze(-2)
        flat_query = query.reshape(batch * windows * steps, views, self.motion_width)
        flat_tokens = motion_tokens.reshape(
            batch * windows * steps, motion_tokens.shape[-2], self.motion_width
        )
        flat_mask = motion_mask.reshape(batch * windows * steps, motion_mask.shape[-1])
        any_motion = flat_mask.any(dim=1)
        safe_mask = flat_mask.clone()
        safe_mask[~any_motion, 0] = True
        attended, attention = self.cross_attention(
            flat_query,
            flat_tokens,
            flat_tokens,
            key_padding_mask=~safe_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = attended * any_motion[:, None, None]
        residual = self.motion_projection(attended).reshape(
            batch, windows, steps, views, width
        )
        residual = residual * any_motion.reshape(batch, windows, steps, 1, 1)
        residual = residual.permute(0, 1, 3, 2, 4)
        strength = 0.25 * torch.sigmoid(self.residual_logit)
        fused = visual_sequence + strength * residual
        return fused, {
            "motion_attention": attention,
            "motion_residual_strength": strength,
            "motion_available": any_motion.reshape(batch, windows, steps),
        }


class UnifiedVisualMotionStudent(nn.Module):
    """One MC3 visual backbone, aligned motion residual and one 40-class head."""

    def __init__(
        self,
        visual: MC3VisualStudent,
        use_skeleton: bool = True,
        use_imu: bool = True,
        motion_width: int = 128,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        if not visual.temporal_modeling:
            raise ValueError("unified fusion requires the temporal MC3 visual anchor")
        self.visual = visual
        self.motion_residual = AlignedMotionResidual(
            visual_width=512,
            motion_width=motion_width,
            dropout=dropout,
            use_skeleton=use_skeleton,
            use_imu=use_imu,
        )

    def forward_from_backbone_sequence(
        self,
        backbone_sequence: torch.Tensor,
        view_valid: torch.Tensor,
        view_quality: torch.Tensor,
        global_time_position: torch.Tensor,
        motion: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        fused, audit = self.motion_residual(
            backbone_sequence, global_time_position, motion
        )
        output = self.visual.forward_from_backbone_sequence(
            fused, view_valid, view_quality, global_time_position
        )
        if self.training and "stage_logits" not in output:
            early, late = output["window_embeddings"][:, 0], output["window_embeddings"][:, 1]
            output["stage_logits"] = torch.stack(
                (
                    self.visual.classifier(early),
                    self.visual.classifier(late),
                    self.visual.classifier(late - early),
                ),
                dim=1,
            )
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
