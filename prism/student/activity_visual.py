"""MMACT RGB ROI adaptation of the exact native temporal visual student."""

from __future__ import annotations

import math

import torch
from torch import nn

from .native.constants import KINETICS_MEAN, KINETICS_STD
from .native.visual import NativeMC3VisualStudent


class ActivityNativeVisualStudent(NativeMC3VisualStudent):
    """native visual graph adapted from three grayscale IR to three RGB ROIs.

    native's source ``view`` axis means ROI type, not camera. The faithful default
    remains scene/person/workspace and therefore six early/late clips. A caller
    may configure another region count only as an explicit ablation/utility.
    The MC3 trunk and all temporal/fusion heads are inherited; no whole-video
    feature pooling or reduced visual head is introduced.
    """

    def __init__(
        self,
        classes: int = 37,
        views: int = 3,
        region_names: tuple[str, ...] = ("scene", "person", "workspace"),
        frames: int = 16,
        dropout: float = 0.18,
        kinetics_pretrained: bool = True,
    ) -> None:
        if views <= 0:
            raise ValueError("views must be positive")
        if len(region_names) != views or len(set(region_names)) != views:
            raise ValueError("region_names must uniquely identify every ROI token")
        super().__init__(
            classes=classes,
            width=512,
            dropout=dropout,
            fusion_mode="gated",
            enable_distillation_projection=False,
            frames=frames,
            kinetics_pretrained=kinetics_pretrained,
            temporal_modeling=True,
            exact_time_modeling=False,
            cross_view_time_modeling=False,
            spatial_region_modeling=False,
            region_temporal_modeling=False,
            structured_region_modeling=False,
        )
        self.views = int(views)
        self.region_names = tuple(region_names)
        self.view_embedding = nn.Parameter(torch.zeros(self.views, 512))
        nn.init.trunc_normal_(self.view_embedding, std=0.02)

    def _encode_backbone_map(self, images: torch.Tensor) -> torch.Tensor:
        expected = (2, self.frames, self.views, 3)
        if images.ndim != 7 or tuple(images.shape[1:5]) != expected:
            raise ValueError(
                "expected RGB images "
                f"[B,2,{self.frames},{self.views},3,H,W], got {tuple(images.shape)}"
            )
        batch, windows, steps, views, channels, height, width = images.shape
        clips = images.permute(0, 1, 3, 4, 2, 5, 6).reshape(
            batch * windows * views, channels, steps, height, width
        )
        value = clips.float()
        # uint8/cached 0..255 tensors follow the source model's scaling; already
        # normalized floating tensors are deliberately not accepted implicitly.
        if images.dtype == torch.uint8:
            value = value.div(255.0)
        elif value.numel() and (value.detach().amin() < 0 or value.detach().amax() > 1):
            raise ValueError("floating RGB input must be in [0,1]")
        mean = value.new_tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1)
        std = value.new_tensor(KINETICS_STD).view(1, 3, 1, 1, 1)
        value = (value - mean) / std
        value = self.stem(value)
        value = self.layer1(value)
        value = self.layer2(value)
        value = self.layer3(value)
        value = self.layer4(value)
        return value.reshape(batch, windows, views, *value.shape[1:])

    def encode_backbone_sequence(self, images: torch.Tensor) -> torch.Tensor:
        value = self._encode_backbone_map(images)
        batch, windows, views = value.shape[:3]
        sequence = value.mean(dim=(-1, -2)).permute(0, 1, 2, 4, 3)
        sequence = sequence.reshape(batch * windows * views, sequence.shape[-2], 512)
        if sequence.shape[1] != self.frames:
            sequence = nn.functional.interpolate(
                sequence.transpose(1, 2), self.frames, mode="linear", align_corners=False
            ).transpose(1, 2)
        return sequence.reshape(batch, windows, views, self.frames, 512)

    def encode_temporal_sequence_from_backbone(
        self,
        backbone_sequence: torch.Tensor,
        view_valid: torch.Tensor | None = None,
        global_time_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (2, self.views, self.frames, 512)
        if backbone_sequence.ndim != 5 or tuple(backbone_sequence.shape[1:]) != expected:
            raise ValueError(f"backbone_sequence must be [B,2,{self.views},{self.frames},512]")
        if view_valid is None or view_valid.shape != backbone_sequence.shape[:-1]:
            raise ValueError("view_valid must match [B,2,views,frames]")
        if global_time_position is not None and global_time_position.shape != (
            len(backbone_sequence),
            2,
            self.frames,
        ):
            raise ValueError("global_time_position must be [B,2,frames]")
        batch = len(backbone_sequence)
        sequence = backbone_sequence.reshape(batch * 2 * self.views, self.frames, 512)
        time_mask = view_valid.reshape(batch * 2 * self.views, self.frames)
        safe_mask = time_mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        assert self.temporal_encoder is not None
        sequence = self.temporal_encoder(
            sequence + self.time_position, src_key_padding_mask=~safe_mask
        )
        sequence = sequence * time_mask.to(sequence.dtype).unsqueeze(-1)
        return (
            sequence.reshape(batch, 2, self.views, self.frames, 512),
            time_mask.reshape(batch, 2, self.views, self.frames),
        )

    def pool_temporal_sequence(
        self, sequence: torch.Tensor, time_mask: torch.Tensor
    ) -> torch.Tensor:
        expected = (2, self.views, self.frames, 512)
        if sequence.ndim != 5 or tuple(sequence.shape[1:]) != expected:
            raise ValueError(f"sequence must be [B,2,{self.views},{self.frames},512]")
        if time_mask.shape != sequence.shape[:-1]:
            raise ValueError("time_mask must match temporal sequence")
        batch = len(sequence)
        sequence = sequence.reshape(batch * 2 * self.views, self.frames, 512)
        time_mask = time_mask.reshape(batch * 2 * self.views, self.frames)
        whole = self._masked_mean(sequence, time_mask)
        midpoint = max(self.frames // 2, 1)
        early = self._masked_mean(sequence[:, :midpoint], time_mask[:, :midpoint])
        late = self._masked_mean(sequence[:, midpoint:], time_mask[:, midpoint:])
        assert self.temporal_fusion is not None
        ordered = self.temporal_fusion(torch.cat((whole, early, late, late - early), dim=1))
        return (whole + ordered).reshape(batch, 2, self.views, 512)

    def _fuse_clips(
        self,
        clip_embeddings: torch.Tensor,
        view_valid: torch.Tensor,
        view_quality: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = len(clip_embeddings)
        if clip_embeddings.shape != (batch, 2, self.views, 512):
            raise ValueError(f"clip_embeddings must be [B,2,{self.views},512]")
        expected_mask = (batch, 2, self.views, self.frames)
        if view_valid.shape != expected_mask or view_quality.shape != expected_mask:
            raise ValueError("validity and quality must be [B,2,views,frames]")
        clips = (
            clip_embeddings
            + self.window_embedding.view(1, 2, 1, 512)
            + self.view_embedding.view(1, 1, self.views, 512)
        )
        clip_mask = view_valid.any(dim=3)
        clip_quality = view_quality.mean(dim=3)
        safe = clip_mask.reshape(batch, 2 * self.views).clone()
        all_missing = ~safe.any(dim=1)
        safe[all_missing, 0] = True
        encoded = self.token_encoder(
            clips.reshape(batch, 2 * self.views, 512), src_key_padding_mask=~safe
        ).reshape(batch, 2, self.views, 512)
        encoded = encoded * clip_mask.unsqueeze(-1)
        gate_input = torch.cat(
            (encoded, clip_quality.unsqueeze(-1), clip_mask.to(encoded.dtype).unsqueeze(-1)),
            dim=-1,
        )
        gate = self.quality_gate(gate_input).squeeze(-1).masked_fill(~clip_mask, -1e4)
        view_weight = torch.softmax(gate, dim=2) * clip_mask
        view_weight = view_weight / view_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
        window_tokens = (encoded * view_weight.unsqueeze(-1)).sum(dim=2)
        early, late = window_tokens[:, 0], window_tokens[:, 1]
        embedding = self.stage_fusion(
            torch.cat((early, late, 0.5 * (early + late), late - early), dim=1)
        )
        output = {
            "logits": self.classifier(embedding),
            "visual_embedding": embedding,
            "clip_embeddings": encoded,
            "clip_mask": clip_mask,
            "window_embeddings": window_tokens,
            "view_weight": view_weight,
        }
        if self.training:
            output["stage_logits"] = torch.stack(
                (self.classifier(early), self.classifier(late), self.classifier(late - early)),
                dim=1,
            )
        return output
