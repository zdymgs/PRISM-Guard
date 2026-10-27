from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import math
import torch
from torch import nn
from torchvision.models.video import MC3_18_Weights, mc3_18
from session_kd.mmact_backbones.constants import KINETICS_MEAN, KINETICS_STD

class MC3VisualStudent(nn.Module):
    """Shared Kinetics-pretrained MC3-18 over six early/late IR clips.

    The model is deliberately interface-compatible with the student pixel student
    so the backbone is the only material change in the nested visual trial.
    Large VideoMAE features and logits remain training targets, never inputs.
    """

    def __init__(self, classes: int=40, width: int=512, dropout: float=0.18, fusion_mode: str='gated', enable_distillation_projection: bool=False, frames: int=12, kinetics_pretrained: bool=True, temporal_modeling: bool=False, exact_time_modeling: bool=False, cross_view_time_modeling: bool=False, spatial_region_modeling: bool=False, region_temporal_modeling: bool=False, structured_region_modeling: bool=False) -> None:
        super().__init__()
        if width != 512:
            raise ValueError('MC3-18 output width is fixed at 512')
        if fusion_mode != 'gated':
            raise ValueError('the controlled MC3 trial supports gated fusion only')
        self.fusion_mode = fusion_mode
        self.frames = int(frames)
        self.temporal_modeling = bool(temporal_modeling)
        self.exact_time_modeling = bool(exact_time_modeling)
        self.cross_view_time_modeling = bool(cross_view_time_modeling)
        self.spatial_region_modeling = bool(spatial_region_modeling)
        self.region_temporal_modeling = bool(region_temporal_modeling)
        self.structured_region_modeling = bool(structured_region_modeling)
        if self.exact_time_modeling and (not self.temporal_modeling):
            raise ValueError('exact source time requires temporal MC3 modeling')
        if self.cross_view_time_modeling and (not self.temporal_modeling):
            raise ValueError('same-time cross-view fusion requires temporal MC3 modeling')
        if self.spatial_region_modeling and (not self.temporal_modeling):
            raise ValueError('spatial region tokens require temporal MC3 modeling')
        if self.region_temporal_modeling and (not self.spatial_region_modeling):
            raise ValueError('region temporal modeling requires spatial region tokens')
        if self.structured_region_modeling and (not self.spatial_region_modeling):
            raise ValueError('structured region modeling requires spatial region tokens')
        if self.structured_region_modeling and self.region_temporal_modeling:
            raise ValueError('structured and region-temporal screens must remain controlled')
        self.freeze_through = 'none'
        video = mc3_18(weights=MC3_18_Weights.KINETICS400_V1 if kinetics_pretrained else None)
        self.stem = video.stem
        self.layer1 = video.layer1
        self.layer2 = video.layer2
        self.layer3 = video.layer3
        self.layer4 = video.layer4
        if self.temporal_modeling:
            self.time_position = nn.Parameter(torch.zeros(1, self.frames, width))
            nn.init.trunc_normal_(self.time_position, std=0.02)
            temporal_layer = nn.TransformerEncoderLayer(d_model=width, nhead=8, dim_feedforward=width * 2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=1)
            self.temporal_fusion = nn.Sequential(nn.LayerNorm(width * 4), nn.Linear(width * 4, width), nn.GELU(), nn.Dropout(dropout))
            self.exact_time_projection = nn.Sequential(nn.LayerNorm(9), nn.Linear(9, width), nn.GELU(), nn.Linear(width, width)) if self.exact_time_modeling else None
            if self.exact_time_projection is not None:
                nn.init.zeros_(self.exact_time_projection[3].weight)
                nn.init.zeros_(self.exact_time_projection[3].bias)
            if self.cross_view_time_modeling:
                cross_view_layer = nn.TransformerEncoderLayer(d_model=width, nhead=8, dim_feedforward=width * 2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
                self.same_time_view_encoder = nn.TransformerEncoder(cross_view_layer, num_layers=1)
                self.same_time_view_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
                nn.init.zeros_(self.same_time_view_projection[1].weight)
                nn.init.zeros_(self.same_time_view_projection[1].bias)
            else:
                self.same_time_view_encoder = None
                self.same_time_view_projection = None
            if self.spatial_region_modeling:
                if self.structured_region_modeling:
                    self.register_parameter('spatial_region_embedding', None)
                    self.spatial_region_encoder = None
                    self.structured_region_projection = nn.Sequential(nn.LayerNorm(width * 3), nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, width))
                    nn.init.zeros_(self.structured_region_projection[3].weight)
                    nn.init.zeros_(self.structured_region_projection[3].bias)
                    self.spatial_region_projection = None
                else:
                    self.spatial_region_embedding = nn.Parameter(torch.zeros(4, width))
                    nn.init.trunc_normal_(self.spatial_region_embedding, std=0.02)
                    spatial_layer = nn.TransformerEncoderLayer(d_model=width, nhead=8, dim_feedforward=width * 2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
                    self.spatial_region_encoder = nn.TransformerEncoder(spatial_layer, num_layers=1)
                    self.structured_region_projection = None
                    self.spatial_region_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
                    nn.init.zeros_(self.spatial_region_projection[1].weight)
                    nn.init.zeros_(self.spatial_region_projection[1].bias)
                if self.region_temporal_modeling:
                    region_temporal_layer = nn.TransformerEncoderLayer(d_model=width, nhead=8, dim_feedforward=width * 2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
                    self.region_temporal_encoder = nn.TransformerEncoder(region_temporal_layer, num_layers=1)
                else:
                    self.region_temporal_encoder = None
            else:
                self.register_parameter('spatial_region_embedding', None)
                self.spatial_region_encoder = None
                self.region_temporal_encoder = None
                self.spatial_region_projection = None
                self.structured_region_projection = None
        else:
            self.register_parameter('time_position', None)
            self.temporal_encoder = None
            self.temporal_fusion = None
            self.exact_time_projection = None
            self.same_time_view_encoder = None
            self.same_time_view_projection = None
            self.register_parameter('spatial_region_embedding', None)
            self.spatial_region_encoder = None
            self.region_temporal_encoder = None
            self.spatial_region_projection = None
            self.structured_region_projection = None
        self.view_embedding = nn.Parameter(torch.zeros(3, width))
        self.window_embedding = nn.Parameter(torch.zeros(2, width))
        nn.init.trunc_normal_(self.view_embedding, std=0.02)
        nn.init.trunc_normal_(self.window_embedding, std=0.02)
        token_layer = nn.TransformerEncoderLayer(d_model=width, nhead=8, dim_feedforward=width * 2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.token_encoder = nn.TransformerEncoder(token_layer, num_layers=1)
        self.quality_gate = nn.Sequential(nn.LayerNorm(width + 2), nn.Linear(width + 2, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
        self.stage_fusion = nn.Sequential(nn.LayerNorm(width * 4), nn.Linear(width * 4, width), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, classes))
        self.distillation_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1024)) if enable_distillation_projection else None

    def backbone_parameters(self) -> list[nn.Parameter]:
        modules = (self.stem, self.layer1, self.layer2, self.layer3, self.layer4)
        return [parameter for module in modules for parameter in module.parameters()]

    def head_parameters(self) -> list[nn.Parameter]:
        backbone_ids = {id(parameter) for parameter in self.backbone_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in backbone_ids]

    def freeze_low_level(self, freeze_through: str='layer2') -> None:
        if freeze_through not in {'layer1', 'layer2', 'layer3'}:
            raise ValueError(f'unknown freeze boundary: {freeze_through}')
        self.freeze_through = freeze_through
        modules = [self.stem, self.layer1]
        if freeze_through in {'layer2', 'layer3'}:
            modules.append(self.layer2)
        if freeze_through == 'layer3':
            modules.append(self.layer3)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self._freeze_batch_norm_statistics()

    def _freeze_batch_norm_statistics(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eval()

    def train(self, mode: bool=True) -> 'MC3VisualStudent':
        super().train(mode)
        if mode:
            self._freeze_batch_norm_statistics()
        return self

    @staticmethod
    def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(sequence.dtype).unsqueeze(-1)
        return (sequence * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def _encode_backbone_map(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 6 or images.shape[1:4] != (2, self.frames, 3):
            raise ValueError(f'expected [B,2,{self.frames},3,H,W], got {tuple(images.shape)}')
        batch, windows, steps, views, height, width = images.shape
        clips = images.permute(0, 1, 3, 2, 4, 5).reshape(batch * windows * views, steps, 1, height, width)
        unit = clips.float().div_(255.0)
        rgb = unit.repeat(1, 1, 3, 1, 1).permute(0, 2, 1, 3, 4)
        mean = rgb.new_tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1)
        std = rgb.new_tensor(KINETICS_STD).view(1, 3, 1, 1, 1)
        value = (rgb - mean) / std
        value = self.stem(value)
        value = self.layer1(value)
        value = self.layer2(value)
        value = self.layer3(value)
        value = self.layer4(value)
        return value.reshape(batch, windows, views, *value.shape[1:])

    def encode_backbone_sequence(self, images: torch.Tensor) -> torch.Tensor:
        """Return the compact layer4 sequence used by all trainable student heads."""
        value = self._encode_backbone_map(images)
        batch, windows, views = value.shape[:3]
        sequence = value.mean(dim=(-1, -2)).permute(0, 1, 2, 4, 3)
        sequence = sequence.reshape(batch * windows * views, sequence.shape[-2], 512)
        if sequence.shape[1] != self.frames:
            sequence = nn.functional.interpolate(sequence.transpose(1, 2), self.frames, mode='linear', align_corners=False).transpose(1, 2)
        return sequence.reshape(batch, windows, views, self.frames, 512)

    def encode_backbone_spatial_sequences(self, images: torch.Tensor, spatial_grid: int=2) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact global and ordered layer4 spatial sequences in one pass."""
        if spatial_grid <= 1:
            raise ValueError('spatial_grid must be greater than one')
        value = self._encode_backbone_map(images)
        batch, windows, views, channels, steps = value.shape[:5]
        compact = value.mean(dim=(-1, -2)).permute(0, 1, 2, 4, 3)
        compact = compact.reshape(batch * windows * views, compact.shape[-2], 512)
        if compact.shape[1] != self.frames:
            compact = nn.functional.interpolate(compact.transpose(1, 2), self.frames, mode='linear', align_corners=False).transpose(1, 2)
        compact = compact.reshape(batch, windows, views, self.frames, 512)
        value = value.reshape(batch * windows * views, channels, steps, *value.shape[-2:])
        regions = nn.functional.adaptive_avg_pool3d(value, output_size=(self.frames, spatial_grid, spatial_grid))
        regions = regions.flatten(-2).permute(0, 2, 3, 1)
        regions = regions.reshape(batch, windows, views, self.frames, spatial_grid * spatial_grid, 512)
        return (compact, regions)

    def encode_backbone_region_sequence(self, images: torch.Tensor, spatial_grid: int=2) -> torch.Tensor:
        """Return ordered layer4 spatial regions at every retained time step."""
        return self.encode_backbone_spatial_sequences(images, spatial_grid)[1]

    def encode_spatial_regions(self, region_sequence: torch.Tensor, view_valid: torch.Tensor | None=None) -> torch.Tensor:
        if region_sequence.ndim != 6 or region_sequence.shape[1:] != (2, 3, self.frames, 4, 512):
            raise ValueError(f'region_sequence must have shape [B,2,3,{self.frames},4,512], got {tuple(region_sequence.shape)}')
        if not self.spatial_region_modeling:
            raise ValueError('model was not configured for spatial region tokens')
        batch = region_sequence.shape[0]
        global_sequence = region_sequence.mean(dim=4)
        if self.structured_region_modeling:
            assert self.structured_region_projection is not None
            top = 0.5 * (region_sequence[..., 0, :] + region_sequence[..., 1, :])
            bottom = 0.5 * (region_sequence[..., 2, :] + region_sequence[..., 3, :])
            left = 0.5 * (region_sequence[..., 0, :] + region_sequence[..., 2, :])
            right = 0.5 * (region_sequence[..., 1, :] + region_sequence[..., 3, :])
            diagonal = 0.5 * (region_sequence[..., 0, :] + region_sequence[..., 3, :] - region_sequence[..., 1, :] - region_sequence[..., 2, :])
            contrast = torch.cat((bottom - top, right - left, diagonal), dim=-1)
            return global_sequence + self.structured_region_projection(contrast)
        assert self.spatial_region_embedding is not None
        assert self.spatial_region_encoder is not None
        assert self.spatial_region_projection is not None
        if self.region_temporal_modeling:
            if view_valid is None or view_valid.shape != (batch, 2, self.frames, 3):
                raise ValueError(f'region temporal modeling requires view_valid [B,2,{self.frames},3]')
            assert self.region_temporal_encoder is not None
            local_trajectory = region_sequence.permute(0, 1, 2, 4, 3, 5).reshape(batch * 2 * 3 * 4, self.frames, 512)
            time_mask = view_valid.permute(0, 1, 3, 2)
            time_mask = time_mask.unsqueeze(3).expand(-1, -1, -1, 4, -1).reshape(batch * 2 * 3 * 4, self.frames)
            safe_mask = time_mask.clone()
            empty = ~safe_mask.any(dim=1)
            safe_mask[empty, 0] = True
            region_position = self.spatial_region_embedding.view(1, 4, 1, 512)
            region_position = region_position.expand(batch * 6, -1, -1, -1).reshape(batch * 2 * 3 * 4, 1, 512)
            local_trajectory = self.region_temporal_encoder(local_trajectory + self.time_position + region_position, src_key_padding_mask=~safe_mask)
            local_trajectory = local_trajectory * time_mask.to(local_trajectory.dtype).unsqueeze(-1)
            region_sequence = local_trajectory.reshape(batch, 2, 3, 4, self.frames, 512).permute(0, 1, 2, 4, 3, 5)
        region_tokens = region_sequence.reshape(-1, 4, 512)
        encoded = self.spatial_region_encoder(region_tokens + self.spatial_region_embedding.unsqueeze(0))
        residual = self.spatial_region_projection(encoded.mean(dim=1)).reshape(batch, 2, 3, self.frames, 512)
        return global_sequence + residual

    def encode_temporal_sequence_from_backbone(self, backbone_sequence: torch.Tensor, view_valid: torch.Tensor | None=None, global_time_position: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return post-temporal, pre-pooling visual tokens and their mask.

        This is the information boundary needed by fusion temporal MoBind.  The
        method deliberately keeps the original temporal encoder unchanged so
        the clip-level student path can be expressed as the same encoding followed
        by :meth:`pool_temporal_sequence`.
        """
        if backbone_sequence.ndim != 5 or backbone_sequence.shape[1:] != (2, 3, self.frames, 512):
            raise ValueError(f'backbone_sequence must have shape [B,2,3,{self.frames},512], got {tuple(backbone_sequence.shape)}')
        if not self.temporal_modeling:
            raise RuntimeError('temporal sequence access requires temporal_modeling')
        if view_valid is None:
            raise ValueError('temporal MC3 requires per-frame view_valid')
        batch = backbone_sequence.shape[0]
        assert self.temporal_encoder is not None and self.temporal_fusion is not None
        if self.cross_view_time_modeling:
            assert self.same_time_view_encoder is not None
            assert self.same_time_view_projection is not None
            same_time = backbone_sequence.permute(0, 1, 3, 2, 4).reshape(batch * 2 * self.frames, 3, 512)
            same_time_mask = view_valid.reshape(batch * 2 * self.frames, 3)
            safe_view_mask = same_time_mask.clone()
            empty_view = ~safe_view_mask.any(dim=1)
            safe_view_mask[empty_view, 0] = True
            mixed_view = self.same_time_view_encoder(same_time + self.view_embedding.unsqueeze(0), src_key_padding_mask=~safe_view_mask)
            same_time = same_time + self.same_time_view_projection(mixed_view) * same_time_mask.to(mixed_view.dtype).unsqueeze(-1)
            backbone_sequence = same_time.reshape(batch, 2, self.frames, 3, 512).permute(0, 1, 3, 2, 4)
        sequence = backbone_sequence.reshape(batch * 6, self.frames, 512)
        time_mask = view_valid.permute(0, 1, 3, 2).reshape(batch * 6, self.frames)
        safe_mask = time_mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        temporal_input = sequence + self.time_position
        if self.exact_time_modeling:
            if global_time_position is None or global_time_position.shape != (batch, 2, self.frames):
                raise ValueError(f'exact temporal MC3 requires global_time_position [B,2,{self.frames}]')
            frequencies = global_time_position.new_tensor((1.0, 2.0, 4.0, 8.0))
            phase = math.pi * global_time_position.unsqueeze(-1) * frequencies
            exact_features = torch.cat((global_time_position.unsqueeze(-1), phase.sin(), phase.cos()), dim=-1)
            exact_time = self.exact_time_projection(exact_features)
            exact_time = exact_time.unsqueeze(2).expand(-1, -1, 3, -1, -1)
            temporal_input = temporal_input + exact_time.reshape(batch * 6, self.frames, 512)
        sequence = self.temporal_encoder(temporal_input, src_key_padding_mask=~safe_mask)
        sequence = sequence * time_mask.to(sequence.dtype).unsqueeze(-1)
        return (sequence.reshape(batch, 2, 3, self.frames, 512), time_mask.reshape(batch, 2, 3, self.frames))

    def pool_temporal_sequence(self, sequence: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        """Apply the unchanged student whole/early/late temporal pooling head."""
        if sequence.ndim != 5 or sequence.shape[1:] != (2, 3, self.frames, 512):
            raise ValueError(f'sequence must have shape [B,2,3,{self.frames},512], got {tuple(sequence.shape)}')
        if time_mask.shape != sequence.shape[:-1]:
            raise ValueError('time_mask must match the temporal sequence grid')
        batch = sequence.shape[0]
        sequence = sequence.reshape(batch * 6, self.frames, 512)
        time_mask = time_mask.reshape(batch * 6, self.frames)
        whole = self._masked_mean(sequence, time_mask)
        midpoint = max(self.frames // 2, 1)
        early = self._masked_mean(sequence[:, :midpoint], time_mask[:, :midpoint])
        late = self._masked_mean(sequence[:, midpoint:], time_mask[:, midpoint:])
        ordered = self.temporal_fusion(torch.cat((whole, early, late, late - early), dim=1))
        return (whole + ordered).reshape(batch, 2, 3, 512)

    def encode_clips_from_backbone_sequence(self, backbone_sequence: torch.Tensor, view_valid: torch.Tensor | None=None, global_time_position: torch.Tensor | None=None) -> torch.Tensor:
        if not self.temporal_modeling:
            if backbone_sequence.ndim != 5 or backbone_sequence.shape[1:] != (2, 3, self.frames, 512):
                raise ValueError(f'backbone_sequence must have shape [B,2,3,{self.frames},512], got {tuple(backbone_sequence.shape)}')
            return backbone_sequence.mean(dim=3)
        sequence, time_mask = self.encode_temporal_sequence_from_backbone(backbone_sequence, view_valid, global_time_position)
        return self.pool_temporal_sequence(sequence, time_mask)

    def encode_clips(self, images: torch.Tensor, view_valid: torch.Tensor | None=None, global_time_position: torch.Tensor | None=None) -> torch.Tensor:
        return self.encode_clips_from_backbone_sequence(self.encode_backbone_sequence(images), view_valid, global_time_position)

    def forward_from_backbone_sequence(self, backbone_sequence: torch.Tensor, view_valid: torch.Tensor, view_quality: torch.Tensor, global_time_position: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
        clip_embeddings = self.encode_clips_from_backbone_sequence(backbone_sequence, view_valid, global_time_position)
        return self._fuse_clips(clip_embeddings, view_valid, view_quality)

    def forward_from_backbone_region_sequence(self, region_sequence: torch.Tensor, view_valid: torch.Tensor, view_quality: torch.Tensor, global_time_position: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
        backbone_sequence = self.encode_spatial_regions(region_sequence, view_valid)
        return self.forward_from_backbone_sequence(backbone_sequence, view_valid, view_quality, global_time_position)

    def _fuse_clips(self, clip_embeddings: torch.Tensor, view_valid: torch.Tensor, view_quality: torch.Tensor) -> dict[str, torch.Tensor]:
        windows, views = (2, 3)
        batch = clip_embeddings.shape[0]
        if clip_embeddings.shape != (batch, windows, views, 512):
            raise ValueError('clip_embeddings must have shape [B,2,3,512]')
        clip_embeddings = clip_embeddings + self.window_embedding.view(1, 2, 1, 512) + self.view_embedding.view(1, 1, 3, 512)
        clip_mask = view_valid.any(dim=2)
        clip_quality = view_quality.mean(dim=2)
        safe_clip_mask = clip_mask.reshape(batch, 6).clone()
        all_missing = ~safe_clip_mask.any(dim=1)
        safe_clip_mask[all_missing, 0] = True
        encoded = self.token_encoder(clip_embeddings.reshape(batch, 6, 512), src_key_padding_mask=~safe_clip_mask).reshape(batch, 2, 3, 512)
        encoded = encoded * clip_mask.unsqueeze(-1)
        gate_input = torch.cat((encoded, clip_quality.unsqueeze(-1), clip_mask.to(encoded.dtype).unsqueeze(-1)), dim=-1)
        gate = self.quality_gate(gate_input).squeeze(-1).masked_fill(~clip_mask, -10000.0)
        view_weight = torch.softmax(gate, dim=2) * clip_mask
        view_weight = view_weight / view_weight.sum(dim=2, keepdim=True).clamp_min(1e-06)
        window_tokens = (encoded * view_weight.unsqueeze(-1)).sum(dim=2)
        early, late = (window_tokens[:, 0], window_tokens[:, 1])
        embedding = self.stage_fusion(torch.cat((early, late, 0.5 * (early + late), late - early), dim=1))
        output = {'logits': self.classifier(embedding), 'visual_embedding': embedding, 'clip_embeddings': encoded, 'clip_mask': clip_mask, 'window_embeddings': window_tokens, 'view_weight': view_weight}
        if self.training:
            output['stage_logits'] = torch.stack((self.classifier(early), self.classifier(late), self.classifier(late - early)), dim=1)
        if self.distillation_projection is not None:
            output['projected_clip_embeddings'] = self.distillation_projection(encoded)
        return output

    def forward(self, images: torch.Tensor, view_valid: torch.Tensor, view_quality: torch.Tensor, global_time_position: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
        if self.spatial_region_modeling:
            return self.forward_from_backbone_region_sequence(self.encode_backbone_region_sequence(images), view_valid, view_quality, global_time_position)
        return self.forward_from_backbone_sequence(self.encode_backbone_sequence(images), view_valid, view_quality, global_time_position)
