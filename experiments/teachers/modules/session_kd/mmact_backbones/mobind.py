from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import math
import torch
from torch import nn
from session_kd.mmact_backbones.constants import H36M_PARENTS, PART_JOINTS
from session_kd.mmact_backbones.visual import MC3VisualStudent
MOTION_PART_JOINTS = tuple((PART_JOINTS[index] for index in range(2, 7)))
MOTION_PARTS = len(MOTION_PART_JOINTS)

class _GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx: object, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return values.view_as(values)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return (-ctx.scale * gradient, None)

def gradient_reverse(values: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientReversal.apply(values, scale)

def _skeleton_adjacency() -> torch.Tensor:
    adjacency = torch.eye(17, dtype=torch.float32)
    for child, parent in enumerate(H36M_PARENTS.tolist()):
        adjacency[child, parent] = 1.0
        adjacency[parent, child] = 1.0
    return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)

def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dimension: int) -> torch.Tensor:
    weight = mask.to(values.dtype).unsqueeze(-1)
    return (values * weight).sum(dim=dimension) / weight.sum(dim=dimension).clamp_min(1.0)

def _masked_mean_max(values: torch.Tensor, mask: torch.Tensor, dimension: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean = _masked_mean(values, mask, dimension)
    maximum = values.masked_fill(~mask.unsqueeze(-1), -10000.0).amax(dim=dimension)
    mask_dimension = dimension if dimension >= 0 else dimension + 1
    maximum = torch.where(mask.any(dim=mask_dimension).unsqueeze(-1), maximum, 0.0)
    return (mean, maximum)

class GraphTemporalBlock(nn.Module):
    """Skeleton-topology spatial block followed by depthwise temporal mixing."""

    def __init__(self, width: int, dropout: float, adaptive_adjacency: bool=False) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(width)
        self.spatial_projection = nn.Linear(width, width)
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal_depthwise = nn.Conv1d(width, width, kernel_size=3, padding=1, groups=width)
        self.temporal_pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('adjacency', _skeleton_adjacency(), persistent=True)
        self.adaptive_adjacency = nn.Parameter(torch.zeros(17, 17)) if adaptive_adjacency else None

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        spatial_input = self.spatial_norm(values)
        adjacency = self.adjacency
        if self.adaptive_adjacency is not None:
            adjacency = adjacency + 0.1 * torch.tanh(self.adaptive_adjacency)
        spatial = torch.einsum('ij,bwtjd->bwtid', adjacency, spatial_input)
        values = values + self.dropout(self.activation(self.spatial_projection(spatial)))
        batch, windows, steps, joints, width = values.shape
        temporal = self.temporal_norm(values).permute(0, 1, 3, 4, 2)
        temporal = temporal.reshape(batch * windows * joints, width, steps)
        temporal = self.temporal_pointwise(self.activation(self.temporal_depthwise(temporal)))
        temporal = temporal.reshape(batch, windows, joints, width, steps)
        temporal = temporal.permute(0, 1, 4, 2, 3)
        values = values + self.dropout(temporal)
        return values * mask.to(values.dtype).unsqueeze(-1)

class SkeletonPartEncoder(nn.Module):
    """Preserve joint topology and temporal order before five-part pooling."""

    def __init__(self, width: int=96, dropout: float=0.12, explicit_time_position: bool=False, multistream_input: bool=False, multistream_residual_strength: float=0.1, adaptive_graph: bool=False) -> None:
        super().__init__()
        self.width = int(width)
        self.explicit_time_position = bool(explicit_time_position)
        self.multistream_input = bool(multistream_input)
        self.adaptive_graph = bool(adaptive_graph)
        self.joint_stem = nn.Sequential(nn.LayerNorm(27), nn.Linear(27, width), nn.GELU(), nn.Dropout(dropout))
        if self.multistream_input:
            self.stream_stems = nn.ModuleList((nn.Sequential(nn.LayerNorm(input_width), nn.Linear(input_width, width), nn.GELU()) for input_width in (6, 6, 6, 6, 3)))
            self.stream_fusion = nn.Sequential(nn.LayerNorm(width * 5), nn.Linear(width * 5, width), nn.GELU(), nn.Dropout(dropout))
            probability = min(max(float(multistream_residual_strength), 0.0001), 1 - 0.0001)
            self.multistream_residual_logit = nn.Parameter(torch.tensor(math.log(probability / (1.0 - probability))))
        else:
            self.stream_stems = None
            self.stream_fusion = None
            self.register_parameter('multistream_residual_logit', None)
        self.blocks = nn.ModuleList((GraphTemporalBlock(width, dropout, adaptive_adjacency=self.adaptive_graph) for _ in range(2)))
        self.part_projection = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU())
        self.relation_projection = nn.Sequential(nn.LayerNorm(37), nn.Linear(37, width * MOTION_PARTS), nn.GELU())
        self.part_embedding = nn.Parameter(torch.zeros(MOTION_PARTS, width))
        nn.init.trunc_normal_(self.part_embedding, std=0.02)
        self.time_projection = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width)) if self.explicit_time_position else None
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, features: torch.Tensor, feature_mask: torch.Tensor, joint_mask: torch.Tensor, relations: torch.Tensor, relation_mask: torch.Tensor, frame_quality: torch.Tensor, time_position: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-2:] != (17, 13):
            raise ValueError('skeleton_features must end with [17,13]')
        quality = frame_quality.unsqueeze(-1).unsqueeze(-1)
        quality = quality.expand(*features.shape[:-1], 1)
        joint_input = torch.cat((features, feature_mask.to(features.dtype), quality), dim=-1)
        encoded = self.joint_stem(joint_input)
        if self.stream_stems is not None and self.stream_fusion is not None:
            numeric_mask = feature_mask.to(features.dtype)
            streams = [stem(torch.cat((features[..., start:end], numeric_mask[..., start:end]), dim=-1)) for stem, (start, end) in zip(self.stream_stems[:4], ((0, 3), (3, 6), (6, 9), (9, 12)), strict=True)]
            streams.append(self.stream_stems[4](torch.cat((features[..., 12:13], numeric_mask[..., 12:13], quality), dim=-1)))
            stream_delta = self.stream_fusion(torch.cat(streams, dim=-1))
            stream_strength = torch.sigmoid(self.multistream_residual_logit)
            encoded = encoded + stream_strength * stream_delta
        if self.time_projection is not None:
            batch, windows, steps = features.shape[:3]
            if time_position is None:
                local = torch.linspace(0.0, 1.0, steps, device=features.device, dtype=features.dtype)
                window_offset = torch.arange(windows, device=features.device, dtype=features.dtype)
                canonical = (window_offset[:, None] + local[None]) / max(windows, 1)
                time_position = canonical.unsqueeze(0).expand(batch, -1, -1)
            if time_position.shape != (batch, windows, steps):
                raise ValueError('time_position must match Skeleton [B,W,T]')
            encoded = encoded + self.time_projection(time_position.to(features.dtype).unsqueeze(-1)).unsqueeze(-2)
        for block in self.blocks:
            encoded = block(encoded, joint_mask)
        part_tokens = []
        part_masks = []
        for indices in MOTION_PART_JOINTS:
            selected = encoded[..., list(indices), :]
            selected_mask = joint_mask[..., list(indices)]
            mean, maximum = _masked_mean_max(selected, selected_mask, -2)
            part_tokens.append(self.part_projection(torch.cat((mean, maximum), dim=-1)))
            part_masks.append(selected_mask.any(dim=-1))
        tokens = torch.stack(part_tokens, dim=-2)
        mask = torch.stack(part_masks, dim=-1)
        relation_input = torch.cat((relations, relation_mask.to(relations.dtype), frame_quality.unsqueeze(-1)), dim=-1)
        relation_tokens = self.relation_projection(relation_input)
        relation_tokens = relation_tokens.reshape(*tokens.shape)
        tokens = tokens + relation_tokens + self.part_embedding
        batch, windows, steps, parts, width = tokens.shape
        temporal = tokens.permute(0, 1, 3, 2, 4).reshape(batch * windows * parts, steps, width)
        temporal_mask = mask.permute(0, 1, 3, 2).reshape(batch * windows * parts, steps)
        safe_mask = temporal_mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        temporal = self.temporal_encoder(temporal, src_key_padding_mask=~safe_mask)
        temporal = temporal * temporal_mask.to(temporal.dtype).unsqueeze(-1)
        tokens = temporal.reshape(batch, windows, parts, steps, width)
        tokens = tokens.permute(0, 1, 3, 2, 4)
        return (tokens, mask)

class TemporalConvBlock(nn.Module):

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, kernel_size=5, padding=2, groups=width)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.norm(values).transpose(1, 2)
        values = self.pointwise(torch.nn.functional.gelu(self.depthwise(values)))
        return residual + self.dropout(values.transpose(1, 2))

class IMUPartEncoder(nn.Module):
    """Per-device raw/compensated dual stream with continuous temporal mixing."""

    def __init__(self, width: int=96, dropout: float=0.12, instance_normalization: bool=False, event_feature_width: int=0) -> None:
        super().__init__()
        self.width = int(width)
        self.instance_normalization = bool(instance_normalization)
        self.event_feature_width = int(event_feature_width)
        half = width // 2
        self.raw_stem = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, half), nn.GELU())
        self.compensated_stem = nn.Sequential(nn.LayerNorm(10), nn.Linear(10, width - half), nn.GELU())
        self.point_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.point_time_encoder = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        self.point_blocks = nn.ModuleList((TemporalConvBlock(width, dropout) for _ in range(2)))
        self.bin_projection = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU())
        self.bin_statistics_projection = nn.Sequential(nn.LayerNorm(52), nn.Linear(52, width), nn.GELU())
        self.global_projection = nn.Sequential(nn.LayerNorm(50), nn.Linear(50, width), nn.GELU())
        self.device_embedding = nn.Parameter(torch.zeros(MOTION_PARTS, width))
        nn.init.trunc_normal_(self.device_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=1)
        device_layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.device_encoder = nn.TransformerEncoder(device_layer, num_layers=1)
        self.statistics_input_width = 3720
        self.statistics_projection = nn.Sequential(nn.LayerNorm(self.statistics_input_width), nn.Linear(self.statistics_input_width, width * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 4, width * 2), nn.GELU())
        self.event_projection = nn.Sequential(nn.LayerNorm(self.event_feature_width), nn.Linear(self.event_feature_width, width * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 4, width * 2), nn.GELU()) if self.event_feature_width > 0 else None

    def event_embedding(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if self.event_projection is None:
            raise RuntimeError('IMU event projection is not configured')
        if features.shape[-1] != self.event_feature_width:
            raise ValueError(f'unexpected IMU event width {features.shape[-1]} (expected {self.event_feature_width})')
        embedding = self.event_projection(torch.nan_to_num(features))
        return embedding * valid.to(embedding.dtype).unsqueeze(-1)

    @staticmethod
    def _masked_statistics(values: torch.Tensor, mask: torch.Tensor, dimensions: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = mask.to(values.dtype)
        while weight.ndim < values.ndim:
            weight = weight.unsqueeze(-1)
        count = weight.sum(dim=dimensions).clamp_min(1.0)
        mean = (values * weight).sum(dim=dimensions) / count
        centered = values - mean[(slice(None),) + tuple((None if index in dimensions else slice(None) for index in range(1, values.ndim)))]
        variance = (centered.square() * weight).sum(dim=dimensions) / count
        minimum = values.masked_fill(~weight.bool(), 10000.0).amin(dim=dimensions)
        maximum = values.masked_fill(~weight.bool(), -10000.0).amax(dim=dimensions)
        present = mask.any(dim=dimensions)
        while present.ndim < minimum.ndim:
            present = present.unsqueeze(-1)
        minimum = torch.where(present, minimum, 0.0)
        maximum = torch.where(present, maximum, 0.0)
        return (mean, variance.clamp_min(1e-06).sqrt(), minimum, maximum)

    @staticmethod
    def _plain_mask_statistics(mask: torch.Tensor, dimensions: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = mask.to(torch.float32)
        return (values.mean(dim=dimensions), values.std(dim=dimensions, unbiased=False), values.amin(dim=dimensions), values.amax(dim=dimensions))

    def statistics_embedding(self, sequences: torch.Tensor, sequence_mask: torch.Tensor, bin_statistics: torch.Tensor, bin_mask: torch.Tensor, global_statistics: torch.Tensor, global_mask: torch.Tensor) -> torch.Tensor:
        """Encode the proven P3-style engineered statistics inside the network."""
        sequence_summary = self._masked_statistics(sequences, sequence_mask, (2, 4))
        sequence_mask_summary = self._plain_mask_statistics(sequence_mask, (2, 4))
        bin_summary = self._masked_statistics(bin_statistics, bin_mask, (2,))
        bin_mask_summary = self._plain_mask_statistics(bin_mask, (2,))
        midpoint = max(1, sequences.shape[2] // 2)
        early_sequence = self._masked_statistics(sequences[:, :, :midpoint], sequence_mask[:, :, :midpoint], (2, 4))[0]
        late_sequence = self._masked_statistics(sequences[:, :, midpoint:], sequence_mask[:, :, midpoint:], (2, 4))[0]
        early_bins = self._masked_statistics(bin_statistics[:, :, :midpoint], bin_mask[:, :, :midpoint], (2,))[0]
        late_bins = self._masked_statistics(bin_statistics[:, :, midpoint:], bin_mask[:, :, midpoint:], (2,))[0]
        if sequences.shape[2] == 1:
            sequence_delta = torch.zeros_like(early_sequence)
            bin_delta = torch.zeros_like(early_bins)
        else:
            sequence_delta = late_sequence - early_sequence
            bin_delta = late_bins - early_bins
        global_present = global_mask.any(dim=-1, keepdim=True).to(global_statistics.dtype)
        flat = torch.cat([*(value.flatten(1) for value in sequence_summary), *(value.flatten(1) for value in sequence_mask_summary), *(value.flatten(1) for value in bin_summary), *(value.flatten(1) for value in bin_mask_summary), sequence_delta.flatten(1), bin_delta.flatten(1), (global_statistics * global_present).flatten(1)], dim=1)
        if flat.shape[1] != self.statistics_input_width:
            raise RuntimeError(f'unexpected IMU statistics width {flat.shape[1]} (expected {self.statistics_input_width})')
        return self.statistics_projection(torch.nan_to_num(flat))

    def forward(self, sequences: torch.Tensor, sequence_mask: torch.Tensor, bin_statistics: torch.Tensor, bin_mask: torch.Tensor, global_statistics: torch.Tensor, global_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if sequences.ndim != 6 or sequences.shape[-3] != MOTION_PARTS:
            raise ValueError('imu_sequences must be [B,W,T,5,P,16]')
        if sequences.shape[-1] != 16:
            raise ValueError('imu_sequences must contain raw6+compensated6+quat4')
        batch, windows, steps, devices, points, _ = sequences.shape
        vectors = sequences[..., :12]
        if self.instance_normalization:
            vector_mask = sequence_mask.to(sequences.dtype).unsqueeze(-1)
            count = vector_mask.sum(dim=(1, 2, 4), keepdim=True).clamp_min(1.0)
            vector_mean = (vectors * vector_mask).sum(dim=(1, 2, 4), keepdim=True) / count
            vector_variance = ((vectors - vector_mean).square() * vector_mask).sum(dim=(1, 2, 4), keepdim=True) / count
            vectors = (vectors - vector_mean) / vector_variance.clamp_min(0.0001).sqrt()
            vectors = vectors * vector_mask
        raw = self.raw_stem(vectors[..., :6])
        compensated_input = torch.cat((vectors[..., 6:12], sequences[..., 12:16]), dim=-1)
        compensated = self.compensated_stem(compensated_input)
        encoded = self.point_projection(torch.cat((raw, compensated), dim=-1))
        encoded = encoded.permute(0, 1, 3, 2, 4, 5).reshape(batch * windows * devices, steps * points, self.width)
        flat_mask = sequence_mask.permute(0, 1, 3, 2, 4).reshape(batch * windows * devices, steps * points)
        position = torch.linspace(0.0, 1.0, steps * points, device=sequences.device, dtype=sequences.dtype).view(1, steps * points, 1)
        encoded = encoded + self.point_time_encoder(position)
        encoded = encoded * flat_mask.to(encoded.dtype).unsqueeze(-1)
        for block in self.point_blocks:
            encoded = block(encoded)
            encoded = encoded * flat_mask.to(encoded.dtype).unsqueeze(-1)
        encoded = encoded.reshape(batch, windows, devices, steps, points, self.width).permute(0, 1, 3, 2, 4, 5)
        mean, maximum = _masked_mean_max(encoded, sequence_mask, -2)
        tokens = self.bin_projection(torch.cat((mean, maximum), dim=-1))
        tokens = tokens + self.bin_statistics_projection(bin_statistics)
        global_input = torch.cat((global_statistics, global_mask), dim=-1)
        global_token = self.global_projection(global_input)
        tokens = tokens + global_token[:, None, None] + self.device_embedding[None, None, None]
        temporal = tokens.permute(0, 1, 3, 2, 4).reshape(batch * windows * devices, steps, self.width)
        temporal_mask = bin_mask.permute(0, 1, 3, 2).reshape(batch * windows * devices, steps)
        safe_mask = temporal_mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        temporal = self.temporal_encoder(temporal, src_key_padding_mask=~safe_mask)
        temporal = temporal * temporal_mask.to(temporal.dtype).unsqueeze(-1)
        tokens = temporal.reshape(batch, windows, devices, steps, self.width).permute(0, 1, 3, 2, 4)
        device_tokens = tokens.reshape(batch * windows * steps, devices, self.width)
        device_mask = bin_mask.reshape(batch * windows * steps, devices)
        safe_device_mask = device_mask.clone()
        empty_device = ~safe_device_mask.any(dim=1)
        safe_device_mask[empty_device, 0] = True
        device_tokens = self.device_encoder(device_tokens, src_key_padding_mask=~safe_device_mask)
        device_tokens = device_tokens * device_mask.to(device_tokens.dtype).unsqueeze(-1)
        tokens = device_tokens.reshape(batch, windows, steps, devices, self.width)
        return (tokens, bin_mask)

class MotionSemanticHead(nn.Module):

    def __init__(self, width: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.width = int(width)
        self.embedding_width = width * (MOTION_PARTS * 4 + 4)
        self.classifier = nn.Sequential(nn.LayerNorm(self.embedding_width), nn.Linear(self.embedding_width, width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, classes))

    def embedding(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(tokens.dtype).unsqueeze(-1)
        part = (tokens * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
        window = (tokens * weight).sum(dim=(2, 3)) / weight.sum(dim=(2, 3)).clamp_min(1.0)
        centered = (tokens - window[:, :, None, None]) * weight
        variance = centered.square().sum(dim=(2, 3)) / weight.sum(dim=(2, 3)).clamp_min(1.0)
        deviation = variance.clamp_min(1e-06).sqrt()
        maximum = tokens.masked_fill(~mask.unsqueeze(-1), -10000.0).amax(dim=(2, 3))
        maximum = torch.where(mask.any(dim=(2, 3)).unsqueeze(-1), maximum, 0.0)
        early_part, late_part = (part[:, 0], part[:, 1])
        return torch.cat((early_part.flatten(1), late_part.flatten(1), (0.5 * (early_part + late_part)).flatten(1), (late_part - early_part).flatten(1), deviation[:, 0], deviation[:, 1], maximum[:, 0], maximum[:, 1]), dim=-1)

    def window_embedding(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weight).sum(dim=(2, 3)) / weight.sum(dim=(2, 3)).clamp_min(1.0)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(tokens, mask))

    def compact_embedding(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(tokens, mask)
        return self.classifier[2](self.classifier[1](self.classifier[0](embedding)))

    def classify_compact(self, compact: torch.Tensor) -> torch.Tensor:
        return self.classifier[4](self.classifier[3](compact))

class MoBindLite(nn.Module):
    """Training-time Skeleton/IMU binding model; final inference may prune one side."""

    def __init__(self, width: int=96, alignment_width: int=64, classes: int=40, dropout: float=0.12, imu_instance_normalization: bool=False, imu_event_feature_width: int=0, skeleton_time_position: bool=False, skeleton_multistream: bool=False, skeleton_multistream_strength: float=0.1, skeleton_adaptive_graph: bool=False, domain_classes: int=0, domain_reversal_scale: float=1.0) -> None:
        super().__init__()
        self.width = int(width)
        self.alignment_width = int(alignment_width)
        self.domain_classes = int(domain_classes)
        self.domain_reversal_scale = float(domain_reversal_scale)
        self.skeleton_encoder = SkeletonPartEncoder(width, dropout, explicit_time_position=skeleton_time_position, multistream_input=skeleton_multistream, multistream_residual_strength=skeleton_multistream_strength, adaptive_graph=skeleton_adaptive_graph)
        self.imu_encoder = IMUPartEncoder(width, dropout, instance_normalization=imu_instance_normalization, event_feature_width=imu_event_feature_width)
        self.skeleton_head = MotionSemanticHead(width, classes, dropout)
        self.imu_head = MotionSemanticHead(width, classes, dropout)
        self.skeleton_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, alignment_width))
        self.imu_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, alignment_width))
        self.skeleton_teacher_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1024))
        self.imu_teacher_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1024))
        domain_head = lambda: nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, self.domain_classes))
        self.skeleton_domain_head = domain_head() if self.domain_classes > 0 else None
        self.imu_domain_head = domain_head() if self.domain_classes > 0 else None
        self.skeleton_mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.imu_mask_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.skeleton_mask_token, std=0.02)
        nn.init.trunc_normal_(self.imu_mask_token, std=0.02)
        context_layer = lambda: nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.skeleton_context = nn.TransformerEncoder(context_layer(), num_layers=1)
        self.imu_context = nn.TransformerEncoder(context_layer(), num_layers=1)
        self.skeleton_reconstruction = nn.Linear(width, width)
        self.imu_reconstruction = nn.Linear(width, width)

    def encode(self, motion: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        skeleton_tokens, skeleton_mask = self.skeleton_encoder(motion['skeleton_features'], motion['skeleton_feature_mask'], motion['skeleton_joint_mask'], motion['skeleton_relations'], motion['skeleton_relation_mask'], motion['skeleton_frame_quality'])
        imu_tokens, imu_mask = self.imu_encoder(motion['imu_sequences'], motion['imu_sequence_mask'], motion['imu_bin_statistics'], motion['imu_bin_mask'], motion['imu_global_statistics'], motion['imu_global_mask'])
        skeleton_compact = self.skeleton_head.compact_embedding(skeleton_tokens, skeleton_mask)
        imu_compact = self.imu_head.compact_embedding(imu_tokens, imu_mask)
        imu_statistics = self.imu_encoder.statistics_embedding(motion['imu_sequences'], motion['imu_sequence_mask'], motion['imu_bin_statistics'], motion['imu_bin_mask'], motion['imu_global_statistics'], motion['imu_global_mask'])
        imu_compact = imu_compact + imu_statistics
        if self.imu_encoder.event_projection is not None:
            imu_compact = imu_compact + self.imu_encoder.event_embedding(motion['imu_event_features'], motion['imu_event_valid'])
        output = {'skeleton_tokens': skeleton_tokens, 'skeleton_mask': skeleton_mask, 'imu_tokens': imu_tokens, 'imu_mask': imu_mask, 'skeleton_alignment': torch.nn.functional.normalize(self.skeleton_projection(skeleton_tokens), dim=-1), 'imu_alignment': torch.nn.functional.normalize(self.imu_projection(imu_tokens), dim=-1), 'skeleton_logits': self.skeleton_head(skeleton_tokens, skeleton_mask), 'imu_logits': self.imu_head.classify_compact(imu_compact), 'skeleton_compact': skeleton_compact, 'imu_compact': imu_compact, 'skeleton_teacher_features': self.skeleton_teacher_projection(self.skeleton_head.window_embedding(skeleton_tokens, skeleton_mask)), 'imu_teacher_features': self.imu_teacher_projection(self.imu_head.window_embedding(imu_tokens, imu_mask))}
        if self.domain_classes > 0:
            assert self.skeleton_domain_head is not None
            assert self.imu_domain_head is not None
            output['skeleton_domain_logits'] = self.skeleton_domain_head(gradient_reverse(skeleton_compact, self.domain_reversal_scale))
            output['imu_domain_logits'] = self.imu_domain_head(gradient_reverse(imu_compact, self.domain_reversal_scale))
        return output

    def masked_prediction(self, tokens: torch.Tensor, valid: torch.Tensor, modality: str, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = tokens.shape[0]
        flat = tokens.reshape(batch, -1, self.width)
        flat_valid = valid.reshape(batch, -1)
        selected = torch.rand_like(flat_valid, dtype=torch.float32) < mask_ratio
        selected = selected & flat_valid
        if modality == 'skeleton':
            mask_token = self.skeleton_mask_token
            context = self.skeleton_context
            reconstruction = self.skeleton_reconstruction
        elif modality == 'imu':
            mask_token = self.imu_mask_token
            context = self.imu_context
            reconstruction = self.imu_reconstruction
        else:
            raise ValueError(modality)
        masked = torch.where(selected.unsqueeze(-1), mask_token, flat)
        safe_valid = flat_valid.clone()
        empty = ~safe_valid.any(dim=1)
        safe_valid[empty, 0] = True
        predicted = reconstruction(context(masked, src_key_padding_mask=~safe_valid))
        return (predicted, flat.detach(), selected)

    def forward(self, motion: dict[str, torch.Tensor], mask_ratio: float=0.0) -> dict[str, torch.Tensor]:
        output = self.encode(motion)
        if self.training and mask_ratio > 0.0:
            for modality in ('skeleton', 'imu'):
                predicted, target, selected = self.masked_prediction(output[f'{modality}_tokens'], output[f'{modality}_mask'], modality, mask_ratio)
                output[f'{modality}_reconstruction'] = predicted
                output[f'{modality}_reconstruction_target'] = target
                output[f'{modality}_reconstruction_mask'] = selected
        return output

class SeparateMotionEncoder(nn.Module):
    """Keep Skeleton and IMU part/time tokens distinct before shared fusion."""

    def __init__(self, skeleton_encoder: SkeletonPartEncoder, imu_encoder: IMUPartEncoder, width: int=96, modality_dropout: float=0.0) -> None:
        super().__init__()
        self.width = int(width)
        self.modality_dropout = float(modality_dropout)
        if not 0.0 <= self.modality_dropout < 0.5:
            raise ValueError('modality_dropout must be in [0, 0.5)')
        self.skeleton_encoder = skeleton_encoder
        self.imu_encoder = imu_encoder
        self.last_skeleton_tokens: torch.Tensor | None = None
        self.last_skeleton_mask: torch.Tensor | None = None
        self.modality_embedding = nn.Parameter(torch.zeros(2, self.width))
        self.statistics_lift = nn.Sequential(nn.LayerNorm(self.width * 2), nn.Linear(self.width * 2, self.width), nn.GELU())

    def fresh_parameters(self) -> list[nn.Parameter]:
        return [self.modality_embedding, *self.statistics_lift.parameters()]

    def forward(self, motion: dict[str, torch.Tensor], global_time_position: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        skeleton_tokens, skeleton_mask = self.skeleton_encoder(motion['skeleton_features'], motion['skeleton_feature_mask'], motion['skeleton_joint_mask'], motion['skeleton_relations'], motion['skeleton_relation_mask'], motion['skeleton_frame_quality'], global_time_position)
        imu_tokens, imu_mask = self.imu_encoder(motion['imu_sequences'], motion['imu_sequence_mask'], motion['imu_bin_statistics'], motion['imu_bin_mask'], motion['imu_global_statistics'], motion['imu_global_mask'])
        if skeleton_tokens.shape != imu_tokens.shape:
            raise ValueError('Skeleton and IMU part/time grids must match')
        self.last_skeleton_tokens = skeleton_tokens
        self.last_skeleton_mask = skeleton_mask
        statistics = self.imu_encoder.statistics_embedding(motion['imu_sequences'], motion['imu_sequence_mask'], motion['imu_bin_statistics'], motion['imu_bin_mask'], motion['imu_global_statistics'], motion['imu_global_mask'])
        if self.imu_encoder.event_projection is not None:
            statistics = statistics + self.imu_encoder.event_embedding(motion['imu_event_features'], motion['imu_event_valid'])
        skeleton_tokens = skeleton_tokens + self.modality_embedding[0]
        imu_tokens = imu_tokens + self.statistics_lift(statistics)[:, None, None, None] + self.modality_embedding[1]
        if self.training and self.modality_dropout > 0.0:
            decision = torch.rand(len(skeleton_tokens), device=skeleton_tokens.device)
            drop_skeleton = decision < self.modality_dropout
            drop_imu = (decision >= self.modality_dropout) & (decision < 2.0 * self.modality_dropout)
            skeleton_tokens = skeleton_tokens.masked_fill(drop_skeleton[:, None, None, None, None], 0.0)
            skeleton_mask = skeleton_mask & ~drop_skeleton[:, None, None, None]
            imu_tokens = imu_tokens.masked_fill(drop_imu[:, None, None, None, None], 0.0)
            imu_mask = imu_mask & ~drop_imu[:, None, None, None]
        return (torch.cat((skeleton_tokens, imu_tokens), dim=3), torch.cat((skeleton_mask, imu_mask), dim=3))

class JointMotionEncoder(nn.Module):
    """Create body-part x time motion tokens with Skeleton as the semantic anchor.

    The pretrained alignment projections expose the shared S/I motion, while the
    original IMU token and engineered trial statistics remain in a private residual
    path.  This avoids forcing both sensors into an identical representation.
    """

    def __init__(self, skeleton_encoder: SkeletonPartEncoder, imu_encoder: IMUPartEncoder, skeleton_alignment: nn.Module, imu_alignment: nn.Module, imu_semantic_head: MotionSemanticHead, width: int=96, alignment_width: int=64, dropout: float=0.12, initial_imu_gate: float=0.35, maximum_imu_residual: float=1.0) -> None:
        super().__init__()
        self.width = int(width)
        self.alignment_width = int(alignment_width)
        self.maximum_imu_residual = float(maximum_imu_residual)
        if not 0.0 < self.maximum_imu_residual <= 1.0:
            raise ValueError('maximum_imu_residual must be in (0, 1]')
        self.skeleton_encoder = skeleton_encoder
        self.imu_encoder = imu_encoder
        self.skeleton_alignment = skeleton_alignment
        self.imu_alignment = imu_alignment
        self.imu_semantic_head = imu_semantic_head
        self.last_token_reliability: torch.Tensor | None = None
        self.last_raw_token_reliability: torch.Tensor | None = None
        self.last_skeleton_tokens: torch.Tensor | None = None
        self.last_skeleton_mask: torch.Tensor | None = None
        self.last_imu_logits: torch.Tensor | None = None
        self.alignment_interaction = nn.Sequential(nn.LayerNorm(self.alignment_width * 4), nn.Linear(self.alignment_width * 4, self.width), nn.GELU())
        self.statistics_lift = nn.Sequential(nn.LayerNorm(self.width * 2), nn.Linear(self.width * 2, self.width), nn.GELU())
        self.imu_residual = nn.Sequential(nn.LayerNorm(self.width * 4), nn.Linear(self.width * 4, self.width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.width * 2, self.width), nn.Dropout(dropout))
        nn.init.normal_(self.imu_residual[4].weight, std=0.005)
        nn.init.zeros_(self.imu_residual[4].bias)
        self.token_reliability = nn.Sequential(nn.LayerNorm(self.width * 4), nn.Linear(self.width * 4, self.width), nn.GELU(), nn.Linear(self.width, 1))
        probability = min(max(float(initial_imu_gate), 0.0001), 1.0 - 0.0001)
        nn.init.zeros_(self.token_reliability[-1].weight)
        nn.init.constant_(self.token_reliability[-1].bias, math.log(probability / (1.0 - probability)))

    def fresh_parameters(self) -> list[nn.Parameter]:
        modules = (self.alignment_interaction, self.statistics_lift, self.imu_residual, self.token_reliability)
        return [parameter for module in modules for parameter in module.parameters()]

    def forward(self, motion: dict[str, torch.Tensor], global_time_position: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        skeleton_tokens, skeleton_mask = self.skeleton_encoder(motion['skeleton_features'], motion['skeleton_feature_mask'], motion['skeleton_joint_mask'], motion['skeleton_relations'], motion['skeleton_relation_mask'], motion['skeleton_frame_quality'], global_time_position)
        imu_tokens, imu_mask = self.imu_encoder(motion['imu_sequences'], motion['imu_sequence_mask'], motion['imu_bin_statistics'], motion['imu_bin_mask'], motion['imu_global_statistics'], motion['imu_global_mask'])
        if skeleton_tokens.shape != imu_tokens.shape:
            raise ValueError('Skeleton and IMU part/time grids must match')
        statistics = self.imu_encoder.statistics_embedding(motion['imu_sequences'], motion['imu_sequence_mask'], motion['imu_bin_statistics'], motion['imu_bin_mask'], motion['imu_global_statistics'], motion['imu_global_mask'])
        if self.imu_encoder.event_projection is not None:
            statistics = statistics + self.imu_encoder.event_embedding(motion['imu_event_features'], motion['imu_event_valid'])
        imu_compact = self.imu_semantic_head.compact_embedding(imu_tokens, imu_mask) + statistics
        self.last_imu_logits = self.imu_semantic_head.classify_compact(imu_compact)
        self.last_skeleton_tokens = skeleton_tokens
        self.last_skeleton_mask = skeleton_mask
        imu_private = imu_tokens + self.statistics_lift(statistics)[:, None, None, None]
        skeleton_aligned = torch.nn.functional.normalize(self.skeleton_alignment(skeleton_tokens), dim=-1)
        imu_aligned = torch.nn.functional.normalize(self.imu_alignment(imu_tokens), dim=-1)
        alignment = self.alignment_interaction(torch.cat((skeleton_aligned, imu_aligned, skeleton_aligned * imu_aligned, (skeleton_aligned - imu_aligned).abs()), dim=-1))
        difference = imu_private - skeleton_tokens
        interaction = torch.cat((imu_private, difference, imu_private * skeleton_tokens, alignment), dim=-1)
        residual = self.imu_residual(interaction)
        reliability = torch.sigmoid(self.token_reliability(torch.cat((skeleton_tokens, imu_private, difference.abs(), skeleton_tokens * imu_private), dim=-1)))
        effective_reliability = self.maximum_imu_residual * reliability
        self.last_raw_token_reliability = reliability
        self.last_token_reliability = effective_reliability.detach().mean()
        both = skeleton_mask & imu_mask
        skeleton_only = skeleton_mask & ~imu_mask
        imu_only = imu_mask & ~skeleton_mask
        joint = skeleton_tokens + effective_reliability * residual
        joint = joint * both.to(joint.dtype).unsqueeze(-1)
        joint = joint + skeleton_tokens * skeleton_only.to(joint.dtype).unsqueeze(-1)
        joint = joint + imu_private * imu_only.to(joint.dtype).unsqueeze(-1)
        return (joint, skeleton_mask | imu_mask)

class MoBindMotionResidual(nn.Module):
    """Fuse one motion encoder into semantic visual clip tokens.

    The first version injected a tiny zero-initialized residual into every
    layer-4 time/view token.  That preserved the visual anchor, but the motion
    gradient had to pass through the complete visual temporal stack and almost
    no complementary evidence reached the classifier.  This adapter instead
    attends over all part/time tokens inside each action window after MC3 has
    formed a semantic clip token, while still using the visual model's single
    token encoder and final classifier.
    """

    def __init__(self, modality: str, encoder: nn.Module, semantic_head: MotionSemanticHead, teacher_projection: nn.Module, visual_width: int=512, motion_width: int=96, dropout: float=0.12, initial_residual_strength: float=0.25, reliability_event_feature_width: int=0, global_fusion_mode: str='additive', reliability_groups: int=1) -> None:
        super().__init__()
        if modality not in {'skeleton', 'imu', 'separate', 'joint'}:
            raise ValueError('motion residual must be skeleton, imu, separate or joint')
        self.modality = modality
        self.encoder = encoder
        self.semantic_head = semantic_head
        self.teacher_projection = teacher_projection
        self.reliability_event_feature_width = int(reliability_event_feature_width)
        if global_fusion_mode not in {'additive', 'conditional'}:
            raise ValueError('global_fusion_mode must be additive or conditional')
        self.global_fusion_mode = global_fusion_mode
        self.reliability_groups = int(reliability_groups)
        if self.reliability_groups <= 0 or visual_width % self.reliability_groups:
            raise ValueError('reliability_groups must divide visual_width')
        self.event_reliability_projection = nn.Sequential(nn.LayerNorm(self.reliability_event_feature_width), nn.Linear(self.reliability_event_feature_width, motion_width), nn.GELU(), nn.Dropout(dropout)) if self.reliability_event_feature_width > 0 else None
        teacher_width = int(teacher_projection[-1].out_features)
        self.semantic_input_width = motion_width * 2 + teacher_width * 4
        self.visual_query = nn.Sequential(nn.LayerNorm(visual_width), nn.Linear(visual_width, motion_width))
        self.time_encoder = nn.Sequential(nn.Linear(1, motion_width), nn.GELU(), nn.Linear(motion_width, motion_width))
        self.motion_projection = nn.Sequential(nn.LayerNorm(motion_width * 4), nn.Linear(motion_width * 4, motion_width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(motion_width * 2, visual_width), nn.Dropout(dropout))
        nn.init.normal_(self.motion_projection[4].weight, std=0.005)
        nn.init.zeros_(self.motion_projection[4].bias)
        local_reliability_width = motion_width * (5 if self.event_reliability_projection is not None else 4)
        self.reliability = nn.Sequential(nn.LayerNorm(local_reliability_width), nn.Linear(local_reliability_width, motion_width), nn.GELU(), nn.Linear(motion_width, self.reliability_groups))
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.constant_(self.reliability[-1].bias, math.log(0.75 / 0.25))
        condition_width = motion_width * 2
        self.global_motion_context = nn.Sequential(nn.LayerNorm(self.semantic_input_width), nn.Linear(self.semantic_input_width, condition_width), nn.GELU()) if self.global_fusion_mode == 'conditional' else None
        self.global_visual_context = nn.Sequential(nn.LayerNorm(visual_width), nn.Linear(visual_width, condition_width), nn.GELU()) if self.global_fusion_mode == 'conditional' else None
        global_projection_input = condition_width * 4 if self.global_fusion_mode == 'conditional' else self.semantic_input_width
        self.global_motion_projection = nn.Sequential(nn.LayerNorm(global_projection_input), nn.Linear(global_projection_input, visual_width), nn.GELU(), nn.Dropout(dropout), nn.Linear(visual_width, visual_width), nn.Dropout(dropout))
        nn.init.normal_(self.global_motion_projection[4].weight, std=0.005)
        nn.init.zeros_(self.global_motion_projection[4].bias)
        global_reliability_width = visual_width * 4 + (motion_width if self.event_reliability_projection is not None else 0)
        self.global_reliability = nn.Sequential(nn.LayerNorm(global_reliability_width), nn.Linear(global_reliability_width, motion_width * 2), nn.GELU(), nn.Linear(motion_width * 2, self.reliability_groups))
        nn.init.zeros_(self.global_reliability[-1].weight)
        nn.init.constant_(self.global_reliability[-1].bias, math.log(0.75 / 0.25))
        initial_probability = float(initial_residual_strength)
        if not 0.0 < initial_probability < 1.0:
            raise ValueError('initial_residual_strength must be between zero and one')
        self.residual_logit = nn.Parameter(torch.tensor(math.log(initial_probability / (1.0 - initial_probability))))

    def strength(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_logit)

    def encode_motion(self, motion: dict[str, torch.Tensor], global_time_position: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.modality == 'skeleton':
            return self.encoder(motion['skeleton_features'], motion['skeleton_feature_mask'], motion['skeleton_joint_mask'], motion['skeleton_relations'], motion['skeleton_relation_mask'], motion['skeleton_frame_quality'], global_time_position)
        if self.modality in {'separate', 'joint'}:
            return self.encoder(motion, global_time_position)
        return self.encoder(motion['imu_sequences'], motion['imu_sequence_mask'], motion['imu_bin_statistics'], motion['imu_bin_mask'], motion['imu_global_statistics'], motion['imu_global_mask'])

    def fuse_global(self, visual_embedding: torch.Tensor, motion_embedding: torch.Tensor, available: torch.Tensor, reliability_context: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.global_fusion_mode == 'conditional':
            assert self.global_motion_context is not None
            assert self.global_visual_context is not None
            motion_context = self.global_motion_context(motion_embedding)
            visual_context = self.global_visual_context(visual_embedding)
            interaction = torch.cat((visual_context, motion_context, visual_context * motion_context, (visual_context - motion_context).abs()), dim=-1)
            projected = self.global_motion_projection(interaction)
        else:
            projected = self.global_motion_projection(motion_embedding)
        reliability_values = [visual_embedding, projected, visual_embedding * projected, (visual_embedding - projected).abs()]
        if self.event_reliability_projection is not None:
            if reliability_context is None:
                raise RuntimeError('event reliability context is required')
            reliability_values.append(reliability_context)
        reliability_input = torch.cat(reliability_values, dim=-1)
        group_gate = torch.sigmoid(self.global_reliability(reliability_input))
        gate = group_gate.repeat_interleave(projected.shape[-1] // self.reliability_groups, dim=-1)
        sample_available = available.any(dim=1, keepdim=True)
        residual = projected * gate * self.strength()
        residual = residual * sample_available.to(residual.dtype)
        return (visual_embedding + residual, group_gate)

    def forward(self, visual_clips: torch.Tensor, global_time_position: torch.Tensor, motion: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if visual_clips.ndim != 4 or visual_clips.shape[1:3] != (2, 3):
            raise ValueError('visual_clips must be [B,2,3,C]')
        original_tokens, token_mask = self.encode_motion(motion, global_time_position)
        reliability_context = None
        if self.event_reliability_projection is not None:
            event_features = motion['imu_event_features']
            if event_features.shape[-1] != self.reliability_event_feature_width:
                raise ValueError('unexpected reliability event feature width')
            reliability_context = self.event_reliability_projection(torch.nan_to_num(event_features))
            reliability_context = reliability_context * motion['imu_event_valid'].to(reliability_context.dtype).unsqueeze(-1)
        if self.modality == 'separate':
            if not isinstance(self.encoder, SeparateMotionEncoder):
                raise RuntimeError('separate modality requires SeparateMotionEncoder')
            if self.encoder.last_skeleton_tokens is None or self.encoder.last_skeleton_mask is None:
                raise RuntimeError('separate encoder audit tensors are unavailable')
            motion_compact = self.semantic_head.compact_embedding(self.encoder.last_skeleton_tokens, self.encoder.last_skeleton_mask)
        else:
            motion_compact = self.semantic_head.compact_embedding(original_tokens, token_mask)
        if self.modality == 'imu':
            motion_statistics = self.encoder.statistics_embedding(motion['imu_sequences'], motion['imu_sequence_mask'], motion['imu_bin_statistics'], motion['imu_bin_mask'], motion['imu_global_statistics'], motion['imu_global_mask'])
            motion_compact = motion_compact + motion_statistics
            if self.encoder.event_projection is not None:
                motion_compact = motion_compact + self.encoder.event_embedding(motion['imu_event_features'], motion['imu_event_valid'])
        motion_logits = self.semantic_head.classify_compact(motion_compact)
        joint_audit: dict[str, torch.Tensor] = {}
        if self.modality == 'joint':
            if not isinstance(self.encoder, JointMotionEncoder):
                raise RuntimeError('joint modality requires JointMotionEncoder')
            if self.encoder.last_skeleton_tokens is None or self.encoder.last_skeleton_mask is None or self.encoder.last_imu_logits is None or (self.encoder.last_raw_token_reliability is None):
                raise RuntimeError('joint encoder audit tensors are unavailable')
            joint_audit = {'joint_skeleton_logits': self.semantic_head(self.encoder.last_skeleton_tokens, self.encoder.last_skeleton_mask), 'joint_imu_logits': self.encoder.last_imu_logits, 'joint_raw_imu_gate': self.encoder.last_raw_token_reliability}
        teacher_windows = self.teacher_projection(self.semantic_head.window_embedding(original_tokens, token_mask))
        teacher_early, teacher_late = (teacher_windows[:, 0], teacher_windows[:, 1])
        teacher_semantic = torch.cat((teacher_early, teacher_late, 0.5 * (teacher_early + teacher_late), teacher_late - teacher_early), dim=-1)
        motion_semantic = torch.cat((motion_compact, teacher_semantic), dim=-1)
        tokens = original_tokens
        batch, windows, views, visual_width = visual_clips.shape
        steps = tokens.shape[2]
        parts = tokens.shape[3]
        if tokens.shape[:3] != (batch, windows, steps) or parts <= 0:
            raise ValueError('motion and visual grids differ')
        if global_time_position.shape != (batch, windows, steps):
            raise ValueError('global_time_position must match motion time bins')
        time = self.time_encoder(global_time_position.unsqueeze(-1))
        tokens = tokens + time.unsqueeze(-2)
        query = self.visual_query(visual_clips)
        flat_tokens = tokens.flatten(2, 3)
        flat_mask = token_mask.flatten(2, 3)
        score = torch.einsum('bwvd,bwld->bwvl', query, flat_tokens)
        score = score / math.sqrt(tokens.shape[-1])
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=-1)
        safe_mask[empty, 0] = True
        score = score.masked_fill(~safe_mask.unsqueeze(-2), -10000.0)
        attention = torch.softmax(score, dim=-1)
        attention = attention * flat_mask.to(attention.dtype).unsqueeze(-2)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-06)
        attended = torch.einsum('bwvl,bwld->bwvd', attention, flat_tokens)
        available = flat_mask.any(dim=-1)
        attended = attended * available.to(attended.dtype).unsqueeze(-1).unsqueeze(-1)
        weight = flat_mask.to(tokens.dtype).unsqueeze(-1)
        window = (flat_tokens * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
        delta = window[:, 1] - window[:, 0]
        signed_delta = torch.stack((-delta, delta), dim=1)
        window = window.unsqueeze(2).expand(-1, -1, views, -1)
        signed_delta = signed_delta.unsqueeze(2).expand(-1, -1, views, -1)
        reliability_values = [query, attended, query * attended, (query - attended).abs()]
        if reliability_context is not None:
            reliability_values.append(reliability_context[:, None, None].expand(-1, windows, views, -1))
        reliability_input = torch.cat(reliability_values, dim=-1)
        local_group_gate = torch.sigmoid(self.reliability(reliability_input))
        local_gate = local_group_gate.repeat_interleave(visual_width // self.reliability_groups, dim=-1)
        adapter_input = torch.cat((attended, window, signed_delta, query * attended), dim=-1)
        strength = self.strength()
        residual = self.motion_projection(adapter_input) * local_gate * strength
        residual = residual * available.to(residual.dtype).unsqueeze(-1).unsqueeze(-1)
        fused = visual_clips + residual
        audit = {'motion_part_attention': attention.reshape(batch, windows, views, steps, parts), 'motion_reliability': local_group_gate, 'motion_residual_strength': strength, 'motion_available': available, 'motion_logits': motion_logits, 'motion_semantic_embedding': motion_semantic, 'motion_reliability_context': reliability_context, 'motion_tokens': tokens, 'motion_token_mask': token_mask}
        audit.update(joint_audit)
        return (fused, audit)

class UnifiedMoBindStudent(nn.Module):
    """One visual model, one aligned motion encoder and one final 40-class head."""

    def __init__(self, visual: MC3VisualStudent, motion_residual: MoBindMotionResidual) -> None:
        super().__init__()
        if not visual.temporal_modeling:
            raise ValueError('MoBind fusion requires temporal visual tokens')
        self.visual = visual
        self.motion_residual = motion_residual

    def forward_from_backbone_sequence(self, backbone_sequence: torch.Tensor, view_valid: torch.Tensor, view_quality: torch.Tensor, global_time_position: torch.Tensor, motion: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual_clips = self.visual.encode_clips_from_backbone_sequence(backbone_sequence, view_valid, global_time_position)
        visual_only = self.visual._fuse_clips(visual_clips, view_valid, view_quality)
        fused_clips, audit = self.motion_residual(visual_clips, global_time_position, motion)
        output = self.visual._fuse_clips(fused_clips, view_valid, view_quality)
        fused_embedding, global_reliability = self.motion_residual.fuse_global(output['visual_embedding'], audit['motion_semantic_embedding'], audit['motion_available'], audit['motion_reliability_context'])
        output['visual_embedding'] = fused_embedding
        output['logits'] = self.visual.classifier(fused_embedding)
        output['unfused_visual_logits'] = visual_only['logits']
        audit['motion_global_reliability'] = global_reliability
        output.update(audit)
        return output

    def forward(self, images: torch.Tensor, view_valid: torch.Tensor, view_quality: torch.Tensor, global_time_position: torch.Tensor, motion: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.forward_from_backbone_sequence(self.visual.encode_backbone_sequence(images), view_valid, view_quality, global_time_position, motion)
