from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def temporal_shift(x: torch.Tensor, batch_size: int, time_steps: int, fold_divisor: int = 8) -> torch.Tensor:
    _, channels, height, width = x.shape
    sequence = x.reshape(batch_size, time_steps, channels, height, width)
    fold = channels // fold_divisor
    if fold == 0 or time_steps == 1:
        return x
    shifted = torch.zeros_like(sequence)
    shifted[:, :-1, :fold] = sequence[:, 1:, :fold]
    shifted[:, 1:, fold : 2 * fold] = sequence[:, :-1, fold : 2 * fold]
    shifted[:, :, 2 * fold :] = sequence[:, :, 2 * fold :]
    return shifted.reshape(batch_size * time_steps, channels, height, width)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class AttentionPool(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(feature_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        attended = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        maximum = sequence.amax(dim=1)
        return torch.cat([attended, maximum], dim=1)


class SpatialTemporalBranch(nn.Module):
    def __init__(self, input_dim: int, dropout: float, grid: tuple[int, int] = (2, 3)) -> None:
        super().__init__()
        self.grid = grid
        self.num_tokens = grid[0] * grid[1]
        self.token_project = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU()
        )
        self.position = nn.Parameter(torch.zeros(1, 1, self.num_tokens, 256))
        self.spatial_score = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.frame_project = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU()
        )
        self.temporal = TemporalBlock(256, 1, dropout * 0.5)
        self.pool = AttentionPool(256)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels = spatial.shape[:3]
        pooled = nn.functional.adaptive_avg_pool2d(
            spatial.reshape(batch_size * time_steps, channels, *spatial.shape[-2:]),
            self.grid,
        )
        tokens = pooled.flatten(2).transpose(1, 2).reshape(
            batch_size, time_steps, self.num_tokens, channels
        )
        tokens = self.token_project(tokens) + self.position
        weights = torch.softmax(self.spatial_score(tokens).squeeze(-1), dim=2)
        attended = torch.sum(tokens * weights.unsqueeze(-1), dim=2)
        maximum = tokens.amax(dim=2)
        sequence = self.frame_project(torch.cat([attended, maximum], dim=-1))
        sequence = self.temporal(sequence.transpose(1, 2)).transpose(1, 2)
        return self.pool(sequence)


class VisualEncoder(nn.Module):
    def __init__(
        self,
        modalities: tuple[str, ...],
        dropout: float,
        depth_input_channels: int = 3,
        use_layer3_spatial: bool = False,
        imagenet_pretrained: bool = False,
        ir_stem_initialization: str = "mean",
        stem_fusion: str = "concat",
        use_ir_motion: bool = False,
        use_ir_local: bool = False,
    ) -> None:
        super().__init__()
        self.use_depth = "depth" in modalities
        self.use_ir = "ir" in modalities
        self.use_ir_motion = bool(use_ir_motion)
        self.use_ir_local = bool(use_ir_local)
        self.depth_input_channels = depth_input_channels
        self.use_layer3_spatial = use_layer3_spatial
        self.stem_fusion = stem_fusion
        if ir_stem_initialization not in {"mean", "sum"}:
            raise ValueError(f"未知 IR stem 初始化：{ir_stem_initialization}")
        if stem_fusion not in {"concat", "ir_depth_residual"}:
            raise ValueError(f"未知视觉 stem 融合：{stem_fusion}")
        if stem_fusion == "ir_depth_residual" and not (
            self.use_depth and self.use_ir
        ):
            raise ValueError("ir_depth_residual 只适用于 Depth+IR")
        if self.use_ir_motion and (not self.use_ir or self.use_depth):
            raise ValueError("IR motion currently requires an IR-only visual branch")
        if self.use_ir_local and (not self.use_ir or self.use_depth):
            raise ValueError("IR local currently requires an IR-only visual branch")
        if self.use_ir_motion and self.use_ir_local:
            raise ValueError("IR motion and IR local residuals are separate experiments")
        if not (self.use_depth or self.use_ir):
            raise ValueError("视觉编码器至少需要 depth 或 ir")

        if self.use_depth and self.use_ir:
            stem_channels = 64 if stem_fusion == "ir_depth_residual" else 32
            self.depth_stem = nn.Conv2d(
                depth_input_channels, stem_channels, 7, stride=2, padding=3, bias=False
            )
            self.ir_stem = nn.Conv2d(
                1, stem_channels, 7, stride=2, padding=3, bias=False
            )
            if stem_fusion == "ir_depth_residual":
                self.depth_residual_scale = nn.Parameter(torch.tensor(0.0))
            else:
                self.register_parameter("depth_residual_scale", None)
        elif self.use_depth:
            self.depth_stem = nn.Conv2d(depth_input_channels, 64, 7, stride=2, padding=3, bias=False)
            self.ir_stem = None
            self.register_parameter("depth_residual_scale", None)
        else:
            self.depth_stem = None
            self.ir_stem = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
            self.register_parameter("depth_residual_scale", None)
        if self.use_ir_motion:
            self.ir_motion_stem = nn.Conv2d(
                1, 64, 7, stride=2, padding=3, bias=False
            )
            self.ir_motion_norm = nn.LayerNorm(1024)
            self.ir_motion_residual_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.ir_motion_stem = None
            self.ir_motion_norm = None
            self.register_parameter("ir_motion_residual_scale", None)
        if self.use_ir_local:
            self.ir_local_stem = nn.Conv2d(
                1, 64, 7, stride=2, padding=3, bias=False
            )
            self.ir_local_norm = nn.LayerNorm(1024)
            self.ir_local_residual_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.ir_local_stem = None
            self.ir_local_norm = None
            self.register_parameter("ir_local_residual_scale", None)

        backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        )
        if imagenet_pretrained:
            with torch.no_grad():
                if self.use_depth and self.use_ir:
                    assert self.depth_stem is not None and self.ir_stem is not None
                    if stem_fusion == "ir_depth_residual":
                        if depth_input_channels == 3:
                            self.depth_stem.weight.copy_(backbone.conv1.weight)
                        else:
                            depth_weight = backbone.conv1.weight.mean(
                                dim=1, keepdim=True
                            )
                            self.depth_stem.weight.copy_(
                                depth_weight.repeat(
                                    1, depth_input_channels, 1, 1
                                )
                                / depth_input_channels
                            )
                        ir_weight = backbone.conv1.weight
                    else:
                        # 两个 32 通道 stem 合并后仍对应预训练 ResNet 的
                        # 64 通道输入。
                        self.depth_stem.weight.copy_(backbone.conv1.weight[:32])
                        ir_weight = backbone.conv1.weight[32:]
                    ir_weight = (
                        ir_weight.sum(dim=1, keepdim=True)
                        if ir_stem_initialization == "sum"
                        else ir_weight.mean(dim=1, keepdim=True)
                    )
                    self.ir_stem.weight.copy_(ir_weight)
                elif self.use_depth:
                    assert self.depth_stem is not None
                    if depth_input_channels == 3:
                        self.depth_stem.weight.copy_(backbone.conv1.weight)
                    else:
                        mean_weight = backbone.conv1.weight.mean(dim=1, keepdim=True)
                        self.depth_stem.weight.copy_(
                            mean_weight.repeat(1, depth_input_channels, 1, 1)
                            / depth_input_channels
                        )
                else:
                    assert self.ir_stem is not None
                    ir_weight = (
                        backbone.conv1.weight.sum(dim=1, keepdim=True)
                        if ir_stem_initialization == "sum"
                        else backbone.conv1.weight.mean(dim=1, keepdim=True)
                    )
                    self.ir_stem.weight.copy_(ir_weight)
                    if self.ir_motion_stem is not None:
                        self.ir_motion_stem.weight.copy_(ir_weight)
                    if self.ir_local_stem is not None:
                        self.ir_local_stem.weight.copy_(ir_weight)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        if use_layer3_spatial:
            self.layer3_spatial = SpatialTemporalBranch(256, dropout)
            self.layer3_residual_project = nn.Sequential(
                nn.Linear(512, 1024), nn.LayerNorm(1024)
            )
            self.layer3_residual_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.layer3_spatial = None
            self.layer3_residual_project = None
            self.register_parameter("layer3_residual_scale", None)
        self.temporal = nn.Sequential(
            TemporalBlock(512, 1, dropout * 0.5),
            TemporalBlock(512, 2, dropout * 0.5),
        )
        self.pool = AttentionPool(512)
        self.output_dim = 1024

    def train(self, mode: bool = True) -> "VisualEncoder":
        super().train(mode)
        if self.use_ir_motion or self.use_ir_local:
            # The motion stream reuses the appearance trunk. Keep the
            # fold-pure appearance BN statistics fixed so motion intensity
            # cannot overwrite them during residual adaptation.
            for module in self.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.eval()
        return self

    def encode_pooled(
        self,
        inputs: dict[str, torch.Tensor],
        return_spatial: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        reference = inputs["depth"] if self.use_depth else inputs["ir"]
        batch_size, time_steps = reference.shape[:2]
        features: list[torch.Tensor] = []
        if self.use_depth:
            depth = inputs["depth"].reshape(
                batch_size * time_steps,
                self.depth_input_channels,
                *reference.shape[-2:],
            )
            features.append(self.depth_stem(depth))
        if self.use_ir:
            ir = inputs["ir"].reshape(batch_size * time_steps, 1, *reference.shape[-2:])
            features.append(self.ir_stem(ir))
        if len(features) == 2 and self.stem_fusion == "ir_depth_residual":
            assert self.depth_residual_scale is not None
            # Features are appended Depth then IR. At zero scale this path
            # exactly reproduces the initialized IR-only encoder.
            x = features[1] + torch.tanh(self.depth_residual_scale) * features[0]
        else:
            x = torch.cat(features, dim=1) if len(features) == 2 else features[0]
        encoded_batch_size = batch_size
        if self.use_ir_motion:
            if "ir_motion" not in inputs:
                raise KeyError("IR motion model requires inputs['ir_motion']")
            assert self.ir_motion_stem is not None
            motion = inputs["ir_motion"].reshape(
                batch_size * time_steps, 1, *reference.shape[-2:]
            )
            x = torch.cat([x, self.ir_motion_stem(motion)], dim=0)
            encoded_batch_size = batch_size * 2
        elif self.use_ir_local:
            if "ir_local" not in inputs:
                raise KeyError("IR local model requires inputs['ir_local']")
            assert self.ir_local_stem is not None
            local_ir = inputs["ir_local"].reshape(
                batch_size * time_steps, 1, *reference.shape[-2:]
            )
            x = torch.cat([x, self.ir_local_stem(local_ir)], dim=0)
            encoded_batch_size = batch_size * 2
        x = self.maxpool(self.relu(self.bn1(x)))
        x = self.layer1(temporal_shift(x, encoded_batch_size, time_steps))
        x = self.layer2(temporal_shift(x, encoded_batch_size, time_steps))
        x = self.layer3(temporal_shift(x, encoded_batch_size, time_steps))
        layer3_spatial = x.reshape(
            encoded_batch_size, time_steps, 256, *x.shape[-2:]
        )
        local = self.layer3_spatial(layer3_spatial) if self.layer3_spatial is not None else None
        x = self.layer4(temporal_shift(x, encoded_batch_size, time_steps))
        spatial = x.reshape(
            encoded_batch_size, time_steps, 512, *x.shape[-2:]
        )
        sequence = spatial.mean(dim=(-2, -1))
        sequence = self.temporal(sequence.transpose(1, 2)).transpose(1, 2)
        pooled = self.pool(sequence)
        if self.use_ir_motion:
            assert (
                self.ir_motion_norm is not None
                and self.ir_motion_residual_scale is not None
            )
            appearance, motion_pooled = pooled[:batch_size], pooled[batch_size:]
            pooled = appearance + torch.tanh(
                self.ir_motion_residual_scale
            ) * self.ir_motion_norm(motion_pooled)
            spatial = spatial[:batch_size]
        elif self.use_ir_local:
            assert (
                self.ir_local_norm is not None
                and self.ir_local_residual_scale is not None
            )
            appearance, local_pooled = pooled[:batch_size], pooled[batch_size:]
            quality = inputs.get("ir_local_quality")
            if quality is None:
                raise KeyError("IR local model requires inputs['ir_local_quality']")
            quality = quality.to(
                device=local_pooled.device, dtype=local_pooled.dtype
            ).reshape(batch_size, 1)
            pooled = appearance + torch.tanh(
                self.ir_local_residual_scale
            ) * quality * self.ir_local_norm(local_pooled)
            spatial = spatial[:batch_size]
        if local is not None:
            assert self.layer3_residual_project is not None
            assert self.layer3_residual_scale is not None
            pooled = pooled + self.layer3_residual_scale * self.layer3_residual_project(local)
        return (pooled, spatial) if return_spatial else pooled

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        return_spatial: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.encode_pooled(inputs, return_spatial=return_spatial)


class SkeletonEncoder(nn.Module):
    def __init__(self, dropout: float, input_dim: int = 4) -> None:
        super().__init__()
        self.joint_embed = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU())
        self.frame_project = nn.Sequential(nn.Linear(17 * 32, 256), nn.LayerNorm(256), nn.GELU())
        self.temporal = nn.Sequential(
            TemporalBlock(256, 1, dropout * 0.5),
            TemporalBlock(256, 2, dropout * 0.5),
        )
        self.pool = AttentionPool(256)
        self.output_dim = 512

    def encode_sequence(self, skeleton: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, joints, channels = skeleton.shape
        x = self.joint_embed(skeleton).reshape(batch_size, time_steps, joints * 32)
        x = self.frame_project(x)
        return self.temporal(x.transpose(1, 2)).transpose(1, 2)

    def encode_pooled(self, skeleton: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encode_sequence(skeleton))

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        return self.encode_pooled(skeleton)


class AlignedMultimodalModel(nn.Module):
    def __init__(
        self,
        modalities: list[str],
        num_classes: int = 40,
        dropout: float = 0.3,
        use_aux_heads: bool = False,
        use_modality_masks: bool = False,
        use_cross_attention: bool = False,
        cross_attention_grid: tuple[int, int] = (2, 3),
        cross_attention_heads: int = 4,
        depth_input_channels: int = 3,
        use_layer3_spatial: bool = False,
        imagenet_pretrained: bool = False,
        ir_stem_initialization: str = "mean",
        visual_stem_fusion: str = "concat",
        skeleton_input_dim: int = 4,
        use_ir_motion: bool = False,
        use_ir_local: bool = False,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.use_aux_heads = use_aux_heads
        self.use_modality_masks = use_modality_masks
        self.use_cross_attention = use_cross_attention
        self.cross_attention_grid = cross_attention_grid
        self.use_visual = "depth" in self.modalities or "ir" in self.modalities
        self.use_skeleton = "skeleton" in self.modalities
        if not (self.use_visual or self.use_skeleton):
            raise ValueError("至少需要一个输入模态")
        self.visual = (
            VisualEncoder(
                self.modalities,
                dropout,
                depth_input_channels,
                use_layer3_spatial,
                imagenet_pretrained,
                ir_stem_initialization,
                visual_stem_fusion,
                use_ir_motion,
                use_ir_local,
            )
            if self.use_visual
            else None
        )
        if self.use_skeleton:
            self.skeleton = SkeletonEncoder(dropout, skeleton_input_dim)
        else:
            self.skeleton = None

        if self.use_visual and self.use_skeleton:
            if use_cross_attention:
                self.depth_token_project = nn.Linear(512, 256)
                self.cross_attention = nn.MultiheadAttention(
                    256,
                    cross_attention_heads,
                    dropout=dropout * 0.5,
                    batch_first=True,
                )
                self.cross_attention_norm = nn.LayerNorm(256)
                self.cross_attention_scale = nn.Parameter(torch.tensor(0.0))
            else:
                self.depth_token_project = None
                self.cross_attention = None
                self.cross_attention_norm = None
                self.register_parameter("cross_attention_scale", None)
            self.visual_project = nn.Sequential(nn.Linear(1024, 384), nn.LayerNorm(384), nn.GELU())
            self.skeleton_project = nn.Sequential(nn.Linear(512, 384), nn.LayerNorm(384), nn.GELU())
            gate_input_dim = 770 if use_modality_masks else 768
            self.gate = nn.Linear(gate_input_dim, 384)
            self.classifier = nn.Sequential(nn.LayerNorm(384 * 3), nn.Dropout(dropout), nn.Linear(384 * 3, num_classes))
            if use_aux_heads:
                self.visual_aux_classifier = nn.Sequential(
                    nn.LayerNorm(384), nn.Dropout(dropout), nn.Linear(384, num_classes)
                )
                self.skeleton_aux_classifier = nn.Sequential(
                    nn.LayerNorm(384), nn.Dropout(dropout), nn.Linear(384, num_classes)
                )
            else:
                self.visual_aux_classifier = None
                self.skeleton_aux_classifier = None
            if use_modality_masks:
                self.visual_missing_token = nn.Parameter(torch.zeros(1, 384))
                self.skeleton_missing_token = nn.Parameter(torch.zeros(1, 384))
            else:
                self.register_parameter("visual_missing_token", None)
                self.register_parameter("skeleton_missing_token", None)
        elif self.use_visual:
            self.classifier = nn.Sequential(nn.LayerNorm(1024), nn.Dropout(dropout), nn.Linear(1024, num_classes))
        else:
            self.classifier = nn.Sequential(nn.LayerNorm(512), nn.Dropout(dropout), nn.Linear(512, num_classes))

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        return_aux: bool = False,
        return_embedding: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if not self.use_visual:
            assert self.skeleton is not None
            embedding = self.skeleton(inputs["skeleton"])
            logits = self.classifier(embedding)
            return (
                {"logits": logits, "embedding": embedding}
                if return_embedding
                else logits
            )
        assert self.visual is not None
        if not self.use_skeleton:
            visual = self.visual(inputs)
            assert isinstance(visual, torch.Tensor)
            logits = self.classifier(visual)
            return (
                {"logits": logits, "embedding": visual}
                if return_embedding
                else logits
            )
        assert self.skeleton is not None
        if self.use_cross_attention:
            visual_output = self.visual(inputs, return_spatial=True)
            assert isinstance(visual_output, tuple)
            visual, spatial = visual_output
            skeleton_sequence = self.skeleton.encode_sequence(inputs["skeleton"])
            batch_size, time_steps = skeleton_sequence.shape[:2]
            pooled_spatial = nn.functional.adaptive_avg_pool2d(
                spatial.reshape(batch_size * time_steps, 512, *spatial.shape[-2:]),
                self.cross_attention_grid,
            )
            depth_tokens = pooled_spatial.flatten(2).transpose(1, 2)
            assert self.depth_token_project is not None and self.cross_attention is not None
            depth_tokens = self.depth_token_project(depth_tokens)
            query = skeleton_sequence.reshape(batch_size * time_steps, 1, 256)
            attended, _ = self.cross_attention(query, depth_tokens, depth_tokens, need_weights=False)
            assert self.cross_attention_norm is not None and self.cross_attention_scale is not None
            aligned = query + self.cross_attention_scale * self.cross_attention_norm(attended)
            skeleton = self.skeleton.pool(aligned.reshape(batch_size, time_steps, 256))
        else:
            visual = self.visual(inputs)
            assert isinstance(visual, torch.Tensor)
            skeleton = self.skeleton(inputs["skeleton"])
        visual = self.visual_project(visual)
        skeleton = self.skeleton_project(skeleton)
        visual_aux = visual
        skeleton_aux = skeleton
        if self.use_modality_masks:
            visual_present = inputs.get(
                "depth_present", torch.ones(len(visual), 1, device=visual.device, dtype=visual.dtype)
            ).to(device=visual.device, dtype=visual.dtype)
            skeleton_present = inputs.get(
                "skeleton_present",
                torch.ones(len(skeleton), 1, device=skeleton.device, dtype=skeleton.dtype),
            ).to(device=skeleton.device, dtype=skeleton.dtype)
            assert self.visual_missing_token is not None and self.skeleton_missing_token is not None
            visual = visual_present * visual + (1.0 - visual_present) * self.visual_missing_token
            skeleton = (
                skeleton_present * skeleton + (1.0 - skeleton_present) * self.skeleton_missing_token
            )
            gate_input = torch.cat([visual, skeleton, visual_present, skeleton_present], dim=1)
        else:
            gate_input = torch.cat([visual, skeleton], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        fused = gate * visual + (1.0 - gate) * skeleton
        embedding = torch.cat([visual, skeleton, fused], dim=1)
        logits = self.classifier(embedding)
        if return_embedding:
            return {"logits": logits, "embedding": embedding}
        if not return_aux:
            return logits
        if not self.use_aux_heads:
            raise RuntimeError("模型没有启用辅助分类头")
        assert self.visual_aux_classifier is not None and self.skeleton_aux_classifier is not None
        return {
            "logits": logits,
            "depth_logits": self.visual_aux_classifier(visual_aux),
            "skeleton_logits": self.skeleton_aux_classifier(skeleton_aux),
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def fp32_size_mb(model: nn.Module) -> float:
    return parameter_count(model) * 4 / (1024**2)
