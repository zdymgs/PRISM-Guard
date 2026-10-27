from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mc3_visual_model import MC3VisualStudent
from mobind_lite_model import (
    MoBindLite,
    MoBindMotionResidual,
    SeparateMotionEncoder,
    UnifiedMoBindStudent,
)


def deployment_model_config(
    visual_config: dict[str, Any],
    pretrain_config: dict[str, Any],
    modality: str,
    fusion_config: dict[str, Any],
) -> dict[str, Any]:
    """Keep only architecture values needed to recreate a standalone Student."""

    if modality != "separate":
        raise ValueError("the frozen challenge deployment architecture is V+separate(S,I)")
    return {
        "visual": dict(visual_config),
        "motion": dict(pretrain_config),
        "fusion": {
            "modality": modality,
            "separate_modality_dropout": float(
                fusion_config.get("separate_modality_dropout", 0.0)
            ),
            "initial_residual_strength": float(
                fusion_config.get("initial_residual_strength", 0.25)
            ),
            "global_fusion_mode": str(
                fusion_config.get("global_fusion_mode", "additive")
            ),
            "reliability_groups": int(fusion_config.get("reliability_groups", 1)),
            "reliability_event_feature_width": 0,
        },
    }


def build_deploy_model(config: dict[str, Any]) -> UnifiedMoBindStudent:
    visual_config = dict(config["visual"])
    if visual_config.get("backbone") != "mc3_18_temporal":
        raise ValueError("challenge deployment requires temporal MC3")
    visual = MC3VisualStudent(
        classes=int(visual_config.get("classes", 40)),
        width=int(visual_config.get("width", 512)),
        dropout=float(visual_config.get("dropout", 0.18)),
        fusion_mode=str(visual_config.get("fusion_mode", "gated")),
        enable_distillation_projection=bool(
            visual_config.get("enable_distillation_projection", False)
        ),
        frames=int(visual_config["frames"]),
        kinetics_pretrained=False,
        temporal_modeling=True,
        exact_time_modeling=bool(visual_config.get("exact_time_modeling", False)),
        cross_view_time_modeling=bool(
            visual_config.get("cross_view_time_modeling", False)
        ),
        spatial_region_modeling=bool(
            visual_config.get("spatial_region_modeling", False)
        ),
        region_temporal_modeling=bool(
            visual_config.get("region_temporal_modeling", False)
        ),
        structured_region_modeling=bool(
            visual_config.get("structured_region_modeling", False)
        ),
    )
    motion_config = dict(config["motion"])
    pretrained_shape = MoBindLite(**motion_config)
    fusion = dict(config["fusion"])
    if fusion.get("modality") != "separate":
        raise ValueError("unsupported frozen challenge deployment modality")
    encoder = SeparateMotionEncoder(
        pretrained_shape.skeleton_encoder,
        pretrained_shape.imu_encoder,
        width=int(motion_config["width"]),
        modality_dropout=float(fusion["separate_modality_dropout"]),
    )
    residual = MoBindMotionResidual(
        "separate",
        encoder,
        pretrained_shape.skeleton_head,
        pretrained_shape.skeleton_teacher_projection,
        visual_width=512,
        motion_width=int(motion_config["width"]),
        dropout=float(motion_config["dropout"]),
        initial_residual_strength=float(fusion["initial_residual_strength"]),
        reliability_event_feature_width=int(
            fusion.get("reliability_event_feature_width", 0)
        ),
        global_fusion_mode=str(fusion["global_fusion_mode"]),
        reliability_groups=int(fusion["reliability_groups"]),
    )
    return UnifiedMoBindStudent(visual, residual)


def load_deploy_checkpoint(
    path: str | Path,
) -> tuple[UnifiedMoBindStudent, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "deployment_model_config" not in checkpoint:
        raise ValueError("checkpoint is not self-contained for challenge deployment")
    model = build_deploy_model(checkpoint["deployment_model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint
