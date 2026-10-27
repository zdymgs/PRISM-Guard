"""The full CUHK-X visual, skeleton, IMU and fusion student."""

from .native.visual import NativeMC3VisualStudent
from .native.mobind import (
    NativeMoBindLite, NativeSeparateMotionEncoder, NativeMoBindMotionResidual,
    NativeUnifiedMoBindStudent,
)


def build_cuhkx_student():
    visual = NativeMC3VisualStudent(
        classes=40, width=512, dropout=0.18, fusion_mode="gated",
        enable_distillation_projection=False, frames=16, kinetics_pretrained=False,
        temporal_modeling=True, exact_time_modeling=False,
        cross_view_time_modeling=False, spatial_region_modeling=False,
        region_temporal_modeling=False, structured_region_modeling=False,
    )
    pretrained = NativeMoBindLite(width=96, alignment_width=64, classes=40, dropout=0.12)
    import torch
    rng = torch.random.get_rng_state()
    encoder = NativeSeparateMotionEncoder(
        pretrained.skeleton_encoder, pretrained.imu_encoder, width=96,
        modality_dropout=0.0,
    )
    torch.random.set_rng_state(rng)
    residual = NativeMoBindMotionResidual(
        "separate", encoder, pretrained.skeleton_head,
        pretrained.skeleton_teacher_projection, visual_width=512,
        motion_width=96, dropout=0.12, initial_residual_strength=0.25,
        reliability_event_feature_width=0, global_fusion_mode="additive",
        reliability_groups=1,
    )
    return NativeUnifiedMoBindStudent(visual, residual)
