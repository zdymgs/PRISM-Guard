"""MMAct multimodal student architecture."""

from __future__ import annotations
import torch
from torch import nn
from .activity_imu import NativeIMUSchema, ActivityNativeIMUEncoder
from .activity_visual import ActivityNativeVisualStudent
from .native.mobind import (
    MotionSemanticHead,
    NativeMoBindMotionResidual,
    NativeSeparateMotionEncoder,
    NativeSkeletonPartEncoder,
    NativeUnifiedMoBindStudent,
)


def build_mmact_student(
    *, classes: int = 37, kinetics_pretrained: bool = False
) -> NativeUnifiedMoBindStudent:
    """Build the complete composition with no network/download side effects."""
    visual = ActivityNativeVisualStudent(classes=classes, kinetics_pretrained=kinetics_pretrained)
    skeleton = NativeSkeletonPartEncoder(width=96, dropout=0.12, explicit_time_position=False)
    imu = ActivityNativeIMUEncoder(NativeIMUSchema.create(), width=96, dropout=0.12)
    encoder = NativeSeparateMotionEncoder(skeleton, imu, width=96, modality_dropout=0.0)
    semantic = MotionSemanticHead(width=96, classes=classes, dropout=0.12)
    teacher_projection = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 1024))
    residual = NativeMoBindMotionResidual(
        "separate",
        encoder,
        semantic,
        teacher_projection,
        visual_width=512,
        motion_width=96,
        dropout=0.12,
    )
    return NativeUnifiedMoBindStudent(visual, residual)
