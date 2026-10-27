"""MM-Fit student architecture."""

from __future__ import annotations
import torch
from torch import nn
from .native.mobind import (
    NativeSkeletonPartEncoder,
    NativeSeparateMotionEncoder,
    MotionSemanticHead,
    NativeMoBindMotionResidual,
    NativeUnifiedMoBindStudent,
)
from .activity_visual import ActivityNativeVisualStudent
from .activity_imu import ActivityNativeIMUEncoder, NativeIMUSchema

REGIONS = ("scene", "person", "body_motion")


class MMFitIMUEncoder(ActivityNativeIMUEncoder):
    """Native raw 6-channel stems, channel masks, temporal/device/statistic paths.

    No compensated acceleration/quaternions are invented. Ear/head evidence is
    put in the contextual torso slot, explicitly not a measured torso sensor.
    Left leg remains absent; eight streams remain four actual devices.
    """

    def __init__(self):
        super().__init__(NativeIMUSchema.create(), width=96, dropout=0.12)


def build_student():
    v = ActivityNativeVisualStudent(classes=11, kinetics_pretrained=False, region_names=REGIONS)
    s = NativeSkeletonPartEncoder(width=96, dropout=0.12)
    i = MMFitIMUEncoder()
    encoder = NativeSeparateMotionEncoder(s, i, width=96, modality_dropout=0.0)
    head = MotionSemanticHead(width=96, classes=11, dropout=0.12)
    projection = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 1024))
    residual = NativeMoBindMotionResidual(
        "separate", encoder, head, projection, visual_width=512, motion_width=96, dropout=0.12
    )
    return NativeUnifiedMoBindStudent(v, residual)
