from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from session_kd.mmact_imu import MMACTIMUEncoder, NativeIMUSchema
from session_kd.mmact_backbones.mobind import MoBindLite, UnifiedMoBindStudent

class MMACTMoBindLite(MoBindLite):
    """Original full Skeleton+IMU binding model with only IMU encoder replaced."""

    def __init__(self, *, classes: int=37, width: int=96, alignment_width: int=64, dropout: float=0.12):
        super().__init__(width=width, alignment_width=alignment_width, classes=classes, dropout=dropout, imu_instance_normalization=False, imu_event_feature_width=0, skeleton_time_position=False, skeleton_multistream=False, skeleton_adaptive_graph=False)
        self.imu_encoder = MMACTIMUEncoder(NativeIMUSchema.create(), width=width, dropout=dropout)

def build_mmact_motion_pretrain(*, classes: int=37) -> MMACTMoBindLite:
    return MMACTMoBindLite(classes=classes)
