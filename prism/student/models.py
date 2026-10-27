"""Unchanged native architectures; no weights and no download on construction."""

from .activity_model import build_mmact_student as build_mmact
from .workout_model import build_student as build_mmfit
from .native.visual import NativeMC3VisualStudent as _VisualStudent
from .native.mobind import NativeUnifiedMoBindStudent as UnifiedStudent


class VisualStudent(_VisualStudent):
    """Explicit local initialization; automatic network downloads are disabled."""

    def __init__(self, *args, kinetics_pretrained=False, **kwargs):
        if kinetics_pretrained:
            raise ValueError("supply authorized local backbone state; downloads are disabled")
        super().__init__(*args, kinetics_pretrained=False, **kwargs)
