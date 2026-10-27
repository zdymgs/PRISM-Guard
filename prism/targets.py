"""Eq. (11): class-aware smoothing without a new target recipe."""

from .core.group import class_aware_hard_targets


def smooth_targets(chosen, classes, eta=5e-4):
    """Eq. (11): q(c)=eta+(1-C*eta) times teacher-class indicator."""
    return class_aware_hard_targets(chosen, classes, eta)
