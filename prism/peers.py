"""Eqs. (6)-(8): posterior alignment and full expert-bank peer aggregation."""

from .core.alignment import align_probabilities
from .core.content import features as expert_bank_features, mean_prob
from .core.group import build_group_features, fit_group_head
from .core.activity_mix import _monotone as activity_monotone
from .core.workout_peers import bout_repeat_peers
