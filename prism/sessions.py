"""Eq. (1) metadata/session preparation; legacy grouping remains explicit."""

from .core.sequence import RecordingMetadata, build_sessions
from .core.activity import _blocks as activity_blocks
from .core.workout import _ordered_bouts as workout_blocks
