"""Eqs. (3)-(5): unchanged dataset-native decoding primitives."""

from .core.sequence import (
    RecordingMetadata,
    TransitionModel,
    DecoderConfig,
    fit_transition_model,
    decode_unique_beam_paths,
    decode_unique_beam,
    decode_sessions,
)
from .core.activity import TransitionModel as FirstOrderTransition, _decode as viterbi
from .core.workout import _viterbi as workout_viterbi
