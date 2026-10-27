from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

@dataclass(frozen=True)
class SequenceConfig:
    gap_seconds: float = 1.5
    transition_weight: float = 0.45
    emission_scale: float = 1.0
    path_mix: float = 0.51

@dataclass(frozen=True)
class TransitionModel:
    start_log_probability: np.ndarray
    end_log_probability: np.ndarray
    transition_log_probability: np.ndarray

    @property
    def num_classes(self) -> int:
        return int(self.transition_log_probability.shape[0])

@dataclass(frozen=True)
class SequenceResult:
    probabilities: np.ndarray
    classes: np.ndarray
    changed: np.ndarray
    bout_ids: np.ndarray
    edge_ids: tuple[tuple[str, ...], ...]

def _vector(value: Sequence[object] | np.ndarray, name: str, n: int | None=None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    if n is not None and len(result) != n:
        raise ValueError(f'{name} must contain {n} rows')
    return result

def _probabilities(value: np.ndarray, name: str, num_classes: int=11) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != num_classes:
        raise ValueError(f'{name} must have shape [N,{num_classes}]')
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f'{name} must be finite and non-negative')
    total = result.sum(axis=1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError(f'every {name} row must have positive mass')
    return result / total

def _ordered_bouts(workout_ids: np.ndarray, timestamps: np.ndarray, gap_seconds: float) -> tuple[list[np.ndarray], np.ndarray]:
    """Return deterministic bouts; invalid/tied timestamps are singleton fallbacks."""
    n = len(workout_ids)
    bouts: list[np.ndarray] = []
    bout_ids = np.full(n, -1, dtype=np.int64)
    for workout in np.unique(workout_ids):
        rows = np.flatnonzero(workout_ids == workout)
        valid = rows[np.isfinite(timestamps[rows])]
        invalid = rows[~np.isfinite(timestamps[rows])]
        order = valid[np.argsort(timestamps[valid], kind='stable')]
        _, counts = np.unique(timestamps[order], return_counts=True)
        tied_times = set(np.unique(timestamps[order])[counts > 1].tolist())
        current: list[int] = []
        previous: float | None = None
        for row in order:
            timestamp = float(timestamps[row])
            if timestamp in tied_times:
                if current:
                    bouts.append(np.asarray(current, dtype=np.int64))
                    current = []
                bouts.append(np.asarray([row], dtype=np.int64))
                previous = None
                continue
            if previous is None or timestamp - previous <= gap_seconds:
                current.append(int(row))
            else:
                bouts.append(np.asarray(current, dtype=np.int64))
                current = [int(row)]
            previous = timestamp
        if current:
            bouts.append(np.asarray(current, dtype=np.int64))
        bouts.extend((np.asarray([row], dtype=np.int64) for row in invalid))
    for bout_id, rows in enumerate(bouts):
        bout_ids[rows] = bout_id
    if n and np.any(bout_ids < 0):
        raise AssertionError('bout builder lost rows')
    return (bouts, bout_ids)

def fit_source_transition(source_workout_ids: Sequence[object] | np.ndarray, source_timestamps_seconds: Sequence[float] | np.ndarray, source_labels: Sequence[int] | np.ndarray, *, num_classes: int=11, smoothing: float=1.0, gap_seconds: float=1.5) -> TransitionModel:
    """Fit a smoothed source-only first-order HMM, including self-loops."""
    if num_classes <= 1 or smoothing <= 0.0 or gap_seconds <= 0.0:
        raise ValueError('num_classes, smoothing, and gap_seconds must be positive')
    labels = _vector(source_labels, 'source_labels').astype(np.int64)
    n = len(labels)
    if n == 0:
        raise ValueError('source_labels must not be empty')
    workouts = _vector(source_workout_ids, 'source_workout_ids', n)
    timestamps = _vector(source_timestamps_seconds, 'source_timestamps_seconds', n).astype(np.float64)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError('source_labels contain an invalid class index')
    bouts, _ = _ordered_bouts(workouts, timestamps, gap_seconds)
    start = np.full(num_classes, smoothing, dtype=np.float64)
    end = np.full(num_classes, smoothing, dtype=np.float64)
    transition = np.full((num_classes, num_classes), smoothing, dtype=np.float64)
    for rows in bouts:
        path = labels[rows]
        start[path[0]] += 1.0
        end[path[-1]] += 1.0
        if len(path) > 1:
            np.add.at(transition, (path[:-1], path[1:]), 1.0)
    start /= start.sum()
    end /= end.sum()
    transition /= transition.sum(axis=1, keepdims=True)
    return TransitionModel(start_log_probability=np.log(start), end_log_probability=np.log(end), transition_log_probability=np.log(transition))

def _viterbi(emission: np.ndarray, model: TransitionModel, transition_weight: float, emission_scale: float) -> np.ndarray:
    length, classes = emission.shape
    log_emission = np.full_like(emission, -np.inf, dtype=np.float64)
    positive = emission > 0.0
    log_emission[positive] = np.log(emission[positive])
    score = emission_scale * log_emission[0] + transition_weight * model.start_log_probability
    trace = np.zeros((length, classes), dtype=np.int64)
    for position in range(1, length):
        candidates = score[:, None] + transition_weight * model.transition_log_probability
        trace[position] = np.argmax(candidates, axis=0)
        score = candidates[trace[position], np.arange(classes)] + emission_scale * log_emission[position]
    last = int(np.argmax(score + transition_weight * model.end_log_probability))
    path = np.empty(length, dtype=np.int64)
    path[-1] = last
    for position in range(length - 1, 0, -1):
        path[position - 1] = trace[position, path[position]]
    return path

def decode_sequence(workout_ids: Sequence[object] | np.ndarray, timestamps_seconds: Sequence[float] | np.ndarray, probabilities: np.ndarray, transition: TransitionModel, *, sample_ids: Sequence[object] | np.ndarray | None=None, config: SequenceConfig=SequenceConfig()) -> SequenceResult:
    """Decode each workout bout with self-loop-capable Viterbi."""
    base = _probabilities(probabilities, 'probabilities', transition.num_classes)
    n = len(base)
    workouts = _vector(workout_ids, 'workout_ids', n)
    timestamps = _vector(timestamps_seconds, 'timestamps_seconds', n).astype(np.float64)
    ids = np.arange(n).astype(str) if sample_ids is None else _vector(sample_ids, 'sample_ids', n).astype(str)
    if len(np.unique(ids)) != n:
        raise ValueError('sample_ids must be unique')
    if config.gap_seconds <= 0.0 or config.transition_weight < 0.0:
        raise ValueError('gap_seconds must be positive and transition_weight non-negative')
    if not np.isfinite(config.emission_scale) or config.emission_scale <= 0.0:
        raise ValueError('emission_scale must be finite and positive')
    if not 0.5 < config.path_mix <= 1.0:
        raise ValueError('path_mix must be in (0.5,1] to preserve Viterbi argmax')
    c = transition.num_classes
    if transition.start_log_probability.shape != (c,) or transition.end_log_probability.shape != (c,) or transition.transition_log_probability.shape != (c, c):
        raise ValueError('transition model shapes are inconsistent')
    if not (np.isfinite(transition.start_log_probability).all() and np.isfinite(transition.end_log_probability).all() and np.isfinite(transition.transition_log_probability).all()):
        raise ValueError('transition model must be finite')
    bouts, bout_ids = _ordered_bouts(workouts, timestamps, config.gap_seconds)
    path = base.argmax(axis=1).astype(np.int64)
    output = base.copy()
    edge_ids: list[list[str]] = [[] for _ in range(n)]
    for rows in bouts:
        if len(rows) > 1:
            path[rows] = _viterbi(base[rows], transition, config.transition_weight, config.emission_scale)
            one_hot = np.eye(c, dtype=np.float64)[path[rows]]
            output[rows] = (1.0 - config.path_mix) * base[rows] + config.path_mix * one_hot
            for left, right in zip(rows[:-1], rows[1:]):
                edge = f'sequence:{workouts[left]}:{ids[left]}->{ids[right]}'
                edge_ids[int(left)].append(edge)
                edge_ids[int(right)].append(edge)
    output /= output.sum(axis=1, keepdims=True)
    classes = output.argmax(axis=1).astype(np.int64)
    if not np.array_equal(classes, path):
        raise AssertionError('heuristic target argmax differs from Viterbi path')
    return SequenceResult(probabilities=output, classes=classes, changed=classes != base.argmax(axis=1), bout_ids=bout_ids, edge_ids=tuple((tuple(values) for values in edge_ids)))
