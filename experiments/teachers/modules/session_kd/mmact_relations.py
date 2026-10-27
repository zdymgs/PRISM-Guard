from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence
import numpy as np

@dataclass(frozen=True)
class RelationConfig:
    peer_weight: float = 0.25
    match_threshold: float = 0.55
    skip_penalty: float = 0.2
    transition_weight: float = 0.45
    path_mix: float = 0.51

@dataclass(frozen=True)
class TransitionModel:
    start_log_probability: np.ndarray
    end_log_probability: np.ndarray
    transition_log_probability: np.ndarray

    @property
    def num_classes(self) -> int:
        return int(self.transition_log_probability.shape[0])

def _vector(value: Sequence[object] | np.ndarray, name: str, n: int | None=None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or (n is not None and len(result) != n):
        suffix = '' if n is None else f' with {n} rows'
        raise ValueError(f'{name} must be one-dimensional{suffix}')
    return result

def _probabilities(value: np.ndarray, name: str, classes: int | None=None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or (classes is not None and result.shape[1] != classes):
        raise ValueError(f'{name} has an invalid shape')
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result.sum(1) <= 0):
        raise ValueError(f'{name} must contain finite non-negative rows with positive mass')
    return result / result.sum(1, keepdims=True)

def bhattacharyya_posterior_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise Bhattacharyya coefficients between class posteriors."""
    a = _probabilities(left, 'left posteriors')
    b = _probabilities(right, 'right posteriors', a.shape[1])
    return np.sqrt(a) @ np.sqrt(b).T

def monotone_bhattacharyya_matches(left_posteriors: np.ndarray, right_posteriors: np.ndarray, *, threshold: float=0.55, skip_penalty: float=0.2) -> tuple[tuple[int, int, float], ...]:
    """Globally align two ordered trials using monotone DP with explicit skips."""
    similarity = bhattacharyya_posterior_similarity(left_posteriors, right_posteriors)
    m, n = similarity.shape
    score = np.full((m + 1, n + 1), -np.inf)
    action = np.zeros((m + 1, n + 1), dtype=np.int8)
    score[0, 0] = 0.0
    score[1:, 0] = -skip_penalty * np.arange(1, m + 1)
    score[0, 1:] = -skip_penalty * np.arange(1, n + 1)
    action[1:, 0] = 1
    action[0, 1:] = 2
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score[i - 1, j - 1] + similarity[i - 1, j - 1] - threshold
            options = (match, score[i - 1, j] - skip_penalty, score[i, j - 1] - skip_penalty)
            action[i, j] = int(np.argmax(options))
            score[i, j] = options[action[i, j]]
    matches: list[tuple[int, int, float]] = []
    i, j = (m, n)
    while i or j:
        move = int(action[i, j])
        if move == 0:
            value = float(similarity[i - 1, j - 1])
            if value >= threshold:
                matches.append((i - 1, j - 1, value))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return tuple(reversed(matches))

def build_repeat_probabilities(subject: np.ndarray, scene: np.ndarray, session: np.ndarray, starts: np.ndarray, ids: np.ndarray, features: np.ndarray, base: np.ndarray, config: RelationConfig) -> tuple[np.ndarray, tuple[tuple[str, ...], ...], tuple[tuple[float, ...], ...]]:
    """Aggregate monotone peers from different sessions of the same subject/scene."""
    peers: list[list[tuple[int, float]]] = [[] for _ in range(len(base))]
    scopes = sorted(set(zip(subject.tolist(), scene.tolist())))
    for subject_id, scene_id in scopes:
        scope = np.flatnonzero((subject == subject_id) & (scene == scene_id))
        session_values = sorted(np.unique(session[scope]).tolist())
        for left_session, right_session in combinations(session_values, 2):
            left = scope[session[scope] == left_session]
            right = scope[session[scope] == right_session]
            left = left[np.lexsort((ids[left], starts[left]))]
            right = right[np.lexsort((ids[right], starts[right]))]
            for li, ri, similarity in monotone_bhattacharyya_matches(base[left], base[right], threshold=config.match_threshold, skip_penalty=config.skip_penalty):
                a, b = (int(left[li]), int(right[ri]))
                peers[a].append((b, similarity))
                peers[b].append((a, similarity))
    output = base.copy()
    peer_ids: list[tuple[str, ...]] = []
    peer_sims: list[tuple[float, ...]] = []
    for row, values in enumerate(peers):
        values.sort(key=lambda item: (ids[item[0]], -item[1]))
        peer_ids.append(tuple((ids[peer] for peer, _ in values)))
        peer_sims.append(tuple((float(value) for _, value in values)))
        if values:
            peer_mean = base[[peer for peer, _ in values]].mean(0)
            output[row] = (1 - config.peer_weight) * base[row] + config.peer_weight * peer_mean
    output /= output.sum(1, keepdims=True)
    return (output, tuple(peer_ids), tuple(peer_sims))

def _blocks(stream_keys: Sequence[tuple[str, str, str]], starts: np.ndarray, ids: np.ndarray, ordinals: np.ndarray | None):
    """Build labeled-action sequences; wall-clock pauses are not missing actions."""
    blocks: list[np.ndarray] = []
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for row, key in enumerate(stream_keys):
        grouped.setdefault(key, []).append(row)
    for key in sorted(grouped):
        rows = np.asarray(grouped[key], dtype=np.int64)
        if ordinals is None:
            rows = rows[np.lexsort((ids[rows], starts[rows]))]
        else:
            rows = rows[np.lexsort((ids[rows], starts[rows], ordinals[rows]))]
        current = [int(rows[0])] if len(rows) else []
        for row in rows[1:]:
            previous = current[-1]
            consecutive = ordinals is None or ordinals[row] == ordinals[previous] + 1
            if consecutive:
                current.append(int(row))
            else:
                blocks.append(np.asarray(current, dtype=np.int64))
                current = [int(row)]
        if current:
            blocks.append(np.asarray(current, dtype=np.int64))
    return blocks

def fit_source_transition(subject_ids: Sequence[object], scene_ids: Sequence[object], session_ids: Sequence[object], starts: Sequence[float], ends: Sequence[float], interval_ids: Sequence[object], labels: Sequence[int], *, num_classes: int, annotation_ordinals: Sequence[int] | None=None, smoothing: float=1.0) -> TransitionModel:
    """Fit source-train transitions only; self-loops are counted and allowed."""
    y = _vector(labels, 'labels').astype(np.int64)
    n = len(y)
    subject = _vector(subject_ids, 'subject_ids', n).astype(str)
    scene = _vector(scene_ids, 'scene_ids', n).astype(str)
    session = _vector(session_ids, 'session_ids', n).astype(str)
    start = _vector(starts, 'starts', n).astype(float)
    end = _vector(ends, 'ends', n).astype(float)
    ids = _vector(interval_ids, 'interval_ids', n).astype(str)
    ordinals = None if annotation_ordinals is None else _vector(annotation_ordinals, 'annotation_ordinals', n).astype(np.int64)
    if n == 0 or smoothing <= 0 or np.any(y < 0) or np.any(y >= num_classes):
        raise ValueError('invalid source labels or transition parameters')
    start_count = np.full(num_classes, smoothing)
    end_count = np.full(num_classes, smoothing)
    transition = np.full((num_classes, num_classes), smoothing)
    streams = list(zip(subject.tolist(), scene.tolist(), session.tolist()))
    for rows in _blocks(streams, start, ids, ordinals):
        path = y[rows]
        start_count[path[0]] += 1
        end_count[path[-1]] += 1
        if len(path) > 1:
            np.add.at(transition, (path[:-1], path[1:]), 1)
    start_count /= start_count.sum()
    end_count /= end_count.sum()
    transition /= transition.sum(1, keepdims=True)
    return TransitionModel(np.log(start_count), np.log(end_count), np.log(transition))

def _decode(emission: np.ndarray, model: TransitionModel, weight: float) -> np.ndarray:
    length, classes = emission.shape
    log_e = np.log(np.clip(emission, 1e-300, 1.0))
    score = log_e[0] + weight * model.start_log_probability
    trace = np.zeros((length, classes), dtype=np.int64)
    for position in range(1, length):
        candidate = score[:, None] + weight * model.transition_log_probability
        trace[position] = np.argmax(candidate, axis=0)
        score = candidate[trace[position], np.arange(classes)] + log_e[position]
    path = np.empty(length, dtype=np.int64)
    path[-1] = np.argmax(score + weight * model.end_log_probability)
    for position in range(length - 1, 0, -1):
        path[position - 1] = trace[position, path[position]]
    return path
