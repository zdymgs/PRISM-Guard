"""Label-firewalled relation adapter for interval-level MMACT predictions.

The module is deliberately data-loader agnostic.  A caller supplies exactly one
row per annotated interval; synchronized camera views must have been aggregated
before this boundary.  Target compilation never accepts labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np


CANDIDATE_NAMES = ("direct", "repeat", "sequence", "combined")


@dataclass(frozen=True)
class RelationConfig:
    peer_weight: float = 0.25
    match_threshold: float = 0.55
    skip_penalty: float = 0.20
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


@dataclass(frozen=True)
class AcceptancePolicy:
    candidate: str
    margin: float
    validation_net_gain: int
    validation_rescues: int
    validation_harms: int
    validation_changed: int


@dataclass(frozen=True)
class RelationResult:
    interval_ids: np.ndarray
    probabilities: Mapping[str, np.ndarray]
    classes: Mapping[str, np.ndarray]
    repeat_peer_ids: tuple[tuple[str, ...], ...]
    repeat_peer_similarities: tuple[tuple[float, ...], ...]
    sequence_block_ids: np.ndarray


@dataclass(frozen=True)
class AcceptedResult:
    probabilities: np.ndarray
    classes: np.ndarray
    accepted: np.ndarray
    candidate: np.ndarray


def _vector(value: Sequence[object] | np.ndarray, name: str, n: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or (n is not None and len(result) != n):
        suffix = "" if n is None else f" with {n} rows"
        raise ValueError(f"{name} must be one-dimensional{suffix}")
    return result


def _probabilities(value: np.ndarray, name: str, classes: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or (classes is not None and result.shape[1] != classes):
        raise ValueError(f"{name} has an invalid shape")
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result.sum(1) <= 0):
        raise ValueError(f"{name} must contain finite non-negative rows with positive mass")
    return result / result.sum(1, keepdims=True)


def _validate_config(config: RelationConfig) -> None:
    if not 0 <= config.peer_weight <= 0.5:
        raise ValueError("peer_weight must be in [0,0.5]")
    if not 0 <= config.match_threshold <= 1:
        raise ValueError("match_threshold must be in [0,1]")
    if config.skip_penalty < 0 or config.transition_weight < 0:
        raise ValueError("penalties and weights must be non-negative")
    if not 0.5 < config.path_mix <= 1:
        raise ValueError("path_mix must be in (0.5,1]")


def _validate_intervals(
    subject_ids: Sequence[object], scene_ids: Sequence[object], session_ids: Sequence[object],
    starts: Sequence[float], ends: Sequence[float], interval_ids: Sequence[object],
    features: np.ndarray, probabilities: np.ndarray,
) -> tuple[np.ndarray, ...]:
    base = _probabilities(probabilities, "probabilities")
    n = len(base)
    subject = _vector(subject_ids, "subject_ids", n).astype(str)
    scene = _vector(scene_ids, "scene_ids", n).astype(str)
    session = _vector(session_ids, "session_ids", n).astype(str)
    start = _vector(starts, "starts", n).astype(np.float64)
    end = _vector(ends, "ends", n).astype(np.float64)
    ids = _vector(interval_ids, "interval_ids", n).astype(str)
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) != n or not np.isfinite(x).all():
        raise ValueError("features must be a finite [N,D] matrix")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms <= 0) or not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("features must be non-zero and L2-normalized per interval")
    if len(np.unique(ids)) != n:
        raise ValueError("interval_ids must be unique")
    if not np.isfinite(start).all() or not np.isfinite(end).all() or np.any(end < start):
        raise ValueError("interval boundaries must be finite with end >= start")
    keys = list(zip(subject, scene, session, start.tolist(), end.tolist()))
    if len(set(keys)) != n:
        raise ValueError("one row per interval is required; aggregate synchronized cameras first")
    return subject, scene, session, start, end, ids, x, base


def bhattacharyya_posterior_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise Bhattacharyya coefficients between class posteriors."""
    a = _probabilities(left, "left posteriors")
    b = _probabilities(right, "right posteriors", a.shape[1])
    return np.sqrt(a) @ np.sqrt(b).T


def monotone_bhattacharyya_matches(
    left_posteriors: np.ndarray, right_posteriors: np.ndarray, *,
    threshold: float = 0.55, skip_penalty: float = 0.20,
) -> tuple[tuple[int, int, float], ...]:
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
    i, j = m, n
    while i or j:
        move = int(action[i, j])
        if move == 0:
            value = float(similarity[i - 1, j - 1])
            if value >= threshold:
                matches.append((i - 1, j - 1, value))
            i -= 1; j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return tuple(reversed(matches))


def build_repeat_probabilities(
    subject: np.ndarray, scene: np.ndarray, session: np.ndarray, starts: np.ndarray,
    ids: np.ndarray, features: np.ndarray, base: np.ndarray, config: RelationConfig,
) -> tuple[np.ndarray, tuple[tuple[str, ...], ...], tuple[tuple[float, ...], ...]]:
    """Aggregate monotone peers from different sessions of the same subject/scene.

    ``features`` is unused; similarity is posterior Bhattacharyya only.
    """
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
            for li, ri, similarity in monotone_bhattacharyya_matches(
                base[left], base[right], threshold=config.match_threshold,
                skip_penalty=config.skip_penalty,
            ):
                a, b = int(left[li]), int(right[ri])
                peers[a].append((b, similarity)); peers[b].append((a, similarity))
    output = base.copy()
    peer_ids: list[tuple[str, ...]] = []
    peer_sims: list[tuple[float, ...]] = []
    for row, values in enumerate(peers):
        values.sort(key=lambda item: (ids[item[0]], -item[1]))
        peer_ids.append(tuple(ids[peer] for peer, _ in values))
        peer_sims.append(tuple(float(value) for _, value in values))
        if values:
            peer_mean = base[[peer for peer, _ in values]].mean(0)
            output[row] = (1 - config.peer_weight) * base[row] + config.peer_weight * peer_mean
    output /= output.sum(1, keepdims=True)
    return output, tuple(peer_ids), tuple(peer_sims)


def _blocks(
    stream_keys: Sequence[tuple[str, str, str]], starts: np.ndarray,
    ids: np.ndarray, ordinals: np.ndarray | None,
):
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
                blocks.append(np.asarray(current, dtype=np.int64)); current = [int(row)]
        if current:
            blocks.append(np.asarray(current, dtype=np.int64))
    return blocks


def fit_source_transition(
    subject_ids: Sequence[object], scene_ids: Sequence[object], session_ids: Sequence[object],
    starts: Sequence[float], ends: Sequence[float], interval_ids: Sequence[object],
    labels: Sequence[int], *, num_classes: int,
    annotation_ordinals: Sequence[int] | None = None, smoothing: float = 1.0,
) -> TransitionModel:
    """Fit source-train transitions only; self-loops are counted and allowed."""
    y = _vector(labels, "labels").astype(np.int64)
    n = len(y)
    subject = _vector(subject_ids, "subject_ids", n).astype(str)
    scene = _vector(scene_ids, "scene_ids", n).astype(str)
    session = _vector(session_ids, "session_ids", n).astype(str)
    start = _vector(starts, "starts", n).astype(float)
    end = _vector(ends, "ends", n).astype(float)
    ids = _vector(interval_ids, "interval_ids", n).astype(str)
    ordinals = None if annotation_ordinals is None else _vector(
        annotation_ordinals, "annotation_ordinals", n
    ).astype(np.int64)
    if n == 0 or smoothing <= 0 or np.any(y < 0) or np.any(y >= num_classes):
        raise ValueError("invalid source labels or transition parameters")
    start_count = np.full(num_classes, smoothing); end_count = np.full(num_classes, smoothing)
    transition = np.full((num_classes, num_classes), smoothing)
    streams = list(zip(subject.tolist(), scene.tolist(), session.tolist()))
    for rows in _blocks(streams, start, ids, ordinals):
        path = y[rows]; start_count[path[0]] += 1; end_count[path[-1]] += 1
        if len(path) > 1:
            np.add.at(transition, (path[:-1], path[1:]), 1)
    start_count /= start_count.sum(); end_count /= end_count.sum(); transition /= transition.sum(1, keepdims=True)
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


def sequence_probabilities(
    subject: np.ndarray, scene: np.ndarray, session: np.ndarray,
    starts: np.ndarray, ends: np.ndarray, ids: np.ndarray,
    emission: np.ndarray, model: TransitionModel, config: RelationConfig,
    annotation_ordinals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if emission.shape[1] != model.num_classes:
        raise ValueError("transition and posterior class counts differ")
    output = emission.copy(); block_ids = np.full(len(emission), -1, dtype=np.int64)
    streams = list(zip(subject.tolist(), scene.tolist(), session.tolist()))
    for block_id, rows in enumerate(_blocks(streams, starts, ids, annotation_ordinals)):
        block_ids[rows] = block_id
        if len(rows) > 1:
            path = _decode(emission[rows], model, config.transition_weight)
            one_hot = np.eye(model.num_classes)[path]
            output[rows] = (1 - config.path_mix) * emission[rows] + config.path_mix * one_hot
    output /= output.sum(1, keepdims=True)
    return output, block_ids


def compile_relation_candidates(
    subject_ids: Sequence[object], scene_ids: Sequence[object], session_ids: Sequence[object],
    starts: Sequence[float], ends: Sequence[float], interval_ids: Sequence[object],
    features: np.ndarray, probabilities: np.ndarray, transition: TransitionModel, *,
    annotation_ordinals: Sequence[int] | None = None,
    config: RelationConfig = RelationConfig(),
) -> RelationResult:
    """Compile four target candidates without accepting or inspecting labels."""
    _validate_config(config)
    subject, scene, session, start, end, ids, x, direct = _validate_intervals(
        subject_ids, scene_ids, session_ids, starts, ends, interval_ids, features, probabilities
    )
    repeat, peer_ids, peer_sims = build_repeat_probabilities(
        subject, scene, session, start, ids, x, direct, config
    )
    ordinals = None if annotation_ordinals is None else _vector(
        annotation_ordinals, "annotation_ordinals", len(ids)
    ).astype(np.int64)
    sequence, blocks = sequence_probabilities(
        subject, scene, session, start, end, ids, direct, transition, config, ordinals
    )
    combined, combined_blocks = sequence_probabilities(
        subject, scene, session, start, end, ids, repeat, transition, config, ordinals
    )
    if not np.array_equal(blocks, combined_blocks):
        raise AssertionError("candidate sequence partitions differ")
    values = {"direct": direct, "repeat": repeat, "sequence": sequence, "combined": combined}
    return RelationResult(ids, values, {name: value.argmax(1) for name, value in values.items()}, peer_ids, peer_sims, blocks)


def _candidate_margin(candidate: np.ndarray, direct: np.ndarray) -> np.ndarray:
    proposed = candidate.argmax(1); base = direct.argmax(1)
    return candidate[np.arange(len(candidate)), proposed] - candidate[np.arange(len(candidate)), base]


def select_acceptance_policy(
    validation: RelationResult, validation_labels: Sequence[int], *,
    margins: Sequence[float] = (0.0, 0.05, 0.10, 0.15),
    candidates: Sequence[str] = ("repeat", "sequence", "combined"),
) -> AcceptancePolicy:
    """Select one finite candidate/margin policy using validation labels only."""
    labels = _vector(validation_labels, "validation_labels", len(validation.interval_ids)).astype(int)
    direct = validation.probabilities["direct"]; base_class = direct.argmax(1)
    best = AcceptancePolicy("direct", float("inf"), 0, 0, 0, 0)
    best_key = (0, 0, 0, 0)
    for rank, name in enumerate(candidates):
        if name not in CANDIDATE_NAMES or name == "direct":
            raise ValueError(f"invalid acceptance candidate: {name}")
        proposal = validation.probabilities[name]; proposed_class = proposal.argmax(1)
        margin = _candidate_margin(proposal, direct)
        for threshold in margins:
            accepted = (proposed_class != base_class) & (margin >= threshold)
            rescues = int(np.sum(accepted & (base_class != labels) & (proposed_class == labels)))
            harms = int(np.sum(accepted & (base_class == labels) & (proposed_class != labels)))
            changed = int(accepted.sum()); net = rescues - harms
            key = (net, rescues, -harms, -rank)
            if net > 0 and key > best_key:
                best_key = key
                best = AcceptancePolicy(name, float(threshold), net, rescues, harms, changed)
    return best


def apply_acceptance(result: RelationResult, policy: AcceptancePolicy) -> AcceptedResult:
    """Apply a frozen source/validation policy; deliberately has no labels argument."""
    direct = result.probabilities["direct"]
    if policy.candidate == "direct":
        return AcceptedResult(direct.copy(), direct.argmax(1), np.zeros(len(direct), bool), np.full(len(direct), "direct"))
    if policy.candidate not in result.probabilities:
        raise ValueError("policy candidate is unavailable")
    proposal = result.probabilities[policy.candidate]
    accepted = (proposal.argmax(1) != direct.argmax(1)) & (_candidate_margin(proposal, direct) >= policy.margin)
    output = direct.copy(); output[accepted] = proposal[accepted]
    names = np.full(len(direct), "direct", dtype=object); names[accepted] = policy.candidate
    return AcceptedResult(output, output.argmax(1), accepted, names)
