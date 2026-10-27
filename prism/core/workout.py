"""Label-free relation compilation for MM-Fit.

The source fitting/selection functions are deliberately separate from target
application.  Nothing in this module reads files or imports an experiment script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


_COSINE_BLOCK_ROWS = 256


@dataclass(frozen=True)
class RepeatConfig:
    min_gap_seconds: float = 10.0
    max_gap_seconds: float = 600.0
    top_k: int = 3
    similarity_threshold: float = 0.0
    mutual: bool = False
    peer_weight: float = 0.5


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
class RepeatResult:
    probabilities: np.ndarray
    classes: np.ndarray
    changed: np.ndarray
    peer_indices: tuple[tuple[int, ...], ...]
    peer_similarities: tuple[tuple[float, ...], ...]
    edge_ids: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SequenceResult:
    probabilities: np.ndarray
    classes: np.ndarray
    changed: np.ndarray
    bout_ids: np.ndarray
    edge_ids: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class RelationCandidates:
    base_probabilities: np.ndarray
    base_classes: np.ndarray
    repeat: RepeatResult
    sequence: SequenceResult
    candidate_probabilities: np.ndarray
    candidate_classes: np.ndarray
    relation_kind: np.ndarray
    relation_available: np.ndarray
    proposal_margin: np.ndarray
    candidate_support_gain: np.ndarray


@dataclass(frozen=True)
class AcceptancePolicy:
    enabled_relations: tuple[str, ...]
    threshold: float
    source_net_gain: int
    source_rescues: int
    source_harms: int
    source_changed: int


@dataclass(frozen=True)
class CompilationResult:
    row_ids: np.ndarray
    workout_ids: np.ndarray
    timestamps_seconds: np.ndarray
    sample_ids: np.ndarray
    candidates: RelationCandidates
    accepted_probabilities: np.ndarray
    accepted_classes: np.ndarray
    accepted_mask: np.ndarray
    accepted_relation_kind: np.ndarray
    accepted_edge_ids: tuple[tuple[str, ...], ...]
    policy: AcceptancePolicy


def _vector(value: Sequence[object] | np.ndarray, name: str, n: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and len(result) != n:
        raise ValueError(f"{name} must contain {n} rows")
    return result


def _probabilities(value: np.ndarray, name: str, num_classes: int = 11) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != num_classes:
        raise ValueError(f"{name} must have shape [N,{num_classes}]")
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    total = result.sum(axis=1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError(f"every {name} row must have positive mass")
    return result / total


def _check_repeat_config(config: RepeatConfig) -> None:
    if config.min_gap_seconds < 10.0:
        raise ValueError("min_gap_seconds must be at least the frozen 10 seconds")
    if config.max_gap_seconds < config.min_gap_seconds:
        raise ValueError("max_gap_seconds must be at least min_gap_seconds")
    if config.top_k <= 0:
        raise ValueError("top_k must be positive")
    if not -1.0 <= config.similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [-1,1]")
    if not 0.0 <= config.peer_weight <= 1.0:
        raise ValueError("peer_weight must be in [0,1]")


def _canonical_edge(workout: object, left: object, right: object) -> str:
    a, b = sorted((str(left), str(right)))
    return f"repeat:{workout}:{a}<->{b}"


def _workout_cosine_top_k(
    rows: np.ndarray,
    features: np.ndarray,
    norms: np.ndarray,
    timestamps: np.ndarray,
    config: RepeatConfig,
    directed: list[list[tuple[int, float]]],
) -> None:
    """Populate exact stable top-k peers using bounded cosine matrix blocks."""
    normalized = np.zeros((len(rows), features.shape[1]), dtype=np.float64)
    valid_norm = norms[rows] > 0.0
    normalized[valid_norm] = features[rows[valid_norm]] / norms[rows[valid_norm], None]
    peer_times = timestamps[rows]
    valid_peer = valid_norm & np.isfinite(peer_times)
    for start in range(0, len(rows), _COSINE_BLOCK_ROWS):
        stop = min(start + _COSINE_BLOCK_ROWS, len(rows))
        query_rows = rows[start:stop]
        similarity = normalized[start:stop] @ normalized.T
        gaps = np.abs(timestamps[query_rows, None] - peer_times[None, :])
        eligible = (
            valid_peer[None, :]
            & (norms[query_rows, None] > 0.0)
            & np.isfinite(timestamps[query_rows, None])
            & (gaps >= config.min_gap_seconds)
            & (gaps <= config.max_gap_seconds)
            & (query_rows[:, None] != rows[None, :])
            & (similarity >= config.similarity_threshold)
        )
        similarity[~eligible] = -np.inf
        block_rows = np.arange(stop - start)
        # Repeated argmax is O(k) per query and preserves the original ordering:
        # greatest similarity first, then the lowest global row index on ties.
        for _ in range(min(config.top_k, len(rows))):
            columns = np.argmax(similarity, axis=1)
            values = similarity[block_rows, columns]
            valid = np.isfinite(values)
            for local_row in np.flatnonzero(valid):
                directed[int(query_rows[local_row])].append(
                    (int(rows[columns[local_row]]), float(values[local_row]))
                )
            similarity[block_rows, columns] = -np.inf


def build_repeat_relations(
    workout_ids: Sequence[object] | np.ndarray,
    timestamps_seconds: Sequence[float] | np.ndarray,
    sample_ids: Sequence[object] | np.ndarray,
    features: np.ndarray,
    probabilities: np.ndarray,
    *,
    config: RepeatConfig = RepeatConfig(),
) -> RepeatResult:
    """Build same-workout, temporally separated cosine peers without labels.

    Used by this module's ``compile_relations``; selects window-level top-k peers.
    """
    _check_repeat_config(config)
    base = _probabilities(probabilities, "probabilities")
    n = len(base)
    workouts = _vector(workout_ids, "workout_ids", n)
    timestamps = _vector(timestamps_seconds, "timestamps_seconds", n).astype(np.float64)
    ids = _vector(sample_ids, "sample_ids", n).astype(str)
    if len(np.unique(ids)) != n:
        raise ValueError("sample_ids must be unique")
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) != n:
        raise ValueError("features must have shape [N,D]")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")

    norms = np.linalg.norm(x, axis=1)
    directed: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for workout in np.unique(workouts):
        rows = np.flatnonzero(workouts == workout)
        _workout_cosine_top_k(
            rows, x, norms, timestamps, config, directed
        )

    directed_sets = [{peer for peer, _ in values} for values in directed]
    peers: list[tuple[int, ...]] = []
    similarities: list[tuple[float, ...]] = []
    edge_ids: list[tuple[str, ...]] = []
    output = base.copy()
    for row, values in enumerate(directed):
        kept = [
            (peer, similarity)
            for peer, similarity in values
            if not config.mutual or row in directed_sets[peer]
        ]
        peer_rows = tuple(peer for peer, _ in kept)
        peers.append(peer_rows)
        similarities.append(tuple(similarity for _, similarity in kept))
        edge_ids.append(
            tuple(_canonical_edge(workouts[row], ids[row], ids[peer]) for peer in peer_rows)
        )
        if peer_rows:
            peer_mean = base[np.asarray(peer_rows, dtype=np.int64)].mean(axis=0)
            output[row] = (1.0 - config.peer_weight) * base[row] + config.peer_weight * peer_mean
    output /= output.sum(axis=1, keepdims=True)
    classes = output.argmax(axis=1).astype(np.int64)
    base_classes = base.argmax(axis=1).astype(np.int64)
    return RepeatResult(
        probabilities=output,
        classes=classes,
        changed=classes != base_classes,
        peer_indices=tuple(peers),
        peer_similarities=tuple(similarities),
        edge_ids=tuple(edge_ids),
    )


def _ordered_bouts(
    workout_ids: np.ndarray, timestamps: np.ndarray, gap_seconds: float
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return deterministic bouts; invalid/tied timestamps are singleton fallbacks."""
    n = len(workout_ids)
    bouts: list[np.ndarray] = []
    bout_ids = np.full(n, -1, dtype=np.int64)
    for workout in np.unique(workout_ids):
        rows = np.flatnonzero(workout_ids == workout)
        valid = rows[np.isfinite(timestamps[rows])]
        invalid = rows[~np.isfinite(timestamps[rows])]
        order = valid[np.argsort(timestamps[valid], kind="stable")]
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
        bouts.extend(np.asarray([row], dtype=np.int64) for row in invalid)
    for bout_id, rows in enumerate(bouts):
        bout_ids[rows] = bout_id
    if n and np.any(bout_ids < 0):
        raise AssertionError("bout builder lost rows")
    return bouts, bout_ids


def fit_source_transition(
    source_workout_ids: Sequence[object] | np.ndarray,
    source_timestamps_seconds: Sequence[float] | np.ndarray,
    source_labels: Sequence[int] | np.ndarray,
    *,
    num_classes: int = 11,
    smoothing: float = 1.0,
    gap_seconds: float = 1.5,
) -> TransitionModel:
    """Fit a smoothed source-only first-order HMM, including self-loops."""
    if num_classes <= 1 or smoothing <= 0.0 or gap_seconds <= 0.0:
        raise ValueError("num_classes, smoothing, and gap_seconds must be positive")
    labels = _vector(source_labels, "source_labels").astype(np.int64)
    n = len(labels)
    if n == 0:
        raise ValueError("source_labels must not be empty")
    workouts = _vector(source_workout_ids, "source_workout_ids", n)
    timestamps = _vector(
        source_timestamps_seconds, "source_timestamps_seconds", n
    ).astype(np.float64)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("source_labels contain an invalid class index")
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
    return TransitionModel(
        start_log_probability=np.log(start),
        end_log_probability=np.log(end),
        transition_log_probability=np.log(transition),
    )


def _viterbi(
    emission: np.ndarray,
    model: TransitionModel,
    transition_weight: float,
    emission_scale: float,
) -> np.ndarray:
    length, classes = emission.shape
    log_emission = np.full_like(emission, -np.inf, dtype=np.float64)
    positive = emission > 0.0
    log_emission[positive] = np.log(emission[positive])
    score = emission_scale * log_emission[0] + transition_weight * model.start_log_probability
    trace = np.zeros((length, classes), dtype=np.int64)
    for position in range(1, length):
        candidates = (
            score[:, None]
            + transition_weight * model.transition_log_probability
        )
        trace[position] = np.argmax(candidates, axis=0)
        score = (
            candidates[trace[position], np.arange(classes)]
            + emission_scale * log_emission[position]
        )
    last = int(
        np.argmax(score + transition_weight * model.end_log_probability)
    )
    path = np.empty(length, dtype=np.int64)
    path[-1] = last
    for position in range(length - 1, 0, -1):
        path[position - 1] = trace[position, path[position]]
    return path


def decode_sequence(
    workout_ids: Sequence[object] | np.ndarray,
    timestamps_seconds: Sequence[float] | np.ndarray,
    probabilities: np.ndarray,
    transition: TransitionModel,
    *,
    sample_ids: Sequence[object] | np.ndarray | None = None,
    config: SequenceConfig = SequenceConfig(),
) -> SequenceResult:
    """Decode each workout bout with self-loop-capable Viterbi."""
    base = _probabilities(probabilities, "probabilities", transition.num_classes)
    n = len(base)
    workouts = _vector(workout_ids, "workout_ids", n)
    timestamps = _vector(timestamps_seconds, "timestamps_seconds", n).astype(np.float64)
    ids = (
        np.arange(n).astype(str)
        if sample_ids is None
        else _vector(sample_ids, "sample_ids", n).astype(str)
    )
    if len(np.unique(ids)) != n:
        raise ValueError("sample_ids must be unique")
    if config.gap_seconds <= 0.0 or config.transition_weight < 0.0:
        raise ValueError("gap_seconds must be positive and transition_weight non-negative")
    if not np.isfinite(config.emission_scale) or config.emission_scale <= 0.0:
        raise ValueError("emission_scale must be finite and positive")
    if not 0.5 < config.path_mix <= 1.0:
        raise ValueError("path_mix must be in (0.5,1] to preserve Viterbi argmax")
    c = transition.num_classes
    if (
        transition.start_log_probability.shape != (c,)
        or transition.end_log_probability.shape != (c,)
        or transition.transition_log_probability.shape != (c, c)
    ):
        raise ValueError("transition model shapes are inconsistent")
    if not (
        np.isfinite(transition.start_log_probability).all()
        and np.isfinite(transition.end_log_probability).all()
        and np.isfinite(transition.transition_log_probability).all()
    ):
        raise ValueError("transition model must be finite")
    bouts, bout_ids = _ordered_bouts(workouts, timestamps, config.gap_seconds)
    path = base.argmax(axis=1).astype(np.int64)
    output = base.copy()
    edge_ids: list[list[str]] = [[] for _ in range(n)]
    for rows in bouts:
        if len(rows) > 1:
            path[rows] = _viterbi(
                base[rows],
                transition,
                config.transition_weight,
                config.emission_scale,
            )
            one_hot = np.eye(c, dtype=np.float64)[path[rows]]
            output[rows] = (
                (1.0 - config.path_mix) * base[rows] + config.path_mix * one_hot
            )
            for left, right in zip(rows[:-1], rows[1:]):
                edge = f"sequence:{workouts[left]}:{ids[left]}->{ids[right]}"
                edge_ids[int(left)].append(edge)
                edge_ids[int(right)].append(edge)
    output /= output.sum(axis=1, keepdims=True)
    classes = output.argmax(axis=1).astype(np.int64)
    if not np.array_equal(classes, path):
        raise AssertionError("heuristic target argmax differs from Viterbi path")
    return SequenceResult(
        probabilities=output,
        classes=classes,
        changed=classes != base.argmax(axis=1),
        bout_ids=bout_ids,
        edge_ids=tuple(tuple(values) for values in edge_ids),
    )


def build_relation_candidates(
    base_probabilities: np.ndarray,
    repeat: RepeatResult,
    sequence: SequenceResult,
    *,
    repeat_enabled: bool = True,
    sequence_enabled: bool = True,
) -> RelationCandidates:
    """Apply repeat precedence without labels."""
    base = _probabilities(base_probabilities, "base_probabilities")
    n = len(base)
    if repeat.probabilities.shape != base.shape or sequence.probabilities.shape != base.shape:
        raise ValueError("base, repeat, and sequence probability shapes must match")
    base_classes = base.argmax(axis=1).astype(np.int64)
    candidate = base.copy()
    relation_kind = np.full(n, "base", dtype="<U8")
    repeat_route = repeat_enabled & repeat.changed
    sequence_route = sequence_enabled & ~repeat_route & sequence.changed
    candidate[repeat_route] = repeat.probabilities[repeat_route]
    candidate[sequence_route] = sequence.probabilities[sequence_route]
    relation_kind[repeat_route] = "repeat"
    relation_kind[sequence_route] = "sequence"
    candidate_classes = candidate.argmax(axis=1).astype(np.int64)
    available = candidate_classes != base_classes
    rows = np.arange(n)
    proposal_margin = np.zeros(n, dtype=np.float64)
    support_gain = np.zeros(n, dtype=np.float64)
    proposal_margin[available] = (
        candidate[rows[available], candidate_classes[available]]
        - candidate[rows[available], base_classes[available]]
    )
    support_gain[available] = (
        candidate[rows[available], candidate_classes[available]]
        - base[rows[available], candidate_classes[available]]
    )
    return RelationCandidates(
        base_probabilities=base,
        base_classes=base_classes,
        repeat=repeat,
        sequence=sequence,
        candidate_probabilities=candidate,
        candidate_classes=candidate_classes,
        relation_kind=relation_kind,
        relation_available=available,
        proposal_margin=proposal_margin,
        candidate_support_gain=support_gain,
    )


def select_acceptance_policy(
    candidates: RelationCandidates,
    source_validation_labels: Sequence[int] | np.ndarray,
    *,
    thresholds: Iterable[float] | None = None,
) -> AcceptancePolicy:
    """Select a source-validation-only relation gate; no positive net disables all."""
    labels = _vector(
        source_validation_labels, "source_validation_labels", len(candidates.base_classes)
    ).astype(np.int64)
    classes = candidates.base_probabilities.shape[1]
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("source_validation_labels contain an invalid class index")
    if thresholds is None:
        values = np.unique(candidates.proposal_margin[candidates.relation_available])
    else:
        values = np.unique(np.asarray(tuple(thresholds), dtype=np.float64))
    if values.size == 0:
        return AcceptancePolicy((), float("inf"), 0, 0, 0, 0)
    if not np.isfinite(values).all():
        raise ValueError("thresholds must be finite")

    best: tuple[tuple[int, int, int, int, int], AcceptancePolicy] | None = None
    relation_sets = (("repeat",), ("sequence",), ("repeat", "sequence"))
    base = candidates.base_classes
    proposal = candidates.candidate_classes
    for relation_rank, enabled in enumerate(relation_sets):
        allowed = np.isin(candidates.relation_kind, enabled)
        for threshold in values:
            route = (
                candidates.relation_available
                & allowed
                & (candidates.proposal_margin >= threshold)
            )
            prediction = base.copy()
            prediction[route] = proposal[route]
            rescue = int(np.sum((base != labels) & (prediction == labels)))
            harm = int(np.sum((base == labels) & (prediction != labels)))
            changed = int(route.sum())
            policy = AcceptancePolicy(
                enabled_relations=enabled,
                threshold=float(threshold),
                source_net_gain=rescue - harm,
                source_rescues=rescue,
                source_harms=harm,
                source_changed=changed,
            )
            key = (rescue - harm, rescue, -harm, -changed, -relation_rank)
            if best is None or key > best[0]:
                best = (key, policy)
    assert best is not None
    return best[1] if best[1].source_net_gain > 0 else AcceptancePolicy((), float("inf"), 0, 0, 0, 0)


def apply_acceptance(
    candidates: RelationCandidates, policy: AcceptancePolicy
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a frozen label-free policy to candidate probabilities."""
    allowed = np.isin(candidates.relation_kind, policy.enabled_relations)
    accepted = (
        candidates.relation_available
        & allowed
        & (candidates.proposal_margin >= policy.threshold)
    )
    probability = candidates.base_probabilities.copy()
    probability[accepted] = candidates.candidate_probabilities[accepted]
    classes = probability.argmax(axis=1).astype(np.int64)
    relation_kind = np.full(len(classes), "base", dtype="<U8")
    relation_kind[accepted] = candidates.relation_kind[accepted]
    changed = classes != candidates.base_classes
    if not np.array_equal(accepted, changed):
        raise AssertionError("accepted rows must be exactly the changed rows")
    return probability, classes, changed, relation_kind


def compile_relations(
    row_ids: Sequence[object] | np.ndarray,
    workout_ids: Sequence[object] | np.ndarray,
    timestamps_seconds: Sequence[float] | np.ndarray,
    sample_ids: Sequence[object] | np.ndarray,
    features: np.ndarray,
    probabilities: np.ndarray,
    transition: TransitionModel,
    policy: AcceptancePolicy,
    *,
    repeat_config: RepeatConfig = RepeatConfig(),
    sequence_config: SequenceConfig = SequenceConfig(),
) -> CompilationResult:
    """Compile all auditable layers without accepting target labels."""
    base = _probabilities(probabilities, "probabilities")
    n = len(base)
    rows = _vector(row_ids, "row_ids", n)
    workouts = _vector(workout_ids, "workout_ids", n)
    timestamps = _vector(timestamps_seconds, "timestamps_seconds", n).astype(np.float64)
    ids = _vector(sample_ids, "sample_ids", n).astype(str)
    repeat = build_repeat_relations(
        workouts, timestamps, ids, features, base, config=repeat_config
    )
    sequence = decode_sequence(
        workouts,
        timestamps,
        base,
        transition,
        sample_ids=ids,
        config=sequence_config,
    )
    candidates = build_relation_candidates(base, repeat, sequence)
    accepted_probability, accepted_classes, accepted_mask, accepted_kind = apply_acceptance(
        candidates, policy
    )
    accepted_edge_ids = tuple(
        repeat.edge_ids[row]
        if accepted_kind[row] == "repeat"
        else sequence.edge_ids[row]
        if accepted_kind[row] == "sequence"
        else ()
        for row in range(n)
    )
    return CompilationResult(
        row_ids=rows.copy(),
        workout_ids=workouts.copy(),
        timestamps_seconds=timestamps.copy(),
        sample_ids=ids.copy(),
        candidates=candidates,
        accepted_probabilities=accepted_probability,
        accepted_classes=accepted_classes,
        accepted_mask=accepted_mask,
        accepted_relation_kind=accepted_kind,
        accepted_edge_ids=accepted_edge_ids,
        policy=policy,
    )


__all__ = [
    "AcceptancePolicy",
    "CompilationResult",
    "RelationCandidates",
    "RepeatConfig",
    "RepeatResult",
    "SequenceConfig",
    "SequenceResult",
    "TransitionModel",
    "apply_acceptance",
    "build_relation_candidates",
    "build_repeat_relations",
    "compile_relations",
    "decode_sequence",
    "fit_source_transition",
    "select_acceptance_policy",
]
