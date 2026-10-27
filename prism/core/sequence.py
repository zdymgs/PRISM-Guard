"""CUHK-X session construction and constrained sequence decoding."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable
import numpy as np

@dataclass(frozen=True)
class RecordingMetadata:
    sample_ids: np.ndarray
    users: np.ndarray
    dates: np.ndarray
    starts: np.ndarray


@dataclass(frozen=True)
class TransitionModel:
    start_log_probability: np.ndarray
    end_log_probability: np.ndarray
    bigram_log_probability: np.ndarray
    trigram_log_probability: np.ndarray


@dataclass(frozen=True)
class DecoderConfig:
    gap_seconds: float
    transition_weight: float
    trigram_backoff: float
    beam_width: int


@dataclass(frozen=True)
class BeamPosterior:
    """Approximate structured posterior retained by the finite beam."""

    paths: np.ndarray
    path_scores: np.ndarray
    path_probability: np.ndarray
    marginals: np.ndarray


def build_sessions(
    indices: Iterable[int] | np.ndarray,
    metadata: RecordingMetadata,
    gap_seconds: float,
    grouping: str,
) -> list[np.ndarray]:
    if grouping not in {"known_user", "anonymous_date"}:
        raise ValueError(f"Unsupported grouping: {grouping}")
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for value in indices:
        index = int(value)
        if not np.isfinite(metadata.starts[index]) or not metadata.dates[index]:
            continue
        if grouping == "known_user":
            key = (metadata.users[index], metadata.dates[index])
        else:
            key = (metadata.dates[index],)
        grouped[key].append(index)

    sessions: list[np.ndarray] = []
    for values in grouped.values():
        ordered = sorted(values, key=lambda index: metadata.starts[index])
        current: list[int] = []
        previous_start: float | None = None
        for index in ordered:
            start = float(metadata.starts[index])
            if (
                current
                and previous_start is not None
                and start - previous_start > float(gap_seconds)
            ):
                sessions.append(np.asarray(current, dtype=np.int64))
                current = []
            current.append(index)
            previous_start = start
        if current:
            sessions.append(np.asarray(current, dtype=np.int64))
    return sessions


def fit_transition_model(
    labels: np.ndarray,
    sessions: list[np.ndarray],
    num_classes: int,
    trigram_backoff: float,
    alpha: float = 0.25,
) -> TransitionModel:
    if trigram_backoff <= 0:
        raise ValueError("trigram_backoff must be positive")
    start_counts = np.full(num_classes, alpha, dtype=np.float64)
    end_counts = np.full(num_classes, alpha, dtype=np.float64)
    bigram_counts = np.full((num_classes, num_classes), alpha, dtype=np.float64)
    trigram_counts = np.zeros(
        (num_classes, num_classes, num_classes), dtype=np.float64
    )
    for session in sessions:
        if len(session) == 0:
            continue
        sequence = labels[session].astype(np.int64, copy=False)
        start_counts[sequence[0]] += 1.0
        end_counts[sequence[-1]] += 1.0
        if len(sequence) >= 2:
            np.add.at(bigram_counts, (sequence[:-1], sequence[1:]), 1.0)
        if len(sequence) >= 3:
            np.add.at(
                trigram_counts,
                (sequence[:-2], sequence[1:-1], sequence[2:]),
                1.0,
            )

    bigram_probability = bigram_counts / bigram_counts.sum(axis=1, keepdims=True)
    trigram_total = trigram_counts.sum(axis=2, keepdims=True)
    trigram_mle = np.divide(
        trigram_counts,
        trigram_total,
        out=np.zeros_like(trigram_counts),
        where=trigram_total > 0,
    )
    interpolation = trigram_total / (trigram_total + float(trigram_backoff))
    trigram_probability = (
        interpolation * trigram_mle
        + (1.0 - interpolation) * bigram_probability[None, :, :]
    )
    return TransitionModel(
        start_log_probability=np.log(start_counts / start_counts.sum()),
        end_log_probability=np.log(end_counts / end_counts.sum()),
        bigram_log_probability=np.log(np.maximum(bigram_probability, 1e-300)),
        trigram_log_probability=np.log(np.maximum(trigram_probability, 1e-300)),
    )


def decode_unique_beam_paths(
    emission_log_probability: np.ndarray,
    model: TransitionModel,
    transition_weight: float,
    beam_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    emission = np.asarray(emission_log_probability, dtype=np.float64)
    if emission.ndim != 2:
        raise ValueError("emission_log_probability must have shape [time, class]")
    length, num_classes = emission.shape
    if length == 0:
        return (
            np.empty((1, 0), dtype=np.int64),
            np.zeros(1, dtype=np.float64),
        )
    if num_classes > 63:
        raise ValueError("The uint64 no-repeat mask supports at most 63 classes")
    if length > num_classes:
        raise ValueError("A no-repeat sequence cannot be longer than num_classes")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    initial_score = emission[0] + transition_weight * model.start_log_probability
    initial_count = min(int(beam_width), num_classes)
    initial_classes = np.argsort(initial_score, kind="stable")[-initial_count:]
    scores = initial_score[initial_classes]
    paths = initial_classes[:, None].astype(np.int64)
    used = np.left_shift(np.uint64(1), initial_classes.astype(np.uint64))
    previous = np.full(initial_count, -1, dtype=np.int64)
    last = initial_classes.astype(np.int64)
    class_bits = np.left_shift(
        np.uint64(1), np.arange(num_classes, dtype=np.uint64)
    )

    for time_index in range(1, length):
        if time_index == 1:
            transition = model.bigram_log_probability[last]
        else:
            transition = model.trigram_log_probability[previous, last]
        candidates = (
            scores[:, None]
            + emission[time_index][None, :]
            + transition_weight * transition
        )
        candidates[(used[:, None] & class_bits[None, :]) != 0] = -np.inf
        flat = candidates.ravel()
        finite_count = int(np.isfinite(flat).sum())
        keep_count = min(int(beam_width), finite_count)
        if keep_count == 0:
            raise RuntimeError("No valid unique-label beam remains")
        kept = np.argsort(flat, kind="stable")[-keep_count:]
        parent = kept // num_classes
        new_class = (kept % num_classes).astype(np.int64)
        scores = flat[kept]
        paths = np.concatenate((paths[parent], new_class[:, None]), axis=1)
        used = used[parent] | class_bits[new_class]
        previous = last[parent]
        last = new_class

    scores = scores + transition_weight * model.end_log_probability[last]
    order = np.argsort(scores, kind="stable")[::-1]
    return paths[order], scores[order]


def decode_unique_beam_posterior(
    emission_log_probability: np.ndarray,
    model: TransitionModel,
    transition_weight: float,
    beam_width: int,
    posterior_temperature: float = 1.0,
) -> BeamPosterior:
    """Marginalize the retained structured paths into uncertainty-aware targets.

    This is an approximation to the full sequence posterior because only the final
    beam is retained. Unlike mixing a MAP one-hot target with emissions at a fixed
    ratio, it preserves ambiguity when several legal sequence paths remain plausible.
    """

    if posterior_temperature <= 0:
        raise ValueError("posterior_temperature must be positive")
    paths, path_scores = decode_unique_beam_paths(
        emission_log_probability,
        model,
        transition_weight=transition_weight,
        beam_width=beam_width,
    )
    scaled = (path_scores - np.max(path_scores)) / float(posterior_temperature)
    path_probability = np.exp(scaled)
    path_probability /= path_probability.sum()
    length, num_classes = np.asarray(emission_log_probability).shape
    marginals = np.zeros((length, num_classes), dtype=np.float64)
    for time_index in range(length):
        np.add.at(marginals[time_index], paths[:, time_index], path_probability)
    return BeamPosterior(
        paths=paths,
        path_scores=path_scores,
        path_probability=path_probability,
        marginals=marginals,
    )


def decode_unique_beam(
    emission_log_probability: np.ndarray,
    model: TransitionModel,
    transition_weight: float,
    beam_width: int,
) -> np.ndarray:
    paths, _ = decode_unique_beam_paths(
        emission_log_probability,
        model,
        transition_weight=transition_weight,
        beam_width=beam_width,
    )
    return paths[0]


def decode_sessions(
    base_log_probability: np.ndarray,
    sessions: list[np.ndarray],
    model: TransitionModel,
    config: DecoderConfig,
) -> np.ndarray:
    predictions = np.asarray(base_log_probability).argmax(axis=1).astype(np.int64)
    for session in sessions:
        predictions[session] = decode_unique_beam(
            base_log_probability[session],
            model,
            transition_weight=config.transition_weight,
            beam_width=config.beam_width,
        )
    return predictions
