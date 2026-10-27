"""Learned own-plus-peer group teacher with expert-bank features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GroupLineage:
    expert_count: int
    expert_names: tuple[str, ...]
    class_count: int
    fit_ids: tuple[str, ...]
    normalizer_training_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]
    posterior_provenance: str


@dataclass
class GroupHead:
    scaler: StandardScaler
    classifier: LogisticRegression
    lineage: GroupLineage

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = _finite_matrix(features, "features")
        partial = self.classifier.predict_proba(self.scaler.transform(x))
        output = np.zeros((len(x), self.lineage.class_count), dtype=np.float64)
        output[:, self.classifier.classes_.astype(np.int64)] = partial
        return output


@dataclass(frozen=True)
class FrozenThreshold:
    threshold: float
    rescues: int
    harms: int
    net: int
    changed: int
    selection_ids: tuple[str, ...]
    posterior_provenance: str


@dataclass(frozen=True)
class HardTargetBundle:
    probabilities: np.ndarray
    prediction: np.ndarray
    q_wrong: float
    q_chosen: float
    threshold: FrozenThreshold


def _finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64)
    if output.ndim != 2 or not np.isfinite(output).all():
        raise ValueError(f"{name} must be a finite matrix")
    return output


def _validate_bank(bank: np.ndarray, expert_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(bank, dtype=np.float64)
    if probability.ndim != 3 or probability.shape[2] < 2:
        raise ValueError("bank must have shape [N,K,C] with C >= 2")
    if np.any(probability < 0) or not np.isfinite(probability).all():
        raise ValueError("bank must contain finite non-negative probabilities")
    if expert_mask is None:
        mask = np.ones(probability.shape[:2], dtype=bool)
    else:
        mask = np.asarray(expert_mask, dtype=bool)
        if mask.shape != probability.shape[:2]:
            raise ValueError("expert_mask must have shape [N,K]")
    sums = probability.sum(2)
    if np.any(mask & (~np.isclose(sums, 1.0, atol=1e-5))):
        raise ValueError("each available bank branch must sum to one")
    # A missing branch has no predictive content.  Requiring zeros prevents a
    # caller from accidentally leaking an unavailable posterior into a mean.
    if np.any((~mask) & (np.abs(probability).sum(2) > 1e-12)):
        raise ValueError("masked bank branches must be zero")
    return probability, mask


def build_group_features(
    bank: np.ndarray,
    reference_prediction: Sequence[int] | np.ndarray,
    peer_indices: Sequence[Sequence[int]],
    *,
    expert_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Construct full expert-bank features with branch-safe missingness.

    For every row, the peer set supplied by the caller contains *other* rows.
    This function adds self exactly once.  Features are own sqrt-posteriors,
    per-expert means of sqrt-posteriors over self plus peers, the mean reference
    one-hot over self plus peers, and the number of other peers.
    """
    probability, mask = _validate_bank(bank, expert_mask)
    n, experts, classes = probability.shape
    reference = np.asarray(reference_prediction, dtype=np.int64)
    if reference.shape != (n,) or np.any((reference < 0) | (reference >= classes)):
        raise ValueError("reference_prediction must contain one valid class per row")
    if len(peer_indices) != n:
        raise ValueError("peer_indices must have one sequence per row")
    sqrt_bank = np.sqrt(probability)
    own = sqrt_bank.reshape(n, experts * classes)
    aggregate = np.zeros_like(sqrt_bank)
    aggregate_reference = np.zeros((n, classes), dtype=np.float64)
    peer_count = np.zeros((n, 1), dtype=np.float64)
    for row, supplied in enumerate(peer_indices):
        other = [int(value) for value in supplied]
        if any(value < 0 or value >= n for value in other):
            raise ValueError("peer index out of range")
        if row in other:
            raise ValueError("peer_indices must not duplicate self; self is added internally")
        if len(set(other)) != len(other):
            raise ValueError("duplicate peer index")
        members = np.asarray([row, *other], dtype=np.int64)
        for expert in range(experts):
            available = members[mask[members, expert]]
            if len(available):
                aggregate[row, expert] = sqrt_bank[available, expert].mean(0)
        counts = np.bincount(reference[members], minlength=classes)
        aggregate_reference[row] = counts / len(members)
        peer_count[row, 0] = len(other)
    return np.concatenate((own, aggregate.reshape(n, -1), aggregate_reference, peer_count), axis=1).astype(np.float32)


def fit_group_head(
    features: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    sample_ids: Sequence[str],
    *,
    fit_indices: Sequence[int],
    holdout_indices: Sequence[int],
    expert_names: Sequence[str],
    class_count: int,
    posterior_provenance: str,
    regularization_c: float = 0.03,
) -> GroupHead:
    """Fit only on explicit source indices; the holdout must be disjoint.

    ``posterior_provenance`` must explicitly assert out-of-fold source bank
    predictions.  This prevents current-model in-sample posteriors from being
    silently used as group-training evidence.
    """
    x = _finite_matrix(features, "features")
    y = np.asarray(labels, dtype=np.int64)
    ids = np.asarray(sample_ids, dtype=str)
    fit = np.asarray(fit_indices, dtype=np.int64)
    holdout = np.asarray(holdout_indices, dtype=np.int64)
    if len(y) != len(x) or len(ids) != len(x):
        raise ValueError("features, labels, and sample_ids must align")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    if not len(fit) or not len(holdout):
        raise ValueError("fit_indices and holdout_indices must both be explicit and non-empty")
    if np.intersect1d(fit, holdout).size:
        raise ValueError("fit_indices and holdout_indices must be disjoint")
    if np.any(fit < 0) or np.any(fit >= len(x)) or np.any(holdout < 0) or np.any(holdout >= len(x)):
        raise ValueError("split index out of range")
    if posterior_provenance != "source_oof":
        raise ValueError("group fitting requires posterior_provenance='source_oof'")
    names = tuple(str(value) for value in expert_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("expert_names must be non-empty and unique")
    expected = 2 * len(names) * class_count + class_count + 1
    if x.shape[1] != expected:
        raise ValueError(f"feature width {x.shape[1]} does not match K={len(names)}, C={class_count}")
    if np.any((y[fit] < 0) | (y[fit] >= class_count)):
        raise ValueError("fit labels are outside class range")
    scaler = StandardScaler().fit(x[fit])
    classifier = LogisticRegression(C=regularization_c, max_iter=1000, solver="lbfgs")
    sample_weight = np.where(x[fit, -1] > 0, 2.0, 1.0)
    classifier.fit(scaler.transform(x[fit]), y[fit], sample_weight=sample_weight)
    training_ids = tuple(ids[fit])
    lineage = GroupLineage(len(names), names, class_count, training_ids, training_ids,
                           tuple(ids[holdout]), posterior_provenance)
    return GroupHead(scaler, classifier, lineage)


def select_replacement_threshold(
    base_prediction: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    oof_group_probability: np.ndarray,
    selection_ids: Sequence[str],
    *,
    posterior_provenance: str,
    candidates: Sequence[float] = (0.0, 0.05, 0.10, 0.15, 0.20),
) -> FrozenThreshold:
    """Freeze a rescue-minus-harm threshold from OOF/held predictions only.

    This learned-group selector has five default candidate thresholds.
    """
    if posterior_provenance not in {"source_oof", "separate_holdout"}:
        raise ValueError("threshold selection requires OOF or separate-holdout posteriors")
    probability = _finite_matrix(oof_group_probability, "oof_group_probability")
    base = np.asarray(base_prediction, dtype=np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    ids = tuple(str(value) for value in selection_ids)
    if base.shape != truth.shape or base.shape != (len(probability),) or len(ids) != len(base):
        raise ValueError("threshold inputs must align")
    proposal = probability.argmax(1)
    margin = probability[np.arange(len(base)), proposal] - probability[np.arange(len(base)), base]
    best: FrozenThreshold | None = None
    for threshold in candidates:
        if not np.isfinite(threshold):
            raise ValueError("threshold candidates must be finite")
        route = (proposal != base) & (margin >= float(threshold))
        rescues = int(np.sum(route & (base != truth) & (proposal == truth)))
        harms = int(np.sum(route & (base == truth) & (proposal != truth)))
        record = FrozenThreshold(float(threshold), rescues, harms, rescues - harms, int(route.sum()), ids, posterior_provenance)
        if best is None or (record.net, -record.harms, -record.changed, record.threshold) > (best.net, -best.harms, -best.changed, best.threshold):
            best = record
    assert best is not None
    if best.net <= 0:
        return FrozenThreshold(float("inf"), 0, 0, 0, 0, ids, posterior_provenance)
    return best


def apply_group_threshold(base_prediction: Sequence[int] | np.ndarray, group_probability: np.ndarray, policy: FrozenThreshold) -> tuple[np.ndarray, np.ndarray]:
    probability = _finite_matrix(group_probability, "group_probability")
    base = np.asarray(base_prediction, dtype=np.int64)
    if base.shape != (len(probability),):
        raise ValueError("base_prediction and group_probability must align")
    proposal = probability.argmax(1)
    margin = probability[np.arange(len(base)), proposal] - probability[np.arange(len(base)), base]
    route = (proposal != base) & (margin >= policy.threshold)
    output = base.copy(); output[route] = proposal[route]
    return output, route


def class_aware_hard_targets(chosen: Sequence[int] | np.ndarray, class_count: int, q_wrong: float = 0.0005) -> np.ndarray:
    """Return smoothed hard targets for an arbitrary class count."""
    labels = np.asarray(chosen, dtype=np.int64)
    q_chosen = 1.0 - (class_count - 1) * float(q_wrong)
    if class_count < 2 or q_wrong < 0 or q_chosen < 0:
        raise ValueError("class_count/q_wrong produce a negative target probability")
    if np.any((labels < 0) | (labels >= class_count)):
        raise ValueError("chosen labels are outside class range")
    output = np.full((len(labels), class_count), q_wrong, dtype=np.float32)
    output[np.arange(len(labels)), labels] = q_chosen
    return output


def compile_hard_target_bundle(
    chosen: Sequence[int] | np.ndarray,
    class_count: int,
    *,
    frozen_threshold: FrozenThreshold,
    q_wrong: float = 0.0005,
) -> HardTargetBundle:
    """Compile targets only with an already frozen replacement policy attached."""
    labels = np.asarray(chosen, dtype=np.int64)
    probability = class_aware_hard_targets(labels, class_count, q_wrong)
    return HardTargetBundle(probability, labels.copy(), float(q_wrong),
                            1.0 - (class_count - 1) * float(q_wrong), frozen_threshold)
