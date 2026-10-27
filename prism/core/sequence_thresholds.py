"""Sequence proposal scores and threshold selection."""
from __future__ import annotations
import numpy as np

SCORE_NAMES = (
    "sequence_minus_base_probability",
    "sequence_probability",
    "negative_base_probability",
    "group_max_probability",
    "group_probability_margin",
)


def gate_features(probability, base, sequence):
    rows = np.arange(len(base))
    ordered = np.sort(probability, axis=1)[:, ::-1]
    return np.column_stack(
        (
            probability[rows, sequence] - probability[rows, base],
            probability[rows, sequence],
            -probability[rows, base],
            ordered[:, 0],
            ordered[:, 0] - ordered[:, 1],
        )
    ).astype(np.float64)


def select_gate(base, sequence, features, labels):
    disagreement = sequence != base
    candidates = []
    for score_index, score_name in enumerate(SCORE_NAMES):
        values = np.unique(
            np.concatenate(
                (
                    np.linspace(-1.0, 1.0, 401),
                    np.quantile(
                        features[disagreement, score_index],
                        [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95],
                    ),
                )
            )
        )
        for threshold in values:
            route = disagreement & (features[:, score_index] >= threshold)
            prediction = base.copy()
            prediction[route] = sequence[route]
            rescue = int(np.sum((base != labels) & (prediction == labels)))
            harm = int(np.sum((base == labels) & (prediction != labels)))
            candidates.append(
                {
                    "score_index": score_index,
                    "score_name": score_name,
                    "threshold": float(threshold),
                    "changed": int(route.sum()),
                    "rescue": rescue,
                    "harm": harm,
                    "net": rescue - harm,
                }
            )
    positive = [row for row in candidates if row["net"] > 0]
    return (
        max(
            positive,
            key=lambda row: (
                row["net"],
                row["rescue"],
                -row["harm"],
                -row["changed"],
            ),
        )
        if positive
        else {
            "score_index": 0,
            "score_name": SCORE_NAMES[0],
            "threshold": 2.0,
            "changed": 0,
            "rescue": 0,
            "harm": 0,
            "net": 0,
        }
    )
