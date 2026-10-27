"""Posterior-margin threshold selection."""
from __future__ import annotations
import numpy as np

def choose_threshold(base, proposal, probability, labels):
    """Select a posterior-margin gate; re-exported by ``prism.gate``.

    Candidates combine a 201-point grid with disagreement-score quantiles.
    """
    rows = np.arange(len(base))
    score = probability[rows, proposal] - probability[rows, base]
    disagreement = proposal != base
    values = np.unique(
        np.concatenate(
            (
                np.linspace(-0.10, 0.90, 201),
                np.quantile(score[disagreement], [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]),
            )
        )
    )
    candidates = []
    for threshold in values:
        route = disagreement & (score >= threshold)
        prediction = base.copy()
        prediction[route] = proposal[route]
        rescue = int(np.sum((base != labels) & (prediction == labels)))
        harm = int(np.sum((base == labels) & (prediction != labels)))
        candidates.append(
            {
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
        else {"threshold": 2.0, "changed": 0, "rescue": 0, "harm": 0, "net": 0}
    )

from .sequence_thresholds import gate_features, select_gate
