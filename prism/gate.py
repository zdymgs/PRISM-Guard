"""Eqs. (9)-(10): acceptance and selection with explicit label roles."""

import numpy as np
from .core.thresholds import choose_threshold, gate_features, select_gate
from .eval.metrics import paired


def accept(current, proposal, score, threshold, *, label_role=None):
    """Eq. (9): apply a threshold set without evaluation labels."""
    if label_role not in (None, "source", "selection"):
        raise PermissionError("evaluation labels cannot set a gate")
    current = np.asarray(current)
    proposal = np.asarray(proposal)
    score = np.asarray(score)
    if current.shape != proposal.shape or score.shape != current.shape:
        raise ValueError("gate vector shapes differ")
    return np.where(score >= threshold, proposal, current)


def select_source_path(current, candidates, labels, *, label_role):
    """CUHK-X source net/rescue/harm/change criterion; native adapters select the other datasets."""
    if label_role not in ("source", "selection"):
        raise PermissionError("evaluation labels cannot select a path")
    records = [
        dict(path=name, **paired(labels, current, pred)) for name, pred in candidates.items()
    ]
    return max(records, key=lambda r: (r["net"], r["rescues"], -r["harms"], -r["changed"]))
