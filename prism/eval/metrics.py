"""Pure, fixed-class diagnostics. No artifact or label I/O."""

import numpy as np
from scipy.stats import binomtest


def integers(x, name):
    a = np.asarray(x)
    if a.ndim != 1 or (a.size and not np.issubdtype(a.dtype, np.integer)):
        raise ValueError(name + " must be a 1D integer vector")
    return a.astype(np.int64)


def metrics(y, pred, classes):
    y, pred = integers(y, "y"), integers(pred, "pred")
    if not isinstance(classes, int) or classes < 2 or y.shape != pred.shape or not len(y):
        raise ValueError("invalid dimensions/classes")
    if np.any((y < 0) | (y >= classes) | (pred < 0) | (pred >= classes)):
        raise ValueError("class out of range")
    cm = np.bincount(y * classes + pred, minlength=classes**2).reshape(classes, classes)
    denom = cm.sum(0) + cm.sum(1)
    f = np.divide(2 * cm.diagonal(), denom, out=np.zeros(classes), where=denom > 0)
    return {
        "n": len(y),
        "correct": int((y == pred).sum()),
        "accuracy": float((y == pred).mean()),
        f"macro_f1_fixed{classes}": float(f.mean()),
    }


def paired(y, base, pred):
    y, base, pred = [integers(x, n) for x, n in ((y, "y"), (base, "base"), (pred, "pred"))]
    if y.shape != base.shape or y.shape != pred.shape:
        raise ValueError("paired shape mismatch")
    a, b = base == y, pred == y
    r, h = int((~a & b).sum()), int((a & ~b).sum())
    return {
        "rescues": r,
        "harms": h,
        "net": r - h,
        "changed": int((base != pred).sum()),
        "mcnemar_exact_p": float(binomtest(r, r + h, 0.5).pvalue) if r + h else 1.0,
    }


def reliability(y, pred, peers, blocks, classes):
    m = metrics(y, pred, classes)
    y, pred = np.asarray(y), np.asarray(pred)
    if len(peers) != len(y):
        raise ValueError("peer row count")
    edges = same = any_peer = correct_peer = same_correct_peer = 0
    for i, pp in enumerate(peers):
        p = integers(pp, "peers")
        if len(set(p)) != len(p) or i in p or np.any((p < 0) | (p >= len(y))):
            raise ValueError("invalid peers")
        edges += len(p)
        same += int((y[p] == y[i]).sum())
        if pred[i] != y[i]:
            any_peer += int(len(p) > 0)
            correct_peer += int(np.any(pred[p] == y[p]))
            same_correct_peer += int(np.any((pred[p] == y[p]) & (y[p] == y[i])))
    count = np.zeros((classes, classes), dtype=np.int64)
    seen = set()
    for bb in blocks:
        b = integers(bb, "block")
        if len(set(b)) != len(b) or seen.intersection(b) or np.any((b < 0) | (b >= len(y))):
            raise ValueError("invalid/overlapping blocks")
        seen.update(b)
        np.add.at(count, (y[b[:-1]], y[b[1:]]), 1)
    rows = count.sum(1)
    p = count / np.maximum(rows[:, None], 1)
    entropy = -(p * np.log2(np.maximum(p, 1e-300))).sum(1)
    errors = len(y) - m["correct"]
    return {
        "n": len(y),
        "directed_peer_edge_count": edges,
        "same_label_edge_count": same,
        "same_label_edge_rate": same / edges if edges else None,
        "direct_error_count": errors,
        "direct_error_rate": errors / len(y),
        "errors_with_any_peer_count": any_peer,
        "errors_with_any_peer_fraction": any_peer / errors if errors else None,
        "errors_with_any_correct_peer_count": correct_peer,
        "errors_with_any_correct_peer_fraction": correct_peer / errors if errors else None,
        "errors_with_same_label_correct_peer_count": same_correct_peer,
        "errors_with_same_label_correct_peer_fraction": (
            same_correct_peer / errors if errors else None
        ),
        "h_next_given_previous_bits": float(rows @ entropy / rows.sum()) if rows.sum() else None,
        "row_entropy_unweighted_mean_bits": float(entropy[rows > 0].mean()) if rows.sum() else None,
        "transition_pair_count": int(rows.sum()),
        "block_count": len(blocks),
    }
