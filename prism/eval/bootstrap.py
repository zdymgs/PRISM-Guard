"""Fixed-class, paired session bootstrap; no data I/O or model selection."""

import numpy as np
from .metrics import metrics, paired


def scores(cm):
    cm = np.asarray(cm)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    den = cm.sum(-1) + cm.sum(-2)
    f = np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0).mean(-1)
    acc = tp.sum(-1) / cm.sum(axis=(-1, -2))
    return acc, f


def cluster_bootstrap(y, base, predictions, cluster_ids, classes, repetitions=2000, seed=20261015):
    y = np.asarray(y)
    base = np.asarray(base)
    clusters = np.asarray(cluster_ids)
    if len(y) == 0 or y.shape != base.shape or y.shape != clusters.shape:
        raise ValueError("invalid vectors")
    for p in [base, *predictions.values()]:
        metrics(y, np.asarray(p), classes)
    _, ix = np.unique(clusters, return_inverse=True)
    n = ix.max() + 1
    weights = (
        np.random.default_rng(seed)
        .multinomial(n, np.full(n, 1 / n), size=repetitions)
        .astype(np.float64)
    )

    def cm(p):
        return np.bincount(
            ix * classes * classes + y * classes + np.asarray(p), minlength=n * classes * classes
        ).reshape(n, classes * classes)

    def resample(c):
        return scores((weights @ c.astype(np.float64)).reshape(repetitions, classes, classes))

    bc = cm(base)
    ba, bf = resample(bc)
    bap, bfp = scores(bc.sum(0).reshape(classes, classes))
    result = {}
    for name, p in predictions.items():
        cc = cm(p)
        aa, ff = resample(cc)
        ap, fp = scores(cc.sum(0).reshape(classes, classes))
        result[name] = {
            "accuracy_difference": float(ap - bap),
            "accuracy_ci95": np.quantile(aa - ba, [0.025, 0.975]).tolist(),
            "macro_f1_difference": float(fp - bfp),
            "macro_f1_ci95": np.quantile(ff - bf, [0.025, 0.975]).tolist(),
            "clusters": int(n),
            "repetitions": repetitions,
            "seed": seed,
            "unit": "fraction; multiply by 100 for pp",
        }
    return result


def summarize(y, predictions, clusters, classes, chain, additional_references=None):
    boot = cluster_bootstrap(y, predictions["direct"], predictions, clusters, classes)
    result = {}
    for name, p in predictions.items():
        result[name] = {
            **metrics(y, p, classes),
            "versus_direct": paired(y, predictions["direct"], p),
            "versus_chain": paired(y, predictions[chain], p),
            "delta_vs_direct_cluster_bootstrap": boot[name],
        }
        if additional_references:
            result[name]["historical_references"] = {
                key: {
                    **paired(y, r, p),
                    "accuracy_difference": metrics(y, p, classes)["accuracy"]
                    - metrics(y, r, classes)["accuracy"],
                    "macro_f1_difference": metrics(y, p, classes)[f"macro_f1_fixed{classes}"]
                    - metrics(y, r, classes)[f"macro_f1_fixed{classes}"],
                }
                for key, r in additional_references.items()
            }
    shuffled = [m for m in result if m.startswith("group_shuffled_peers_seed")]
    aggregate = {}
    if shuffled:
        for key in ("accuracy", f"macro_f1_fixed{classes}", "correct"):
            vals = [result[m][key] for m in shuffled]
            aggregate[key] = {
                "mean": float(np.mean(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        aggregate["methods"] = shuffled
    return {"methods": result, "shuffled_five_seed_summary": aggregate}
