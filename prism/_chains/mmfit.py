"""MM-Fit native pseudo-bout, learned-peer and sequence target construction."""

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, f1_score

from ..core.group import build_group_features, fit_group_head
from ..core.workout import SequenceConfig, _ordered_bouts, decode_sequence, fit_source_transition
from ..core.workout_peers import bout_repeat_peers


CLASSES = 11
EXPERTS = ("rgb", "pose", "imu", "video_mae_large", "video_jepa", "joint")
GRID = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95)
PATHS = ("direct", "group", "sequence", "direct_sequence", "direct_sequence_nogate", "online_forward")


def gate(d, p, threshold):
    base = d.argmax(1)
    proposed = p.argmax(1)
    margin = p[np.arange(len(p)), proposed] - p[np.arange(len(p)), base]
    use = (base != proposed) & (margin >= threshold)
    return np.where(use, proposed, base)


def _score(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=np.arange(CLASSES), average="macro", zero_division=0)),
        "correct": int((y == p).sum()),
    }


def select(y, d, p):
    records = []
    for threshold in GRID:
        pred = gate(d, p, threshold)
        records.append(dict(threshold=threshold, changed=int((pred != d.argmax(1)).sum()),
                            **_score(y, pred)))
    best = max(records, key=lambda r: (r["macro_f1"], r["accuracy"], -r["changed"]))
    return best["threshold"], records


def forward(prob, work, time, model):
    out = np.empty(len(prob), np.int64)
    for rows in _ordered_bouts(work, time, 1.5)[0]:
        alpha = 0.2 * np.log(np.maximum(prob[rows[0]], 1e-300)) + model.start_log_probability
        alpha -= logsumexp(alpha)
        out[rows[0]] = alpha.argmax()
        for i in rows[1:]:
            alpha = 0.2 * np.log(np.maximum(prob[i], 1e-300)) + logsumexp(
                alpha[:, None] + model.transition_log_probability, axis=0
            )
            alpha -= logsumexp(alpha)
            out[i] = alpha.argmax()
    return out


def _historical_metrics(y, pred):
    matrix = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    np.add.at(matrix, (y.astype(int), pred.astype(int)), 1)
    denom = matrix.sum(1) + matrix.sum(0)
    f1 = np.divide(2 * np.diag(matrix), denom, out=np.zeros(CLASSES), where=denom > 0)
    return {"accuracy": float(np.mean(y == pred)), "macro_f1_fixed11": float(f1.mean())}


def _historical_selection(layers, labels, direct):
    records = []
    best = None
    base = direct.argmax(1)
    for name, probability in layers.items():
        if name == "old_average_peer_targets":
            continue
        for threshold in GRID:
            if name == "direct" and threshold != 0:
                continue
            proposal = probability.argmax(1)
            margin = probability[np.arange(len(base)), proposal] - probability[np.arange(len(base)), base]
            accepted = (proposal != base) & (margin >= threshold)
            final = direct.copy()
            final[accepted] = probability[accepted]
            pred = final.argmax(1)
            metrics = _historical_metrics(labels, pred)
            rescues = int(np.sum((base != labels) & (pred == labels)))
            harms = int(np.sum((base == labels) & (pred != labels)))
            record = {
                "name": name, "margin": threshold, "changed": int(accepted.sum()),
                "rescues": rescues, "harms": harms, "accuracy": metrics["accuracy"],
                "macro_f1_fixed11": metrics["macro_f1_fixed11"],
            }
            records.append(record)
            key = (metrics["macro_f1_fixed11"], metrics["accuracy"], -int(accepted.sum()))
            if best is None or key > best[0]:
                best = key, record
    return best[1], records


def _shuffle_bout_peers(work, time, direct, original, seed):
    rng = np.random.default_rng(seed)
    out = [[] for _ in time]
    retained = total = 0
    for workout in np.unique(work):
        order = np.flatnonzero(work == workout)
        order = order[np.argsort(time[order], kind="stable")]
        cut = np.r_[0, np.flatnonzero(
            (np.diff(direct.argmax(1)[order]) != 0) | (np.diff(time[order]) > 1.5)
        ) + 1, len(order)]
        bouts = [order[a:b] for a, b in zip(cut[:-1], cut[1:])]
        for j, rows in enumerate(bouts):
            eligible = [
                k for k, bout in enumerate(bouts)
                if k != j and 10 <= max(
                    time[bout[0]] - time[rows[-1]], time[rows[0]] - time[bout[-1]]
                ) <= 600
            ]
            count = len(original[rows[0]])
            assert all(len(original[r]) == count for r in rows)
            assert len(eligible) >= count
            chosen = rng.choice(eligible, size=count, replace=False)
            for position, row in enumerate(rows):
                phase = position / max(len(rows) - 1, 1)
                out[row] = tuple(
                    int(bouts[k][round(phase * (len(bouts[k]) - 1))]) for k in chosen
                )
                retained += len(set(out[row]) & set(original[row]))
                total += count
                assert len(out[row]) == len(original[row]) and row not in out[row]
    return tuple(tuple(pp) for pp in out), {
        "edges": total, "original_edges_retained": retained,
    }


def build_targets(source, selection, target, config, *, label_role="selection",
                  include_controls=False):
    """Fit the source group/transition models and replay selection-set path search."""
    if label_role not in {"source", "selection"}:
        raise PermissionError("path selection requires source or selection labels")
    if "label" in target:
        raise ValueError("target bank must not contain labels")
    weights = np.asarray(config.get("direct_weights", [0, 0.25, 0, 0.25, 0, 0.5]), np.float32)
    splits = {}
    for name, data in (("source", source), ("selection", selection), ("target", target)):
        bank = np.asarray(data["expert_bank"])
        work = np.asarray(data["workout"])
        time = np.asarray(data["time"], float)
        direct = np.einsum("k,nkc->nc", weights, bank)
        peers, _ = bout_repeat_peers(work, time, data["descriptor"], direct)
        features = build_group_features(bank, direct.argmax(1), peers)
        average = direct.copy()
        for row, pp in enumerate(peers):
            if pp:
                average[row] = 0.75 * direct[row] + 0.25 * direct[list(pp)].mean(0)
        splits[name] = {
            "direct": direct, "work": work, "time": time, "features": features,
            "average": average, "ids": np.asarray(data["record_id"], str),
        }
        if include_controls:
            splits[name].update(bank=bank, peers=peers)
    src, val = splits["source"], splits["selection"]
    labels = np.asarray(source["label"], int)
    vlabels = np.asarray(selection["label"], int)
    combined = np.concatenate((src["features"], val["features"]))
    nsource = len(labels)
    head = fit_group_head(
        combined, np.r_[labels, vlabels], np.r_[src["ids"], val["ids"]],
        fit_indices=np.arange(nsource), holdout_indices=np.arange(nsource, len(combined)),
        expert_names=EXPERTS, class_count=CLASSES, posterior_provenance="source_oof",
        regularization_c=0.03,
    )
    transition = fit_source_transition(src["work"], src["time"], labels, num_classes=CLASSES)
    cfg = SequenceConfig(gap_seconds=1.5, transition_weight=0.1, emission_scale=0.2, path_mix=0.51)
    for name in ("selection", "target"):
        s = splits[name]
        group = head.predict_proba(s["features"]).astype(np.float32)
        layers = {"direct": s["direct"], "old_average_peer_targets": s["average"], "group": group}
        for transition_weight in (0.1, 0.3):
            emission = 0.65 * group + 0.35 * s["direct"]
            seq = decode_sequence(
                s["work"], s["time"], emission, transition, sample_ids=s["ids"],
                config=SequenceConfig(gap_seconds=1.5, transition_weight=transition_weight,
                                      path_mix=0.51, emission_scale=0.2),
            ).probabilities.astype(np.float32)
            layers[f"sequence_{transition_weight}"] = seq
            route = group.argmax(1) != s["average"].argmax(1)
            precedence = seq.copy()
            precedence[route] = group[route]
            layers[f"precedence_{transition_weight}"] = precedence
        direct_seq = decode_sequence(
            s["work"], s["time"], s["direct"], transition, sample_ids=s["ids"], config=cfg
        ).probabilities
        s.update(layers=layers, g=group, groupseq=layers["sequence_0.1"], seq=direct_seq)
        s["pred"] = {
            "direct": s["direct"].argmax(1),
            "group": gate(s["direct"], group, config.get("DG_margin", 0.95)),
            "sequence": layers["sequence_0.1"].argmax(1),
            "direct_sequence_nogate": direct_seq.argmax(1),
            "online_forward": forward(s["direct"], s["work"], s["time"], transition),
            "repeat_precedence": layers["precedence_0.1"].argmax(1),
            "precedence_0.1": layers["precedence_0.1"].argmax(1),
            "precedence_0.3": layers["precedence_0.3"].argmax(1),
        }
    threshold, ds_records = select(vlabels, val["direct"], val["seq"])
    for name in ("selection", "target"):
        s = splits[name]
        s["pred"]["direct_sequence"] = gate(s["direct"], s["seq"], threshold)
    path_scores, all_candidates = [], []
    for name in PATHS:
        if name in ("group", "sequence", "direct_sequence"):
            key = {"group": "g", "sequence": "groupseq", "direct_sequence": "seq"}[name]
            _, records = select(vlabels, val["direct"], val[key])
            records = [dict(method=name, **record) for record in records]
        else:
            pred = val["pred"][name]
            records = [dict(method=name, threshold=None, **_score(vlabels, pred),
                            changed=int((pred != val["pred"]["direct"]).sum()))]
        all_candidates.extend(records)
        path_scores.append(max(records, key=lambda r: (
            r["macro_f1"], r["accuracy"], -r["changed"]
        )))
    winner = max(path_scores, key=lambda r: (r["macro_f1"], r["accuracy"], -r["changed"]))
    historical, historical_records = _historical_selection(val["layers"], vlabels, val["direct"])
    for name in ("selection", "target"):
        s = splits[name]
        method = winner["method"]
        s["pred"]["selected"] = (
            gate(s["direct"], s[{"group": "g", "sequence": "groupseq", "direct_sequence": "seq"}[method]],
                 winner["threshold"])
            if method in ("group", "sequence", "direct_sequence") else s["pred"][method].copy()
        )
        s["pred"]["historical_selected"] = gate(
            s["direct"], s["layers"][historical["name"]], historical["margin"]
        )
    result = {
        "predictions": splits["target"]["pred"], "selection_predictions": val["pred"],
        "selected_path": winner["method"], "selection_records": all_candidates,
        "gates": {"group": config.get("DG_margin", 0.95), "direct_sequence": threshold,
                  "direct_sequence_records": ds_records, "selected": winner,
                  "repeat_precedence": None},
        "historical_selected": historical, "historical_selection_records": historical_records,
    }
    if include_controls:
        fixed_threshold = None
        shuffle_records = {}
        for name in ("selection", "target"):
            s = splits[name]
            direct = s["direct"]
            base = direct.argmax(1)
            empty = tuple(() for _ in range(len(direct)))
            no_peers = head.predict_proba(build_group_features(s["bank"], base, empty))
            controls = {
                "order_only": decode_sequence(
                    s["work"], s["time"], np.full_like(direct, 1 / CLASSES),
                    transition, sample_ids=s["ids"], config=cfg,
                ).classes,
                "group_no_peers": gate(direct, no_peers, .95),
                "group_nogate": s["g"].argmax(1),
            }
            sq = np.sqrt(s["bank"].astype(float))
            fixed = np.stack([
                sq[[i, *pp]].mean(axis=(0, 1)) for i, pp in enumerate(s["peers"])
            ])
            fixed /= fixed.sum(1, keepdims=True)
            if name == "selection":
                fixed_threshold, fixed_records = select(vlabels, direct, fixed)
            controls["group_fixed_mean"] = gate(direct, fixed, fixed_threshold)
            shuffle_records[name] = {}
            for seed in range(2026101501, 2026101506):
                peers, record = _shuffle_bout_peers(
                    s["work"], s["time"], direct, s["peers"], seed
                )
                proposal = head.predict_proba(build_group_features(s["bank"], base, peers))
                controls[f"group_shuffled_peers_seed_{seed}"] = gate(direct, proposal, .95)
                shuffle_records[name][str(seed)] = record
            result["selection_controls" if name == "selection" else "controls"] = controls
        result["control_policies"] = {
            "fixed_mean_threshold": fixed_threshold,
            "fixed_mean_records": fixed_records, "shuffle_records": shuffle_records,
        }
    return result
