"""CUHK-X target construction from source OOF and held expert banks."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from ..core.activity import TransitionModel as FirstOrder, _decode
from ..core.content import apply_group, apply_seq, crossfit, features, prob, stack_fit
from ..core.sequence import (
    DecoderConfig,
    RecordingMetadata,
    build_sessions,
    decode_sessions,
    fit_transition_model,
)
from ..core.sequence_thresholds import gate_features, select_gate
from ..core.thresholds import choose_threshold
from ..eval.metrics import metrics, paired
from .repeats import graph


def _select_group(base, probability, labels):
    if np.array_equal(base, probability.argmax(1)):
        return {"threshold": 2., "net": 0, "rescue": 0, "harm": 0, "changed": 0}
    return choose_threshold(base, probability.argmax(1), probability, labels)


def _select_sequence(base, probability, proposal, labels):
    if np.array_equal(base, proposal):
        return {
            "score_index": 0, "threshold": 2., "net": 0,
            "rescue": 0, "harm": 0, "changed": 0,
        }
    return select_gate(base, proposal, gate_features(probability, base, proposal), labels)


def _forward_filter(emission, transition, chunks):
    """Causal Bayesian filter with raw transitions and no end prior or gate."""
    out = np.empty_like(emission, dtype=np.float64)
    for chunk in chunks:
        alpha = transition.start_log_probability.copy()
        for t, i in enumerate(chunk):
            if t:
                alpha = logsumexp(alpha[:, None] + transition.transition_log_probability, axis=0)
            alpha = alpha + np.log(np.maximum(emission[i], 1e-300))
            alpha -= logsumexp(alpha)
            out[i] = np.exp(alpha)
    return out


def build_cuhkx_targets(
    source_bank,
    target_bank,
    source_labels,
    inner_fold,
    source_metadata: RecordingMetadata,
    target_metadata: RecordingMetadata,
    *,
    label_role="source",
    include_controls=False,
):
    """Construct six paths and select one using source or selection labels.

    Banks have shape ``[rows, experts, 40]`` in the same expert order. Source
    expert predictions are already out of fold. ``inner_fold`` contains their
    existing three-way, subject-disjoint partition. Metadata rows align with
    each bank. Transition fitting uses all source labels; DIRECT and group
    stacking use the supplied inner partition.
    """
    if label_role not in ("source", "selection"):
        raise PermissionError("evaluation labels cannot select a path")
    sbank, tbank = np.asarray(source_bank), np.asarray(target_bank)
    if sbank.ndim != 3 or tbank.ndim != 3 or sbank.shape[2] != 40 or tbank.shape[2] != 40:
        raise ValueError("CUHK-X expert banks must have shape [rows, experts, 40]")
    y, inner = np.asarray(source_labels), np.asarray(inner_fold)
    sid, tid = source_metadata.sample_ids, target_metadata.sample_ids
    su, tu = source_metadata.users, target_metadata.users
    assert not set(sid) & set(tid) and not set(su) & set(tu)

    with threadpool_limits(2):
        cv = [
            (np.flatnonzero(inner != j), np.flatnonzero(inner == j))
            for j in range(3)
        ]
        assert all(not set(su[tr]) & set(su[va]) for tr, va in cv)
        sx = np.sqrt(sbank).reshape(len(sid), -1)
        tx = np.sqrt(tbank).reshape(len(tid), -1)
        sd = crossfit(sx, y, cv)
        dm = stack_fit(sx, y)
        td = prob(dm, tx)
        sb, tb = sd.argmax(1), td.argmax(1)
        sp, sm, _ = graph(sid, sb, sbank, source_metadata)
        tp, tm, _ = graph(tid, tb, tbank, target_metadata)
        sfx, tfx = features(sbank, sb, sp), features(tbank, tb, tp)
        sg = crossfit(sfx, y, cv, True)
        gm = stack_fit(sfx, y, True)
        tg = prob(gm, tfx)
        gg = _select_group(sb, sg, y)
        sclass, tclass = apply_group(sb, sg, gg), apply_group(tb, tg, gg)
        sblocks = build_sessions(np.arange(len(sid)), sm, 30., "anonymous_date")
        tblocks = build_sessions(np.arange(len(tid)), tm, 30., "anonymous_date")
        tr = fit_transition_model(y, sblocks, 40, 1., alpha=.25)
        cfg = DecoderConfig(30., .45, 1., 50)
        first = FirstOrder(
            tr.start_log_probability, tr.end_log_probability, tr.bigram_log_probability
        )

        def decode(p, b, blocks):
            return decode_sessions(
                np.log(np.maximum(.65 * p + .35 * np.eye(40)[b], 1e-9)),
                blocks, tr, cfg,
            )

        sq, tq = decode(sg, sclass, sblocks), decode(tg, tclass, tblocks)
        qgate = _select_sequence(sclass, sg, sq, y)
        ss, ts = apply_seq(sclass, sg, sq, qgate), apply_seq(tclass, tg, tq, qgate)
        ds, dt = decode(sd, sb, sblocks), decode(td, tb, tblocks)
        dgate = _select_sequence(sb, sd, ds, y)
        dsg, dtg = apply_seq(sb, sd, ds, dgate), apply_seq(tb, td, dt, dgate)

        def online(p, blocks):
            covered = set()
            for chunk in blocks:
                covered.update(chunk)
            singleton = [np.array([i]) for i in range(len(p)) if i not in covered]
            return _forward_filter(p, first, blocks + singleton).argmax(1)

        predictions = {
            "direct": tb,
            "group": tclass,
            "sequence": ts,
            "direct_sequence": dtg,
            "direct_sequence_nogate": dt,
            "online_forward": online(td, tblocks),
        }
        paths = {
            "direct": sb,
            "group": sclass,
            "sequence": ss,
            "direct_sequence": dsg,
            "direct_sequence_nogate": ds,
            "online_forward": online(sd, sblocks),
        }
        records = [
            {"path": name, **metrics(y, pp, 40), **paired(y, sb, pp)}
            for name, pp in paths.items()
        ]
        winner = max(
            records,
            key=lambda row: (row["net"], row["rescues"], -row["harms"], -row["changed"]),
        )

        if include_controls:
            # E1R controls share the already fitted source-only heads and gates.
            def mean_probability(bank, peers):
                p = np.stack([
                    np.sqrt(bank[[i, *pp]]).mean(axis=(0, 1))
                    for i, pp in enumerate(peers)
                ])
                return p / p.sum(1, keepdims=True)

            fixed_gate = _select_group(sb, mean_probability(sbank, sp), y)
            no_peers = [[] for _ in tid]
            position, _, _ = graph(tid, tb, tbank, target_metadata, mode="position")
            emission = .65 * td + .35 * np.eye(40)[tb]
            naive = tb.copy()
            for chunk in tblocks:
                naive[chunk] = _decode(emission[chunk], first, .45)
            controls = {
                "equal_mean": tbank.mean(1).argmax(1),
                "order_only": decode_sessions(
                    np.full((len(tid), 40), -np.log(40)), tblocks, tr, cfg
                ),
                "direct_viterbi_naive": naive,
                "group_no_peers": apply_group(tb, prob(gm, features(tbank, tb, no_peers)), gg),
                "group_nogate": tg.argmax(1),
                "group_fixed_mean": apply_group(tb, mean_probability(tbank, tp), fixed_gate),
                "group_posalign_peers": apply_group(
                    tb, prob(gm, features(tbank, tb, position)), gg
                ),
            }
            for seed in range(2026101501, 2026101506):
                peers, _, _ = graph(
                    tid, tb, tbank, target_metadata, mode="shuffle", seed=seed
                )
                controls[f"group_shuffled_peers_seed{seed}"] = apply_group(
                    tb, prob(gm, features(tbank, tb, peers)), gg
                )

    result = {
        "predictions": {name: value.astype(np.int64) for name, value in predictions.items()},
        "source_predictions": {name: value.astype(np.int64) for name, value in paths.items()},
        "selected_path": winner["path"],
        "selection_records": records,
        "gates": {
            "group_gate": gg,
            "sequence_gate": qgate,
            "direct_sequence_gate": dgate,
        },
    }
    if include_controls:
        result["controls"] = {name: p.astype(np.int64) for name, p in controls.items()}
        result["control_policies"] = {"fixed_mean_gate": fixed_gate}
    return result
