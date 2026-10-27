"""MMAct expert-bank, learned-peer and sequence target construction."""

from itertools import combinations

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import f1_score

from ..core.activity import _blocks, _decode, fit_source_transition
from ..core.activity_mix import _monotone, weighted_available_mix
from ..core.group import build_group_features, fit_group_head


CLASSES = 37
MARGINS = (0.0, 0.05, 0.1, 0.15)
PATHS = (
    "direct", "group", "sequence", "direct_sequence_gated",
    "direct_sequence_nogate", "online_forward",
)


def combined_similarity(geometry, posterior):
    geometry = np.asarray(geometry, np.float64)
    posterior = np.asarray(posterior, np.float64)
    if geometry.ndim != 2 or len(geometry) != len(posterior) or not np.isfinite(geometry).all():
        raise ValueError("geometry mismatch")
    if not np.allclose(np.linalg.norm(geometry, axis=1), 1, atol=1e-5):
        raise ValueError("geometry must be L2 normalized")
    cosine = np.clip(geometry @ geometry.T, -1, 1)
    cosine = (cosine + 1) / 2
    bc = np.sqrt(posterior) @ np.sqrt(posterior).T
    return 0.5 * cosine + 0.5 * bc


def geometry_posterior_peers(meta, geometry, direct):
    n = len(direct)
    sim = combined_similarity(geometry, direct)
    peers = [[] for _ in range(n)]
    subject = np.asarray(meta["subject"], str)
    scene = np.asarray(meta["scene"], str)
    session = np.asarray(meta["session"], str)
    starts = np.asarray(meta["start_ms"], float)
    ids = np.asarray(meta["interval_id"], str)
    for sub, scn in sorted(set(zip(subject, scene))):
        scope = np.flatnonzero((subject == sub) & (scene == scn))
        sessions = sorted(set(session[scope]))
        for a, b in combinations(sessions, 2):
            left = scope[session[scope] == a]
            right = scope[session[scope] == b]
            left = left[np.lexsort((ids[left], starts[left]))]
            right = right[np.lexsort((ids[right], starts[right]))]
            for x, y, value in _monotone(sim, left, right):
                peers[x].append((y, value))
                peers[y].append((x, value))
    out, peer_ids, peer_sim = [], [], []
    for p in peers:
        p = sorted(p, key=lambda x: (ids[x[0]], -x[1]))
        out.append([x[0] for x in p])
        peer_ids.append(tuple(ids[x[0]] for x in p))
        peer_sim.append(tuple(x[1] for x in p))
    return out, tuple(peer_ids), tuple(peer_sim)


def sequence_from_group(meta, group, transition):
    hard = np.eye(CLASSES, dtype=np.float64)[group.argmax(1)]
    emission = 0.65 * group + 0.35 * hard
    seq = emission.copy()
    classes = group.argmax(1)
    block = np.full(len(group), -1, np.int64)
    keys = list(zip(meta["subject"], meta["scene"], meta["session"]))
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(tuple(key), []).append(i)
    bid = 0
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda i: (
            meta["ordinal"][i], meta["start_ms"][i], meta["interval_id"][i]
        ))
        chunks, cur = [], []
        for i in ordered:
            if cur and meta["ordinal"][i] != meta["ordinal"][cur[-1]] + 1:
                chunks.append(cur)
                cur = []
            cur.append(i)
        if cur:
            chunks.append(cur)
        for chunk in chunks:
            ix = np.asarray(chunk)
            block[ix] = bid
            bid += 1
            if len(ix) > 1:
                classes[ix] = _decode(emission[ix], transition, 0.45)
                seq[ix] = 0.49 * emission[ix] + 0.51 * np.eye(CLASSES)[classes[ix]]
    return seq / seq.sum(1, keepdims=True), classes, block


def apply_margin(direct, candidate, margin):
    base = direct.argmax(1)
    proposed = candidate.argmax(1)
    score = candidate[np.arange(len(base)), proposed] - candidate[np.arange(len(base)), base]
    changed = (proposed != base) & (score >= margin)
    out = direct.copy()
    out[changed] = candidate[changed]
    return out, changed


def _score(y, p):
    return {
        "macro_f1_fixed37": float(f1_score(
            y, p, labels=np.arange(CLASSES), average="macro", zero_division=0
        )),
        "accuracy": float(np.mean(y == p)),
        "correct": int(np.sum(y == p)),
    }


def _gated_choice(direct, proposal, y):
    rows = []
    for margin in MARGINS:
        p, changed = apply_margin(direct, proposal, margin)
        rows.append(dict(margin=margin, changed=int(changed.sum()), **_score(y, p.argmax(1))))
    best = max(rows, key=lambda r: (
        r["macro_f1_fixed37"], r["accuracy"], -r["changed"], -r["margin"]
    ))
    return best, rows


def _meta(data):
    return {
        "subject": np.asarray(data["subject"]),
        "scene": np.asarray(data["scene"]),
        "session": np.asarray(data["session"]),
        "start_ms": np.asarray(data["start_ms"], np.int64),
        "end_ms": np.asarray(data["end_ms"], np.int64),
        "interval_id": np.asarray(data["record_id"], str),
        "ordinal": np.asarray(data["ordinal"], np.int64),
    }


def _forward_filter(emission, transition, chunks):
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


def _peers_from_similarity(meta, similarity):
    out = [[] for _ in range(len(similarity))]
    subject, scene, session, starts, ids = (
        meta[k] for k in ("subject", "scene", "session", "start_ms", "interval_id")
    )
    for sub, scn in sorted(set(zip(subject, scene))):
        scope = np.flatnonzero((subject == sub) & (scene == scn))
        for a, b in combinations(sorted(set(session[scope])), 2):
            left, right = scope[session[scope] == a], scope[session[scope] == b]
            left = left[np.lexsort((ids[left], starts[left]))]
            right = right[np.lexsort((ids[right], starts[right]))]
            for x, y, _ in _monotone(similarity, left, right):
                out[x].append(y)
                out[y].append(x)
    return [sorted(p, key=lambda x: ids[x]) for p in out]


def _shuffle_peers(meta, peers, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i, pp in enumerate(peers):
        used = []
        for j in pp:
            pool = np.flatnonzero(
                (meta["subject"] == meta["subject"][i])
                & (meta["scene"] == meta["scene"][i])
                & (meta["session"] == meta["session"][j])
            )
            pool = np.asarray([x for x in pool if x not in used and x != i])
            if not len(pool):
                raise ValueError("empty compatible shuffle pool")
            alternatives = pool[pool != j]
            used.append(int(rng.choice(alternatives if len(alternatives) else pool)))
        out.append(used)
    assert all(len(a) == len(b) for a, b in zip(peers, out))
    return out


def _fixed_mean(bank, mask, peers):
    out = np.zeros((len(bank), CLASSES))
    sq = np.sqrt(bank)
    for i, pp in enumerate(peers):
        members = np.asarray([i, *pp])
        expert_means = []
        for k in range(bank.shape[1]):
            valid = members[mask[members, k]]
            if len(valid):
                expert_means.append(sq[valid, k].mean(0))
        out[i] = np.mean(expert_means, axis=0)
    return out / out.sum(1, keepdims=True)


def build_targets(source, selection, target, config, *, label_role="selection",
                  include_controls=False):
    """Fit on the source OOF bank and select margins/paths on the selection bank."""
    if label_role not in {"source", "selection"}:
        raise PermissionError("path selection requires source or selection labels")
    if "label" in target:
        raise ValueError("target bank must not contain labels")
    mixture = {
        "names": [source["expert_names"][i] for i in config["direct_expert_indices"]],
        "weights": config["direct_expert_weights"],
    }
    splits = {}
    for name, data in (("source", source), ("selection", selection), ("target", target)):
        bank = np.asarray(data["expert_bank"])
        mask = np.asarray(data["availability"], bool)
        expert_names = np.asarray(data["expert_names"], str)
        meta = _meta(data)
        direct = weighted_available_mix(bank, mask, expert_names, mixture)
        peers, peer_ids, _ = geometry_posterior_peers(meta, data["geometry"], direct)
        features = build_group_features(bank, direct.argmax(1), peers, expert_mask=mask)
        splits[name] = {
            "bank": bank, "mask": mask, "meta": meta, "direct": direct,
            "features": features, "peer_ids": peer_ids,
        }
        if include_controls:
            splits[name].update(peers=peers, geometry=np.asarray(data["geometry"]))
    src, val = splits["source"], splits["selection"]
    labels = np.asarray(source["label"], int)
    vlabels = np.asarray(selection["label"], int)
    joint = np.concatenate((src["features"], val["features"]))
    nsource = len(labels)
    head = fit_group_head(
        joint, np.r_[labels, np.full(len(vlabels), -1)],
        np.r_[source["record_id"], selection["record_id"]],
        fit_indices=np.arange(nsource), holdout_indices=np.arange(nsource, len(joint)),
        expert_names=np.asarray(source["expert_names"], str), class_count=CLASSES,
        posterior_provenance="source_oof", regularization_c=0.03,
    )
    sm = src["meta"]
    transition = fit_source_transition(
        subject_ids=sm["subject"], scene_ids=sm["scene"], session_ids=sm["session"],
        starts=sm["start_ms"], ends=sm["end_ms"], interval_ids=sm["interval_id"],
        labels=labels, num_classes=CLASSES, annotation_ordinals=sm["ordinal"],
    )
    for name in ("selection", "target"):
        s = splits[name]
        group = head.predict_proba(s["features"])
        sequence, _, _ = sequence_from_group(s["meta"], group, transition)
        # The decomposition reads these two layers through the frozen float32 bank.
        s["group"] = group.astype(np.float32).astype(float)
        s["sequence"] = sequence.astype(np.float32).astype(float)
        direct = s["direct"].astype(np.float32).astype(float)
        chunks = _blocks(
            list(zip(s["meta"]["subject"], s["meta"]["scene"], s["meta"]["session"])),
            s["meta"]["start_ms"], s["meta"]["interval_id"], s["meta"]["ordinal"],
        )
        decoded = np.empty(len(direct), dtype=int)
        for chunk in chunks:
            decoded[chunk] = (_decode(direct[chunk], transition, 0.45)
                              if len(chunk) > 1 else direct[chunk].argmax(1))
        s["direct_sequence_gated"] = 0.49 * s["direct"] + 0.51 * np.eye(CLASSES)[decoded]
        s["pred"] = {
            "direct": direct.argmax(1), "group": s["group"].argmax(1),
            "sequence": s["sequence"].argmax(1), "direct_sequence_nogate": decoded,
            "online_forward": _forward_filter(direct, transition, chunks).argmax(1),
        }
    chosen, ds_records = _gated_choice(val["direct"], val["direct_sequence_gated"], vlabels)
    for name in ("selection", "target"):
        s = splits[name]
        s["pred"]["direct_sequence"] = apply_margin(
            s["direct"], s["direct_sequence_gated"], chosen["margin"]
        )[0].argmax(1)
        s["pred"]["direct_sequence_gated"] = s["pred"]["direct_sequence"].copy()
    records = []
    for rank, path in enumerate(PATHS):
        if path in ("group", "sequence", "direct_sequence_gated"):
            for margin in MARGINS:
                pred, changed = apply_margin(val["direct"], val[path], margin)
                records.append(dict(path=path, rank=rank, margin=margin,
                                    changed=int(changed.sum()), **_score(vlabels, pred.argmax(1))))
        else:
            p = val["pred"][path]
            records.append(dict(path=path, rank=rank, margin=0.0,
                                changed=int(np.sum(p != val["pred"]["direct"])), **_score(vlabels, p)))
    winner = max(records, key=lambda r: (
        r["macro_f1_fixed37"], r["accuracy"], -r["changed"], -r["rank"], -r["margin"]
    ))
    for name in ("selection", "target"):
        s = splits[name]
        path = winner["path"]
        s["pred"]["selected"] = (
            apply_margin(s["direct"], s[path], winner["margin"])[0].argmax(1)
            if path in ("group", "sequence", "direct_sequence_gated")
            else s["pred"][path].copy()
        )
    result = {
        "predictions": splits["target"]["pred"],
        "selection_predictions": val["pred"],
        "selected_path": winner["path"], "selection_records": records,
        "gates": {"direct_sequence": chosen["margin"], "direct_sequence_records": ds_records,
                  "selected": winner},
        "direct_selection": mixture,
    }
    if include_controls:
        prior = np.bincount(labels, minlength=CLASSES) / len(labels)
        mean_policy = None
        for name in ("selection", "target"):
            s = splits[name]
            direct, meta = s["direct"], s["meta"]
            base = direct.argmax(1)
            group = head.predict_proba(s["features"])
            empty = [[] for _ in range(len(direct))]
            controls = {
                "group": group.argmax(1), "group_nogate": group.argmax(1),
                "match_combined": group.argmax(1),
                "group_no_peers": head.predict_proba(build_group_features(
                    s["bank"], base, empty, expert_mask=s["mask"]
                )).argmax(1),
                # E3 uses the ungated MAP path under this name; the main path
                # dictionary and its margin-selection order are unchanged.
                "direct_sequence": s["pred"]["direct_sequence_nogate"].copy(),
                "direct_sequence_gated": s["pred"]["direct_sequence_gated"].copy(),
            }
            chunks = _blocks(
                list(zip(meta["subject"], meta["scene"], meta["session"])),
                meta["start_ms"], meta["interval_id"], meta["ordinal"],
            )
            for method, emission in (
                ("order_only", np.full_like(direct, 1 / CLASSES)),
                ("order_prior", np.broadcast_to(prior, direct.shape)),
            ):
                pred = np.empty(len(direct), dtype=int)
                for chunk in chunks:
                    pred[chunk] = (_decode(emission[chunk], transition, .45)
                                   if len(chunk) > 1 else emission[chunk].argmax(1))
                controls[method] = pred
            geometry = s["geometry"].astype(float)
            cosine = (np.clip(geometry @ geometry.T, -1, 1) + 1) / 2
            bc = np.sqrt(direct) @ np.sqrt(direct).T
            variants = {
                "match_geometry_only": _peers_from_similarity(meta, cosine),
                "match_posterior_only": _peers_from_similarity(meta, bc),
            }
            variants.update({
                f"group_shuffled_peers_seed{seed}": _shuffle_peers(meta, s["peers"], seed)
                for seed in range(2026101501, 2026101506)
            })
            for method, pp in variants.items():
                proposal = head.predict_proba(build_group_features(
                    s["bank"], base, pp, expert_mask=s["mask"]
                ))
                controls[method] = apply_margin(direct, proposal, 0.)[0].argmax(1)
            mean = _fixed_mean(s["bank"], s["mask"], s["peers"])
            if name == "selection":
                mean_policy, mean_records = _gated_choice(direct, mean, vlabels)
            controls["group_fixed_mean"] = apply_margin(
                direct, mean, mean_policy["margin"]
            )[0].argmax(1)
            result["selection_controls" if name == "selection" else "controls"] = controls
        result["control_policies"] = {
            "fixed_mean_gate": mean_policy, "fixed_mean_records": mean_records,
        }
    return result
