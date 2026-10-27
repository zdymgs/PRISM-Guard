"""Compute the CUHK-X statistics used by Figures 3 and 5–7."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', help='Locally generated teacher prediction NPZ.')
    parser.add_argument('metadata', help='Recording metadata CSV.')
    parser.add_argument('class_manifest', help='Class ID/name CSV.')
    parser.add_argument('output', help='Output aggregate JSON.')
    args = parser.parse_args(argv)
    import csv
    import json
    import sys
    from collections import Counter, defaultdict

    import numpy as np

    PRED, META, MAN, OUT = args.predictions, args.metadata, args.class_manifest, args.output
    from pathlib import Path
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    z = np.load(PRED)
    ids = z["evaluation_ids"].astype(str)
    clusters = z["evaluation_clusters"].astype(str)
    fold = z["evaluation_outer_fold"]
    P = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("evaluation_prediction__")}
    meta = {r["sample_id"]: r for r in csv.DictReader(open(META, encoding="utf-8-sig"))}
    names = {int(r["class_id"]): r["class_label"] for r in csv.DictReader(open(MAN, encoding="utf-8-sig"))}
    y = np.array([int(meta[s]["class_id"]) for s in ids])
    users = np.array([meta[s]["user_id"] for s in ids])
    C = 40


    def macro_f1(t, p):
        f = []
        for c in range(C):
            tp = np.sum((p == c) & (t == c)); fp = np.sum((p == c) & (t != c)); fn = np.sum((p != c) & (t == c))
            f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        return 100 * np.mean(f), np.array(f) * 100


    # selected path per outer fold (selected on source labels): fold 0 full chain, folds 1-2 D->S
    sel = P["selected"] if "selected" in P else np.where(fold == 0, P["sequence"], P["direct_sequence"])
    out = {"folds_users": {int(f): sorted(set(users[fold == f])) for f in np.unique(fold)}}
    out["check"] = {"direct": macro_f1(y, P["direct"])[0], "selected": macro_f1(y, sel)[0],
                    "full_chain": macro_f1(y, P["sequence"])[0], "equal_mean": macro_f1(y, P["equal_mean"])[0]}

    # A1 per fold
    a1 = {}
    for f in np.unique(fold):
        m = fold == f
        a1[int(f)] = {k: macro_f1(y[m], v[m])[0] for k, v in
                      [("direct", P["direct"]), ("D->S", P["direct_sequence"]), ("D->G", P["group"]),
                       ("full", P["sequence"]), ("selected", sel), ("naive_viterbi", P["direct_viterbi_naive"]),
                       ("online", P["online_forward"])]}
        a1[int(f)]["n"] = int(m.sum())
    out["A1_per_fold"] = a1

    # A2 per-class F1 change Direct -> selected
    _, fd = macro_f1(y, P["direct"]); _, fs = macro_f1(y, sel)
    out["A2_per_class"] = sorted([{"class": names[c], "direct": fd[c], "prism": fs[c], "delta": fs[c] - fd[c],
                                   "support": int(np.sum(y == c))} for c in range(C)], key=lambda r: -r["delta"])

    # A3 confused pairs (true -> predicted) under Direct, with counts under PRISM
    cd = Counter((int(a), int(b)) for a, b in zip(y, P["direct"]) if a != b)
    cs = Counter((int(a), int(b)) for a, b in zip(y, sel) if a != b)
    top = cd.most_common(12)
    out["A3_pairs"] = [{"true": names[a], "pred": names[b], "direct": n, "prism": cs.get((a, b), 0)} for (a, b), n in top]
    # small confusion block for the classes involved in the top pairs
    cls = []
    for (a, b), _ in cd.most_common(6):
        for c in (a, b):
            if c not in cls:
                cls.append(c)
    cls = cls[:8]
    def block(pred):
        M = np.zeros((len(cls), len(cls)), int)
        for i, a in enumerate(cls):
            for j, b in enumerate(cls):
                M[i, j] = int(np.sum((y == a) & (pred == b)))
        return M.tolist()
    out["A3_block"] = {"classes": [names[c] for c in cls], "direct": block(P["direct"]), "prism": block(sel)}

    # A4 rescues and harms relative to Direct, per path and fold
    a4 = {}
    for f in np.unique(fold):
        m = fold == f; b = P["direct"][m]; t = y[m]
        a4[int(f)] = {}
        for k in ["direct_sequence", "group", "sequence", "direct_viterbi_naive", "online_forward"]:
            p = P[k][m]
            a4[int(f)][k] = {"rescue": int(np.sum((p == t) & (b != t))), "harm": int(np.sum((b == t) & (p != t))),
                             "changed": int(np.sum(p != b))}
    out["A4_rescue_harm"] = a4
    if "selected" in P:
        for f in np.unique(fold):
            m = fold == f; b = P["direct"][m]; t = y[m]; p = sel[m]
            a4[int(f)]["selected"] = {
                "rescue": int(np.sum((p == t) & (b != t))),
                "harm": int(np.sum((b == t) & (p != t))),
                "changed": int(np.sum(p != b)),
            }

    # A5 session length vs gain (accuracy, selected vs direct)
    sess = defaultdict(list)
    for i, c in enumerate(clusters):
        sess[c].append(i)
    rows = []
    for c, idx in sess.items():
        idx = np.array(idx)
        rows.append({"len": len(idx), "direct": float(np.mean(P["direct"][idx] == y[idx])),
                     "prism": float(np.mean(sel[idx] == y[idx])), "n": len(idx)})
    buckets = {"1-3": (1, 3), "4-6": (4, 6), "7-9": (7, 9), "10+": (10, 999)}
    a5 = {}
    for name, (lo, hi) in buckets.items():
        rs = [r for r in rows if lo <= r["len"] <= hi]
        if not rs:
            continue
        w = np.array([r["n"] for r in rs], float)
        a5[name] = {"sessions": len(rs), "clips": int(w.sum()),
                    "direct_acc": float(np.average([r["direct"] for r in rs], weights=w) * 100),
                    "prism_acc": float(np.average([r["prism"] for r in rs], weights=w) * 100),
                    "gains": [100 * (r["prism"] - r["direct"]) for r in rs]}
    out["A5_session_length"] = a5
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps(out["check"], indent=1))
    print({k: {kk: round(vv, 2) for kk, vv in v.items() if kk != "n"} for k, v in a1.items()})


if __name__ == "__main__":
    main()
