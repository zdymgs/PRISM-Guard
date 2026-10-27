"""Compute session co-occurrence and confusion counts used in the paper."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', help='Locally generated teacher prediction NPZ.')
    parser.add_argument('metadata', help='Recording metadata CSV.')
    parser.add_argument('class_manifest', help='Class ID/name CSV.')
    args = parser.parse_args(argv)
    import csv, sys
    from collections import Counter, defaultdict
    import numpy as np
    z = np.load(args.predictions); ids = z["evaluation_ids"].astype(str); cl = z["evaluation_clusters"].astype(str); fold = z["evaluation_outer_fold"]
    meta = {r["sample_id"]: r for r in csv.DictReader(open(args.metadata, encoding="utf-8-sig"))}
    names = {int(r["class_id"]): r["class_label"] for r in csv.DictReader(open(args.class_manifest, encoding="utf-8-sig"))}
    inv = {v: k for k, v in names.items()}
    y = np.array([int(meta[s]["class_id"]) for s in ids])
    P = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("evaluation_prediction__")}
    sel = P["selected"] if "selected" in P else np.where(fold == 0, P["sequence"], P["direct_sequence"])
    sess = defaultdict(set)
    for c, t in zip(cl, y): sess[c].add(t)
    for a, b in [("Use_a_mobile_phone", "Play_games"), ("Turn_pages", "Read_documents"), ("Mop_the_floor", "Sweep_the_floor"),
                 ("Take_and_use_tableware", "Stir_drinks"), ("Check_the_time", "Take_body_temperature"), ("Eat_food", "Take_medicine")]:
        A, B = inv[a], inv[b]
        na = sum(A in s for s in sess.values()); nb = sum(B in s for s in sess.values()); both = sum(A in s and B in s for s in sess.values())
        print(f"{a} / {b}: sessions with A {na}, with B {nb}, both {both}")
    m = y == inv["Wipe_bowls"]
    print("Wipe_bowls n", m.sum(), "direct ->", Counter(names[int(p)] for p in P["direct"][m]).most_common(4))
    print("Wipe_bowls prism ->", Counter(names[int(p)] for p in sel[m]).most_common(4))
    print("pred as Wipe_bowls (direct/prism):", Counter(names[int(t)] for t, p in zip(y, P["direct"]) if names[int(p)] == "Wipe_bowls").most_common(4),
          Counter(names[int(t)] for t, p in zip(y, sel) if names[int(p)] == "Wipe_bowls").most_common(4))


if __name__ == "__main__":
    main()
