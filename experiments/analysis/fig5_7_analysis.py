"""Render Figures 5–7: rescue scatter, class lollipop and session outcomes."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', help='Aggregate JSON from analysis_stats.py.')
    parser.add_argument('output', help='Figure output directory.')
    args = parser.parse_args(argv)
    import json
    import sys
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    res = json.load(open(args.results, encoding="utf-8"))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
    W = 3.45
    folds = ["0", "1", "2"]


    def save(fig, name):
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


    def pretty(n):
        return n.replace("_", " ").replace("Take body temperature", "Take temperature").replace(
            "Take and use tableware", "Use tableware")


    # ---- Fig. 5: rescued vs harmed scatter
    a4 = res["A4_rescue_harm"]
    selp = {"0": "sequence", "1": "direct_sequence", "2": "direct_sequence"}


    def tot(key, field):
        return sum(a4[f][key if key != "selected" else ("selected" if "selected" in a4[f] else selp[f])][field] for f in folds)


    pts = [("selected", "PRISM (selected)", "#2E3F5C", "o", 1),
           ("direct_sequence", "D→S (decoder)", "#F0A04B", "o", 1),
           ("sequence", "D→G→S (full chain)", "#C0504D", "o", 1),
           ("group", "D→G (peers)", "#7FB3D5", "o", 1),
           ("direct_viterbi_naive", "Naive Viterbi", "#8C8C8C", "s", 0),
           ("online_forward", "Online filtering", "#8C8C8C", "D", 0)]
    fig, ax = plt.subplots(figsize=(W, 2.45))
    xs = np.linspace(0, 100, 50)
    for net, tx, ty in [(0, 74, 70), (50, 2, 57), (100, 2, 107)]:
        ax.plot(xs, xs + net, color="#C8C8C8", lw=0.7, ls=(0, (4, 3)), zorder=0)
        ax.text(tx, ty, "net 0" if net == 0 else f"net +{net}", fontsize=6.8, color="#888", rotation=27, va="bottom")
    place = {"selected": (64, 150, "left"), "direct_sequence": (30, 140, "right"), "sequence": (64, 172, "left"),
             "group": (40, 18, "left"), "direct_viterbi_naive": (27, 66, "left"), "online_forward": (84, 104, "left")}
    for key, lab, col, mk, filled in pts:
        r, h, ch = tot(key, "rescue"), tot(key, "harm"), tot(key, "changed")
        ax.scatter(h, r, s=14 + ch * 0.22, marker=mk, color=col if filled else "white", edgecolor=col, linewidth=1.2, zorder=3)
        tx, ty, ha = place[key]
        ax.annotate(f"{lab}  (+{r - h})", (h, r), xytext=(tx, ty), textcoords="data", fontsize=7,
                    color=col if filled else "#555", ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#999", shrinkA=0, shrinkB=3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 190)
    ax.set_xlabel("Clips harmed (correct label broken)"); ax.set_ylabel("Clips rescued (error corrected)")
    ax.text(2, 186, "marker area: clips changed\n(+n): net gain", fontsize=6.5, color="#777", ha="left", va="top")
    fig.tight_layout(); save(fig, "fig5_rescue_scatter")

    # ---- Fig. 6: lollipop of per-class F1 change, all classes
    rows = sorted(res["A2_per_class"], key=lambda r: -r["delta"])
    d = np.array([r["delta"] for r in rows]); x = np.arange(len(rows))
    PUR, GOLD = "#5B4B8A", "#C9A227"
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for xi, di in zip(x, d):
        col = PUR if di > 0 else (GOLD if di < 0 else "#AAAAAA")
        ax.vlines(xi, 0, di, color=col, lw=1.3)
        ax.scatter(xi, di, s=14, color=col, zorder=3)
    ax.axhline(0, color="#555", lw=0.6)
    names = [pretty(r["class"]) for r in rows]
    cls = [r["class"] for r in rows]
    lab_pos = [(0, 6, 46), (1, 9, 36), (2, 12, 29), (3, 15, 22),
               (cls.index("Take_medicine"), 23, 15), (cls.index("Eat_food"), 24, 9.5)]
    for i, tx, ty in lab_pos:
        ax.annotate(f"{names[i]} ({d[i]:+.1f})", (x[i], d[i]), xytext=(tx, ty), textcoords="data", fontsize=6.8,
                    color=PUR, ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="#B9B0D6",
                                                                         shrinkA=0, shrinkB=2))
    i = len(rows) - 1
    ax.annotate(f"{names[i]} ({d[i]:+.1f})", (x[i], d[i]), xytext=(-6, 0), textcoords="offset points",
                fontsize=6.8, color="#9C7A10", ha="right", va="center")
    n_up, n_zero, n_down = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
    ax.text(len(rows) - 1, 46, f"{n_up} up, {n_zero} unchanged, {n_down} down", ha="right", fontsize=7, color="#444")
    ax.set_xticks([]); ax.set_xlim(-1, len(rows))
    ax.set_ylim(-22, 52); ax.set_ylabel("Change in per-class F1 (points)")
    ax.set_xlabel("40 CUHK-X classes, sorted by change")
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout(); save(fig, "fig6_class_change")

    # ---- Fig. 7: session outcomes by length, 100% stacked bars
    a5 = res["A5_session_length"]; keys = list(a5)
    imp = np.array([sum(g > 0 for g in a5[k]["gains"]) for k in keys], float)
    wor = np.array([sum(g < 0 for g in a5[k]["gains"]) for k in keys], float)
    tot_s = np.array([a5[k]["sessions"] for k in keys], float)
    unc = tot_s - imp - wor
    gain = [a5[k]["prism_acc"] - a5[k]["direct_acc"] for k in keys]
    TEAL, AMBER = "#2A9D8F", "#E9A23B"
    fig, ax = plt.subplots(figsize=(W, 1.75))
    y = np.arange(len(keys))[::-1]
    left = np.zeros(len(keys))
    for vals, col, lab in [(imp, TEAL, "improved"), (unc, "#D9D9D9", "unchanged"), (wor, AMBER, "worse")]:
        share = 100 * vals / tot_s
        ax.barh(y, share, left=left, color=col, height=0.6, label=lab, edgecolor="white", linewidth=0.5)
        for yi, l, sh in zip(y, left, share):
            if sh >= 9:
                ax.text(l + sh / 2, yi, f"{sh:.0f}%", ha="center", va="center", fontsize=6.8,
                        color="white" if col != "#D9D9D9" else "#444")
        left += share
    for yi, g in zip(y, gain):
        ax.text(103, yi, f"+{g:.1f}", va="center", fontsize=7.5, color="#1F5F57", fontweight="bold")
    ax.text(103, len(keys) - 0.35, "gain", fontsize=6.8, color="#1F5F57")
    ax.set_yticks(y, [f"{k} clips ({int(t)})" for k, t in zip(keys, tot_s)])
    ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of sessions (%)")
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1, columnspacing=0.8)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.tight_layout(); save(fig, "fig7_session_outcomes")
    print("OK", n_up, n_zero, n_down, imp, wor, unc)


if __name__ == "__main__":
    main()
