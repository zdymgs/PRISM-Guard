"""Render Figure 3: CUHK-X teacher targets per outer fold."""
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
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
    W, W2 = 3.45, 7.1
    GREEN, RED, NAVY, GRAY = "#3A9E6B", "#D9534F", "#2E3F5C", "#9AA0A6"
    folds = ["0", "1", "2"]


    def save(fig, name):
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


    def pretty(n):
        return n.replace("_", " ").replace("Take body temperature", "Take temperature").replace(
            "Listen to music with headphones", "Listen to music").replace("Take and use tableware", "Use tableware")


    # ---- per-fold grouped bars
    fig, ax = plt.subplots(figsize=(W, 2.25))
    series = [("direct", "Direct", "#B8C2CC"), ("D->G", "D→G", "#7FB3D5"), ("D->S", "D→S", "#F0A04B"),
              ("full", "D→G→S", "#C0504D"), ("selected", "PRISM (selected)", NAVY)]
    x = np.arange(3); w = 0.16
    for k, (key, lab, col) in enumerate(series):
        ax.bar(x + (k - 2) * w, [res["A1_per_fold"][f][key] for f in folds], w, label=lab, color=col,
               edgecolor="white", linewidth=0.4)
    for i, f in enumerate(folds):
        d = res["A1_per_fold"][f]
        ax.text(i + 2 * w, d["selected"] + 0.5, f"+{d['selected'] - d['direct']:.1f}", ha="center", fontsize=7.5,
                color=NAVY, fontweight="bold")
    ax.set_xticks(x, [f"Fold {f}\n({res['A1_per_fold'][f]['n']} clips)" for f in folds])
    ax.set_ylim(65, 95); ax.set_ylabel("Macro-F1 (%)")
    ax.legend(ncol=3, frameon=False, loc="upper left", handlelength=1.0, columnspacing=0.6, borderaxespad=0.1)
    fig.tight_layout(); save(fig, "fig3_folds")


if __name__ == "__main__":
    main()
