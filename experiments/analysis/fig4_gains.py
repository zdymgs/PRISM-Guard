"""Render Figure 4 from locally computed metrics.

Input CSV columns: dataset, experiment, split, method, delta_macro_f1,
macro_f1_ci95_lower, macro_f1_ci95_upper (metrics are fractions).
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_names = {f.name for f in fm.fontManager.ttflist}
LATIN = ['Liberation Serif'] if 'Liberation Serif' in _names else ['Times New Roman', 'DejaVu Serif']
CJK = [n for n in ['Noto Serif CJK SC', 'Noto Serif CJK JP', 'Source Han Serif SC', 'SimSun', 'Songti SC'] if n in _names][:1]
plt.rcParams.update({
    'font.family': LATIN,
    'mathtext.fontset': 'stix', 'font.size': 7, 'axes.linewidth': 0.5,
    'pdf.fonttype': 42, 'axes.unicode_minus': False,
})

EXP = {'CUHK-X': ('E1R_clean23', 'all_2470'), 'MMAct': ('E3_component_ablation', 'F0'),
       'MM-Fit': ('E2_information_decomposition', 'TEST')}
SELECTED_METHOD = {'CUHK-X': 'selected', 'MMAct': 'direct_sequence', 'MM-Fit': 'direct_sequence'}
SELECTED_SOURCE = {'CUHK-X': ('E5_selected_path_completed', 'all_2470')}

PRISM = '#A4400F'; SEQ = '#C0761C'; PEER = '#2F6B9A'; CTRL = '#8C8C8C'; INK = '#222222'
BAND = '#FBF0E4'

# (label, method key, kind, colour)   kind: 'prism' filled + bold, 'ctrl' hollow grey, 'shuf' seed range
ROWS = [
    ('PRISM (selected path)',     'SELECTED',             'prism', PRISM),
    ('Full chain D$\\to$G$\\to$S', 'sequence',             'prism', PRISM),
    'Sequence step',
    ('D$\\to$S: protocol decoding', 'direct_sequence',     'prism', SEQ),
    ('Online filtering',          'online_forward',       'ctrl',  CTRL),
    ('Naive Viterbi',             'direct_viterbi_naive', 'ctrl',  CTRL),
    'Peer step and gate',
    ('D$\\to$G: learned peers',    'group',                'prism', PEER),
    ('Fixed-mean peers',          'group_fixed_mean',     'ctrl',  CTRL),
    ('No peers',                  'group_no_peers',       'ctrl',  CTRL),
    ('Random peers (5 seeds)',    'SHUF',                 'shuf',  CTRL),
    ('D$\\to$G without gate',      'group_nogate',         'ctrl',  CTRL),
]
ZH = {  # zh labels, same order as ROWS
    'PRISM (selected path)': 'PRISM（选中路径）', 'Full chain D$\\to$G$\\to$S': '完整链 D\u2192G\u2192S',
    'Sequence step': '序列步', 'D$\\to$S: protocol decoding': 'D\u2192S：协议约束解码',
    'Online filtering': '在线前向滤波', 'Naive Viterbi': '朴素 Viterbi', 'Peer step and gate': '同伴步与门控',
    'D$\\to$G: learned peers': 'D\u2192G：学习式同伴', 'Fixed-mean peers': '同伴：固定均值', 'No peers': '去掉同伴',
    'Random peers (5 seeds)': '随机同伴（5 个种子）', 'D$\\to$G without gate': 'D\u2192G 去掉门控',
    'not run': '未运行', 'collapse': '崩溃',
    'xlabel': '相对 Direct 的 \u0394Macro-F1（百分点），会话级 bootstrap 95% 置信区间',
}
XLIM = {'CUHK-X': (-5.6, 8.4), 'MMAct': (-3.6, 15.4), 'MM-Fit': (-7.6, 3.0)}
TICKS = {'CUHK-X': [-4, 0, 4, 8], 'MMAct': [0, 5, 10, 15], 'MM-Fit': [-6, -3, 0]}
TICKLABEL = lambda v: ('%g' % v).replace('-', '\u2212')


def load_real(path):
    import pandas as pd
    m = pd.read_csv(path, encoding='utf-8-sig')
    D = {}
    for ds, (exp, sp) in EXP.items():
        sub = m[(m.dataset == ds) & (m.experiment == exp) & (m.split == sp)]
        D[ds] = {}
        for _, r in sub.iterrows():
            D[ds][r.method] = {'d': float(r.delta_macro_f1) * 100,
                               'ci': [float(r.macro_f1_ci95_lower) * 100, float(r.macro_f1_ci95_upper) * 100]}
        sh = sub[sub.method.str.contains('shuffled')].delta_macro_f1.astype(float) * 100
        if len(sh):
            D[ds]['SHUF'] = {'d': sh.mean(), 'ci': [sh.min(), sh.max()]}
        if ds in SELECTED_SOURCE:
            e2, s2 = SELECTED_SOURCE[ds]
            r2 = m[(m.dataset == ds) & (m.experiment == e2) & (m.split == s2) & (m.method == SELECTED_METHOD[ds])]
            assert len(r2) == 1, f'{ds}: expected 1 selected row, got {len(r2)}'
            r2 = r2.iloc[0]
            D[ds][SELECTED_METHOD[ds]] = {'d': float(r2.delta_macro_f1) * 100,
                                          'ci': [float(r2.macro_f1_ci95_lower) * 100, float(r2.macro_f1_ci95_upper) * 100]}
        sel = SELECTED_METHOD[ds]
        if sel in D[ds]:
            D[ds]['SELECTED'] = D[ds][sel]
        else:
            raise ValueError(f'{ds}: selected method {sel!r} is missing from the supplied metrics')
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', required=True, help='Locally computed metrics CSV.'); ap.add_argument('--out', required=True)
    ap.add_argument('--lang', choices=['en', 'zh'], default='en')
    args = ap.parse_args()
    zh = args.lang == 'zh'
    tr = (lambda t: ZH.get(t, t)) if zh else (lambda t: t)
    if zh:
        plt.rcParams['font.family'] = LATIN + CJK
    D = load_real(args.metrics)

    n = len(ROWS); ys = np.arange(n)[::-1].astype(float)
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.55), sharey=True,
                            gridspec_kw=dict(width_ratios=[1.0, 1.25, 0.95]))
    plt.subplots_adjust(left=0.215, right=0.995, top=0.92, bottom=0.12, wspace=0.07)
    for ax, ds in zip(axs, EXP):
        lo, hi = XLIM[ds]; span = hi - lo
        for y, r in zip(ys, ROWS):
            if isinstance(r, tuple) and r[2] == 'prism':
                ax.axhspan(y - 0.5, y + 0.5, color=BAND, lw=0, zorder=0)
        ax.axvline(0, color='#7a7a7a', lw=0.6, zorder=1)
        for y, r in zip(ys, ROWS):
            if isinstance(r, str):
                ax.axhline(y - 0.18, color='#d0d0d0', lw=0.4, zorder=1)
                continue
            label, key, kind, c = r
            v = D[ds].get(key)
            if v is None:
                ax.text(lo + 0.03 * span, y, tr('not run'), va='center', ha='left', fontsize=5.8,
                        color='#a0a0a0', style='italic')
                continue
            d = v['d']; ci = v.get('ci')
            if d < lo:  # off-axis value (MM-Fit online filtering)
                ax.annotate('', xy=(lo + 0.01 * span, y), xytext=(lo + 0.17 * span, y),
                            arrowprops=dict(arrowstyle='-|>', color='#B23A3A', lw=0.8, mutation_scale=6))
                ax.text(lo + 0.19 * span, y, f'{d:.1f}'.replace('-', '\u2212'), va='center', ha='left',
                        fontsize=6, color='#B23A3A', fontweight='bold')
                continue
            if ci is not None and ci[1] > ci[0]:
                ax.plot([max(ci[0], lo), min(ci[1], hi)], [y, y], color=c,
                        lw=2.0 if kind == 'prism' else (1.8 if kind == 'shuf' else 0.9),
                        alpha=0.45 if kind == 'shuf' else 1.0, solid_capstyle='butt', zorder=2)
            if kind == 'prism':
                ax.plot(d, y, 'o', ms=4.4, color=c, mec='white', mew=0.5, zorder=3)
            elif kind == 'shuf':
                ax.plot(d, y, 'D', ms=3.4, mfc='white', mec=c, mew=0.8, zorder=3)
            else:
                ax.plot(d, y, 'o', ms=3.6, mfc='white', mec=c, mew=0.8, zorder=3)
            right = max(d, ci[1] if ci else d)
            txt = ('0.0' if abs(d) < 0.05 else f'{d:+.1f}').replace('-', '\u2212')
            if right + 0.04 * span < hi - 0.02 * span:
                ax.text(right + 0.03 * span, y, txt, va='center', ha='left', fontsize=5.9,
                        color=c if kind == 'prism' else '#777777',
                        fontweight='bold' if kind == 'prism' else 'normal', zorder=4)
            else:
                left = min(d, ci[0] if ci else d)
                ax.text(left - 0.03 * span, y, txt, va='center', ha='right', fontsize=5.9,
                        color=c if kind == 'prism' else '#777777',
                        fontweight='bold' if kind == 'prism' else 'normal', zorder=4)
        ax.set_xlim(lo, hi); ax.set_ylim(-0.6, n - 0.4)
        ax.set_xticks(TICKS[ds]); ax.set_xticklabels([TICKLABEL(t) for t in TICKS[ds]])
        ax.set_title(ds, fontsize=8, fontweight='bold', pad=3)
        for s in ['top', 'right', 'left']:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis='x', length=2, pad=1.5, labelsize=6.5)
        ax.tick_params(axis='y', length=0)
    ax0 = axs[0]
    real = [r for r in ROWS if isinstance(r, tuple)]
    ax0.set_yticks([y for y, r in zip(ys, ROWS) if isinstance(r, tuple)])
    ax0.set_yticklabels([tr(r[0]) for r in real], fontsize=6.6)
    for t, r in zip(ax0.get_yticklabels(), real):
        t.set_color(r[3] if r[2] == 'prism' else '#555555')
        if r[2] == 'prism':
            t.set_fontweight('bold')
    for y, r in zip(ys, ROWS):  # group headers on the separator rows
        if isinstance(r, str):
            ax0.text(-0.03, y + 0.05, tr(r), transform=ax0.get_yaxis_transform(), ha='right', va='center',
                     fontsize=6.2, color='#555555', fontstyle='normal' if zh else 'italic')
    fig.supxlabel(tr('xlabel') if zh else '$\\Delta$macro-F1 vs. Direct (points), 95% session-level bootstrap interval',
                  fontsize=6.8, y=0.01)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=300)
    print('wrote', out)


if __name__ == '__main__':
    main()
