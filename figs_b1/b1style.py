"""Common style and render checks for the paper-B1 figures.

Single-column width 3.4 in, Okabe-Ito colours (distinguishable with colour-vision deficiency),
numerics as points with error bars, analytic predictions as lines, panel labels (a)(b)(c).
check(fig, name) saves PDF + PNG and reports: overlapping text, text outside the figure,
and legends that cover data points.
"""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'figs_b1', 'out'); os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(ROOT, 'results', 'b1_figs'); os.makedirs(DATA, exist_ok=True)
W1 = 3.4  # single column (in)

# Okabe-Ito
K, OR, SB, GR, YE, BL, VE, PU = '#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'
SEQ = [BL, VE, GR, PU, OR, SB, K]
MK = ['o', 's', '^', 'D', 'v', 'p', 'h']

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'mathtext.fontset': 'dejavusans',
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'lines.linewidth': 1.2, 'lines.markersize': 4, 'errorbar.capsize': 1.5,
    'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.borderaxespad': 0.3, 'legend.labelspacing': 0.25,
    'savefig.dpi': 300, 'pdf.fonttype': 42, 'figure.dpi': 150,
})

def label(ax, s, x=-0.02, y=1.0, **kw):
    """Panel label (a), (b), ... in the upper-left corner outside the axes frame."""
    ax.text(x, y, f'({s})', transform=ax.transAxes, ha='right', va='bottom', fontsize=9, fontweight='bold', **kw)

def pts(ax, x, y, yerr=None, c=BL, m='o', ms=4, **kw):
    """Numerical data: filled markers with a thin white edge, optional error bars."""
    return ax.errorbar(x, y, yerr=yerr, fmt=m, color=c, ms=ms, mec='white', mew=0.5, elinewidth=0.7, capsize=1.5, ls='none', **kw)

def _texts(fig):
    arts = list(fig.texts)
    for ax in fig.axes:
        if not ax.get_visible(): continue
        if not ax.axison:
            arts += list(ax.texts) + ([ax.get_legend()] if ax.get_legend() else []); continue
        x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
        tx = [t for t in ax.get_xticklabels() if t.get_text() and x0 - 1e-9*abs(x0) <= t.get_position()[0] <= x1 + 1e-9*abs(x1)]
        ty = [t for t in ax.get_yticklabels() if t.get_text() and y0 - 1e-9*abs(y0) <= t.get_position()[1] <= y1 + 1e-9*abs(y1)]
        arts += [ax.title, ax.xaxis.label, ax.yaxis.label] + tx + ty + list(ax.texts)
        if ax.get_legend(): arts.append(ax.get_legend())
    return [a for a in arts if a.get_visible() and (not hasattr(a, 'get_text') or a.get_text())]

def check(fig, name, allow_legend_on=()):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    arts = _texts(fig); boxes = [(a, a.get_window_extent(r)) for a in arts]
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.width > 0 and b.width > 0 and a.overlaps(b):
                ov = max(0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
                if ov > 2: bad.append(('text-text', str(boxes[i][0])[:45], str(boxes[j][0])[:45]))
    # nothing is clipped (bbox_inches='tight'), but the saved width must stay within one column
    tb = fig.get_tightbbox(r); wid = tb.width
    if wid > fig.get_figwidth()*1.03: bad.append(('too-wide', f'{wid:.2f} in > {fig.get_figwidth():.2f} in', ''))
    # legends or free text covering data points
    for ax in fig.axes:
        covers = ([ax.get_legend()] if ax.get_legend() else []) + list(ax.texts)
        data = []
        for ln in ax.get_lines():
            if not ln.get_visible() or ln.get_label() in allow_legend_on: continue
            if ln.get_transform() != ax.transData: continue   # axhline/axvline live in blended coordinates
            xy = np.column_stack(ln.get_data()).astype(float)
            if len(xy): data.append((ln, ax.transData.transform(xy)))
        for col in ax.collections:
            off = col.get_offsets()
            if len(off) and col.get_visible(): data.append((col, ax.transData.transform(np.asarray(off, float))))
        for c_ in covers:
            if c_ is None or not c_.get_visible(): continue
            bb = c_.get_window_extent(r).expanded(1.0, 1.0)
            for art, d in data:
                if getattr(art, 'get_gid', lambda: None)() == 'nocheck': continue
                inside = (d[:, 0] > bb.x0) & (d[:, 0] < bb.x1) & (d[:, 1] > bb.y0) & (d[:, 1] < bb.y1)
                if inside.any():
                    bad.append(('covers-data', str(c_)[:45], str(art.get_label())[:30])); break
    print(f'{name}: render check', 'OK' if not bad else f'{len(bad)} issues:', *bad[:8], sep='\n  ', flush=True)
    fig.savefig(os.path.join(OUT, f'{name}.pdf'), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(OUT, f'{name}.png'), bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig); return bad
