"""Fig. 7: lambda is a static property of the pattern: one minus the counterflow momentum of its internal relaxation.
(a) 1D soliton lattice vs the homogenization formula; (b) skyrmion lattices: static and two-field predictions vs the exact
dynamic overlaps, displacement-only theory fails; (c) the counterflow cancels mainly the co-rotating momentum.
Data: results/b1_figs/fig7_csl.json, results/b1_lambda_alpha/static_P_*.json, ritz_P_*.log, results/b1_todo/circ_split.log."""
import json, os, re, glob, numpy as np, matplotlib.pyplot as plt
from b1style import *
LA = os.path.join(ROOT, 'results', 'b1_lambda_alpha')
fig = plt.figure(figsize=(W1, 3.0))
gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.5)
a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, :])

# (a) 1D theorem
cs = json.load(open(os.path.join(DATA, 'fig7_csl.json')))
r5 = cs['5.0']; B5 = np.array([r['B'] for r in r5])
a.plot(B5*1e2, [r['formula'] for r in r5], color=K, label='formula')
pts(a, B5*1e2, [r['num'] for r in r5], c=BL, label=r'$K_\perp=5$')
r2 = cs['0.2']; a.plot([r['B']*1e2 for r in r2], [r['formula'] for r in r2], color=VE, lw=0.8, ls=':')
a.plot([r['B']*1e2 for r in r2], [r['num'] for r in r2], 's', color=VE, mfc='none', mew=0.8, ms=3.5, label=r'$K_\perp=0.2$')
a.set_xlabel(r'$B$ ($10^{-2}J$)'); a.set_ylabel(r'$\lambda$ (1D)'); a.set_ylim(0, 1.08); a.set_xlim(-0.2, 4.2); a.set_yticks([0.5, 1.0])
a.legend(loc='upper right', fontsize=5.3)

# (b) skyrmion lattices: predictions vs exact dynamic overlaps
dyn, sta = {}, {}
for f in glob.glob(os.path.join(LA, 'static_P_*.json')):
    for key, v in json.load(open(f)).items(): dyn[key] = v['dyn']; sta[key] = v['static']
ritz = {}
for f in glob.glob(os.path.join(LA, 'ritz_P_*.log')):
    for l in open(f):
        m = re.match(r'(\d+_[\d.]+): (.*)', l.strip())
        if not m: continue
        g = dict(disp=re.search(r'disp P=2: tau=([\d.]+) P=([\d.]+)', l), two=re.search(r'2f P=2: tau=([\d.]+) P=([\d.]+)', l), aff=re.search(r'2f P=0: tau=([\d.]+) P=([\d.]+)', l))
        ritz[m.group(1)] = {k_: (float(v.group(1)), float(v.group(2))) for k_, v in g.items()}
keys = [k_ for k_ in ritz if k_ in dyn]
b.plot([0, 1.05], [0, 1.05], color='0.6', lw=0.6, zorder=0)
for j, (nm, src, col, mk) in enumerate((('static mode', None, BL, 'o'), ('two-field', 'two', GR, '^'), ('displacement only', 'disp', VE, 's'), ('affine', 'aff', '0.45', 'D'))):
    xs, ys, xt, yt = [], [], [], []
    for k_ in keys:
        xs.append(dyn[k_][1]); xt.append(dyn[k_][0])
        if src is None: ys.append(sta[k_][1]); yt.append(sta[k_][0])
        else: ys.append(ritz[k_][src][1]); yt.append(ritz[k_][src][0])
    b.plot(xs, ys, mk, color=col, ms=3.2, mec='white', mew=0.4, ls='none', label=nm)
    b.plot(xt, yt, mk, color=col, ms=3.2, mfc='none', mew=0.6, ls='none')
b.set_xlabel(r'exact $\kappa_P$ (filled), $\kappa_\tau$ (open)'); b.set_ylabel('prediction'); b.set_xlim(0, 1.0); b.set_ylim(0, 1.55); b.set_yticks([0, 0.5, 1.0]); b.set_xticks([0.5, 1.0])
b.legend(loc='upper left', fontsize=5.0, ncol=2, handletextpad=0.05, columnspacing=0.4, borderaxespad=0.15)

# (c) circular decomposition
rows = []
for l in open(os.path.join(ROOT, 'results', 'b1_todo', 'circ_split.log')):
    m = re.search(r'B=([\d.]+): P=([\d.]+) r=([\d.]+) \| pi\+=([\d.]+) pi-=([\d.]+) \| from split: kappa_P=([\d.]+) kappa_tau=([\d.]+)', l)
    if m: rows.append([float(x) for x in m.groups()])
R = np.array(rows); B = R[:, 0]
c.plot(B, [dyn[f'19_{x:g}'][1] for x in B], color=K, lw=1.0, label=r'$\kappa_P$ (exact)')
c.plot(B, [dyn[f'19_{x:g}'][0] for x in B], color=K, lw=1.0, ls='--', label=r'$\kappa_\tau$ (exact)')
pts(c, B, R[:, 3], c=BL, m='o', label=r'$\pi_+$ co-rotating')
pts(c, B, R[:, 4], c=VE, m='s', label=r'$\pi_-$ counter-rotating')
c.set_xlabel(r'$B\ (D^2/J)$'); c.set_ylabel(r'momentum $\pi$'); c.set_xlim(0.27, 0.88); c.set_ylim(0, 0.95)
c.legend(loc='upper right', fontsize=5.8, ncol=2, columnspacing=0.8)
label(a, 'a', x=-0.33, y=1.0); label(b, 'b', x=-0.33, y=1.0); label(c, 'c', x=-0.09, y=1.04)
check(fig, 'fig7_lambda')
print('lattices:', keys)
