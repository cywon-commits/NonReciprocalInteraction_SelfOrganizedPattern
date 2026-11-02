"""Fig. 1: a current is a fixed nonreciprocal bond bias; in the co-moving frame it enters only as w1 (T-odd) and w2 (T-even),
and the Landau-Lifshitz drive W2 = (beta - alpha) u points along u whatever the Hall angle.
Panel (b) uses the Thiele tensors of the B = 0.5 D^2/J lattice from results/b1_todo/dD_ana.log."""
import os, re, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from b1style import *

fig = plt.figure(figsize=(W1, 1.85))
a = fig.add_axes([0.0, 0.0, 0.56, 0.86]); b = fig.add_axes([0.62, 0.0, 0.38, 0.86])

def arrow(ax, p, q, col=K, lw=1.0, ms=6, ls='-', **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, color=col, lw=lw, ls=ls, shrinkA=0, shrinkB=0, **kw))

# (a) cycloid with one highlighted bond and the two nonreciprocal couplings
a.set_xlim(-0.4, 6.4); a.set_ylim(-2.2, 1.75); a.set_aspect('equal'); a.axis('off')
xs = np.arange(0, 6.01, 1.0); q = 2*np.pi/8; L_ = 0.42
for i, x in enumerate(xs):
    th = q*i + 0.35; col = BL if i in (2, 3) else '0.6'
    arrow(a, (x - L_*np.sin(th), -L_*np.cos(th)), (x + L_*np.sin(th), L_*np.cos(th)), col=col, lw=1.1, ms=6)
a.text(2, -0.5, r'$\mathbf{S}_i$', ha='center', va='top', color=BL); a.text(3, -0.5, r'$\mathbf{S}_j$', ha='center', va='top', color=BL)
arrow(a, (2.15, 0.6), (2.85, 0.6), col=VE, lw=1.0, ms=6, connectionstyle='arc3,rad=-0.5')
arrow(a, (2.85, -1.12), (2.15, -1.12), col=GR, lw=1.0, ms=6)
a.text(2.15, 0.98, r'$\kappa_r$: receiver-dependent', color=VE, ha='center', va='bottom', fontsize=6.5)
a.text(2.5, -1.3, r'$\kappa_f$: field coupling', color=GR, ha='center', va='top', fontsize=6.5)
arrow(a, (4.7, 1.0), (6.1, 1.0), col=K, lw=1.4, ms=8)
a.text(5.4, 1.2, r'fixed bias $\hat{\mathbf{j}}$', ha='center', va='bottom', fontsize=6.5)
a.text(3.0, -2.15, r'$u=-2a\kappa_r,\;\; \beta u=-2a\kappa_f$', ha='center', va='bottom', fontsize=7)

# (b) slot vectors of a driven skyrmion lattice from the exact Thiele projection
line = open(os.path.join(ROOT, 'results', 'b1_todo', 'dD_ana.log')).readline()
G = float(re.search(r'G=(-?[\d.]+)', line).group(1)); Dxx = float(re.search(r'Dxx=([\d.]+)', line).group(1)); Dyy = float(re.search(r'Dyy=([\d.]+)', line).group(1))
al, be = 0.3, 0.6; Gm = np.array([[0, G], [-G, 0]]); Dm = np.diag([Dxx, Dyy]); u = np.array([1.0, 0])
v = np.linalg.solve(Gm - al*Dm, (Gm - be*Dm) @ u); w1 = v - u; w2 = be*u - al*v; W2 = w2 + al*w1
b.set_xlim(-0.45, 1.45); b.set_ylim(-0.5, 1.2); b.set_aspect('equal'); b.axis('off')
for vec, col, lab, off in ((u, K, r'$\mathbf{u}$', (0.03, 0.0)), (v, '0.45', r'$\mathbf{v}$', (0.03, 0.05)),
                           (w1, VE, r'$\mathbf{w}_1$', (-0.2, 0.0)), (w2, BL, r'$\mathbf{w}_2$', (0.05, -0.08)),
                           (W2, GR, r'$\mathbf{W}_2=(\beta-\alpha)\mathbf{u}$', (-0.42, -0.45))):
    arrow(b, (0, 0), tuple(vec), col=col, lw=1.3, ms=7)
    b.text(vec[0] + off[0], vec[1] + off[1], lab, color=col, fontsize=7, ha='left', va='bottom')
b.plot([0], [0], 'o', ms=2.5, color=K)
b.text(-0.42, 1.18, r'$\alpha=0.3,\ \beta=0.6$' + '\nSkX, $B=0.5D^2/J$', fontsize=6.5, va='top')
label(a, 'a', x=0.06, y=0.93); label(b, 'b', x=0.02, y=0.98)
check(fig, 'fig1_concept')
print('w1.w2 =', w1 @ w2, ' v =', v, ' W2 =', W2)
