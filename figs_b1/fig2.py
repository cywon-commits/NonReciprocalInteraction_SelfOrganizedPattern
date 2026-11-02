"""Fig. 2: the residual magnetic group of the driven pattern fixes its first-order response, and the relevant vector is w1, not u.
(a) group-subgroup steps with BNS labels (results/b1_residual_group/residual_bns.json); (b) translation-free change of m_z of a
driven Neel skyrmion (w1 along x); (c) the distortion axis stays perpendicular to w1 for every damping and current direction,
while its angle to u follows the Hall rotation. Data: results/b1_figs/fig2_distortion.json, fig2_map.npy (data_fig2.py)."""
import json, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from b1style import *
G = {r['case']: r['bns'] for r in json.load(open(os.path.join(ROOT, 'results', 'b1_residual_group', 'residual_bns.json')))}
def bns(key): return next(v for k, v in G.items() if k.startswith(key))
fig = plt.figure(figsize=(W1, 3.0))
a = fig.add_axes([0.0, 0.62, 1.0, 0.38]); b = fig.add_axes([0.03, 0.0, 0.4, 0.55]); c = fig.add_axes([0.62, 0.09, 0.36, 0.46])

# (a) trees
a.set_xlim(0, 10); a.set_ylim(0, 4); a.axis('off')
def box(x, y, s, col=K, fs=6.3):
    a.text(x, y, s, ha='center', va='center', fontsize=fs, color=col, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=col, lw=0.6))
def arr(p, q, s=None, side=1):
    a.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=6, lw=0.7, color='0.35', shrinkA=7, shrinkB=7))
    if s: a.text((p[0] + q[0])/2 + 0.12*side, (p[1] + q[1])/2 + 0.05, s, fontsize=5.6, ha='left' if side > 0 else 'right', va='center', color='0.25')
box(2.4, 3.55, f"Néel cycloid  $P_ama2$ ({bns('Neel cycloid q||x')})")
box(1.1, 1.75, f"$Pm'a2'$ ({bns('Neel cycloid, w1 || q')})", col=BL); box(3.75, 1.75, f"$Pm'm2'$ ({bns('Neel cycloid, w1 ⊥ q')})", col=VE)
arr((2.0, 3.35), (1.2, 1.97), r'$\mathbf{u}\parallel\mathbf{q}$', side=-1); arr((2.8, 3.35), (3.65, 1.97), r'$\mathbf{u}\perp\mathbf{q}$', side=1)
a.text(1.1, 1.25, r'cone $M\parallel\hat z\times\mathbf{u}$', ha='center', va='top', fontsize=5.8, color=BL)
a.text(3.75, 1.25, 'exact cycloid, no cone', ha='center', va='top', fontsize=5.8, color=VE)
a.text(2.4, 0.55, r'$M_z$ forbidden by $T\,t_{\lambda/2}$', ha='center', va='center', fontsize=5.6, color='0.3')
box(7.6, 3.55, f"Néel SkX  $P6m'm'$ ({bns('Neel SkX, unbiased')})")
box(6.4, 1.75, f"$Cm'$ ({bns('Neel SkX, w1||x')})", col=BL); box(8.85, 1.75, '$P1$', col='0.4')
arr((7.2, 3.35), (6.5, 1.97), r"$\mathbf{w}_1\perp m'$", side=-1); arr((8.0, 3.35), (8.75, 1.97), r'generic $\mathbf{w}_1$', side=1)
a.text(7.6, 1.0, r'$\mathbf{d}\perp\mathbf{w}_1$ in both', ha='center', va='center', fontsize=5.8, color=BL)
a.text(7.6, 0.45, r"(linear response of $6m'm'$)", ha='center', va='center', fontsize=5.6, color='0.3')
a.plot([5.05, 5.05], [0.3, 3.9], color='0.8', lw=0.6)

# (b) map of the translation-free dm_z, centred on the on-site skyrmion
dz = np.load(os.path.join(DATA, 'fig2_map.npy')); dj = json.load(open(os.path.join(DATA, 'fig2_distortion.json')))
Lx, Ly = dz.shape; R = 7
sub = np.roll(np.roll(dz, R, 0), R, 1)[:2*R + 1, :2*R + 1]
vm = np.abs(sub).max()
im = b.imshow(sub.T, origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm, extent=(-R - 0.5, R + 0.5, -R - 0.5, R + 0.5))
b.set_xticks([]); b.set_yticks([]); b.set_aspect('equal')
mp = dj['map']
for vec, col, lab, sc in ((mp['w1'], K, r'$\mathbf{w}_1$', 600), (mp['u'], '0.4', r'$\mathbf{u}$', 220)):
    v = np.array(vec)*sc; o = np.array([-5.6, -5.4])
    b.add_patch(FancyArrowPatch(tuple(o), tuple(o + v), arrowstyle='-|>', mutation_scale=6, lw=1.0, color=col))
    b.text(*(o + v + np.array([0.3, 0.0])), lab, fontsize=6.5, color=col, va='center')
b.text(0, R + 0.9, r'$\delta m_z$, translation removed', ha='center', va='bottom', fontsize=6)
S0 = np.load(os.path.join(ROOT, 'states', 'S_B0.5.npy'))[..., 2]; s0 = np.roll(np.roll(S0, R, 0), R, 1)[:2*R + 1, :2*R + 1]
gg = np.arange(-R, R + 1); b.contour(gg, gg, s0.T, levels=[0], colors='k', linewidths=0.6)
r0 = [r for r in dj['rows'] if r['alpha'] == 0.3 and r['theta_u'] == 0][0]; dv = np.array(r0['d']); dv = dv/np.linalg.norm(dv)*4.0
b.add_patch(FancyArrowPatch(tuple(-dv), tuple(dv), arrowstyle='-|>', mutation_scale=7, lw=1.1, color=GR))
b.text(dv[0] + 0.3, dv[1], r'$\mathbf{d}$', color=GR, fontsize=7, va='center')

# (c) distortion axis relative to w1 and to u
for j, (th, mk) in enumerate(((0, 'o'), (30, '^'), (90, 's'))):
    rs = [r for r in dj['rows'] if r['theta_u'] == th]
    al = np.array([r['alpha'] for r in rs]); dw = np.array([(r['ang_d'] - r['ang_w1']) % 180 for r in rs])
    du = np.array([(r['ang_d'] - th) % 180 for r in rs])
    c.plot(al + (j - 1)*0.012, dw, mk, color=BL, ms=3.8, mec='white', mew=0.4, ls='none', label=r'to $\mathbf{w}_1$' if j == 0 else None)
    c.plot(al + (j - 1)*0.012, du, mk, color=VE, ms=3.8, mfc='none', mew=0.8, ls='none', label=r'to $\mathbf{u}$' if j == 0 else None)
c.axhline(90, color=BL, lw=0.8, ls='--')
c.set_xlabel(r'$\alpha$ ($\beta=2\alpha$)'); c.set_ylabel(r'angle of $\mathbf{d}$ (deg)'); c.set_xlim(0.0, 0.7); c.set_xticks([0.2, 0.4, 0.6]); c.set_ylim(0, 180); c.set_yticks([0, 90, 180])
c.legend(loc='lower left', fontsize=5.6)
label(a, 'a', x=0.04, y=0.92); label(b, 'b', x=0.05, y=1.08); label(c, 'c', x=-0.3, y=1.02)
check(fig, 'fig2_symmetry')
