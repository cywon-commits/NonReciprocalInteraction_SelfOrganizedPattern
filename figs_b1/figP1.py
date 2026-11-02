"""Phenomenon figure P1: the three driven chiral patterns of the paper and the first-order effect of a current on each.
(a) Bloch helix (chain along x, exact driven cone of code/helix): unbiased and at u_eff = 0.9 u_L, where the spins tilt into a cone
along q. (b) Neel skyrmion lattice (states/S_B0.5.npy, 2x2 cells): current u, Hall velocity v and the slot vector w1 from the
Thiele projection. (c) Diagonal Neel stripes (P = 30, converged co-moving state of code/stripe/conv.py) with a current along them."""
import sys, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from b1style import *
sys.path.insert(0, os.path.join(ROOT, 'code', 'helix')); sys.path.insert(0, os.path.join(ROOT, 'code', 'skx')); sys.path.insert(0, os.path.join(ROOT, 'code', 'stripe'))
from helix import predict, cone
from skx import GD
fig = plt.figure(figsize=(W1, 3.3))
a1 = fig.add_axes([0.0, 0.64, 1.0, 0.34])
b = fig.add_axes([0.02, 0.0, 0.46, 0.58]); c = fig.add_axes([0.53, 0.0, 0.46, 0.58])

# (a) helix and driven cone (N = 40 shown of an exact N = 200, n = 3 cone)
J = 1.0; N, n = 200, 3; th = 2*np.pi*n/N; D = J*np.tan(th); uL = D/np.sqrt(2)
Lh = 1.15
def proj(S):            # oblique view: chain axis x to the right, y into the page, z up
    return Lh*(S[:, 0] + 0.42*S[:, 1]), Lh*(S[:, 2] + 0.32*S[:, 1])
for row, (ue, col) in enumerate(((0.0, '0.45'), (0.9*uL, BL))):
    u = ue/(0.3/0.1 - 1); M, _ = predict(J, D, th, u, 0.1, 0.3); S, _ = cone(N, n, M)
    S = S[::2][:34]; xs = np.arange(len(S))*0.95; y0 = -2.9*row; px, py = proj(S)
    a1.plot([xs[0] - 0.4, xs[-1] + 0.4], [y0, y0], color='0.8', lw=0.5, zorder=0)
    for x0, dx_, dy_, sz in zip(xs, px, py, S[:, 1]):
        a1.add_patch(FancyArrowPatch((x0, y0), (x0 + dx_, y0 + dy_), arrowstyle='-|>', mutation_scale=4.5, lw=0.8, color=col, alpha=0.55 + 0.45*(sz < 0), shrinkA=0, shrinkB=0))
    a1.text(-0.8, y0, ['$u=0$', r'$u_{\rm eff}=0.9\,u_L$'][row], fontsize=6.5, color=col, ha='right', va='center')
    if row == 1: a1.text(xs[-1], y0 - 1.0, rf'cone $M={M:.2f}$ along $\mathbf{{q}}$', fontsize=6.5, color=col, ha='right', va='top')
a1.add_patch(FancyArrowPatch((xs[-1] + 1.4, 0.9), (xs[-1] + 4.6, 0.9), arrowstyle='-|>', mutation_scale=7, lw=1.2, color=K))
a1.text(xs[-1] + 4.8, 0.9, r'$\mathbf{u}\parallel\mathbf{q}$', fontsize=6.5, va='center')
a1.set_xlim(-7.2, xs[-1] + 7.0); a1.set_ylim(-4.4, 1.6); a1.set_aspect('equal'); a1.axis('off')

# (b) skyrmion lattice with u, v, w1
S0 = np.load(os.path.join(ROOT, 'states', 'S_B0.5.npy')); S = np.tile(S0, (3, 2, 1))[5:55, 8:58]; Lx, Ly = S.shape[:2]
b.imshow(S[..., 2].T, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=(-0.5, Lx - 0.5, -0.5, Ly - 0.5))
X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing='ij'); sk = (X % 3 == 1) & (Y % 3 == 1)
b.quiver(X[sk], Y[sk], S[..., 0][sk], S[..., 1][sk], color='k', scale=22, width=0.006, headwidth=3.5, headlength=3.5, headaxislength=3)
Gm, Dm, _ = GD(S0); al, be = 0.3, 0.6; uv = np.array([1.0, 0]); V = np.linalg.solve(Gm - al*Dm, (Gm - be*Dm) @ uv); w1 = V - uv
o = np.array([2.0, Ly + 3.0]); sc = 10
for vec, col, lab, off in ((uv, K, r'$\mathbf{u}$', (0.6, -0.8)), (V, '0.45', r'$\mathbf{v}$', (0.6, 0.6)), (w1/np.linalg.norm(w1)*0.6, VE, r'$\mathbf{w}_1$', (-2.8, 0.8))):
    b.add_patch(FancyArrowPatch(tuple(o), tuple(o + sc*vec), arrowstyle='-|>', mutation_scale=7, lw=1.2, color=col, clip_on=False))
    b.text(*(o + sc*vec + np.array(off)), lab, color=col, fontsize=7, va='center')
b.set_xlim(-0.5, Lx - 0.5); b.set_ylim(-0.5, Ly + 11); b.set_xticks([]); b.set_yticks([]); b.set_aspect('equal')
for sp in b.spines.values(): sp.set_visible(False)
b.plot([-0.5, Lx - 0.5, Lx - 0.5, -0.5, -0.5], [-0.5, -0.5, Ly - 0.5, Ly - 0.5, -0.5], color='k', lw=0.6)

# (c) diagonal stripes, current along the stripes
from conv import make, cycloid, steady
P = 30; m, th_s, Ds = make(P); S1, r1 = steady(m, cycloid(P, th_s), np.zeros(2))
L = 50; I_, J_ = np.meshgrid(np.arange(L), np.arange(L), indexing='ij'); St = S1[(I_ + J_) % P]
c.imshow(St[..., 2].T, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=(-0.5, L - 0.5, -0.5, L - 0.5))
sk = (I_ % 3 == 0) & (J_ % 3 == 0)
c.quiver(I_[sk], J_[sk], St[..., 0][sk], St[..., 1][sk], color='k', scale=30, width=0.006, headwidth=3.5, headlength=3.5, headaxislength=3)
e = np.array([1, -1])/np.sqrt(2); o = np.array([3.0, L + 7.5])
c.add_patch(FancyArrowPatch(tuple(o), tuple(o + 9*e), arrowstyle='-|>', mutation_scale=7, lw=1.2, color=K, clip_on=False))
c.text(o[0] + 9*e[0] + 1.0, o[1] + 9*e[1], r'$\mathbf{u}$ along the stripes', fontsize=6.5, va='center')
c.set_xlim(-0.5, L - 0.5); c.set_ylim(-0.5, L + 9); c.set_xticks([]); c.set_yticks([]); c.set_aspect('equal')
for sp in c.spines.values(): sp.set_visible(False)
c.plot([-0.5, L - 0.5, L - 0.5, -0.5, -0.5], [-0.5, -0.5, L - 0.5, L - 0.5, -0.5], color='k', lw=0.6)
fig.text(0.01, 0.98, '(a)', fontsize=9, fontweight='bold', va='top'); fig.text(0.01, 0.63, '(b)', fontsize=9, fontweight='bold', va='top'); fig.text(0.52, 0.63, '(c)', fontsize=9, fontweight='bold', va='top')
check(fig, 'figP1_patterns')
