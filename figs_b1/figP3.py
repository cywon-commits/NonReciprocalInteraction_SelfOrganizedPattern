"""Phenomenon figure P3: what lambda is in real space. A longitudinal strain wave (wavelength 38 = two cells, displacement along x of
amplitude A = 2 sites, exaggerated) imposed on the skyrmion lattice: affine (skyrmions deform with the lattice) vs statically
relaxed (same skyrmion displacement, internal canting relaxed), and their difference, the counterflow. Bottom: Doppler-momentum
density of the phonon (its own elliptic polarization) averaged across the strip. Data: results/b1_figs/phen_relax_B*.npz (figs_b1/phen/relax_maps.py)."""
import os, numpy as np, matplotlib.pyplot as plt
from b1style import *
A = 2.0
fig = plt.figure(figsize=(W1, 3.55))
rows = ((0.3, 0.773), (0.85, 0.107))
for r, (B, lam) in enumerate(rows):
    d = np.load(os.path.join(DATA, f'phen_relax_B{B}.npz')); S0 = d['S']; Lx, Ly = S0.shape[:2]
    y0 = 0.7 - 0.37*r
    panels = []
    for c, nm in enumerate(('affine', 'relaxed', 'diff')):
        ax = fig.add_axes([0.07 + 0.31*c, y0, 0.29, 0.29*Lx/Ly*0 + 0.27]); panels.append(ax)
        if nm == 'diff':
            dd = np.real(d['relaxed'] - d['affine'])*A
            vm = np.abs(dd[..., 2]).max()
            ax.imshow(dd[..., 2].T, origin='lower', cmap='PuOr_r', vmin=-vm, vmax=vm, extent=(-0.5, Lx - 0.5, -0.5, Ly - 0.5), interpolation='nearest')
            X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing='ij'); sk = (X % 3 == 1) & (Y % 3 == 1)
            ax.quiver(X[sk], Y[sk], dd[..., 0][sk], dd[..., 1][sk], color='k', scale=8, width=0.007, headwidth=3.5, headlength=3.5, headaxislength=3)
        else:
            Sd = S0 + A*np.real(d[nm]); Sd /= np.linalg.norm(Sd, axis=-1)[..., None]
            ax.imshow(Sd[..., 2].T, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=(-0.5, Lx - 0.5, -0.5, Ly - 0.5), interpolation='nearest')
            ax.contour(np.arange(Lx), np.arange(Ly), S0[..., 2].T, levels=[0], colors='0.25', linewidths=0.4, linestyles=':')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
        if r == 0: ax.set_title(['affine', 'relaxed', 'relaxed − affine'][c], fontsize=7, pad=2)
    fig.text(0.035, y0 + 0.135, f'$B={B}$\n$\\lambda={lam:.2f}$', fontsize=6.5, ha='center', va='center', rotation=90)
# bottom: momentum density averaged across the strip
ax = fig.add_axes([0.13, 0.06, 0.84, 0.17])
for r, (B, lam) in enumerate(rows):
    d = np.load(os.path.join(DATA, f'phen_relax_B{B}.npz')); Lx = d['S'].shape[0]; kx = float(d['kx'])
    for nm, ls in (('affine', '--'), ('relaxed', '-')):
        p = d[nm + '_pol_p'].mean(1)/d[nm + '_pol_n'].mean()/kx
        ax.plot(np.arange(Lx), p, ls=ls, color=SEQ[r], lw=1.0, label=f'$B={B}$, {nm} ({p.mean():.2f})')
ax.set_xlim(0, Lx - 1); ax.set_ylim(-0.2, 2.4); ax.set_yticks([0, 1, 2]); ax.set_xlabel(r'$x$ (sites)'); ax.set_ylabel(r'$p(x)/k$', fontsize=7)
ax.legend(loc='upper right', fontsize=5.0, ncol=2, columnspacing=0.8)
fig.text(0.0, 0.995, '(a)', fontsize=9, fontweight='bold', va='top'); fig.text(0.0, 0.63, '(b)', fontsize=9, fontweight='bold', va='top'); fig.text(0.0, 0.27, '(c)', fontsize=9, fontweight='bold', va='top')
check(fig, 'figP3_counterflow')
