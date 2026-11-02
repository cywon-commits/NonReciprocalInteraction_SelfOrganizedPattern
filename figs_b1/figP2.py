"""Phenomenon figure P2: what the Doppler instability does in real space.
(a) Skyrmion lattice above threshold (u_eff = 0.15 = 1.5 u*(2pi/76), 76x66 box, co-moving frame): snapshots; arrows in the
middle panel are the skyrmion displacements since t = 0 (x 4), a compression wave along the current.
(b) Satellite amplitude |m_z(k = 2pi/76)| against time: dynamics (points) and the Bloch growth rate of the same mode (line);
below threshold (u_eff = 0.05) the noise decays. (c, d) Stripes (P = 30) driven along themselves at 1.5 u* and 3 u*
(L = 256): bending wave, saturated zigzag, and at 3 u* dislocations and partial reorientation.
Data: results/b1_figs/phen_skx_u*.npz, phen_skx_mode.json, phen_stripe_f*.npz (figs_b1/phen/*.py)."""
import os, json, numpy as np, matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.patches import FancyArrowPatch
from b1style import *
d = np.load(os.path.join(DATA, 'phen_skx_u0.15.npz')); t = d['t']; mz = d['mz'].astype(float)
d0 = np.load(os.path.join(DATA, 'phen_skx_u0.05.npz')); mode = json.load(open(os.path.join(DATA, 'phen_skx_mode.json')))
def centres(m):
    Lx, Ly = m.shape; big = np.tile(m, (2, 2)); lab, n = ndimage.label(big < 0); cs = ndimage.center_of_mass(big < 0, lab, range(1, n + 1))
    return np.array([c for c in cs if c[0] < Lx and c[1] < Ly])
def match(c0, c1, L):
    out = []
    for p in c0:
        dd = (c1 - p + np.array(L)/2) % np.array(L) - np.array(L)/2; j = np.argmin((dd**2).sum(1)); out.append(dd[j])
    out = np.array(out); return out - out.mean(0)
fig = plt.figure(figsize=(W1, 3.95))
ax_s = [fig.add_axes([0.0 + 0.335*i, 0.67, 0.32, 0.29]) for i in range(3)]
bb = fig.add_axes([0.15, 0.475, 0.83, 0.135])
cs_ = [fig.add_axes([0.03 + 0.195*i, 0.0, 0.185, 0.34]) for i in range(5)]

# (a) skyrmion snapshots
pick = [0, int(np.argmin(abs(t - 16000))), int(np.argmin(abs(t - 18500)))]
Lx, Ly = mz[0].shape; c0 = centres(mz[0])
for k, (ax, i) in enumerate(zip(ax_s, pick)):
    ax.imshow(mz[i].T, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=(-0.5, Lx - 0.5, -0.5, Ly - 0.5), interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'$t={t[i]:.0f}$', fontsize=7, pad=2, loc='right' if k == 0 else 'center')
    if k == 1:
        dv = match(c0, centres(mz[i]), (Lx, Ly))
        for p, v in zip(c0, dv):
            ax.add_patch(FancyArrowPatch(tuple(p), tuple(p + 4*v), arrowstyle='-|>', mutation_scale=5, lw=0.9, color=K))
ax_s[0].add_patch(FancyArrowPatch((4, Ly + 4), (22, Ly + 4), arrowstyle='-|>', mutation_scale=6, lw=1.0, color=K, clip_on=False))
ax_s[0].text(24, Ly + 4, r'$\mathbf{u}$', fontsize=7, va='center')

# (b) growth of the k = 2 pi / 76 satellite
pts(bb, t[::2], d['sat'][::2], c=VE, m='o', ms=3, label=r'$u_{\rm eff}=0.15$')
pts(bb, d0['t'], d0['sat'], c=BL, m='s', ms=3, label=r'$u_{\rm eff}=0.05$')
g = mode['modes'][0]['growth']; tt = np.linspace(2000, 15000, 10); i0 = int(np.argmin(abs(t - 8000)))
bb.plot(tt, d['sat'][i0]*np.exp(g*(tt - t[i0])), color=K, lw=1.0, label=rf'Bloch, $\Gamma={g*1e4:.2f}\times10^{{-4}}$')
bb.set_yscale('log'); bb.set_xlabel('$t$'); bb.set_ylabel(r'$|m_z(k_1)|$'); bb.set_xlim(0, 20000); bb.set_ylim(1e-19, 1e6)
bb.set_yticks([1e-15, 1e-8, 1e-1]); bb.legend(loc='upper left', fontsize=5.0, ncol=3, columnspacing=0.8, handletextpad=0.3)

# (c, d) stripes in the (across, along) frame of the sheared box
def real(m):           # column n (across) is shifted by n/2 along the stripes to undo the shear
    Nn, Nm = m.shape; out = np.empty_like(m)
    for n in range(Nn): out[n] = np.roll(m[n], -(n//2))
    return out
for j, (fac, times) in enumerate(((1.5, (0, 8000, 16000)), (3.0, (1500, 4000)))):
    f = np.load(os.path.join(DATA, f'phen_stripe_f{fac}.npz')); ts = f['t']; ms = f['mz'].astype(float)
    for k, tt_ in enumerate(times):
        ax = cs_[k if j == 0 else 3 + k]; i = int(np.argmin(abs(ts - tt_))); im = real(ms[i])[:, :90]
        ax.imshow(im.T, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=(0, im.shape[0]/np.sqrt(2), 0, im.shape[1]*np.sqrt(2)), interpolation='nearest', aspect='equal')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'{fac:g}$u^*$, {ts[i]:.0f}', fontsize=5.6, pad=2)
fig.add_artist(FancyArrowPatch((0.012, 0.08), (0.012, 0.2), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=6, lw=1.0, color=K))
fig.text(0.016, 0.22, r'$\mathbf{u}$', fontsize=7, ha='center', va='bottom')
fig.text(0.0, 0.995, '(a)', fontsize=9, fontweight='bold', va='top'); fig.text(0.0, 0.64, '(b)', fontsize=9, fontweight='bold', va='top')
fig.text(0.0, 0.38, '(c)', fontsize=9, fontweight='bold', va='top'); fig.text(0.61, 0.38, '(d)', fontsize=9, fontweight='bold', va='top')
check(fig, 'figP2_breakdown')
