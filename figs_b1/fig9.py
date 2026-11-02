"""Fig. 9: exchange-even couplings pump the lattice phonons with zero threshold through the static relaxed mode, dual to the
Zhang-Li Doppler term. (a) damping-like SOT: x k is constant; (b) the pumping lives entirely on the internal relaxation
(affine part exactly zero); (c) second-order nonreciprocal stiffness of delta D: threshold delta D* proportional to k.
Data: results/b1_figs/sot_k.log (code/b1/b1_k.py), sot_static.log (code/b1/b1_static.py), results/b1_todo/dD_alpha.log."""
import os, re, numpy as np, matplotlib.pyplot as plt
from b1style import *
fig = plt.figure(figsize=(W1, 3.0))
gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.55)
a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, :])

# (a) pumping ratio x = split/Gamma0 against k for three directions (n = y, k || x is z x n)
rows = [re.search(r'k=([\d.]+) angle=\s*(\d+): Gamma0=([\d.e+-]+) omega=([\d.e+-]+)\s+half-split=([\d.e+-]+)\s+x=split/Gamma0=([\d.]+)', l)
        for l in open(os.path.join(DATA, 'sot_k.log'))]
S = np.array([[float(g) for g in m.groups()] for m in rows if m])
kk = np.geomspace(0.006, 0.045, 20)
for j, (ang, col, mk) in enumerate(((0, VE, 'o'), (45, GR, '^'), (90, BL, 's'))):
    s = S[S[:, 1] == ang]
    pts(a, s[:, 0], s[:, 5], c=col, m=mk, label=rf'$\theta_k={ang}^\circ$')
    a.plot(kk, np.mean(s[:, 5]*s[:, 0])/kk, color=col, lw=0.7, ls='--')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel(r'$k$'); a.set_ylabel(r'$x=\Delta\Gamma/\Gamma_0$')
a.set_xlim(0.006, 0.045); a.set_ylim(0.02, 3)
import matplotlib.ticker as mt
for ax_ in (a,):
    ax_.xaxis.set_minor_formatter(mt.NullFormatter()); ax_.set_xticks([0.01, 0.03]); ax_.xaxis.set_major_formatter(mt.FormatStrFormatter('%g'))
    ax_.yaxis.set_major_formatter(mt.FormatStrFormatter('%g'))
a.legend(loc='lower left', fontsize=5.0, handletextpad=0.1, borderaxespad=0.1)

# (b) exact vs static relaxed vs affine mode (first-order generator, alpha = 0)
R = []
for l in open(os.path.join(DATA, 'sot_static.log')):
    m = re.search(r'k=([\d.]+): .*exact \[np.float64\(([-\d.]+)\).*relaxed \[np.float64\(([-\d.]+)\).*affine \[np.float64\(([-\d.]+)\)', l)
    if m: R.append([float(g) for g in m.groups()])
R = np.array(R); kl = np.linspace(0, 0.036, 10); slope = np.sum(R[:, 0]*R[:, 1])/np.sum(R[:, 0]**2)
b.plot(kl, slope*kl, color='0.6', lw=0.7, zorder=0)
b.plot(R[:, 0], R[:, 1], 'o', color=VE, ms=6, mfc='none', mew=1.0, label='exact mode')
b.plot(R[:, 0], R[:, 2], 'x', color=K, ms=3.5, mew=0.9, label='static relaxed', zorder=5)
pts(b, R[:, 0], np.abs(R[:, 3]), c=BL, m='s', label='affine')
b.set_xlabel(r'$k$'); b.set_ylabel(r'$\mathrm{Re}\,\partial\lambda/\partial c$'); b.set_xlim(0, 0.036); b.set_ylim(-0.002, 0.033)
b.set_xticks([0, 0.01, 0.02, 0.03]); b.set_xticklabels(['0', '0.01', '0.02', '0.03'])
b.legend(loc='upper left', fontsize=5.0, handletextpad=0.1)

# (c) delta D second-order stiffness: threshold delta D* = Gamma0 / (half-split per delta D), alpha = 0.3
dd = []
for l in open(os.path.join(ROOT, 'results', 'b1_todo', 'dD_alpha.log')):
    m = re.search(r'k=([\d.]+)\|\|(x|y): .*half-split/k a=0: ([\d.]+) a=0.3: ([\d.]+)', l)
    if m: dd.append((float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))))
G0 = {(round(r[0], 4), int(r[1])): r[2] for r in S}
kl = np.geomspace(0.008, 0.07, 20); c.plot(kl, 2*kl, color=K, lw=0.9, label=r'$2k$')
for dirn, ang, col, mk, nm in (('x', 0, VE, 'o', r'$\mathbf{k}\perp\mathbf{n}$ (Doppler shift $\times\,\alpha$)'), ('y', 90, BL, 's', r'$\mathbf{k}\parallel\mathbf{n}$ (intrinsic)')):
    ks = np.array([r[0] for r in dd if r[1] == dirn]); sp = np.array([r[3] for r in dd if r[1] == dirn])*ks
    g0 = np.array([G0[(round(k, 4), ang)] for k in ks])
    pts(c, ks, g0/sp, c=col, m=mk, label=nm)
    if dirn == 'y': c.plot(kl, np.mean(g0/sp/ks)*kl, color=col, lw=0.7, ls='--')
c.set_xscale('log'); c.set_yscale('log'); c.set_xlabel(r'$k$'); c.set_ylabel(r'threshold $\delta D^*$')
c.set_xlim(0.008, 0.07); c.set_ylim(0.01, 0.3)
c.xaxis.set_minor_formatter(mt.NullFormatter()); c.set_xticks([0.01, 0.02, 0.05]); c.xaxis.set_major_formatter(mt.FormatStrFormatter('%g'))
c.yaxis.set_major_formatter(mt.FormatStrFormatter('%g')); c.yaxis.set_minor_formatter(mt.NullFormatter()); c.set_yticks([0.01, 0.03, 0.1, 0.3])
c.legend(loc='upper left', fontsize=5.8)
label(a, 'a', x=-0.36, y=1.0); label(b, 'b', x=-0.38, y=1.0); label(c, 'c', x=-0.1, y=1.0)
check(fig, 'fig9_sot')
print('x*k (theta=0):', np.round(S[S[:, 1] == 0][:, 5]*S[S[:, 1] == 0][:, 0], 5), ' dD*/k:', [(r[0], r[1]) for r in dd])
