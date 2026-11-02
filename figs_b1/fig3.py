"""Fig. 3: a pattern has no precessional Walker breakdown; its static branch ends in a Landau instability before it unwinds.
Analytic curves; the lattice point in (d) is read from results/b1_figs/fig4_helix.json."""
import json, os, numpy as np, matplotlib.pyplot as plt
from b1style import *

fig, ax = plt.subplots(2, 2, figsize=(W1, 2.75), gridspec_kw=dict(hspace=0.62, wspace=0.5))
(a, b), (c, d) = ax

# (a) domain wall: restoring force of the tilt angle is bounded
phi = np.linspace(0, np.pi, 300)
a.plot(phi/np.pi, np.sin(2*phi), color=K)
for lev, col, ls in ((0.6, BL, '--'), (1.3, VE, ':')):
    a.axhline(lev, color=col, ls=ls, lw=1)
p0 = 0.5*np.arcsin(0.6)
pts(a, [p0/np.pi], [0.6], c=BL)
a.text(0.97, 1.38, 'no root: precession', color=VE, ha='right', va='bottom', fontsize=6.5, transform=a.get_yaxis_transform())
a.set_xlim(0, 1); a.set_ylim(-1.15, 1.85); a.set_xticks([0, 0.25, 0.5, 0.75, 1]); a.set_xticklabels(['0', '', r'$\pi/2$', '', r'$\pi$'])
a.set_xlabel(r'wall tilt $\varphi$'); a.set_ylabel(r'force / $F_{\max}$')
a.set_title('wall: bounded', pad=3)

# (b) pattern: restoring force of the cone magnetization grows linearly up to |M| = 1
M = np.linspace(0, 1, 100)
b.plot(M, M, color=K)
for lev, col, ls in ((0.6, BL, '--'), (0.95, VE, ':')):
    b.axhline(lev, color=col, ls=ls, lw=1); pts(b, [lev], [lev], c=col)
b.set_xlim(0, 1.05); b.set_ylim(0, 1.15)
b.set_xlabel(r'cone magnetization $M$'); b.set_ylabel(r'force / $2\Delta$')
b.set_title('pattern: unbounded', pad=3)
b.set_xticks([0, 0.5, 1]); b.set_yticks([0.5, 1])

# (c) helix: phonon velocity on the driven cone meets the Doppler line before the cone closes
x = np.linspace(0, 1, 300)
c.plot(x, np.sqrt(1 - x**2), color=K)
c.plot(x, x, color=VE, ls='--')
c.text(0.83, 0.68, r'$c(M)$', fontsize=6.5)
c.text(0.36, 0.2, r'$\lambda u_{\rm eff}$', color=VE, fontsize=6.5)
xs = 1/np.sqrt(2)
c.axvline(xs, color='0.6', lw=0.6, zorder=0); c.axvline(1, color='0.6', lw=0.6, ls=':', zorder=0)
c.text(xs, 1.14, r'$u_L$', ha='center', va='bottom', fontsize=7); c.text(1.02, 1.14, r'$u_{\rm unw}$', ha='right', va='bottom', fontsize=7)
c.set_xlim(0, 1.08); c.set_ylim(0, 1.12); c.set_xticks([0, 0.5, 1]); c.set_yticks([0.5, 1]); c.set_xlabel(r'$s\,u_{\rm eff}/D$'); c.set_ylabel(r'velocity / $(D/s)$')

# (d) with easy-axis anisotropy: u_L < u_unw for every K
q = np.linspace(-0.8, 3, 300); r = np.sqrt(1 + q)
d.plot(q, r**2/np.sqrt(r**2 + 1), color=K, label=r'$u_L$')
d.plot(q, r**2, color=VE, ls='--', label=r'$u_{\rm unw}$')
h = json.load(open(os.path.join(DATA, 'fig4_helix.json')))['finite'][-1]      # longest chain, N = 800
pts(d, [0], [h['ueff_star']/h['D']], c=BL, label='lattice, $N=800$')
d.set_xlabel(r'$2AK/D^2$'); d.set_ylabel(r'threshold / $(D/s)$'); d.set_xlim(-0.8, 3); d.set_ylim(0, 6.2)
d.legend(loc='upper left', handlelength=1.4, bbox_to_anchor=(0.0, 1.0))
for s_, a_ in zip('abcd', (a, b, c, d)): label(a_, s_, x=-0.3, y=1.02)
check(fig, 'fig3_walker')
