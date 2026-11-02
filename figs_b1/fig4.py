"""Fig. 4: the driven helix breaks at u_L = D/(sqrt2 s) for every (alpha, beta); the threshold depends on beta only through u_eff.
Data: results/b1_figs/fig4_helix.json (data_fig4.py)."""
import json, os, numpy as np, matplotlib.pyplot as plt
from b1style import *
d = json.load(open(os.path.join(DATA, 'fig4_helix.json')))
h = d['N200']; D = h['D']; kmin = 2*np.pi/200; upred = np.sqrt((D*D + kmin**2)/2)
fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 1.85), gridspec_kw=dict(wspace=0.55))
ins = a.inset_axes([0.21, 0.72, 0.38, 0.2])
for i, s in enumerate(h['sets']):
    ue = np.array(s['ueff']); g = np.array(s['growth'])/s['alpha']; lab = f"({s['alpha']}, {s['beta']})"
    a.plot(ue/upred, g*1e3, MK[i], color=SEQ[i], ms=5.5 - 0.8*i, mfc='none', mew=0.8, ls='none', label=lab)
    ins.plot([s['u_star']/upred], [1], MK[i], color=SEQ[i], ms=4 - 0.5*i, mfc='none', mew=0.7); ins.plot([s['ueff_star']/upred], [0], MK[i], color=SEQ[i], ms=4 - 0.5*i, mfc='none', mew=0.7)
a.axhline(0, color='0.5', lw=0.6); a.axvline(1, color=K, lw=0.9, zorder=0)
a.set_xlabel(r'$u_{\rm eff}/u_L$'); a.set_ylabel(r'$\Gamma_{\max}/\alpha$ ($10^{-3}$)')
a.set_xlim(0.72, 1.17); a.set_ylim(-0.15, 2.3)
ins.set_ylim(-0.6, 1.6); ins.set_yticks([0, 1]); ins.set_yticklabels([r'$u_{\rm eff}^*$', r'$u^*$']); ins.set_title(r'threshold / $u_L$', fontsize=5.5, pad=1.5)
ins.tick_params(labelsize=5, length=1.5, pad=1); ins.set_xticks([-1, 0, 1]); ins.set_xlim(-1.3, 1.3); ins.grid(False)

a.legend(loc='lower left', bbox_to_anchor=(0.0, 0.12), fontsize=4.8, handletextpad=0.1, labelspacing=0.1, borderaxespad=0.1)
# (b) finite chains: k_min correction of the Landau criterion
f = d['finite']; N = np.array([r['N'] for r in f]); Dn = f[0]['D']
Nc = np.geomspace(150, 3000, 100)
b.plot(1/Nc*1e3, np.sqrt((Dn**2 + (2*np.pi/Nc)**2)/2)/(Dn/np.sqrt(2)), color=K, label=r'$\sqrt{(D^2+k_{\min}^2)/2}$')
pts(b, 1/N*1e3, np.array([r['ueff_star'] for r in f])/(Dn/np.sqrt(2)), c=BL, label='exact Bloch')
b.axhline(1, color='0.5', lw=0.6, ls='--')
b.set_xlabel(r'$1/N$ ($10^{-3}$)'); b.set_ylabel(r'$u_{\rm eff}^*/(D/\sqrt{2}s)$'); b.set_xlim(0, 6.2); b.set_ylim(0.99, 1.095)
b.legend(loc='upper left', fontsize=5.5)
label(a, 'a', x=-0.28, y=1.0); label(b, 'b', x=-0.4, y=1.0)
check(fig, 'fig4_helix')
print('raw u* spread:', [round(s['u_star']/upred, 3) for s in h['sets']], ' u_eff* spread:', [round(s['ueff_star']/upred, 4) for s in h['sets']])
