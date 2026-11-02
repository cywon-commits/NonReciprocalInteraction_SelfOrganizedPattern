"""Fig. 5: the skyrmion-lattice threshold is u_eff* = c2 k / lambda with lambda = kappa_P (dissipative overlap), independent of the
Hall angle; damping enters only at order alpha^2.
Data: results/b1_figs/uk_B0.5_nk*.log (lam_alpha.py at k = 2pi/(19 nk)), results/b1_lambda_alpha/lam_alpha_B*_nk20*.json."""
import json, os, re, glob, numpy as np, matplotlib.pyplot as plt
from b1style import *
LA = os.path.join(ROOT, 'results', 'b1_lambda_alpha')
def load(B):
    out = {}
    for f in sorted(glob.glob(os.path.join(LA, f'lam_alpha_B{B}_nk20*.json'))):
        d = json.load(open(f)); kP = abs(d['KP'][0])/(d['P']*d['k']); kt = abs(d['Kt'][0])/d['k']
        for a_, r in d['res'].items():
            if 'uA' in r: out[float(a_)] = d['wn']/(d['k']*abs(r['uA']))
        base = dict(k=d['k'], wn=d['wn'], kP=kP, kt=kt, P=d['P'])
    return base, dict(sorted(out.items()))
Bs = [0.3, 0.5, 0.7, 0.85]; dat = {B: load(B) for B in Bs}

fig = plt.figure(figsize=(W1, 2.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.62, wspace=0.5)
a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, :])

# (a) threshold proportional to k (B = 0.5, alpha = 0.3, beta = 0.6)
ks, us, pred = [], [], []
for nk in (40, 20, 10, 5):
    if nk == 20:
        base, _ = dat[0.5]; d = json.load(open(os.path.join(LA, 'lam_alpha_B0.5_nk20_hi.json')))
        k, wn, kP, u = d['k'], d['wn'], abs(d['KP'][0])/(d['P']*d['k']), abs(d['res']['0.3']['uA'])
    else:
        L = open(os.path.join(DATA, f'uk_B0.5_nk{nk}.log')).read().splitlines()
        k = float(re.search(r'k=([\d.]+)', L[0]).group(1)); wnk = float(re.search(r'wn/k=([\d.]+)', L[0]).group(1)); wn = wnk*k
        kP = abs(float(re.search(r'kap_P=\(([-\d.]+)', L[0]).group(1))); u = abs(json.loads(L[1])['uA'])
    ks.append(k); us.append(u); pred.append(wn/(k*kP))
ks, us, pred = map(np.array, (ks, us, pred))
c2 = dat[0.5][0]['wn']/dat[0.5][0]['k']**2; lam = dat[0.5][0]['kP']
kk = np.geomspace(0.006, 0.09, 50)
a.plot(kk, c2*kk/lam, color=K, label=r'$c_2k/\kappa_P$')
a.plot(kk, c2*kk, color='0.55', ls='--', label='affine ($\\lambda=1$)')
pts(a, ks, us, c=BL, label='exact Bloch')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel(r'$k$'); a.set_ylabel(r'$u_{\rm eff}^*$')
a.set_xlim(0.006, 0.09); a.set_ylim(0.003, 0.4)
a.legend(loc='upper left', fontsize=5.2)
import matplotlib.ticker as mt
a.xaxis.set_minor_formatter(mt.NullFormatter()); a.set_xticks([0.01, 0.03]); a.set_xticklabels(['0.01', '0.03'])
top = a.secondary_xaxis('top', functions=(lambda k: 2*np.pi/np.maximum(k, 1e-9), lambda L: 2*np.pi/np.maximum(L, 1e-9)))
top.set_xticks([100, 300, 1000]); top.set_xticklabels(['100', '300', '1000']); top.tick_params(labelsize=6, pad=1)
top.set_xlabel(r'box $L=2\pi/k$', fontsize=6.5, labelpad=2)

# (b) lambda(B): thresholds sit on kappa_P, not on kappa_tau
Bx = np.array(Bs); kP = np.array([dat[B][0]['kP'] for B in Bs]); kt = np.array([dat[B][0]['kt'] for B in Bs])
b.plot(Bx, kP, color=K, label=r'$\kappa_P$')
b.plot(Bx, kt, color=VE, ls='--', label=r'$\kappa_\tau$')
for i, al in enumerate((0.05, 0.3)):
    y = [dat[B][1][al] for B in Bs]
    b.plot(Bx + (i - 0.5)*0.012, y, MK[i], color=(BL, GR)[i], ms=3.5, mec='white', mew=0.4, ls='none', label=rf'$\lambda$, $\alpha={al}$')
b.set_xlabel(r'$B\ (D^2/J)$'); b.set_ylabel('Doppler efficiency'); b.set_ylim(0, 1.2); b.set_xlim(0.25, 0.9); b.set_xticks([0.3, 0.5, 0.7])
b.legend(loc='upper right', fontsize=5.5)

# (c) alpha^2 dependence
for i, B in enumerate(Bs):
    lam_a = dat[B][1]; al = np.array(list(lam_a)); l = np.array(list(lam_a.values())); i0 = np.argmin(al)
    x = al**2 - al[i0]**2; y = (l - l[i0])/l[i0]*100; fit = al <= 0.3
    cc = np.sum(x[fit]*y[fit])/np.sum(x[fit]**2)
    xx = np.linspace(0, 0.36, 50); c.plot(xx, cc*xx, color=SEQ[i], lw=0.8, ls='--')
    pts(c, x[fit], y[fit], c=SEQ[i], m=MK[i], label=rf'$B={B}$: $c={-cc/100:.2f}$')
    if (~fit).any(): c.plot(x[~fit], y[~fit], MK[i], color=SEQ[i], mfc='none', mew=0.8, ms=4)
c.axhline(0, color='0.5', lw=0.5)
c.set_xlabel(r'$\alpha^2-\alpha_0^2$  ($\alpha_0=0.05$)'); c.set_ylabel(r'$\Delta\lambda/\lambda$ (%)'); c.set_xlim(0, 0.37); c.set_ylim(-22, 5)
c.legend(loc='lower left', fontsize=6, ncol=2, columnspacing=1.0)
c.axvspan(0, 0.01, color=SB, alpha=0.25, lw=0, zorder=0); c.text(0.013, 2.0, r'$\alpha\leq0.1$', fontsize=6, color=BL, va='center')
label(a, 'a', x=-0.33, y=1.12); label(b, 'b', x=-0.3, y=1.12); label(c, 'c', x=-0.11, y=1.0)
check(fig, 'fig5_skx')
print('u* =', np.round(us, 5), ' pred =', np.round(pred, 5), ' ratio', np.round(us/pred, 4))
