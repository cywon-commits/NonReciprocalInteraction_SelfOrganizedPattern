"""Fig. 6: stripes have a quadratic Goldstone mode; the cone only scales the bending velocity by sqrt(1 - M^2), and every finite-box
threshold collapses onto u* sin(phi) = sqrt(c0^2 (1 - M^2) + (2 pi c_b / L)^2).
Data: results/b1_todo/pitch{30,60,120}.log, oblique30*.log (exact Bloch), results_request6/C1_summary.json (dynamics in sheared boxes)."""
import json, os, re, numpy as np, matplotlib.pyplot as plt
from b1style import *
T = os.path.join(ROOT, 'results', 'b1_todo')
par, cm = {}, {}
for P in (30, 60, 120):
    L = open(os.path.join(T, f'pitch{P}.log')).read().splitlines()
    m0 = re.search(r'D=([\d.]+): u=0 resid=\S+ c0=([\d.]+) c0/D=[\d.]+ c_b=([\d.]+)', L[0])
    par[P] = dict(D=float(m0.group(1)), c0=float(m0.group(2)), cb=float(m0.group(3)))
    cm[P] = [(0.0, 1.0)] + [(float(re.search(r' M=([\d.]+)', l).group(1)), float(re.search(r' c=([\d.]+)', l).group(1))/par[P]['c0']) for l in L[1:] if ' M=' in l]
fig = plt.figure(figsize=(W1, 2.95))
gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.5)
a = fig.add_subplot(gs[0, 0]); c = fig.add_subplot(gs[0, 1]); b = fig.add_subplot(gs[1, :])

# (a) bending velocity on the driven cone, k-resolved ratio omega(k; M)/omega(k; 0) (results/b1_figs/pitch_ratio_P*.json)
M2 = np.linspace(0, 0.17, 50); a.plot(M2, M2, color=K, label=r'$1-M^2$ ($\eta=1$)')
eta = {}
for i, P in enumerate((30, 60)):
    d = json.load(open(os.path.join(DATA, f'pitch_ratio_P{P}.json')))
    x = np.array([r['M']**2 for r in d['rows']]); R = np.array([r['ratio'] for r in d['rows']])
    y = 1 - (R**2).mean(1); e = (R**2).std(1)
    eta[P] = np.sum(x*y)/np.sum(x*x)
    pts(a, x, y, yerr=e, c=SEQ[i], m=MK[i], label=f'$P={P}$')
eb = np.mean(list(eta.values())); a.plot(M2, eb*M2, color='0.5', ls='--', lw=0.9, label=rf'$\eta={eb:.2f}$')
a.set_xlabel(r'$M^2$'); a.set_ylabel(r'$1-c(M)^2/c_0^2$'); a.set_xlim(0, 0.17); a.set_ylim(0, 0.25); a.set_yticks([0.1, 0.2]); a.set_xticks([0, 0.05, 0.1, 0.15])
a.legend(loc='upper left', fontsize=5.3)

# (c) threshold against current angle, P = 30 (k -> 0 closed form; Bloch thresholds at k = 0.01)
p = par[30]; ph = np.radians(np.linspace(4, 90, 200))
c.plot(np.degrees(ph), p['c0']/np.sqrt(np.sin(ph)**2 + eb*(p['c0']/p['D'])**2*np.cos(ph)**2)/p['D'], color=K, label='closed form')
c.plot(np.degrees(ph), np.ones_like(ph), color='0.55', ls='--', lw=0.8, label=r'unwinding $D$')
ob = []
for f in ('oblique30.log', 'oblique30b.log'):
    for l in open(os.path.join(T, f)):
        m = re.match(r'phi=(\d+) k=([\d.]+): u_eff\*=([\d.]+)', l)
        if m and float(m.group(2)) == 0.01: ob.append((float(m.group(1)), float(m.group(3))))
ob = np.array(ob); pts(c, ob[:, 0], ob[:, 1]/p['D'], c=BL, label='Bloch, $k=0.01$')
c.set_yscale('log'); c.set_xlabel(r'angle $\varphi(\mathbf{u},\mathbf{q})$ (deg)'); c.set_ylabel(r'$u_{\rm eff}^*/D$')
c.set_xlim(0, 92); c.set_ylim(0.15, 4.0); c.set_xticks([0, 30, 60, 90])
import matplotlib.ticker as mt
for ax_ in (c,):
    ax_.yaxis.set_major_formatter(mt.FormatStrFormatter('%g')); ax_.yaxis.set_minor_formatter(mt.NullFormatter()); ax_.set_yticks([0.2, 0.5, 1])
c.legend(loc='upper right', fontsize=5.0)

# (b) finite boxes: data collapse
C1 = json.load(open(os.path.join(ROOT, 'results_request6', 'C1_summary.json')))
xx = np.geomspace(0.03, 6, 100)
b.plot(xx, np.sqrt(1 + xx**2), color=K, label=r'$\sqrt{1+X^2}$')
b.plot(xx, xx, color='0.55', ls=':', lw=0.8); b.axhline(1, color='0.55', ls=':', lw=0.8)
for i, P in enumerate((30, 60)):
    for j, phi in enumerate((90, 60, 30)):
        rs = [r for r in C1 if r['P'] == P and r['phi'] == phi]
        s = np.sin(np.radians(phi)); c0 = par[P]['c0']; cb = par[P]['cb']
        X = np.array([2*np.pi*cb/(r['L']*c0*np.sqrt(1 - eb*r['M']**2)) for r in rs])
        Y = np.array([r['u_dyn']*s/(c0*np.sqrt(1 - eb*r['M']**2)) for r in rs])
        b.plot(X, Y, MK[j], color=SEQ[i], ms=4 if P == 30 else 3.5, mfc=SEQ[i] if P == 30 else 'none', mew=0.8, ls='none',
               label=rf'$P={P}$, $\varphi={phi}^\circ$')
b.set_xscale('log'); b.set_yscale('log'); b.set_xlim(0.03, 6); b.set_ylim(0.8, 7)
b.set_xlabel(r'$X=2\pi c_b/(c_M L)$,  $c_M=c_0\sqrt{1-\eta M^2}$'); b.set_ylabel(r'$Y=u_{\rm eff}^*\sin\varphi\,/\,c_M$')
b.legend(loc='upper left', fontsize=5.5, ncol=2, columnspacing=0.8)
b.xaxis.set_major_formatter(mt.FormatStrFormatter('%g')); b.yaxis.set_major_formatter(mt.FormatStrFormatter('%g')); b.yaxis.set_minor_formatter(mt.NullFormatter()); b.set_yticks([1, 2, 5])
label(a, 'a', x=-0.33, y=1.0); label(c, 'b', x=-0.33, y=1.0); label(b, 'c', x=-0.11, y=1.0)
check(fig, 'fig6_stripes')
print(par, eta)
