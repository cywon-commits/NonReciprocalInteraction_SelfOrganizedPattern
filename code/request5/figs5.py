import sys, os, json, glob, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
sys.path.insert(0, here); sys.path.insert(1, os.path.join(root, 'code', 'request'))
from style import plt, C
from matplotlib.ticker import NullFormatter
os.environ.setdefault('MODEL_FORM', 'pw')                          # primary dispersion model of task B
from margin5 import margin
from weylB import solve
# Figures 2-6 of request 5 (figure 1 is written by anaA5.py).
O = 'results_request5'; LS = {'8x2': 152, '12x3': 228, '16x4': 304, '24x6': 456, '32x8': 608}; NX = {'8x2': 8, '12x3': 12, '16x4': 16, '24x6': 24, '32x8': 32}
DYN = {'8x2': 0.362, '12x3': 0.347, '16x4': 0.333, '24x6': 0.316, '32x8': 0.305}; FLQ = {'8x2': 0.362, '12x3': 0.345, '16x4': 0.332}
c1 = json.load(open(f'{O}/C1_summary.json')) if os.path.exists(f'{O}/C1_summary.json') else {}
if c1.get('gamma_c'): FLQ['24x6'] = c1['gamma_c']
# ---- figure 2: alpha dependence of the soft branch (lambda along theta*, alpha = 0.15 scaled by 0.15/0.3)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a_, Lx in zip(ax, (152, 144, 136)):
    for i, f in enumerate(sorted(glob.glob(f'{O}/A/disp_nx8_Lx{Lx}_s*_a0.15.json'))):
        d15 = json.load(open(f)); d3 = json.load(open(f.replace('a0.15', 'a0.3'))); km = np.array(d3['km']); th = np.array(d3['theta'])
        P3 = np.array(d3['polar'])[..., 1]; P15 = np.array(d15['polar'])[..., 1]; j = int(np.nanargmax(np.nanmax(P3, 0)))
        a_.plot(km, P3[:, j]*1e4, '-', color=C[i], label=f"gamma={d3['gamma']:.4f}, alpha 0.3")
        a_.plot(km, P15[:, j]*1e4*0.5, 'o', ms=4, mfc='none', color=C[i], label=r'alpha 0.15 $\times$ 0.5')
    a_.axhline(0, color='#52514e', lw=0.8); a_.set_xlim(0, 0.16); a_.set_ylim(-1.5, 3.5); a_.set_xlabel(r'$|k|$ along $\theta^*$'); a_.set_ylabel(r'$\lambda$ ($10^{-4}$)')
    a_.set_title(f"Lx = {Lx}, eps = {Lx/152-1:+.3f}", fontsize=10); a_.legend(fontsize=7)
fig.savefig(f'{O}/fig2_alpha.png'); plt.close(fig)
# ---- figure 3: gamma_c^pred(L) against Floquet and dynamics
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
for fn, key, mk, col, ls, lab in (('B_shift_pw', 'gc_pred_sym', 's', C[2], '-', 'local theory: piecewise model, symmetric'), ('B_shift_pw', 'gc_pred_weyl', 'D', C[3], '-', 'piecewise, Weyl'),
                                  ('B_shift', 'gc_pred_sym', 's', C[2], ':', 'linear model, symmetric'), ('B_shift', 'gc_pred_weyl', 'D', C[3], ':', 'linear model, Weyl')):
    B = json.load(open(f'{O}/{fn}.json')); L, g, e = [], [], []
    for sz, rows in B.items():
        v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        if v: L.append(LS[sz]); g.append(np.median(v)); e.append(np.ptp(v)/2)
    ax.errorbar(L, g, yerr=e, fmt=mk+ls, color=col, label=lab, capsize=3, mfc='none' if ls == ':' else col, alpha=1 if ls == '-' else 0.6)
ax.plot([LS[s] for s in DYN], list(DYN.values()), 'o--', color=C[0], label='dynamics (request 3)')
ax.plot([LS[s] for s in FLQ], list(FLQ.values()), 'x', ms=11, mew=2.5, color=C[1], label='Floquet (request 4, 5)')
ax.axhline(0.2933, color='#52514e', lw=0.8, ls=':'); ax.text(470, 0.295, r'uniform $\gamma_s(0)$', fontsize=8)
ax.set_xscale('log'); ax.set_xticks(list(LS.values())); ax.set_xticklabels(list(LS.values())); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel(r'$L_x$'); ax.set_ylabel(r'$\gamma_c$ (shear peak of the wave)')
ax.legend(fontsize=8); fig.savefig(f'{O}/fig3_gc_pred.png'); plt.close(fig)
# ---- figure 4: predicted eigenfunction against the Floquet envelope (just above the Floquet threshold)
C0 = json.load(open(f'{O}/C0_summary.json')); C0.update(c1.get('c0', {}))
picks = [('8x2', 'results_request4/C/states/8x2/S_r1.0850.npy', 'C1/8x2/f_r1.0850'), ('12x3', 'results_request4/C/states/12x3/S_r1.0785.npy', 'C1/12x3/f_r1.0785'),
         ('16x4', 'results_request4/C/states/16x4/S_r1.0770.npy', 'C1/16x4/f_r1.0770')]
if c1.get('pick'): picks.append(('24x6', c1['pick']['state'], c1['pick']['key']))
fig, ax = plt.subplots(1, len(picks), figsize=(4.6*len(picks), 3.8))
for a_, (sz, f, key) in zip(np.atleast_1d(ax), picks):
    nx = NX[sz]; Lx = 19*nx; Ly = 33*nx/4; mg = margin(np.load(f).astype(float)); xs = np.array(mg['xs'])[::10]; m = np.array(mg['m'])[::10]; e = np.array(mg['e'])[::10]
    xr = ((xs-mg['x_peak'])+Lx/2) % Lx-Lx/2; o = np.argsort(xr)
    for sc, col in (('sym', C[2]), ('weyl', C[3])):
        (lam, n, ky, xc, rms, kx, w), _ = solve(xs, m, e, Lx, Ly, sc); a_.plot(xr[o], (w/w.max())[o], color=col, label=f'{sc}: n={n}, rms {rms:.0f}')
    a_.plot(xr[o], np.clip(np.array(mg['m'])[::10][o]*Lx/30, -0.5, None), color='#9a9893', lw=1, label=r'$m(x)L/30$')
    c = [v for k_, v in C0.items() if key in k_]
    if c: a_.plot(c[0]['env_x'], c[0]['env_w'], 'o', ms=4, color=C[1], label=f"Floquet |d|$^2$, rms {c[0]['rms']:.0f}")
    a_.set_xlim(-Lx/2, Lx/2); a_.set_ylim(-0.5, 1.1); a_.set_xlabel('x - x_peak'); a_.set_title(f'{sz} ({os.path.basename(f)})', fontsize=9); a_.legend(fontsize=7)
fig.savefig(f'{O}/fig4_eigenfunction.png'); plt.close(fig)
# ---- figure 5: sigma/sqrt(L) and sigma k* against L (Floquet just above threshold, and local theory)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
fl = {}
for k_, v in C0.items():
    sz = k_.split('/')[-2]
    if sz in LS and v['Gamma'][0] > 0 and (sz not in fl or v['r'] < fl[sz]['r']): fl[sz] = v
Lf = [LS[s] for s in fl]; sf = np.array([fl[s]['rms'] for s in fl]); kf = np.array([fl[s]['kabs'] for s in fl])
ax[0].plot(Lf, sf/np.sqrt(Lf), 'o', color=C[1], label='Floquet (displacement field)')
ax[1].plot(Lf, sf*kf, 'o', color=C[1], label='Floquet')
Bw = json.load(open(f'{O}/B_weyl_pw.json')); pr = {}
for f, v in Bw.items():
    if v['lam'] > 0 and v['gamma_wave'] < 0.4 and (v['size'] not in pr or v['gamma_wave'] < pr[v['size']]['gamma_wave']): pr[v['size']] = v
Lp = [LS[s] for s in pr]; sp = np.array([pr[s]['rms'] for s in pr]); kp = np.array([np.hypot(pr[s]['ky'], pr[s]['kx']) for s in pr])
ax[0].plot(Lp, sp/np.sqrt(Lp), 'D', color=C[3], label='local theory (Weyl), first unstable state'); ax[1].plot(Lp, sp*kp, 'D', color=C[3], label='local theory')
for v, ls, lab in ((1.24, '--', r'hypothesis 0.2: 1.24$\sqrt{L}$'), (0.88, ':', r'hypothesis 0.3: 0.88$\sqrt{L}$')): ax[0].axhline(v, color='#52514e', ls=ls, lw=1, label=lab)
ax[1].axhline(np.sqrt(2), color='#52514e', ls=':', lw=1, label=r'$\sqrt{2}$ (hypothesis 0.3)')
for a_ in ax: a_.set_xscale('log'); a_.set_xticks(list(LS.values())); a_.set_xticklabels(list(LS.values())); a_.xaxis.set_minor_formatter(NullFormatter()); a_.set_xlabel(r'$L_x$'); a_.legend(fontsize=7)
ax[0].set_ylabel(r'$\sigma/\sqrt{L}$'); ax[1].set_ylabel(r'$\sigma\,|k^*|$'); fig.savefig(f'{O}/fig5_width.png'); plt.close(fig)
# ---- figure 6: margin profiles m(x) L under external strain
D = json.load(open(f'{O}/D_margin.json'))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); seen = set()
for r in D:
    a_ = ax[0 if r['size'] == '16x4' else 1]; key = (r['size'], r['eps0'])
    if key in seen: continue
    seen.add(key); xs = np.array(r['xs']); Lx = r['Lx']; xr = ((xs-r['x_peak'])/Lx+0.5) % 1-0.5; o = np.argsort(xr)
    col = {-0.03: C[0], 0.0: C[2], 0.03: C[1]}[r['eps0']]
    a_.plot(xr[o], (np.array(r['m'])*Lx)[o], color=col, label=f"eps0 = {r['eps0']:+.2f}: m_max L {r['mmaxL']:.1f}, m2 L^2 {r['m2L2']:.1f}")
for a_, t in zip(ax, ('16x4', '12x3')):
    a_.axhline(0, color='#52514e', lw=0.8); a_.set_ylim(-60, 35); a_.set_xlabel('(x - x_peak)/L'); a_.set_ylabel('m(x) L'); a_.set_title(t + ', states just below threshold', fontsize=10); a_.legend(fontsize=7)
fig.savefig(f'{O}/fig6_margin_eps0.png'); plt.close(fig)
