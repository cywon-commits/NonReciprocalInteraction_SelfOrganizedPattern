import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'request'))
from style import plt, C
# Task A analysis of the uniform-cell dispersions (dispA.py) and the M-point probes (mpoint.py).
#  - soft branch lambda(k, theta) on the polar grid; theta*(state) = direction of the largest lambda at the smallest |k| (long-wave);
#  - c(theta) = lambda/k^2 at |k| = 0.01 (log interpolation, request-4 ctheta convention) -> gamma_s(eps) from c = 0 on each line,
#    a = dc/dgamma (linear fit over the line);
#  - along theta*: fit lambda(k) = c k^2 - d k^4 + e k^6 over |k| <= kfit -> d; k*, lambda* directly from the grid maximum
#    (over k and theta) and from the fit (k* = sqrt(c/2d), lambda* = c^2/4d);
#  - effective confinement constants: long-wave form d/a and l_K = 2 sqrt(2 d/a) (Gaussian test function, request section 0.3);
#    finite-k band form (section 0.2) for comparison: Gamma' = d lambda*/d gamma, D_k = -1/2 d2 lambda/dk_x^2 at k* (fixed k_y),
#    l_K = sqrt(D_k/(2 Gamma'));
#  - alpha = 0.15 against 0.3: lambda * alpha and the ratio d/a.
GS = lambda e: 0.2932+0.437*e-0.74*e**2
def load(alpha):
    out = []
    for f in sorted(glob.glob(f'results_request5/A/disp_nx8_Lx*_s*_a{alpha}.json')):
        d = json.load(open(f))
        if 'polar' not in d: continue
        d['P'] = np.array(d['polar'])[..., 1]; d['km'] = np.array(d['km']); d['th'] = np.array(d['theta']); out.append(d)
    return out
def c_at(d, k0=0.01):
    # lambda/k^2 at |k| = k0 for every theta (linear interpolation in log k)
    lk = np.log(d['km']); y = d['P']/d['km'][:, None]**2
    return np.array([np.interp(np.log(k0), lk, y[:, j]) for j in range(len(d['th']))])
def along(d, th):
    j = int(np.argmin(abs(d['th']-th))); return d['km'], d['P'][:, j]
R = {}
for alpha in ('0.3', '0.15'):
    S = load(alpha); rows = []
    for d in S:
        cth = c_at(d); j = int(np.nanargmax(cth)); th = float(d['th'][j]); c01 = float(cth[j])
        k, lam = along(d, th); ok = np.isfinite(lam) & (k <= 0.12)
        A_ = np.stack([k[ok]**2, -k[ok]**4, k[ok]**6], 1); cf = np.linalg.lstsq(A_, lam[ok], rcond=None)[0]
        i, jj = np.unravel_index(np.nanargmax(d['P']), d['P'].shape)
        rows.append(dict(Lx=d['Lx'], s=d['s'], gamma=d['gamma'], exx=d['exx'], m=d['gamma']-GS(d['exx']), theta=th, c01=c01, c_fit=float(cf[0]), d_fit=float(cf[1]),
                         e_fit=float(cf[2]), kstar=float(d['km'][i]), thstar=float(d['th'][jj]), lamstar=float(d['P'][i, jj]), c_theta=cth.tolist()))
    R[alpha] = rows
# per line: gamma_s from c01 = 0, a = dc/dgamma, d near threshold
lines = {}
for r in R['0.3']: lines.setdefault(r['Lx'], []).append(r)
summ = {}
for Lx, rows in sorted(lines.items()):
    rows.sort(key=lambda r: r['gamma']); g = np.array([r['gamma'] for r in rows]); c = np.array([r['c01'] for r in rows])
    i = int(np.where((c[:-1] < 0) & (c[1:] > 0))[0][0]); gs = g[i]-c[i]*(g[i+1]-g[i])/(c[i+1]-c[i])
    near = np.abs(g-gs) < 0.025; a = float(np.polyfit(g[near], c[near], 1)[0])
    pos = [r for r in rows if r['c_fit'] > 0 and r['gamma']-gs > 0.002]
    dvals = [r['d_fit'] for r in pos[:4]]; dmean = float(np.mean(dvals)) if dvals else np.nan
    # finite-k band form: Gamma' = d lambda*/d gamma over the positive states, D_k from the curvature of lambda(k_x) at k*
    gp = [r for r in rows if r['lamstar'] > 0]
    Gp = float(np.polyfit([r['gamma'] for r in gp], [r['lamstar'] for r in gp], 1)[0]) if len(gp) > 2 else np.nan
    lam2 = np.polyfit([r['gamma']-gs for r in gp[:5]], [r['lamstar'] for r in gp[:5]], 2) if len(gp) >= 4 else [np.nan]*3
    summ[Lx] = dict(exx=rows[0]['exx'], gamma_s=float(gs), theta_s=float(np.interp(gs, g, [r['theta'] for r in rows])), a=a, d=dmean, d_over_a=dmean/a,
                    lK_longwave=2*np.sqrt(2*dmean/a), Gprime_band=Gp, lamstar_quad=list(map(float, lam2)),
                    kstar=[(round(r['gamma']-gs, 4), r['kstar'], r['lamstar'], r['thstar']) for r in rows])
    print(f"Lx={Lx} eps={rows[0]['exx']:+.4f}: gamma_s={gs:.4f} theta*={summ[Lx]['theta_s']:.0f} a=dc/dgamma={a:.3f} d={dmean:.2f} d/a={dmean/a:.2f} l_K(long-wave)={2*np.sqrt(2*dmean/a):.2f}"
          f" | lambda* vs delta quadratic coef {lam2[0]:.3f}", flush=True)
    for r in rows: print(f"    gamma={r['gamma']:.4f} delta={r['gamma']-gs:+.4f} theta={r['theta']:.0f} c01={r['c01']:+.4f} c_fit={r['c_fit']:+.4f} d={r['d_fit']:.2f} k*={r['kstar']:.4f} lam*={r['lamstar']:+.2e} th*={r['thstar']:.0f}")
# alpha comparison
cmp = []
for r15 in R['0.15']:
    r3 = [r for r in R['0.3'] if r['Lx'] == r15['Lx'] and r['s'] == r15['s']]
    if r3: r3 = r3[0]; cmp.append(dict(Lx=r15['Lx'], s=r15['s'], ratio_lamstar=r15['lamstar']/r3['lamstar'] if r3['lamstar'] else np.nan, ratio_c=r15['c_fit']/r3['c_fit'],
                                       ratio_d=r15['d_fit']/r3['d_fit'], doa15=r15['d_fit']/r15['c_fit'], doa3=r3['d_fit']/r3['c_fit']))
for c_ in cmp: print('alpha 0.15/0.3', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c_.items()})
# M point
MP = []
for f in sorted(glob.glob('results_request5/A/mpoint_nx8_*_a0.3.json')):
    d = json.load(open(f)); a_ = np.array(d['data']); atM = a_[(a_[:, 0] == 0) & (a_[:, 1] == 0)][0, 2]; MP.append(dict(Lx=d['Lx'], s=d['s'], gamma=d['gamma'], exx=d['exx'], lam_M=float(atM)))
print('M point lambda range:', min(m['lam_M'] for m in MP), max(m['lam_M'] for m in MP))
json.dump(dict(states=R, lines=summ, alpha_cmp=cmp, mpoint=MP), open('results_request5/A_summary.json', 'w'), indent=1)
# ---- figure 1: lambda(k, theta) maps on the eps = 0 and eps = -0.105 lines, k*(delta), lambda*(delta)
fig, ax = plt.subplots(2, 4, figsize=(20, 8.5))
for row, Lx in enumerate((152, 136)):
    rows = sorted(lines[Lx], key=lambda r: r['gamma']); gs = summ[Lx]['gamma_s']
    picks = [min(rows, key=lambda r: abs(r['gamma']-gs-dl)) for dl in (0.0, 0.01, 0.02)]
    for i, r in enumerate(picks):
        d = [x for x in load('0.3') if x['Lx'] == Lx and x['s'] == r['s']][0]
        im = ax[row, i].pcolormesh(d['th'], d['km'], d['P']*1e4, shading='nearest', cmap='RdBu_r', vmin=-np.nanmax(abs(d['P']))*1e4, vmax=np.nanmax(abs(d['P']))*1e4)
        ax[row, i].set_yscale('log'); ax[row, i].set_xlabel(r'$\theta$ (deg)'); ax[row, i].set_ylabel('|k|'); plt.colorbar(im, ax=ax[row, i], label=r'$\lambda$ ($10^{-4}$)')
        ax[row, i].set_title(f"eps={r['exx']:+.3f}, delta={r['gamma']-gs:+.4f}", fontsize=9)
    dl = np.array([r['gamma']-gs for r in rows]); ks = np.array([r['kstar'] for r in rows]); ls_ = np.array([r['lamstar'] for r in rows])
    pos = (dl > 0) & (ls_ > 0)
    a3 = ax[row, 3]; a3.loglog(dl[pos], ks[pos], 'o-', color=C[0], label='k*'); a3.loglog(dl[pos], ls_[pos]*100, 's-', color=C[1], label=r'$\lambda^*\times 100$')
    x = np.geomspace(dl[pos].min(), dl[pos].max(), 10); a3.loglog(x, ks[pos][0]*(x/dl[pos][0])**0.5, ':', color=C[0], lw=1, label=r'$\propto\delta^{1/2}$')
    a3.loglog(x, ls_[pos][0]*100*(x/dl[pos][0])**2, ':', color=C[1], lw=1, label=r'$\propto\delta^{2}$'); a3.set_xlabel(r'$\delta = \gamma - \gamma_s(\varepsilon)$'); a3.legend(fontsize=7)
fig.savefig('results_request5/fig1_dispersion.png')
