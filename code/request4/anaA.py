import sys, os, json, glob, numpy as np
from scipy.optimize import curve_fit, brentq
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
os.environ.setdefault('SKX_CELL', os.path.join(root, 'states', 'S_B0.5.npy'))
sys.path.insert(0, os.path.join(root, 'code', 'request3')); sys.path.insert(1, os.path.join(root, 'code', 'request2')); sys.path.insert(2, os.path.join(root, 'code', 'request'))
from chain_tools import centroids
from strain import tri_strain
from regist import topo
from style import plt, C
# Task A analysis.
#  1. gamma_s(eps_xx): zero of the long-wave coefficient c_max(gamma) (ctheta.py, |k| = 0.01, maximised over the direction)
#     on each eps_xx line, linear interpolation between the bracketing gamma (and a quadratic through 3 points as a check);
#     also the finite-k threshold of scanA.py (gs_*.json) for comparison.
#  2. Peak strain of the request-3 states (hpeak.py method: y-uniform 1-3 harmonic profiles of the triangle gradients; values at the
#     maximum of du_y/dx): eps_xx, du_x/dy, du_y/dy, rotation, ratio r = eps_xx/gamma; y dependence (spread over y of the
#     triangle gradients next to the peak); disk average (R = 37.5) around the peak.
#  3. Self-consistent local threshold gamma = gamma_s(r gamma) and fits of gamma_c(L) (request 3) with gamma_inf fixed there.
O = 'results_request4/A'; GC = json.load(open('results_request3/final_summary.json'))['T0']
# ---- 1 ----
lines = {}
for f in glob.glob(f'{O}/ctheta_u0.000_nx4_Lx*_s*.json'):
    d = json.load(open(f)); lines.setdefault(d['Lx'], []).append(d)
GSe = []
for Lx, L in sorted(lines.items()):
    L.sort(key=lambda d: d['gamma']); g = np.array([d['gamma'] for d in L]); c = np.array([d['c_max'] for d in L])
    i = int(np.where((c[:-1] < 0) & (c[1:] > 0))[0][0]); gl = g[i]-c[i]*(g[i+1]-g[i])/(c[i+1]-c[i])
    j0 = max(0, min(i-1, len(g)-3)); q = np.polyfit(g[j0:j0+3], c[j0:j0+3], 2); rr = np.roots(q).real; rr = rr[(rr > g[i]-0.02) & (rr < g[i+1]+0.02)]
    gq = float(rr[np.argmin(abs(rr-gl))]) if len(rr) else np.nan
    fk = json.load(open(f'{O}/gs_u0.000_nx4_Lx{Lx}.json'))['gamma_s_lin']
    GSe.append(dict(Lx=Lx, exx=L[0]['exx'], gamma_s=float(gl), gamma_s_quad=gq, gamma_s_finite_k=fk, theta=float(np.interp(gl, g, [d['theta'] for d in L])),
                    bracket=[float(g[i]), float(g[i+1])], dc_dgamma=float((c[i+1]-c[i])/(g[i+1]-g[i]))))
e = np.array([d['exx'] for d in GSe]); gs = np.array([d['gamma_s'] for d in GSe]); P1 = np.polyfit(e, gs, 1); P2 = np.polyfit(e, gs, 2)
gs_of = lambda x: float(np.polyval(P2, x))
print('gamma_s(eps):', [(round(d['exx'], 4), round(d['gamma_s'], 4), round(d['gamma_s_quad'], 4), round(d['gamma_s_finite_k'], 4), d['theta']) for d in GSe])
print('linear fit: gamma_s(0) = %.4f, slope = %.3f; quadratic:' % (P1[1], P1[0]), np.round(P2, 4))
# ---- 2 ----
def peak(X, u, Lx, Ly):
    st = tri_strain(X, u, Lx, Ly); c = centroids(X, Lx, Ly); k = 2*np.pi/Lx
    comp = dict(gamma=st[:, 6], dxux=st[:, 7], dyuy=st[:, 1], dyux=2*st[:, 2]-st[:, 6])
    M = np.stack([np.ones(len(c))]+[fn(n*k*c[:, 0]) for n in (1, 2, 3) for fn in (np.cos, np.sin)], 1)
    xs = np.arange(400)*Lx/400; Bm = np.stack([np.ones(400)]+[fn(n*k*xs) for n in (1, 2, 3) for fn in (np.cos, np.sin)], 1)
    prof = {kk: Bm@np.linalg.lstsq(M, v, rcond=None)[0] for kk, v in comp.items()}
    g = prof['gamma']; sg = np.sign(g[np.argmax(abs(g))]); i = int(np.argmax(sg*g)); xp = xs[i]
    out = {kk: float(sg*prof[kk][i]) for kk in comp}; out['omega'] = 0.5*(out['gamma']-out['dyux'])
    dx = (c[:, 0]-xp+Lx/2) % Lx-Lx/2; near = np.abs(dx) < 10
    out['y_spread_gamma'] = float(np.std(sg*comp['gamma'][near])); out['y_spread_dxux'] = float(np.std(sg*comp['dxux'][near]))
    disk = np.abs(dx) < 37.5
    out['disk_gamma'] = float(np.mean(sg*comp['gamma'][disk])); out['disk_dxux'] = float(np.mean(sg*comp['dxux'][disk]))
    out['r'] = out['dxux']/out['gamma']; out['max_abs_dxux'] = float(abs(prof['dxux']).max()); return out
PK = {}
for size, dn in [('8x2', 'T0/8x2_s*'), ('12x3', 'T0/12x3_s*'), ('16x4', 'T0/16x4_s*'), ('24x6', 'T0/24x6_s*'), ('32x8', 'A32/chain*')]:
    rows = []
    for d in sorted(glob.glob(f'results_request3/{dn}')):
        S_ = sorted(json.load(open(f)) for f in glob.glob(f'{d}/sum_r*.json')) if False else sorted([json.load(open(f)) for f in glob.glob(f'{d}/sum_r*.json')], key=lambda q: q['r'])
        ev = [q for q in S_ if q['event']]
        if not ev: continue
        below = [q for q in S_ if q['r'] < ev[0]['r']][-2:]
        for q in below:                              # last two intact steps (gamma_c bracket)
            S = np.load(f"{d}/S_r{q['r']:.4f}.npy").astype(float); t = topo(S)
            if t['intact']: o = peak(t['X'], t['u'], *S.shape[:2]); o.update(src=f"{d}/S_r{q['r']:.4f}", kind='step'); rows.append(o)
        fb = f"{d}/slipbuf_r{ev[0]['r']:.4f}.npz"   # last 5 records before the slip
        if os.path.exists(fb):
            z = np.load(fb, allow_pickle=True); Lx, Ly = 19*ev[0]['nx'], 33*ev[0]['ny']
            for X, u in list(zip(z['X'], z['u']))[-5:]:
                o = peak(np.asarray(X, float), np.asarray(u, float), Lx, Ly); o.update(src=fb, kind='pre-slip'); rows.append(o)
    PK[size] = rows
    r_ = np.array([o['r'] for o in rows]); print(size, 'n', len(rows), 'r = %.3f +- %.3f' % (r_.mean(), r_.std()), 'gamma %.3f..%.3f' % (min(o['gamma'] for o in rows), max(o['gamma'] for o in rows)),
          'dyux %.4f dyuy %.4f' % (np.mean([o['dyux'] for o in rows]), np.mean([o['dyuy'] for o in rows])), 'y-spread gamma %.3f' % np.mean([o['y_spread_gamma'] for o in rows]),
          'disk r %.3f' % np.mean([o['disk_dxux']/o['disk_gamma'] for o in rows]), flush=True)
# ---- 3 ----
res = dict(gamma_s_eps=GSe, fit_lin=P1.tolist(), fit_quad=P2.tolist(), peaks={k: v for k, v in PK.items()})
sc = {}
for size, rows in PK.items():
    r_ = np.mean([o['r'] for o in rows]); g_sc = brentq(lambda g: g-gs_of(r_*g), 0.2, 0.35); sc[size] = dict(r=float(r_), gamma_s_Hpeak=float(g_sc))
r_all = np.mean([np.mean([o['r'] for o in rows]) for rows in PK.values()]); g_all = brentq(lambda g: g-gs_of(r_all*g), 0.2, 0.35)
sc['all'] = dict(r=float(r_all), gamma_s_Hpeak=float(g_all)); res['self_consistent'] = sc; print('self-consistent gamma_s(H_peak):', sc)
L = np.array([GC[f'{s}_main']['Lx'] for s in ('8x2', '12x3', '16x4', '24x6', '32x8')], float)
g = np.array([GC[f'{s}_main']['gamma_c']['est'] for s in ('8x2', '12x3', '16x4', '24x6', '32x8')]); er = np.array([GC[f'{s}_main']['gamma_c']['err'] for s in ('8x2', '12x3', '16x4', '24x6', '32x8')])
F = {}
for name, gi in [('gamma_s(H_peak)', g_all), ('0.281 (request)', 0.281), ('gamma_s(0) long-wave', gs_of(0.0)), ('0.308 (request 2)', 0.308)]:
    f1 = lambda x, A: gi+A/x; P, _ = curve_fit(f1, L, g, p0=[10], sigma=er, absolute_sigma=True); c1 = float(np.sum(((f1(L, *P)-g)/er)**2))
    f2 = lambda x, A, B_: gi+A/x+B_/x**2; P2_, _ = curve_fit(f2, L, g, p0=[10, 0], sigma=er, absolute_sigma=True); c2 = float(np.sum(((f2(L, *P2_)-g)/er)**2))
    f3 = lambda x, A, p: gi+A*x**-p; P3, C3 = curve_fit(f3, L, g, p0=[5, 1], sigma=er, absolute_sigma=True, maxfev=20000); c3 = float(np.sum(((f3(L, *P3)-g)/er)**2))
    F[name] = dict(gamma_inf=float(gi), A_over_L=dict(A=float(P[0]), chi2=c1, dof=4), A_L_B_L2=dict(A=float(P2_[0]), B=float(P2_[1]), chi2=c2, dof=3),
                   p_free=dict(p=float(P3[1]), p_err=float(np.sqrt(C3[1, 1])), A=float(P3[0]), chi2=c3, dof=3), DcL=((g-gi)*L).tolist())
    print(name, round(gi, 4), 'A/L chi2 %.1f/4' % c1, 'A/L+B/L2 chi2 %.1f/3' % c2, 'p free %.2f +- %.2f chi2 %.1f/3' % (P3[1], np.sqrt(C3[1, 1]), c3), 'Dc*L', np.round((g-gi)*L, 1))
res['fits_gamma_c'] = F; json.dump(res, open('results_request4/A_summary.json', 'w'), indent=1)
# ---- figures ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].plot(e, gs, 'o', color=C[0], label=r'$\gamma_s(\varepsilon_{xx})$: long-wave $c=0$')
ax[0].plot(e, [d['gamma_s_finite_k'] for d in GSe], 's', mfc='none', color=C[1], label='finite-k scan (interpolated max Re)')
for d in json.load(open('results_request4/bundle/S_c2s11.json')).values(): ax[0].plot(d['exx'], d['gamma'], 'x', color=C[3], ms=9)
x = np.linspace(-0.09, 0.035, 50); ax[0].plot(x, np.polyval(P2, x), '-', color=C[0], lw=1)
for size in PK: ax[0].plot(x, x/sc[size]['r'], ':', lw=0.8, color='#52514e')
ax[0].plot(r_all*g_all, g_all, '*', color=C[3], ms=14, label=fr'$\gamma_s(H_{{peak}})$ = {g_all:.3f} (r = {r_all:.3f})')
ax[0].set_xlabel(r'$\varepsilon_{xx}$'); ax[0].set_ylabel(r'$\gamma$'); ax[0].legend(fontsize=7); ax[0].set_ylim(0.23, 0.33)
ax[0].text(-0.085, 0.235, 'dotted: peak states eps = r gamma', fontsize=7)
xL = np.linspace(0, 1/140, 50)
ax[1].errorbar(1/L, g, yerr=er, fmt='D', color=C[3], capsize=3, label=r'$\gamma_c(L)$ (request 3)')
for i, (name, Fd) in enumerate(F.items()):
    ax[1].plot(xL, Fd['gamma_inf']+Fd['A_L_B_L2']['A']*xL+Fd['A_L_B_L2']['B']*xL**2, '-', lw=0.9, color=C[i], label=f"{name}: A/L+B/L$^2$, $\\chi^2$={Fd['A_L_B_L2']['chi2']:.1f}/3")
ax[1].set_xlabel(r'$1/L_x$'); ax[1].set_ylabel(r'$\gamma_c$'); ax[1].set_xlim(0, None); ax[1].legend(fontsize=6)
fig.savefig('results_request4/fig1_gamma_s_map.png')
