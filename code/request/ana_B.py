import sys, os, glob, json, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, C
# Task B: finite-T winding staircases (N = 200), hysteresis area vs T; first-passage times (Arrhenius) for re-winding and slipping.
O = 'results_request/B_temp'; Ts = ['0.002', '0.005', '0.01', '0.02']
def nrun(tag):
    out = []
    for f in sorted(glob.glob(f'{O}/stairs/rec_{tag}_*.npy')):
        u = float(re.search(r'_([\d.]+)\.npy', f).group(1)); r = np.load(f); q = r[len(r)*3//4:]
        n = np.where(q[:, 6] > 0.9999, 0.0, q[:, 1]).mean(); out.append((u, n))
    return np.array(out)
res = {}; ug = np.round(np.arange(0, 0.2001, 0.005), 4)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for i, T in enumerate(['0'] + Ts):
    ups, dns = [], []
    if T == '0':   # T = 0 reference: the bundled N = 200 staircase runs at the same 0.005 grid (results/hyst)
        for br, L in [('st_up', ups), ('st_dn', dns)]:
            a = []
            for f in sorted(glob.glob(f'results/hyst/rec/rec_{br}_*.npy')):
                u = float(re.search(r'_([\d.]+)\.npy', f).group(1)); r = np.load(f); a.append((u, 0.0 if r[-1, 5] > 0.9999 else r[-1, 4]))
            L.append(np.array(sorted(a)))
    else:
        for s in range(1, 6):
            a, b = nrun(f'up_T{T}_s{s}'), nrun(f'dn_T{T}_s{s}')
            if len(a) == 29: ups.append(a)
            if len(b) == 40: dns.append(b)
    if not ups or not dns: continue
    def on_grid(L, fill_low):
        M = []
        for a in L:
            d = dict((round(u, 4), n) for u, n in a); M.append([d.get(u, fill_low if u < a[:, 0].min() else np.nan) for u in ug])
        return np.array(M)
    U = on_grid(ups, 3.0); Dn = on_grid(dns, np.nan)
    nu, nd = np.nanmean(U, 0), np.nanmean(Dn, 0); m = ~np.isnan(nu) & ~np.isnan(nd)
    area = float(np.trapezoid(nu[m]-nd[m], ug[m]))
    res[T] = dict(n_up=nu.tolist(), n_dn=nd.tolist(), u=ug.tolist(), seeds=(len(ups), len(dns)), hyst_area=area, n_dn_at_u0=float(nd[0]))
    ax[0].plot(ug, nu, '-', color=C[i], label=f'T={T}'); ax[0].plot(ug, nd, '--', color=C[i])
ax[0].set_xlabel(r'$u_{\rm eff}$'); ax[0].set_ylabel('winding n (seed mean)'); ax[0].legend(fontsize=8, title='solid up / dashed down', title_fontsize=8)
TT = [float(T) for T in res]; ax[1].plot(TT, [res[T]['hyst_area'] for T in res], 'o-', color=C[0])
ax[1].set_xlabel('T'); ax[1].set_ylabel(r'hysteresis area $\int (n_{up}-n_{dn})\,du$')
# first-passage times
W = {}
for f in glob.glob(f'{O}/wait/sum_w*.json'):
    m = re.search(r'sum_(w[13])_T([\d.]+)_u([\d.]+)_s(\d+)\.json', f); kind, T, u = m.group(1), float(m.group(2)), float(m.group(3))
    d = json.load(open(f))[0]; t = d['slips'][0][0] if d['slips'] else None; to = d['slips'][0][2] if d['slips'] else None
    W.setdefault((kind, u, T), []).append((t, d['t'], to))
arr = []
for (kind, u, T), L in sorted(W.items()):
    ts = [x[0] for x in L if x[0] is not None]; cens = [x[1] for x in L if x[0] is None]
    # MLE of exponential mean with right-censoring
    tau = (sum(ts)+sum(cens))/len(ts) if ts else None
    arr.append(dict(kind=kind, u=u, T=T, n=len(L), n_events=len(ts), tau=tau, to=[x[2] for x in L if x[2] is not None]))
res['wait'] = arr; fits = {}
for j, (kind, u) in enumerate(sorted({(a['kind'], a['u']) for a in arr})):
    s = [a for a in arr if a['kind'] == kind and a['u'] == u and a['tau'] and a['n_events'] >= 3]
    if len(s) < 3: continue
    x = np.array([1/a['T'] for a in s]); y = np.log([a['tau'] for a in s]); p = np.polyfit(x, y, 1)
    fits[f'{kind}_u{u}'] = dict(dE=p[0], ln_tau0=p[1]); lab = ('n=1→2' if kind == 'w1' else 'n=3→2') + f', u={u}: ΔE={p[0]:.3f}'
    ax[2].plot(x, y, 'o', color=C[j]); ax[2].plot(x, np.polyval(p, x), '-', color=C[j], lw=1, label=lab)
res['arrhenius'] = fits
ax[2].set_xlabel('1/T'); ax[2].set_ylabel(r'ln $\tau$ (first passage)'); ax[2].legend(fontsize=7)
fig.savefig(f'{O}/fig_B_temperature.png'); json.dump(res, open(f'{O}/B_summary.json', 'w'), indent=1)
for T in [k for k in res if k not in ('wait', 'arrhenius')]: print('T', T, 'seeds', res[T]['seeds'], 'hyst area %.4f' % res[T]['hyst_area'], 'n_dn(u=0) %.2f' % res[T]['n_dn_at_u0'])
for a in arr: print(a)
print(fits)
