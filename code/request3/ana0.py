import sys, os, json, glob, numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'request'))
from style import plt, C
# Tasks 0 and A: gamma* per chain (first step with a topological event), per definition (gmax, gbar, gwave) and window
# (n = 1, 3, 5 records before the event), mean +- standard error over seeds per size, and fits gamma*(L) = g_inf + A Lx^-p
# (p free and fixed at 1/2, 1, 4/3, 2). Also the saturated (intact) universal curve gamma(u/u_c) per size and alpha.
O = 'results_request3'; GS = 0.30776
def chain(dn):
    S = [json.load(open(f)) for f in sorted(glob.glob(f'{dn}/sum_r*.json'))]
    S.sort(key=lambda d: d['r']); ev = [d for d in S if d['event']]
    first = ev[0] if ev else None
    return dict(dir=dn, steps=S, first=first, r_star=first['r'] if first else None, r_below=max([d['r'] for d in S if not d['event']], default=None),
                permanent=first['permanent'] if first else None, star=first['star'] if first else None,
                n_transient=sum(d['transient'] for d in S), sat={d['r']: d['sat'] for d in S if not d['event']},
                equil={d['r']: d['equilibrated'] for d in S if not d['event']})
def collect(pattern):
    out = {}
    for dn in sorted(glob.glob(pattern)):
        if not glob.glob(f'{dn}/sum_r*.json'): continue
        c = chain(dn); s = json.load(open(glob.glob(f'{dn}/sum_r*.json')[0])); key = f"{s['nx']}x{s['ny']}"
        out.setdefault(key, []).append(c)
    return out
def stats(chains, key):
    v = np.array([c['star'][key] for c in chains if c['star'] and c['star'].get(key) is not None])
    if len(v) == 0: return None
    return dict(mean=float(v.mean()), se=float(v.std(ddof=1)/np.sqrt(len(v))) if len(v) > 1 else None, n=len(v), values=v.tolist())
def fits(L, g, e):
    e = np.where(np.isfinite(e) & (e > 0), e, np.nanmax(e[np.isfinite(e)]) if np.isfinite(e).any() else 0.01)
    res = {}
    for p in (0.5, 1.0, 4/3, 2.0):
        f = lambda x, gi, A: gi+A*x**-p; P, Cv = curve_fit(f, L, g, p0=[0.3, 1.0], sigma=e, absolute_sigma=True)
        chi2 = float(np.sum(((f(L, *P)-g)/e)**2)); res[f'p={p:.3g}'] = dict(g_inf=float(P[0]), g_inf_err=float(np.sqrt(Cv[0, 0])), A=float(P[1]), chi2=chi2, dof=len(L)-2,
                                                                          pred608=float(f(608, *P)), pred912=float(f(912, *P)))
    if len(L) >= 4:
        try:
            f = lambda x, gi, A, p: gi+A*x**-p; P, Cv = curve_fit(f, L, g, p0=[0.3, 3.0, 1.0], sigma=e, absolute_sigma=True, maxfev=20000)
            res['free'] = dict(g_inf=float(P[0]), g_inf_err=float(np.sqrt(Cv[0, 0])), A=float(P[1]), p=float(P[2]), p_err=float(np.sqrt(Cv[2, 2])),
                               chi2=float(np.sum(((f(L, *P)-g)/e)**2)), dof=len(L)-3)
        except Exception as ex: res['free'] = dict(error=str(ex))
    return res
if __name__ == '__main__':
    T0 = collect(f'{O}/T0/*'); A32 = collect(f'{O}/A32/chain')
    for k, v in A32.items(): T0.setdefault(k, []).extend(v)
    # request-2 chain gamma* (gmax, mean of 5 records) as an extra repeat where the new definition matches
    R2 = json.load(open('results_request2/dyn_summary.json'))['u_s']
    res = dict(sizes={}, fits={})
    keys = [f'{d}_n{n}' for d in ('gmax', 'gbar', 'gwave') for n in (1, 3, 5)]
    for size, chs in sorted(T0.items(), key=lambda kv: int(kv[0].split('x')[0])):
        res['sizes'][size] = dict(Lx=19*int(size.split('x')[0]), chains=[dict(dir=c['dir'], r_star=c['r_star'], r_below=c['r_below'], permanent=c['permanent'],
                                  n_transient=c['n_transient'], star=c['star'], sat_gmax={str(r): s['gmax'] for r, s in c['sat'].items()}, equil={str(r): e for r, e in c['equil'].items()}) for c in chs],
                                  stats={k: stats(chs, k) for k in keys}, request2_gstar=R2.get(size, {}).get('gamma_star'))
    for k in keys:
        pts = [(v['Lx'], v['stats'][k]['mean'], v['stats'][k]['se'] or np.nan) for v in res['sizes'].values() if v['stats'][k]]
        if len(pts) >= 3:
            L, g, e = map(np.array, zip(*pts)); res['fits'][k] = fits(L.astype(float), g, e.astype(float))
    json.dump(res, open(f'{O}/T0_summary.json', 'w'), indent=1)
    for size, v in res['sizes'].items():
        print(size, 'r*', [c['r_star'] for c in v['chains']], 'perm', [c['permanent'] for c in v['chains']],
              {k: (round(s['mean'], 4), round(s['se'], 4) if s['se'] else None, s['n']) for k, s in v['stats'].items() if s and k.endswith('n5')})
    for k, f in res['fits'].items():
        if k.endswith('n5'): print(k, {p: {q: round(x, 4) if isinstance(x, float) else x for q, x in d.items()} for p, d in f.items()})
    # figure: gamma* vs Lx^-p for the three definitions
    if res['fits']:
        fig, ax = plt.subplots(1, 4, figsize=(18, 4.3))
        for j, p in enumerate((0.5, 1.0, 4/3, 2.0)):
            for i, d in enumerate(('gmax', 'gbar', 'gwave')):
                k = f'{d}_n5'; pts = [(v['Lx'], v['stats'][k]['mean'], v['stats'][k]['se'] or 0) for v in res['sizes'].values() if v['stats'][k]]
                if not pts: continue
                L, g, e = map(np.array, zip(*pts)); ax[j].errorbar(L**-p, g, yerr=e, fmt='o', color=C[i], label=d, capsize=3)
                if k in res['fits']:
                    F = res['fits'][k][f'p={p:.3g}']; x = np.linspace(0, (L**-p).max()*1.1, 50); ax[j].plot(x, F['g_inf']+F['A']*x, '-', color=C[i], lw=1)
            ax[j].axhline(GS, color='#52514e', ls='--', lw=1); ax[j].set_xlim(0, None); ax[j].set_xlabel(f'$L_x^{{-{p:.3g}}}$'); ax[j].set_title(f'p = {p:.3g}', fontsize=10)
        ax[0].set_ylabel(r'$\gamma^*$ (mean of 5 records before the event)'); ax[0].legend(fontsize=8)
        fig.savefig(f'{O}/fig_gstar_vs_L.png')
