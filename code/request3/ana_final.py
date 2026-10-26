import sys, os, json, glob, numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'request'))
from style import plt, C
# Request 3, final analysis: tasks 0/A (gamma* and the dynamic threshold gamma_c per size, fits in L^-p), task B (links of the
# residence-time argument), task C (alpha dependence). Inputs: sum_*.json of the chains, B_growth.json (anaG.py),
# B_links.json (anaB.py), gprime.json, uc/*.json, C/*/uc.json. Output: final_summary.json and the report figures.
O = 'results_request3'; GS = 0.30776; GTH = 2e-5          # growth detectable above the noise when Gamma > GTH and a grows
DEFS = ('gmax', 'gbar', 'gwave')
def load_chain(dn):
    S = sorted([json.load(open(f)) for f in glob.glob(f'{dn}/sum_r*.json')], key=lambda d: d['r']); return S
def size_of(d): return f"{d['nx']}x{d['ny']}"
# ---------- gamma* per chain with a step error ----------
def gstar(S, key):
    ev = [d for d in S if d['event']]
    if not ev: return None
    e = ev[0]; below = [d for d in S if d['r'] < e['r'] and not d['event']]
    g = e['star'][f'{key}_n5']; gb = below[-1]['sat'][key] if below else np.nan; ge = e['sat'][key]
    # the slip happened somewhere between the saturated strain of the last intact step and that of the slip step
    return dict(r=e['r'], r_below=below[-1]['r'] if below else None, g=g, g_lo=gb, g_sat=ge, step=float(abs(ge-gb)) if below else np.nan,
                permanent=e['permanent'], n1=e['star'][f'{key}_n1'], n3=e['star'][f'{key}_n3'])
def combine(vals, steps):
    v = np.array(vals); se = v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0.0; st = np.nanmean(steps)/2 if len(steps) else 0.0
    return dict(mean=float(v.mean()), se=float(se), step_err=float(st), err=float(np.hypot(se, st)), n=len(v), values=v.tolist())
# ---------- dynamic threshold gamma_c from the growth rate per step ----------
def gamma_c(res):
    res = sorted(res, key=lambda d: d['r']); grow = [d for d in res if d['Gamma'] > GTH and d['a_end'] > 1e-7]
    if not grow: return None
    hi = grow[0]; lo = [d for d in res if d['r'] < hi['r']]
    lo = lo[-1] if lo else None
    if lo is None: return None
    # linear Gamma(gamma) through the first growing step and the last flat one (Gamma = 0 there within the noise) is an upper
    # bound of the crossing; with two growing steps the linear extrapolation to Gamma = 0 is the estimate, bracketed by lo/hi
    g2 = [d for d in grow[1:2]]
    if g2 and g2[0]['gpeak'] > hi['gpeak']:
        s = (g2[0]['Gamma']-hi['Gamma'])/(g2[0]['gpeak']-hi['gpeak']); est = hi['gpeak']-hi['Gamma']/s if s > 0 else 0.5*(lo['gpeak']+hi['gpeak'])
        est = float(min(max(est, lo['gpeak']), hi['gpeak']))
    else: est = 0.5*(lo['gpeak']+hi['gpeak']); s = None
    return dict(lo=lo['gpeak'], hi=hi['gpeak'], est=est, err=0.5*(hi['gpeak']-lo['gpeak']), Gamma_hi=hi['Gamma'], slope=s, r_lo=lo['r'], r_hi=hi['r'])
def fitp(L, g, e):
    out = {}
    for p in (0.5, 1.0, 4/3, 2.0):
        f = lambda x, gi, A: gi+A*x**-p; P, Cv = curve_fit(f, L, g, p0=[0.3, 1.0], sigma=e, absolute_sigma=True)
        out[f'{p:.3g}'] = dict(g_inf=float(P[0]), g_inf_err=float(np.sqrt(Cv[0, 0])), A=float(P[1]), chi2=float(np.sum(((f(L, *P)-g)/e)**2)), dof=len(L)-2,
                               pred912=float(f(912, *P)))
    try:
        f = lambda x, gi, A, p: gi+A*x**-p; P, Cv = curve_fit(f, L, g, p0=[0.3, 5.0, 1.0], sigma=e, absolute_sigma=True, maxfev=50000)
        out['free'] = dict(g_inf=float(P[0]), g_inf_err=float(np.sqrt(Cv[0, 0])), p=float(P[2]), p_err=float(np.sqrt(Cv[2, 2])), A=float(P[1]),
                           chi2=float(np.sum(((f(L, *P)-g)/e)**2)), dof=len(L)-3)
    except Exception as ex: out['free'] = dict(error=str(ex))
    # free p with g_inf fixed at gamma_s(0)
    try:
        f = lambda x, A, p: GS+A*x**-p; P, Cv = curve_fit(f, L, g, p0=[5.0, 1.0], sigma=e, absolute_sigma=True, maxfev=50000)
        out['free_ginf=gs'] = dict(p=float(P[1]), p_err=float(np.sqrt(Cv[1, 1])), A=float(P[0]), chi2=float(np.sum(((f(L, *P)-g)/e)**2)), dof=len(L)-2)
    except Exception as ex: out['free_ginf=gs'] = dict(error=str(ex))
    return out
if __name__ == '__main__':
    R = dict()
    G = json.load(open(f'{O}/B_growth.json')) if os.path.exists(f'{O}/B_growth.json') else {}
    # ---- task 0 / A ----
    chains = {}
    for dn in sorted(glob.glob(f'{O}/T0/*'))+sorted(glob.glob(f'{O}/A32/chain*')):
        S = load_chain(dn)
        if S and any(d['event'] for d in S): chains[dn] = S
    sizes = {}
    for dn, S in chains.items():
        key = size_of(S[0]); grp = 'hold2x' if dn.endswith('8x2_long') else 'main'
        sizes.setdefault((key, grp), []).append(dn)
    T0 = {}
    for (key, grp), dns in sorted(sizes.items(), key=lambda kv: (int(kv[0][0].split('x')[0]), kv[0][1])):
        Lx = 19*int(key.split('x')[0]); ent = dict(Lx=Lx, group=grp, chains=dns)
        for k in DEFS:
            gs = [gstar(chains[dn], k) for dn in dns]; gs = [x for x in gs if x]
            ent[k] = combine([x['g'] for x in gs], [x['step'] for x in gs]); ent[k+'_chains'] = gs
        gcs = [gamma_c(G[dn]) for dn in dns if dn in G]; gcs = [x for x in gcs if x]
        if gcs:
            v = np.array([x['est'] for x in gcs]); ent['gamma_c'] = dict(est=float(v.mean()), err=float(max(np.mean([x['err'] for x in gcs]), v.std(ddof=1) if len(v) > 1 else 0)),
                                                                          lo=float(np.mean([x['lo'] for x in gcs])), hi=float(np.mean([x['hi'] for x in gcs])), chains=gcs)
        ent['r_star'] = [x['r'] for x in ent['gmax_chains']]
        T0[f'{key}_{grp}'] = ent
    R['T0'] = T0
    main = [v for v in T0.values() if v['group'] == 'main']
    R['fits'] = {}
    for k in DEFS:
        L = np.array([v['Lx'] for v in main], float); g = np.array([v[k]['mean'] for v in main]); e = np.array([v[k]['err'] for v in main])
        R['fits'][f'gstar_{k}'] = fitp(L, g, np.maximum(e, 1e-3))
    mc = [v for v in main if 'gamma_c' in v]
    if len(mc) >= 3:
        L = np.array([v['Lx'] for v in mc], float); g = np.array([v['gamma_c']['est'] for v in mc]); e = np.array([v['gamma_c']['err'] for v in mc])
        R['fits']['gamma_c'] = fitp(L, g, np.maximum(e, 1e-3))
    # ---- task C ----
    TC = {}
    for a in ('0.15', '0.3', '0.6'):
        for size in ('12x3', '16x4'):
            dns = sorted(glob.glob(f'{O}/C/a{a}/{size}_s*')) if a != '0.3' else [dn for dn in sorted(glob.glob(f'{O}/T0/{size}_s*'))]
            dns = [dn for dn in dns if load_chain(dn) and any(d['event'] for d in load_chain(dn))]
            if not dns: continue
            ent = dict(alpha=float(a), size=size, chains=dns)
            for k in DEFS:
                gs = [gstar(load_chain(dn), k) for dn in dns]; ent[k] = combine([x['g'] for x in gs], [x['step'] for x in gs]); ent[k+'_chains'] = gs
            gcs = [gamma_c(G[dn]) for dn in dns if dn in G]; gcs = [x for x in gcs if x]
            if gcs: ent['gamma_c'] = dict(est=float(np.mean([x['est'] for x in gcs])), err=float(np.mean([x['err'] for x in gcs])), lo=float(np.mean([x['lo'] for x in gcs])), hi=float(np.mean([x['hi'] for x in gcs])))
            # universal curve gamma_max(u/u_c) of the intact steps
            S0 = load_chain(dns[0]); ent['univ'] = [(d['r'], d['sat']['gmax'], d['sat']['gwave']) for d in S0 if not d['event']]
            TC[f'a{a}_{size}'] = ent
    R['C'] = TC
    gp = json.load(open(f'{O}/gprime.json')); R['Gprime'] = {a: gp[f'Gprime_a{a}'] for a in ('0.15', '0.3', '0.6')}
    R['uc'] = {}
    for a in ('0.15', '0.6'):
        d = json.load(open(f'{O}/C/a{a}/uc.json')); R['uc'][a] = dict(u_c_4x1=d['u_c_4x1'], ucLx=d['u_c_4x1']*76, ratio_lab_bloch=d['ratio_lab_bloch'], u_bloch_12x3=d['u_bloch_12x3'])
    R['uc']['0.3'] = dict(ucLx=7.57, u_bloch_12x3=None)
    # ---- task B ----
    if os.path.exists(f'{O}/B_links.json'):
        BL = json.load(open(f'{O}/B_links.json'))
        R['B_links'] = {f: {k: v for k, v in d.items() if k != 'series'} for f, d in BL.items()}
    R['B_growth'] = {dn: [{k: d[k] for k in ('r', 'alpha', 'Lx', 'event', 'Gamma', 'Gamma_err', 'gpeak', 'a0', 'a_end')} for d in res] for dn, res in G.items()}
    json.dump(R, open(f'{O}/final_summary.json', 'w'), indent=1)
    # ---------- printout ----------
    for key, v in T0.items():
        print(key, 'Lx', v['Lx'], 'r*', v['r_star'], {k: (round(v[k]['mean'], 4), round(v[k]['err'], 4), v[k]['n']) for k in DEFS}, 'gamma_c', {q: round(v['gamma_c'][q], 4) for q in ('est', 'err', 'lo', 'hi')} if 'gamma_c' in v else None)
    for k, f in R['fits'].items():
        print(k); [print('   ', p, {q: round(x, 4) if isinstance(x, float) else x for q, x in d.items()}) for p, d in f.items()]
    for k, v in TC.items():
        print(k, {q: (round(v[q]['mean'], 4), round(v[q]['err'], 4), v[q]['n']) for q in DEFS}, 'gamma_c', {q: round(v['gamma_c'][q], 4) for q in ('est', 'lo', 'hi')} if 'gamma_c' in v else None, 'r*', [x['r'] for x in v['gmax_chains']])
    # ---------- figures ----------
    # 1. gamma* and gamma_c vs Lx^-p
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
    for j, p in enumerate((0.5, 1.0, 4/3, 2.0)):
        for i, k in enumerate(DEFS):
            L = np.array([v['Lx'] for v in main], float); g = np.array([v[k]['mean'] for v in main]); e = np.array([v[k]['err'] for v in main])
            ax[j].errorbar(L**-p, g, yerr=e, fmt='o', color=C[i], capsize=3, label=r'$\gamma^*$ '+k)
            F = R['fits'][f'gstar_{k}'][f'{p:.3g}']; x = np.linspace(0, (L**-p).max()*1.08, 50); ax[j].plot(x, F['g_inf']+F['A']*x, '-', color=C[i], lw=0.8)
        if 'gamma_c' in R['fits']:
            L = np.array([v['Lx'] for v in mc], float); g = np.array([v['gamma_c']['est'] for v in mc]); e = np.array([v['gamma_c']['err'] for v in mc])
            ax[j].errorbar(L**-p, g, yerr=e, fmt='D', color=C[3], capsize=3, label=r'$\gamma_c$ ($\Gamma=0$, gwave)')
            F = R['fits']['gamma_c'][f'{p:.3g}']; x = np.linspace(0, (L**-p).max()*1.08, 50); ax[j].plot(x, F['g_inf']+F['A']*x, '--', color=C[3], lw=0.8)
        ax[j].axhline(GS, color='#52514e', ls=':', lw=1); ax[j].set_xlim(0, None); ax[j].set_xlabel(f'$L_x^{{-{p:.3g}}}$'); ax[j].set_title(f'p = {p:.3g}', fontsize=10)
    ax[0].set_ylabel('strain at the slip'); ax[0].legend(fontsize=7)
    fig.savefig(f'{O}/fig1_gstar_vs_L.png')
    # 2. growth rate vs strain per size (task B: Gamma vs Delta)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, (key, v) in enumerate([(k, v) for k, v in T0.items() if v['group'] == 'main']):
        for dn in v['chains']:
            if dn not in G: continue
            res = [d for d in G[dn] if d['r'] >= 1.06]; g = np.array([d['gpeak'] for d in res]); Gm = np.array([d['Gamma'] for d in res])
            ax[0].plot(g, Gm*1e4, 'o-', color=C[i], ms=4, lw=0.8, label=key.split('_')[0] if dn == v['chains'][0] else None)
            if 'gamma_c' in v: ax[1].plot(g-v['gamma_c']['est'], Gm*1e4, 'o', color=C[i], ms=4)
    for a_ in ax: a_.axhline(0, color='#52514e', lw=0.6)
    ax[0].axvline(GS, color='#52514e', ls=':', lw=1); ax[0].set_xlabel(r'$\gamma_{wave}$ (peak of the strain wave)'); ax[0].set_ylabel(r'soft-mode growth rate $\Gamma$ ($10^{-4}$)'); ax[0].legend(fontsize=8)
    x = np.linspace(0, 0.012, 10); ax[1].plot(x, R['Gprime']['0.3']*x*1e4, '--', color='#52514e', lw=1, label=r"static $\Gamma'(\gamma-\gamma_s)$, $\alpha$ = 0.3")
    ax[1].set_xlabel(r'$\gamma_{wave}-\gamma_c(L)$'); ax[1].set_ylabel(r'$\Gamma$ ($10^{-4}$)'); ax[1].legend(fontsize=8)
    fig.savefig(f'{O}/fig2_growth.png')
    # 3. task C: gamma*, gamma_c and the universal curve vs alpha
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for i, size in enumerate(('12x3', '16x4')):
        al = [v['alpha'] for k, v in TC.items() if v['size'] == size]; gm = [v['gmax']['mean'] for k, v in TC.items() if v['size'] == size]; ge = [v['gmax']['err'] for k, v in TC.items() if v['size'] == size]
        ax[0].errorbar(al, gm, yerr=ge, fmt='o-', color=C[i], capsize=3, label=fr'$\gamma^*_{{max}}$ {size}')
        gc = [(v['alpha'], v['gamma_c']['est'], 0.5*(v['gamma_c']['hi']-v['gamma_c']['lo'])) for k, v in TC.items() if v['size'] == size and 'gamma_c' in v]
        if gc: a_, g_, e_ = zip(*gc); ax[0].errorbar(a_, g_, yerr=e_, fmt='D--', color=C[i], capsize=3, mfc='none', label=fr'$\gamma_c$ {size}')
    ax[0].axhline(GS, color='#52514e', ls=':', lw=1); ax[0].set_xscale('log'); ax[0].set_xlabel(r'$\alpha$ ($\beta = 2\alpha$)'); ax[0].set_ylabel('strain'); ax[0].legend(fontsize=7)
    al = np.array([0.15, 0.3, 0.6]); Gp = np.array([R['Gprime'][a] for a in ('0.15', '0.3', '0.6')])
    ax[1].plot(al, Gp, 's-', color=C[2], label=r"static $\Gamma' = d\lambda/d\gamma$")
    ax[1].plot(al, Gp[1]*(al/(1+al**2))/(0.3/(1+0.09)), ':', color='#52514e', label=r"$\propto \alpha/(1+\alpha^2)$ (request)")
    ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel(r'$\alpha$'); ax[1].set_ylabel(r"$\Gamma'$"); ax[1].legend(fontsize=7)
    for i, (k, v) in enumerate(TC.items()):
        if v['size'] != '12x3': continue
        r_, g_, w_ = zip(*v['univ']); ax[2].plot(r_, g_, 'o-', color=C[i], ms=4, label=fr'$\alpha$ = {v["alpha"]}')
    ax[2].axhline(GS, color='#52514e', ls=':', lw=1); ax[2].set_xlabel(r'$u/u_c(\alpha)$'); ax[2].set_ylabel(r'$\gamma_{max}$ (12x3, intact steps)'); ax[2].legend(fontsize=7)
    fig.savefig(f'{O}/fig3_alpha.png')
