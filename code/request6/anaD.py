import json, glob, numpy as np
# Request 6, task D summary: drift V/dD, static deformation, and the phonon pair at each k: damping Re of the two branches
# (lambda = Re + i Im, sorted by Im), their mean (damping) and half difference (nonreciprocal split), linear in dD -> slopes;
# threshold dD*(k) where the less damped branch reaches Re = 0 (linear extrapolation of the small-dD slope).
out = []
for f in sorted(glob.glob('results_request6/D/dD_*.json')):
    d = json.load(open(f)); rows = d['rows']; dD = np.array([r['dD'] for r in rows]); res = dict(form=d['form'], Lx=d['Lx'], n=d['n'], ks={})
    V = np.array([r['V'] for r in rows]); sel = dD > 0
    res['V_per_dD'] = (np.polyfit(dD, V, 1)[0]).tolist() if sel.sum() > 1 else None
    res['static'] = [dict(dD=r['dD'], ellip=r['static']['ellip'], dS_max=r['static']['dS_max'], mz=r['static']['mz_mean'], mn=r['static']['mn_mean'], resid=r['resid']) for r in rows]
    for key in rows[0]['phonon']:
        pr = np.array([[p[0][0], p[1][0], p[0][1], p[1][1]] for p in (r['phonon'][key] for r in rows)])
        mean = pr[:, :2].mean(1); split = (pr[:, 1]-pr[:, 0])/2; top = pr[:, :2].max(1)
        s_mean = np.polyfit(dD[:4], mean[:4], 1)[0]; s_split = np.polyfit(dD[:4], split[:4], 1)[0]; s_top = np.polyfit(dD[:4], top[:4], 1)[0]
        thr = -top[0]/s_top if s_top > 0 else float('inf')
        res['ks'][key] = dict(Re0=float(mean[0]), Im0=float(abs(pr[0, 2])), dmean_ddD=float(s_mean), dsplit_ddD=float(s_split), dtop_ddD=float(s_top), dD_star=float(thr),
                              top_at_max=float(top[-1]), raw=pr.tolist())
    out.append(res)
    print(f"{d['form']:5s} Lx={d['Lx']} n={d['n']}: V/dD = {np.round(res['V_per_dD'], 4) if res['V_per_dD'] else None}")
    for key, v in res['ks'].items():
        print(f"     {key:10s} Re0={v['Re0']:+.3e} Im0={v['Im0']:.3e} | d(mean)/dD={v['dmean_ddD']:+.3e} d(split)/dD={v['dsplit_ddD']:+.3e} -> dD*={v['dD_star']:.4f}")
json.dump(out, open('results_request6/D_summary.json', 'w'), indent=1)
