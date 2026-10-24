import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'request'))
from style import plt, C
# Task A summary: gamma_s(u) of the uniformly sheared lattice from the Bloch scans (scanS.py).
# Criterion: a k point has more eigenvalues with Re > 1e-7 than the unsheared lattice at the same u (this removes the
# long-wave Doppler branch, which is already unstable at gamma = 0 for u > 0). gamma_s is bracketed by the last stable and
# the first unstable s; inside the bracket it is interpolated with the soft-pair frequency: below the threshold the soft
# mode is a damped gyrotropic pair Re = -Gamma, Im = +-omega, and omega^2 goes to zero linearly at the softening.
O = 'results_request2/A_S'; res = {}
for f in sorted(glob.glob(f'{O}/S_u*_nx1_Lx19.json')):
    d = json.load(open(f)); u = float(f.split('_u')[1].split('_')[0]); ks = sorted((k for k in d if 'maxRe' in d[k]), key=int)
    if not ks or '0' not in ks: continue
    base = np.array(d['0']['n_unstable']); rows = []
    for k in ks:
        x = d[k]; nun = np.array(x['n_unstable'])
        # physical k of each grid point (cell A1 = (Lx, s), A2 = (0, Ly)); exclude the long-wave Doppler zone |k|^2 < 2.5 u |k_x|
        # (the acoustic branch is unstable for |k|^2 < 0.83 u |k_x| at gamma = 0; factor 3 margin for its shift under shear)
        Bk = 2*np.pi*np.linalg.inv(np.array([[x['Lx'], x['s']], [0, 33.]])).T; nk = x['nk']
        kv = np.array([((i/nk)-np.round(i/nk))*Bk[0]+(j/nk)*Bk[1] for i, j in x['kgrid']])
        dop = (kv**2).sum(1) < 2.5*u*np.abs(kv[:, 0])+1e-12
        new = (nun > base) & ~dop
        mr = np.array(x['maxRe']); order_ = [i for i in np.argsort(-mr) if new[i]]
        rows.append(dict(s=int(k), gamma=x['gamma'], unstable=bool(new.any()), k_soft=[x['kgrid'][i] for i in order_][:4], n_new=int(new.sum()),
                         maxRe_new=float(np.array(x['maxRe'])[new].max()) if new.any() else None, dE=x['E']-d['0']['E'], Q=x['Q'], shapes=x['shapes'], V=x['V']))
    st = [r for r in rows if not r['unstable']]; un = [r for r in rows if r['unstable']]
    g_lo = max(r['gamma'] for r in st) if st else None; g_hi = min(r['gamma'] for r in un) if un else None
    gint = None
    if un:
        # soft k: the first unstable k of the first unstable s; omega^2 of the top pair at that k on the stable side
        s1 = min(r['s'] for r in un); ksoft = [r for r in rows if r['s'] == s1][0]['k_soft'][0]; idx = d[str(s1)]['kgrid'].index(ksoft)
        pts = [(d[str(r['s'])]['gamma'], d[str(r['s'])]['top'][idx][0][1]**2) for r in rows if r['s'] < s1 and r['s'] >= s1-3]
        pts = [p for p in pts if p[1] > 1e-12]
        if len(pts) >= 2:
            g, w2 = np.array(pts).T; p = np.polyfit(g[-2:], w2[-2:], 1); gint = float(-p[1]/p[0])
            if not (g_lo <= gint <= g_hi): gint = None
    res[f'{u:g}'] = dict(u=u, rows=rows, gamma_stable_max=g_lo, gamma_unstable_min=g_hi, gamma_s_interp=gint)
json.dump(res, open(f'{O}/S_summary.json', 'w'), indent=1)
for u, r in res.items():
    print(f"u={u}: stable up to {r['gamma_stable_max']}, unstable from {r['gamma_unstable_min']}, interpolated gamma_s = {r['gamma_s_interp']}",
          [(round(x['gamma'], 3), x['k_soft'][:1]) for x in r['rows'] if x['unstable']][:2])
