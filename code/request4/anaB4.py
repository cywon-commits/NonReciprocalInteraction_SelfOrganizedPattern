import sys, os, json, glob, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
sys.path.insert(0, os.path.join(root, 'code', 'request3'))
from chain_tools import softamp, kband, profile, critical
# Task B: per step of the strained chains (run4.py) the soft-mode growth rate (request 3 anaG.py definition: whole-lattice
# amplitude in the soft band from the registry snapshots every 500, slope of ln a after the first 20 % of the step and before an
# event), gamma_wave, and the total du_x/dx at the peak (exxp, record column 17). gamma_c = the gamma_wave bracket between the last
# flat step and the first growing step (Gamma > 2e-5 and a > 1e-7), as in request 3.
GTH = 2e-5
def series(dn, tag, Lx, Ly, K):
    t, a, g = [], [], []
    for fn in (f'{dn}/traj_{tag}.npz', f'{dn}/slipbuf_{tag}.npz'):
        if not os.path.exists(fn): continue
        z = np.load(fn)
        for tt, X, u in zip(z['t'], z['X'], z['u']):
            ok = ~np.isnan(X[:, 0]); X = X[ok].astype(float); u = u[ok].astype(float)
            if tt in t or len(X) == 0: continue
            t.append(float(tt)); a.append(softamp(X, u, Lx, Ly, Lx/2, Lx, K)[0]); xs, p = profile(X, u, Lx, Ly); g.append(critical(xs, p, Lx)['gpeak'])
    o = np.argsort(t); return np.array(t)[o], np.array(a)[o], np.array(g)[o]
def rate(t, a, t_end):
    m = (t > t[0]+0.2*(t_end-t[0])) & (t <= t_end) & np.isfinite(a) & (a > 0)
    if m.sum() < 4: return np.nan
    return float(np.polyfit(t[m], np.log(a[m]), 1)[0])
out = {}
for dn in sorted(glob.glob('results_request4/B/*/*_s*')):
    S = sorted([json.load(open(f)) for f in glob.glob(f'{dn}/sum_r*.json')], key=lambda d: d['r'])
    if not S: continue
    res = []
    for d in S:
        tag = f"r{d['r']:.4f}"; Lx, Ly = 19*d['nx'], 33*d['ny']; K = kband(Lx, Ly)
        t, a, g = series(dn, tag, Lx, Ly, K); t_end = d['t_event']-1 if d['event'] else t[-1]
        rec = np.load(f'{dn}/rec_{tag}.npy'); pre = rec[rec[:, 0] <= t_end]; q = pre[len(pre)//2:]
        res.append(dict(r=d['r'], eps0=d['eps0'], event=d['event'], Gamma=rate(t, a, t_end), gpeak=float(np.mean(g[(t > t[0]+0.5*(t_end-t[0])) & (t <= t_end)])),
                        exxp=float(np.nanmean(q[:, 17])), a_end=float(a[t <= t_end][-1]), star=d['star'], sat=d['sat']))
        print(dn, tag, 'ev' if d['event'] else '  ', 'gpeak %.4f exxp %+.4f Gamma %.1e a_end %.0e' % (res[-1]['gpeak'], res[-1]['exxp'], res[-1]['Gamma'], res[-1]['a_end']), flush=True)
    grow = [x for x in res if x['Gamma'] > GTH and x['a_end'] > 1e-7]
    gc = None
    if grow:
        hi = grow[0]; lo = [x for x in res if x['r'] < hi['r']]
        if lo: lo = lo[-1]; gc = dict(lo=lo['gpeak'], hi=hi['gpeak'], est=0.5*(lo['gpeak']+hi['gpeak']), exxp_lo=lo['exxp'], exxp_hi=hi['exxp'], r_lo=lo['r'], r_hi=hi['r'])
    out[dn] = dict(steps=res, gamma_c=gc); print(dn, 'gamma_c', gc, flush=True)
json.dump(out, open('results_request4/B_summary.json', 'w'), indent=1)
