import sys, os, json, glob, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from chain_tools import softamp, kband, profile, critical
# Task B, growth rate per chain step: the soft-mode amplitude of the whole lattice (band 0.06 < |k| < 0.11, 0.045 < |k_y| < 0.1,
# after removing the smooth field) from the registry snapshots kept every 500 time units (traj_*.npz) plus the ring before an
# event. Gamma(r) = slope of ln a over the part of the step after the first 20 % and before the event (or the end), where
# a is above 3x its initial (noise) value or, when it never grows, over the same span (then Gamma <= 0 within the noise).
def series(dn, tag, Lx, Ly, K):
    t, a, g = [], [], []
    z = np.load(f'{dn}/traj_{tag}.npz', allow_pickle=True)
    for tt, X, u in zip(z['t'], z['X'], z['u']):
        X = np.asarray(X, float); u = np.asarray(u, float); t.append(float(tt)); a.append(softamp(X, u, Lx, Ly, Lx/2, Lx, K)[0])
        xs, p = profile(X, u, Lx, Ly); g.append(critical(xs, p, Lx)['gpeak'])
    fb = f'{dn}/slipbuf_{tag}.npz'
    if os.path.exists(fb):
        zb = np.load(fb, allow_pickle=True)
        for tt, X, u in zip(zb['t'], zb['X'], zb['u']):
            if tt <= max(t, default=-1): continue
            X = np.asarray(X, float); u = np.asarray(u, float); t.append(float(tt)); a.append(softamp(X, u, Lx, Ly, Lx/2, Lx, K)[0])
            xs, p = profile(X, u, Lx, Ly); g.append(critical(xs, p, Lx)['gpeak'])
    o = np.argsort(t); return np.array(t)[o], np.array(a)[o], np.array(g)[o]
def rate(t, a, t_end):
    m = (t > t[0]+0.2*(t_end-t[0])) & (t <= t_end) & np.isfinite(a) & (a > 0)
    if m.sum() < 4: return np.nan, np.nan
    P, C = np.polyfit(t[m], np.log(a[m]), 1, cov=True); return float(P[0]), float(np.sqrt(C[0, 0]))
if __name__ == '__main__':
    out = {}
    dirs = sorted(glob.glob('results_request3/T0/*'))+sorted(glob.glob('results_request3/A32/chain'))+sorted(glob.glob('results_request3/C/*/12x3_s*'))
    for dn in dirs:
        S = sorted(glob.glob(f'{dn}/sum_r*.json'))
        if not S: continue
        res = []
        for f in S:
            d = json.load(open(f)); tag = os.path.basename(f)[4:-5]; Lx, Ly = 19*d['nx'], 33*d['ny']
            if not os.path.exists(f'{dn}/traj_{tag}.npz'): continue
            K = kband(Lx, Ly); t, a, g = series(dn, tag, Lx, Ly, K)
            t_end = d['t_event']-1 if d['event'] else t[-1]; G, eG = rate(t, a, t_end)
            gp = float(np.mean(g[(t > t[0]+0.5*(t_end-t[0])) & (t <= t_end)]))
            res.append(dict(r=d['r'], alpha=d['alpha'], Lx=Lx, event=d['event'], Gamma=G, Gamma_err=eG, gpeak=gp, a0=float(a[0]), a_end=float(a[t <= t_end][-1]),
                            t=t.tolist(), a=a.tolist(), g=g.tolist()))
            print(dn, tag, f"ev={d['event']} gpeak={gp:.4f} Gamma={G:.2e}+-{eG:.1e} a0={a[0]:.1e} a_end={res[-1]['a_end']:.1e}", flush=True)
        out[dn] = res
    json.dump(out, open('results_request3/B_growth.json', 'w'))
