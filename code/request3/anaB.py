import sys, os, json, glob, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here); sys.path.insert(1, os.path.join(here, '..', 'request2'))
from chain_tools import profile, critical, softamp, kband, unwrap_x, GS
from regist import topo
# Task B: links of the residence-time argument measured on the records just before each slip.
# Input: a slip buffer (request 2: slip_*.npz with spin snapshots S; request 3: slipbuf_*.npz with registry X, u) plus,
# when present, the request-3 traj_*.npz of the same step (registry every 500).
# Per buffer:
#   v       : drift of the profile peak x_peak(t) relative to the lattice (reference coordinates), from a linear fit over the
#             intact records; v*Lx and the sign relative to the current (electrons along -x for u > 0 ... sign kept as is);
#   w, Delta: width of the region with profile > gamma_s(0) and Delta = gpeak - gamma_s(0) (mean over intact records);
#   a(t)    : soft-mode amplitude in a window fixed in the lattice at the slip site x_s (|x - x_s| < 37.5), x_s chosen as the
#             window of largest amplitude in the first broken record;
#   Delta_loc(t) = profile(x_s, t) - gamma_s(0); Gamma(t) = d ln a / dt (central differences) paired with Delta_loc(t);
#   I_slip  = ln a(t_last intact) - ln a(t_enter): cumulative growth since the window last entered the critical region
#             (Delta_loc crossing zero upward), the direct test of Gamma * t_res ~ const.
HW = 37.5
def load_buffer(f, Lx, Ly):
    z = np.load(f, allow_pickle=True); out = []
    if 'S' in z and z['S'].ndim == 4:
        for t, S in zip(z['t'], z['S']):
            r = topo(S.astype(float)); out.append((float(t), r['intact'], r['X'], r['u']))
    else:
        for t, X, u in zip(z['t'], z['X'], z['u']): out.append((float(t), True, np.asarray(X, float), np.asarray(u, float)))
        if 'S' in z:
            r = topo(z['S'].astype(float)); out.append((out[-1][0]+(out[-1][0]-out[-2][0]), r['intact'], r['X'], r['u']))
    return out
def analyse(recs, Lx, Ly, extra=()):
    K = kband(Lx, Ly); intact = [r for r in recs if r[1]]; broken = [r for r in recs if not r[1]]
    if len(intact)+len(extra) < 4: return None
    sign = None; rows = []
    for t, ok, X, u in list(extra)+intact:
        xs, p = profile(X, u, Lx, Ly); c = critical(xs, p, Lx, sign); sign = c['sign']; rows.append((t, xs, sign*p, c, X, u))
    rows.sort(key=lambda r: r[0])
    t = np.array([r[0] for r in rows]); xp = unwrap_x([r[3]['xpeak'] for r in rows], Lx)
    last = t > t[-1]-6000; v = float(np.polyfit(t[last], xp[last], 1)[0]) if last.sum() > 3 else np.nan
    # slip site
    if broken:
        tb, _, Xb, ub = broken[0]; grid = np.arange(0, Lx, 4.0)
        amps = [softamp(Xb, ub, Lx, Ly, x0, HW, K)[0] for x0 in grid]; xsl = float(grid[int(np.nanargmax(amps))])
    else:
        xsl = rows[-1][3]['xpeak']; tb = None
    a = np.array([softamp(X, u, Lx, Ly, xsl, HW, K)[0] for _, _, _, _, X, u in rows])
    dl = np.array([p[int(round(xsl/Lx*len(xs))) % len(xs)]-GS for _, xs, p, _, _, _ in rows])
    la = np.log(a); G = np.gradient(la, t)
    # the growing amplitude turns out to be spread over the whole lattice with its maximum riding on the strain wave:
    # also measure it in a window that moves with the wave (centred on x_peak) and as the median over windows along x
    grid = np.arange(0, Lx, Lx/24)
    aw = np.array([softamp(X, u, Lx, Ly, c['xpeak'], HW, K)[0] for _, _, _, c, X, u in rows])
    am = np.array([np.nanmedian([softamp(X, u, Lx, Ly, x0, HW, K)[0] for x0 in grid]) for _, _, _, _, X, u in rows])
    fit = t > t[-1]-2000.5
    Gw = float(np.polyfit(t[fit], np.log(aw[fit]), 1)[0]) if fit.sum() > 3 else np.nan
    Gm = float(np.polyfit(t[fit], np.log(am[fit]), 1)[0]) if fit.sum() > 3 else np.nan
    # entry into the critical region
    i_end = len(t)-1; i0 = i_end
    while i0 > 0 and dl[i0-1] > 0: i0 -= 1
    I = float(la[i_end]-la[i0]) if dl[i_end] > 0 else np.nan
    tres = float(t[i_end]-t[i0])
    crit = [r[3] for r in rows[-10:]]
    return dict(Lx=Lx, Ly=Ly, t_first_broken=tb, x_slip=xsl, v=v, vLx=v*Lx, w=float(np.mean([c['w'] for c in crit])),
                gpeak=float(np.mean([c['gpeak'] for c in crit])), Delta=float(np.mean([c['gpeak'] for c in crit])-GS),
                I_slip=I, t_in_critical=tres, Gamma_wave=Gw, Gamma_median=Gm, a_wave_end=float(aw[-1]), a_median_end=float(am[-1]), a_end=float(a[-1]), a_enter=float(a[i0]),
                series=dict(t=t.tolist(), a=a.tolist(), a_wave=aw.tolist(), a_median=am.tolist(), Delta_loc=dl.tolist(), Gamma=G.tolist(), xpeak=xp.tolist(), gpeak=[r[3]['gpeak'] for r in rows], w=[r[3]['w'] for r in rows]))
if __name__ == '__main__':
    res = {}
    # request 2 buffers (spin snapshots, 20 records every 200 before the slip)
    for f in sorted(glob.glob('results_request2/B_univ/*/slip_chain_r*.npz')):
        size = f.split('/')[-2]; nx, ny = map(int, size.split('x'))
        if nx < 8: continue
        d = analyse(load_buffer(f, 19*nx, 33*ny), 19*nx, 33*ny)
        if d is None: continue
        res[f] = d
        print(f, {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items() if k != 'series'}, flush=True)
    # request 3 buffers (registry ring + traj of the step)
    for f in sorted(glob.glob('results_request3/**/slipbuf_r*.npz', recursive=True)):
        dn = os.path.dirname(f); s = json.load(open(glob.glob(f'{dn}/sum_*.json')[0])); nx, ny = s['nx'], s['ny']; Lx, Ly = 19*nx, 33*ny
        if nx < 8: continue                                    # 4x1: no wavevector of the soft-mode band fits
        tag = os.path.basename(f)[8:-4]; ft = f'{dn}/traj_{tag}.npz'; extra = []
        recs = load_buffer(f, Lx, Ly)
        if os.path.exists(ft):
            z = np.load(ft, allow_pickle=True); t0 = recs[0][0]
            extra = [(float(t), True, np.asarray(X, float), np.asarray(u, float)) for t, X, u in zip(z['t'], z['X'], z['u']) if t < t0]
        d = analyse(recs, Lx, Ly, extra)
        if d is None: continue
        res[f] = d
        print(f, {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items() if k != 'series'}, flush=True)
    json.dump(res, open('results_request3/B_links.json', 'w'))
