import sys, os, json, glob, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..'); sys.path.insert(0, here); sys.path.insert(1, os.path.join(root, 'code', 'request3')); sys.path.insert(2, os.path.join(root, 'code', 'request2'))
from regist_s import topo_s
from chain_tools import centroids
from strain import tri_strain
# Task B(iii): the local strain at the threshold state of each chain, measured (a) at the peak of du_y/dx (as hpeak.py),
# (b) averaged over the window where the critical Floquet vector lives (task C3: centre 10-18 sites downstream of the peak,
# half width ~15) and (c) over the strip |x - x_peak| < 37.5. Totals relative to the natural lattice (eps0 included).
# For each: local threshold gamma_s(eps_xx) from the task A map and Delta_c L = (gamma_c - gamma_s) L.
AS = json.load(open('results_request4/A_summary.json')); P2 = np.array(AS['fit_quad'])
def prof(S, sx):
    t = topo_s(S, sx); X, u = t['X'], t['u']; Lx, Ly = S.shape[0]/sx, S.shape[1]; st = tri_strain(X, u, Lx, Ly); c = centroids(X, Lx, Ly); k = 2*np.pi/Lx
    M = np.stack([np.ones(len(c))]+[f(n*k*c[:, 0]) for n in (1, 2, 3) for f in (np.cos, np.sin)], 1); xs = np.arange(400)*Lx/400
    B = np.stack([np.ones(400)]+[f(n*k*xs) for n in (1, 2, 3) for f in (np.cos, np.sin)], 1)
    g = B@np.linalg.lstsq(M, st[:, 6], rcond=None)[0]; e = B@np.linalg.lstsq(M, st[:, 7], rcond=None)[0]; s = np.sign(g[np.argmax(abs(g))])
    return xs, s*g, sx*(1+s*e)-1 if s > 0 else sx*(1-e)-1, Lx
out = {}
cases = [('eps0=-0.03', 'results_request4/B/16x4/m03_s1/S_r1.1000.npy', 295/304, 0.3219), ('eps0=0', 'results_request3/T0/16x4_s1/S_r1.0775.npy', 1.0, 0.3333),
         ('eps0=+0.03', 'results_request4/B/16x4/p03f_s1/S_r1.0630.npy', 313/304, 0.3481)]
GC = json.load(open('results_request3/final_summary.json'))['T0']
cases += [(f'size {sz}', f, 1.0, GC[f'{sz}_main']['gamma_c']['est']) for sz, f in [('8x2', 'results_request3/T0/8x2_s1/S_r1.0850.npy'), ('12x3', 'results_request3/T0/12x3_s1/S_r1.0775.npy'),
          ('24x6', 'results_request3/T0/24x6_s1/S_r1.0700.npy'), ('32x8', 'results_request3/A32/chain/S_r1.0650.npy')]]
cases += [('12x3 eps0=+0.03', 'results_request4/B/12x3/p03_s1/S_r1.0675.npy', 235/228, 0.3610), ('12x3 eps0=-0.03', 'results_request4/B/12x3/m03_s1/S_r1.1050.npy', 221/228, 0.3363)]
for name, f, sx, gc in cases:
    xs, g, e, Lx = prof(np.load(f).astype(float), sx); i = int(np.argmax(g)); xp = xs[i]; dx = (xs-xp+Lx/2) % Lx-Lx/2
    res = dict(gamma_peak=float(g[i]), exx_peak=float(e[i]), exx_mode=float(e[(dx > 0) & (dx < 30)].mean()), exx_mode_m=float(e[(dx < 0) & (dx > -30)].mean()),
               exx_strip=float(e[np.abs(dx) < 37.5].mean()), gamma_c=gc)
    for k in ('exx_peak', 'exx_mode', 'exx_mode_m', 'exx_strip'):
        gs = float(np.polyval(P2, res[k])); res['gs_'+k] = gs; res['DcL_'+k] = (gc-gs)*Lx
    out[name] = res; print(name, {k: round(v, 4) for k, v in res.items()})
json.dump(out, open('results_request4/B_localavg.json', 'w'), indent=1)
