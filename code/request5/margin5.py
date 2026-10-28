import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
os.environ.setdefault('SKX_CELL', os.path.join(root, 'states', 'S_B0.5.npy'))
sys.path.insert(0, os.path.join(root, 'code', 'request3')); sys.path.insert(1, os.path.join(root, 'code', 'request2')); sys.path.insert(2, os.path.join(root, 'code', 'request4'))
from chain_tools import centroids
from strain import tri_strain
from regist_s import topo_s
# Task D (and the margin profile used by task B): the bundle's margin.py for boxes with an external strain eps0 = sx - 1.
# The registry is taken against the stretched reference lattice (regist_s.topo_s), the gradients are in natural coordinates, the
# total du_x/dx = sx (1 + du_x/dX) - 1 includes eps0, and the margin is m(x) = gamma(x) - gamma_s(eps_xx(x)) with the request-4 map
# gamma_s(e) = 0.2932 + 0.437 e - 0.74 e^2. Same y-uniform 1-3 harmonic profiles, same curvature stencil (40 of 4000 points) as margin.py.
gs = lambda e: 0.2932+0.437*e-0.74*e**2
def margin(S, sx=1.0, n=4000):
    t = topo_s(S, sx); X, u = t['X'], t['u']; Lx, Ly = S.shape[0]/sx, S.shape[1]; st = tri_strain(X, u, Lx, Ly); c = centroids(X, Lx, Ly); k = 2*np.pi/Lx
    M = np.stack([np.ones(len(c))]+[fn(nn*k*c[:, 0]) for nn in (1, 2, 3) for fn in (np.cos, np.sin)], 1)
    xs = np.arange(n)*Lx/n; Bm = np.stack([np.ones(n)]+[fn(nn*k*xs) for nn in (1, 2, 3) for fn in (np.cos, np.sin)], 1)
    g = Bm@np.linalg.lstsq(M, st[:, 6], rcond=None)[0]; e = Bm@np.linalg.lstsq(M, st[:, 7], rcond=None)[0]
    s = np.sign(g[np.argmax(abs(g))]); g = s*g; e = sx*(1+e)-1
    ip = int(np.argmax(g)); m = g-gs(e); im = int(np.argmax(m)); d = ((xs[im]-xs[ip]+Lx/2) % Lx)-Lx/2
    h = Lx/n; st_ = n//100; m2 = -(np.roll(m, -st_)[im]-2*m[im]+np.roll(m, st_)[im])/(st_*h)**2
    return dict(Lx=float(Lx), g_peak=float(g[ip]), eps_peak=float(e[ip]), m_max=float(m[im]), offset=float(d), offset_L=float(d/Lx), eps_at_max=float(e[im]),
                g_at_max=float(g[im]), mmaxL=float(m[im]*Lx), m2L2=float(m2*Lx**2), xs=xs.tolist(), m=m.tolist(), g=g.tolist(), e=e.tolist(), x_peak=float(xs[ip]))
if __name__ == '__main__':
    B4 = 'results_request4/B'; R3 = 'results_request3'
    cases = [('16x4', 0.0, f'{R3}/T0/16x4_s1/S_r1.0775.npy', 1.0), ('16x4', -0.03, f'{B4}/16x4/m03_s1/S_r1.1000.npy', 295/304), ('16x4', -0.03, f'{B4}/16x4/m03_s2/S_r1.1000.npy', 295/304),
             ('16x4', +0.03, f'{B4}/16x4/p03f_s1/S_r1.0630.npy', 313/304), ('16x4', +0.03, f'{B4}/16x4/p03f_s2/S_r1.0630.npy', 313/304),
             ('12x3', 0.0, f'{R3}/T0/12x3_s1/S_r1.0775.npy', 1.0), ('12x3', -0.03, f'{B4}/12x3/m03_s1/S_r1.1050.npy', 221/228), ('12x3', +0.03, f'{B4}/12x3/p03_s1/S_r1.0675.npy', 235/228)]
    out = []
    for size, e0, f, sx in cases:
        r = margin(np.load(f).astype(float), sx); r.update(size=size, eps0=e0, sx=sx, file=f); out.append(r)
        print(f"{size} eps0={e0:+.2f} sx={sx:.4f}: g_peak={r['g_peak']:.4f} eps_peak={r['eps_peak']:+.3f} | m_max={r['m_max']:+.4f} at {r['offset']:+.1f} ({r['offset_L']:+.3f} L) eps there={r['eps_at_max']:+.3f} | m_max*L={r['mmaxL']:.1f} m2*L^2={r['m2L2']:.2f}", flush=True)
    json.dump(out, open('results_request5/D_margin.json', 'w'))
