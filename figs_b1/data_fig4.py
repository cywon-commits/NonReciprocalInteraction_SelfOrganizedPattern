"""Data for Fig. 4 (helix): growth rate vs u_eff for five (alpha, beta) and finite-chain thresholds.
Uses code/helix/helix.py (exact driven cone, exact Jacobian). Writes results/b1_figs/fig4_helix.json."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'helix'))
from helix import predict, cone, jac, rhs
from b1style import DATA
J = 1.0
def setup(N, n):
    th = 2*np.pi*n/N; return th, J*np.tan(th)
def growth(N, n, ueff, a, b):
    th, D = setup(N, n); u = ueff/(b/a - 1); M, Om = predict(J, D, th, u, a, b); S, _ = cone(N, n, M)
    res = np.abs(rhs(S, J, D, u, a, b, Om)).max()
    return np.linalg.eigvals(jac(S, (J, D, u, a, b, Om))).real.max(), M, res
def thresh(N, n, a, b):
    th, D = setup(N, n); lo, hi = 0.6*D, 0.9*D
    for _ in range(32):
        mid = (lo + hi)/2
        if growth(N, n, mid, a, b)[0] > 1e-10: hi = mid
        else: lo = mid
    return (lo + hi)/2
out = {}
N, n = 200, 3; th, D = setup(N, n); uL = D/np.sqrt(2)
sets = [(0.05, 0.15), (0.1, 0.3), (0.2, 0.5), (0.1, -0.1), (0.3, 0.0)]
out['N200'] = dict(D=D, uL_cont=uL, uunw_lat=2*(np.sqrt(J*J + D*D) - J)/np.sin(th), sets=[])
for a, b in sets:
    t = thresh(N, n, a, b); xs = np.linspace(0.85, 1.15, 13)*t
    g = [growth(N, n, x, a, b) for x in xs]
    out['N200']['sets'].append(dict(alpha=a, beta=b, ueff_star=t, u_star=t/(b/a - 1), ueff=xs.tolist(),
                                    growth=[x[0] for x in g], M=[x[1] for x in g], resid=max(x[2] for x in g)))
    print(a, b, t, max(x[2] for x in g), flush=True)
out['finite'] = []
for N, n in [(200, 3), (400, 6), (800, 12)]:
    th, D = setup(N, n); t = thresh(N, n, 0.1, 0.3); kmin = 2*np.pi/N
    out['finite'].append(dict(N=N, D=D, ueff_star=t, pred=np.sqrt((D*D + kmin**2)/2), cont=D/np.sqrt(2)))
    print(N, t, np.sqrt((D*D + kmin**2)/2), flush=True)
json.dump(out, open(os.path.join(DATA, 'fig4_helix.json'), 'w'), indent=1)
