import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hyst'))
from helix import predict, cone, jac
# Linear (Landau) threshold u_L(n) and unwinding threshold u_unw(n) of the driven cone for winding n in a ring of N sites,
# D = J tan(2 pi 3/200) (same pitch family as N = 200, n = 3).  usage: python thr_N.py N n1,n2,... out.json
J = 1.0; a, b = 0.3, 0.6; D = J*np.tan(2*np.pi*3/200)
N = int(sys.argv[1]); ns = [int(x) for x in sys.argv[2].split(',')]; out = sys.argv[3]
res = {}
for n in ns:
    th = 2*np.pi*n/N; G = J*np.cos(th)+D*np.sin(th); uunw = 2*(G-J)/np.sin(th) if n > 0 else 2*D
    def g(ue):
        u = ue/(b/a-1); M, Om = predict(J, D, th, u, a, b)
        if abs(M) >= 1: return 1.0
        S, _ = cone(N, n, M); return np.linalg.eigvals(jac(S, (J, D, u, a, b, Om))).real.max()
    lo, hi = 0.3*uunw, 0.999*uunw
    if g(hi) <= 1e-10: res[n] = dict(u_L=None, u_unw=uunw); print(N, n, 'stable to unwinding', uunw, flush=True); continue
    for _ in range(16):
        mid = (lo+hi)/2
        if g(mid) > 1e-10: hi = mid
        else: lo = mid
    res[n] = dict(u_L=hi, u_unw=uunw); print(f"N={N} n={n}: u_L={hi:.5f} u_unw={uunw:.5f}", flush=True)
    json.dump(res, open(out, 'w'), indent=1)
