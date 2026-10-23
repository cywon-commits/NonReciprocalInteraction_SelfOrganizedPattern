import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'skx'))
from skx import Model, find_V, bloch_jac
# Linear threshold u_c(size) of the driven SkX: comoving stationary state of the 19x33 cell (find_V),
# Bloch Jacobian max growth over the k allowed in an (nx*19) x (ny*33) periodic supercell.
# usage: python bloch_uc.py nx ny [out.json]
J, D = 1.0, 0.4; B = 0.5*D*D/J; a, b = 0.3, 0.6
nx, ny = int(sys.argv[1]), int(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else f"results_request/A_size/uc_{nx}x{ny}.json"
m = Model(19, 33, J, D, B)
S0 = np.load(os.path.join(os.path.dirname(__file__), '..', '..', 'states', 'S_B0.5.npy'))
# allowed nonzero Bloch k of the supercell, restricted to the low shells (kx index 1..3); kx=0 has no Doppler term
ks = [(2*np.pi*mx/(19*nx), 2*np.pi*my/(33*ny)) for mx in range(1, min(nx, 3)+1) for my in range(0, ny)]
cache = {}
def growth(ue):
    if ue in cache: return cache[ue]
    u = ue/(b/a-1)
    S, V, w1, w2, r = find_V(m, S0.copy(), np.array([u, 0.]), a, b, iters=12, n=1500)
    g = []
    for k in ks:
        lam = np.linalg.eigvals(bloch_jac(m, S, w1, w2, a, k)); g.append(lam.real.max())
    g = np.array(g); cache[ue] = (g.max(), ks[int(g.argmax())], g, V, r)
    print(f"{nx}x{ny} u_eff={ue:.5f} maxRe={g.max():+.3e} at k={np.round(ks[int(g.argmax())],5)} V={np.round(V,5)} resid={r:.1e}", flush=True)
    return cache[ue]
t0 = time.time()
kmin = 2*np.pi/(19*nx); guess = 0.0994*kmin/(2*np.pi/76)
lo, hi = 0.85*guess, 1.15*guess
while growth(lo)[0] > 0: lo *= 0.85
while growth(hi)[0] < 0: hi *= 1.15
for _ in range(10):
    mid = 0.5*(lo+hi)
    if growth(mid)[0] > 0: hi = mid
    else: lo = mid
uc = 0.5*(lo+hi)
# growth-rate curve above threshold (for hold times)
curve = []
for r_ in [1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0]:
    g = growth(r_*uc); curve.append([r_*uc, g[0], list(g[1])])
res = dict(nx=nx, ny=ny, Lx=19*nx, Ly=33*ny, kmin=kmin, u_c=uc, bracket=[lo, hi], ks=ks,
           growth_curve=curve, wall=time.time()-t0)
os.makedirs(os.path.dirname(out) or ".", exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
print(f"RESULT {nx}x{ny}: u_c={uc:.5f}  (u_c*Lx={uc*19*nx:.3f})  wall={time.time()-t0:.0f}s", flush=True)
