import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'skx'))
from skx import Model, find_V, bloch_jac
# Bloch-Jacobian threshold u_c for an nx x ny tiling (lowest kx shells, all ky of the box), damping alpha, beta = 2 alpha.
# usage: python bloch_uc3.py nx ny alpha out.json
J, D = 1.0, 0.4; B = 0.5*D*D/J; nx, ny, a = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); b = 2*a; out = sys.argv[4]
m = Model(19, 33, J, D, B); S0 = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'states', 'S_B0.5.npy'))
ks = [(2*np.pi*mx/(19*nx), 2*np.pi*my/(33*ny)) for mx in range(1, min(nx, 2)+1) for my in range(0, min(ny, 3))]
def g(ue):
    S, V, w1, w2, r = find_V(m, S0.copy(), np.array([ue/(b/a-1), 0.]), a, b)
    return max(np.linalg.eigvals(bloch_jac(m, S, w1, w2, a, k)).real.max() for k in ks)
guess = 7.7/(19*nx); lo, hi = 0.9*guess, 1.1*guess; pts = []
for _ in range(9):
    mid = 0.5*(lo+hi); gm = g(mid); pts.append((mid, gm)); print(nx, ny, a, f'u={mid:.6f} maxRe={gm:+.3e}', flush=True)
    if gm > 0: hi = mid
    else: lo = mid
json.dump(dict(nx=nx, ny=ny, alpha=a, u_c_bloch=0.5*(lo+hi), points=pts), open(out, 'w'), indent=1)
print('RESULT', nx, ny, a, 0.5*(lo+hi))
