import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..'); sys.path.insert(0, here); sys.path.insert(1, os.path.join(root, 'code', 'request2'))
from twist import TwistModel, find_V, link_blocks
from sparse_bloch import soft_eigs
# Bloch threshold of the driven lattice in an nx x 1 box stretched along x to Lx (external strain eps0 = Lx/(19 nx) - 1),
# alpha = 0.3, beta = 2 alpha. The k1 strain-wave mode of the box (wavelength Lx along x) is a k = 0 mode of the box, so the
# eigenvalues at k = 0 are scanned (unstable when max Re lambda > 1e-7: translation Goldstone and near-neutral modes stay below); bisection in u.
# usage: python bloch_uc_s.py nx Lx out.json
J, D = 1.0, 0.4; B = 0.5*D*D/J; a = 0.3; b = 2*a; nx, Lx, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; Lx0 = 19*nx; Ly = 33
S0 = np.tile(np.load(os.path.join(root, 'states', 'S_B0.5.npy')), (nx, 1, 1))
if Lx != Lx0:
    xs = np.arange(Lx)*Lx0/Lx; x0 = np.floor(xs).astype(int); f = (xs-x0)[:, None, None]
    S0 = (1-f)*S0[x0 % Lx0]+f*S0[(x0+1) % Lx0]; S0 /= np.linalg.norm(S0, axis=-1)[..., None]
m = TwistModel(Lx, Ly, 0, J, D, B); S0 = S0.reshape(-1, 3)
def g(ue):
    S, V, w1, w2, r = find_V(m, S0.copy(), np.array([ue, 0.]), a, b)
    bl, _ = link_blocks(m, S, w1, w2, a); ev = soft_eigs(m, bl, (0., 0.), nev=10, sigma=1e-5)
    return float(ev.real.max())
guess = 7.72/Lx; lo, hi = 0.93*guess, 1.07*guess; pts = []
for _ in range(9):
    mid = 0.5*(lo+hi); gm = g(mid); pts.append((mid, gm)); print(nx, Lx, f'u={mid:.6f} maxRe={gm:+.3e}', flush=True)
    if gm > 1e-7: hi = mid              # Goldstone and near-neutral modes (Re lambda ~ 1e-9 .. 1e-8) do not count
    else: lo = mid
json.dump(dict(nx=nx, Lx=Lx, eps0=Lx/Lx0-1, u_c_bloch=0.5*(lo+hi), ucLx=0.5*(lo+hi)*Lx, points=pts), open(out, 'w'), indent=1)
print('RESULT', nx, Lx, 0.5*(lo+hi), 0.5*(lo+hi)*Lx)
