import sys, os, json, glob, argparse, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
os.environ.setdefault('SKX_CELL', os.path.join(root, 'states', 'S_B0.5.npy'))
sys.path.insert(0, os.path.join(root, 'code', 'kt')); sys.path.insert(1, os.path.join(root, 'code', 'request2')); sys.path.insert(2, os.path.join(root, 'code', 'request3'))
from fastllg import Integrator
from regist import topo
from chain_tools import profile, critical
# Task C: saturated strain-wave states at given u/u_c without the soft-mode noise of the noisy chains.
# The saturated state is uniform along y up to the lattice period 33 (the strain wave has k_y = 0), so every perturbation
# with another k_y (the soft band has |k_y| = 0.05-0.1) is removed by averaging the spins over the y translations by 33 j
# ("cleaning"). Chain over the requested u/u_c: start from the request-3 state of the first value, then each value is
# integrated deterministically (T = 0) for --hold, cleaning every --clean time units; the state and its gamma_wave are saved.
p = argparse.ArgumentParser()
p.add_argument('--start', required=True); p.add_argument('--uc', type=float, required=True); p.add_argument('--rs', required=True)
p.add_argument('--hold', type=float, required=True); p.add_argument('--clean', type=float, default=1e4); p.add_argument('--alpha', type=float, default=0.3)
p.add_argument('--dt', type=float, default=0.2); p.add_argument('--out', required=True)
A = p.parse_args(); J, D = 1.0, 0.4; B = 0.5*D*D/J; a = A.alpha; b = 2*a; os.makedirs(A.out, exist_ok=True)
def clean(S):
    Ly = S.shape[1]; n = Ly//33; M = np.mean([np.roll(S, -33*j, 1) for j in range(n)], 0); M /= np.linalg.norm(M, axis=-1)[..., None]
    return np.tile(M[:, :33], (1, n, 1))
S = clean(np.load(A.start).astype(float)); Lx, Ly = S.shape[:2]
for r in [float(x) for x in A.rs.split(',')]:
    f = f'{A.out}/S_r{r:.4f}.npy'
    if os.path.exists(f): S = np.load(f).astype(float); continue
    u = r*A.uc; I = Integrator(S, J, D, B, a, np.array([-u, 0.]), np.array([b*u, 0.]), 0.0, A.dt); t = 0.0; gw = []
    while t < A.hold-1e-9:
        I.run(int(round(A.clean/A.dt))); t += A.clean; I.S[...] = clean(I.S)
        tp = topo(I.S); xs, pr = profile(tp['X'], tp['u'], Lx, Ly); gw.append((t, critical(xs, pr, Lx)['gpeak'], tp['intact']))
    S = I.S.copy(); np.save(f, S.astype(np.float32))
    g = np.array([x[1] for x in gw]); n = len(g)
    json.dump(dict(r=r, u=u, hold=A.hold, gamma_wave=float(g[-1]), drift_last_quarter=float(g[-1]-g[3*n//4]) if n > 4 else None, intact=bool(gw[-1][2]), series=gw),
              open(f.replace('.npy', '.json'), 'w'))
    print(f'r={r} gamma_wave={g[-1]:.4f} change over last quarter={g[-1]-g[3*n//4] if n > 4 else 0:+.4f} intact={gw[-1][2]}', flush=True)
