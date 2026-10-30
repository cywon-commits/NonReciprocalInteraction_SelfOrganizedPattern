import sys, os, time, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
sys.path.insert(0, os.path.join(root, 'code', 'kt'))
from fastllg import Integrator
from melt import sk_centers
# Request 6, task A: large driven SkX above threshold at T = 0 (square lattice, D = 0.4, B = 0.5 D^2/J, alpha = 0.3, beta = 0.6,
# u_eff = u (beta/alpha - 1) = u, current along x, lab frame). Box nx x ny unit cells of 19 x 33 (2 skyrmions each):
# 10x5 (N = 100, 190 x 165), 20x10 (400, 380 x 330), 30x15 (900, 570 x 495), all nearly square.
# Start: ideal lattice + 1e-2 noise (as code/kt/run_T0.py). Every 50 time units the skyrmion centres (sk_centers) are stored
# (plain float arrays, NaN padded); the spin state is checkpointed every 1000 time units (restart-safe).
J, D = 1.0, 0.4; B = 0.5*D*D; a, b = 0.3, 0.6; dt = 0.2
ue, nx, ny, tmax = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]); out = sys.argv[5]
u = ue/(b/a-1); w1 = np.array([-u, 0.]); w2 = np.array([b*u, 0.]); tag = f'u{ue}_{nx}x{ny}'
ck = f'{out}/ckpt_{tag}.npz'; fo = f'{out}/centres_{tag}.npz'
if os.path.exists(ck):
    z = np.load(ck); S = z['S'].astype(float); t = float(z['t']); T_ = list(z['times']); Cs = list(z['C'])
else:
    S = np.tile(np.load(os.path.join(root, 'states', 'S_B0.5.npy')), (nx, ny, 1)); S += 1e-2*np.random.default_rng(7).standard_normal(S.shape); t = 0.0; T_, Cs = [], []
S /= np.linalg.norm(S, axis=-1)[..., None]; I = Integrator(S, J, D, B, a, w1, w2, 0.0, dt); NMAX = 2*nx*ny+60; t0 = time.time()
while t < tmax-1e-9:
    I.run(int(round(50/dt))); t += 50
    C = sk_centers(I.S)[:, :2]; pad = np.full((NMAX, 2), np.nan); pad[:min(len(C), NMAX)] = C[:NMAX]; T_.append(t); Cs.append(pad)
    if int(round(t)) % 1000 == 0:
        np.savez(ck+'.tmp.npz', S=I.S.astype(np.float32), t=t, times=np.array(T_), C=np.array(Cs)); os.replace(ck+'.tmp.npz', ck)
        np.savez(fo, times=np.array(T_), C=np.array(Cs), Lx=19*nx, Ly=33*ny, ue=ue)
        print(f't={t:.0f} N={len(C)} wall={time.time()-t0:.0f}', flush=True)
np.savez(fo, times=np.array(T_), C=np.array(Cs), Lx=19*nx, Ly=33*ny, ue=ue); np.save(f'{out}/last_{tag}.npy', I.S.astype(np.float32)); print('done', flush=True)
