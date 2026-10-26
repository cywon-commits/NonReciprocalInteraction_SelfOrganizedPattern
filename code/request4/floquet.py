import sys, os, json, time, argparse, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
sys.path.insert(0, os.path.join(root, 'code', 'kt'))
from fastllg import Integrator
# Task C1: noise-free linear stability of the saturated strain-wave state (Floquet / Lyapunov exponents).
# The state is periodic in the frame of the strain wave with period T = 19/v (one lattice cell passes the wave, v L_x = 12.2),
# up to the discreteness of the grid, so the exponents are measured by subspace iteration over time windows of length T:
#   q_j (unit tangent vectors, j = 1..K) -> integrate S + eps q_j and S for T with the full nonlinear Heun integrator
#   (T = 0, no noise) -> d_j = (S_j(T) - S(T))/eps projected on the tangent planes -> remove the neutral directions
#   (uniform translations dS/dx, dS/dy and the time shift dS/dt) -> QR -> ln|R_jj| accumulated.
# After the transient the averages ln|R_jj|/T are the leading exponents Gamma_j (Floquet exponents of the periodic state);
# |mu_j| = exp(Gamma_j T). Threshold: Gamma_1 crosses zero.
# Also stored: the leading vector's soft-band weight and its envelope along x (for task C3).
p = argparse.ArgumentParser()
p.add_argument('--start', required=True); p.add_argument('--uc', type=float, required=True); p.add_argument('--r', type=float, required=True)
p.add_argument('--alpha', type=float, default=0.3); p.add_argument('--dt', type=float, default=0.2); p.add_argument('--relax', type=float, default=0.0)
p.add_argument('--T', type=float, default=None); p.add_argument('--K', type=int, default=6); p.add_argument('--iters', type=int, default=30)
p.add_argument('--eps', type=float, default=1e-5); p.add_argument('--out', required=True); p.add_argument('--seed', type=int, default=0)
A = p.parse_args()
J, D = 1.0, 0.4; B = 0.5*D*D/J; a = A.alpha; b = 2*a; u = A.r*A.uc; w1 = np.array([-u, 0.]); w2 = np.array([b*u, 0.])
S = np.load(A.start).astype(float); S /= np.linalg.norm(S, axis=-1)[..., None]; Lx, Ly = S.shape[:2]
T = A.T or 19*Lx/12.2; nT = int(round(T/A.dt)); T = nT*A.dt
def integ(S0, n):
    I = Integrator(S0.copy(), J, D, B, a, w1, w2, 0.0, A.dt); I.run(n); return I.S          # copy: the integrator works in place
def tang(S0, d): return d-np.sum(d*S0, -1)[..., None]*S0
def grad(S0): return (np.roll(S0, -1, 0)-np.roll(S0, 1, 0))/2, (np.roll(S0, -1, 1)-np.roll(S0, 1, 1))/2
t0 = time.time()
if A.relax > 0: S = integ(S, int(round(A.relax/A.dt)))
rng = np.random.default_rng(A.seed); Q = [tang(S, rng.standard_normal(S.shape)) for _ in range(A.K)]
def neutral(S0):
    gx, gy = grad(S0); S1 = integ(S0, 1); st = tang(S0, (S1-S0)/A.dt)
    B_ = []
    for v in (gx, gy, st):
        v = tang(S0, v).ravel()
        for w in B_: v = v-np.dot(w, v)*w
        B_.append(v/np.linalg.norm(v))
    return B_
def orth(vs, S0):
    nb = neutral(S0); M = np.stack([v.ravel() for v in vs], 1)
    for w in nb: M = M-np.outer(w, w@M)
    q, r = np.linalg.qr(M); return [q[:, j].reshape(S0.shape) for j in range(q.shape[1])], np.diag(r)
Q, _ = orth(Q, S); logs = []; hist = []
for it in range(A.iters):
    Sn = integ(S, nT); D_ = []
    for q in Q:
        Sp = S+A.eps*q; Sp /= np.linalg.norm(Sp, axis=-1)[..., None]; D_.append(tang(Sn, (integ(Sp, nT)-Sn)/A.eps))
    Q, rd = orth(D_, Sn); S = Sn; g = np.log(np.abs(rd))/T; logs.append(g); hist.append(g.tolist())
    print(f'it={it} Gamma_top={np.round(g[:4], 7)} wall={time.time()-t0:.0f}', flush=True)
L = np.array(logs); half = L[len(L)//2:]; Gm = half.mean(0); Ge = half.std(0)/np.sqrt(len(half))
# leading vector: soft-band weight (2D FFT of the in-plane perturbation) and envelope along x
q = Q[0]; F = np.fft.fft2(q, axes=(0, 1)); P = (np.abs(F)**2).sum(-1)
kx = 2*np.pi*np.fft.fftfreq(Lx); ky = 2*np.pi*np.fft.fftfreq(Ly); KX, KY = np.meshgrid(kx, ky, indexing='ij'); kk = np.hypot(KX, KY)
band = (kk > 0.06) & (kk < 0.11) & (np.abs(KY) > 0.045); low = kk < 0.05
env = np.sqrt((q**2).sum(-1).sum(1)); env = env/env.max()
# smooth envelope (box filter of one lattice cell) and its width (fraction above half maximum)
envs = np.convolve(np.concatenate([env[-9:], env, env[:9]]), np.ones(19)/19, 'valid'); width = float((envs > 0.5*envs.max()).sum())
res = dict(start=A.start, r=A.r, u=u, alpha=a, Lx=Lx, Ly=Ly, T=T, K=A.K, iters=A.iters, relax=A.relax, Gamma=Gm.tolist(), Gamma_err=Ge.tolist(),
           mu=np.exp(Gm*T).tolist(), hist=hist, band_weight=float(P[band].sum()/P.sum()), low_weight=float(P[low].sum()/P.sum()),
           k_peak=[float(KX.ravel()[np.argmax(P)]), float(KY.ravel()[np.argmax(P)])], env=envs.tolist(), env_x_peak=float(np.argmax(envs)), env_width=width,
           wall=time.time()-t0)
os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True); json.dump(res, open(A.out, 'w'))
np.save(A.out.replace('.json', '_S.npy'), S.astype(np.float32)); np.save(A.out.replace('.json', '_q0.npy'), Q[0].astype(np.float32))
print('RESULT', A.r, 'Gamma', np.round(Gm[:4], 7), '+-', np.round(Ge[:4], 7), 'band', round(res['band_weight'], 3), 'width', width, flush=True)
