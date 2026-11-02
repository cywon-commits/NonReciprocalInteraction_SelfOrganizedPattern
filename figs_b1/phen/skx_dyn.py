"""Phenomenon data: a driven skyrmion lattice below and above the Doppler threshold, co-moving frame (exact Thiele V of the
19x33 cell), 76x66 box (16 skyrmions), B = 0.5 D^2/J, alpha = 0.3, beta = 0.6 (u_eff = u), T = 0, noise 1e-3 at t = 0.
Saves m_z snapshots and the satellite amplitude |S_z(k = 2pi/76 x)| to results/b1_figs/phen_skx_u{u}.npz."""
import sys, os, numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.join(R, 'code', 'kt')); sys.path.insert(0, os.path.join(R, 'code', 'skx'))
from fastllg import Integrator
from skx import GD
u = float(sys.argv[1]); tmax = float(sys.argv[2]); every = float(sys.argv[3])
J, D, a, b = 1.0, 0.4, 0.3, 0.6; B = 0.5*D*D
S0 = np.load(os.path.join(R, 'states', 'S_B0.5.npy')); Gm, Dm, _ = GD(S0); uv = np.array([u, 0.])
V = np.linalg.solve(Gm - a*Dm, (Gm - b*Dm) @ uv); w1 = V - uv; w2 = b*uv - a*V
S = np.tile(S0, (4, 2, 1)); S = S + 1e-3*np.random.default_rng(1).standard_normal(S.shape); S /= np.linalg.norm(S, axis=-1)[..., None]
dt = 0.1; I = Integrator(S, J, D, B, a, w1, w2, 0.0, dt); Lx, Ly = S.shape[:2]
ts, mz, sat, bragg = [], [], [], []
x = np.arange(Lx)[:, None]
def obs(S):
    f = np.abs(np.fft.fft2(S[..., 2]))/(Lx*Ly); return f[1, 0], f[4, 2] if True else 0
t = 0.0
while t <= tmax + 1e-9:
    ts.append(t); mz.append(I.S[..., 2].astype(np.float16)); s1, br = obs(I.S); sat.append(s1); bragg.append(br)
    print(f'u={u} t={t:.0f} sat={s1:.2e} bragg={br:.3e}', flush=True)
    I.run(int(round(every/dt))); t += every
np.savez_compressed(os.path.join(R, 'results', 'b1_figs', f'phen_skx_u{u}.npz'), t=np.array(ts), mz=np.array(mz), sat=np.array(sat), bragg=np.array(bragg), V=V, w1=w1, w2=w2)
