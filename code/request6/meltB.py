import sys, os, time, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
sys.path.insert(0, os.path.join(root, 'code', 'kt')); sys.path.insert(1, here)
from fastllg import Integrator
from melt import sk_centers
from anaA6 import snapshot, defects
# Request 6, task B: finite-temperature melting of the SkX below the Doppler threshold (B = 0.5 D^2/J, alpha = 0.3, beta = 0.6,
# u_eff = u along x, lab frame). Heating ramp T = Tlo ... Thi and cooling back in steps dT, each step held `hold` time units
# (first 40 % discarded); every 100 time units: Psi6 = |<psi6>|, local |psi6|, defect fractions (anaA6.defects), translational
# order Psi_T = mean over the three G (directions from the global psi6 phase, |G| from the density) of |<exp(i G.r)>|, N.
# Heun dt = 0.15 (as code/kt). Restart-safe: checkpoint after every step. Optional 10th argument 'heat': heating leg only
# (extra seeds for the heating T_m).
J, D = 1.0, 0.4; B = 0.5*D*D; a, b = 0.3, 0.6; dt = 0.15
ue, nx, ny = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); Tlo, Thi, dT, hold = map(float, sys.argv[4:8]); seed = int(sys.argv[8]); out = sys.argv[9]
u = ue/(b/a-1); w1 = np.array([-u, 0.]); w2 = np.array([b*u, 0.]); tag = f'u{ue}_{nx}x{ny}_s{seed}'+('_heat' if len(sys.argv) > 10 and sys.argv[10] == 'heat' else ''); ck = f'{out}/ckpt_{tag}.npz'; fo = f'{out}/melt_{tag}.json'
up = np.round(np.arange(Tlo, Thi+1e-9, dT), 4); heat_only = len(sys.argv) > 10 and sys.argv[10] == 'heat'
Ts = list(up) if heat_only else list(up)+list(up[::-1][1:]); Lx, Ly = 19*nx, 33*ny
if os.path.exists(ck):
    z = np.load(ck); S = z['S'].astype(float); done = int(z['done']); res = json.load(open(fo))['steps']
else:
    S = np.tile(np.load(os.path.join(root, 'states', 'S_B0.5.npy')), (nx, ny, 1)); done = 0; res = []
S /= np.linalg.norm(S, axis=-1)[..., None]
def measure(S):
    C = sk_centers(S); C = C[C[:, 2] >= 8][:, :2]; N = len(C); z, psi, ro, co = snapshot(C, Lx, Ly); d = defects(z, ro, co)
    a0 = np.sqrt(2*Lx*Ly/(np.sqrt(3)*N)); G0 = 4*np.pi/(np.sqrt(3)*a0); t6 = np.angle(psi.sum())/6
    PT = np.mean([abs(np.exp(1j*(C@(G0*np.array([np.cos(t6+np.radians(30+60*k)), np.sin(t6+np.radians(30+60*k))])))).mean()) for k in range(3)])
    return [N, abs(psi.mean()), np.abs(psi).mean(), d['f_def'], d['disc'], d['disl'], d['bound'], PT]
t0 = time.time()
for i in range(done, len(Ts)):
    T = float(Ts[i]); I = Integrator(S, J, D, B, a, w1, w2, T, dt, seed=1000*seed+i); rec = []
    for k in range(int(hold/100)):
        I.run(int(round(100/dt))); rec.append(measure(I.S))
    S = I.S; R = np.array(rec[int(0.4*len(rec)):]); m = R.mean(0); e = R.std(0)/np.sqrt(len(R)/5)
    res.append(dict(T=T, leg='heat' if i < len(up) else 'cool', N=m[0], Psi6=m[1], Psi6_err=e[1], psi6_loc=m[2], f_def=m[3], f_def_err=e[3], disc=m[4], disl=m[5], bound=m[6],
                    PsiT=m[7], PsiT_err=e[7], series=np.array(rec)[:, [1, 3, 7]].tolist()))
    json.dump(dict(ue=ue, nx=nx, ny=ny, Lx=Lx, Ly=Ly, hold=hold, dt=dt, seed=seed, steps=res), open(fo, 'w'))
    np.savez(ck+'.tmp.npz', S=S.astype(np.float32), done=i+1); os.replace(ck+'.tmp.npz', ck)
    print(f"{res[-1]['leg']} T={T:.3f}: N={m[0]:.1f} Psi6={m[1]:.3f} f_def={m[3]:.3f} PsiT={m[7]:.3f} wall={time.time()-t0:.0f}", flush=True)
print('done', flush=True)
