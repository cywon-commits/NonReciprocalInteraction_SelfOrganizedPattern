import sys, os, json, time, argparse, collections, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
os.environ.setdefault('SKX_CELL', os.path.join(root, 'states', 'S_B0.5.npy'))
sys.path.insert(0, here); sys.path.insert(1, os.path.join(root, 'code', 'kt'))
from fastllg import Integrator
from orderpar import order, energy
from strain import tri_strain
from regist import topo
# Request 2, tasks B-D: lab-frame driven SkX on an nx x ny tiling of the 19x33 cell at fixed u_eff (T = 0).
# Topological registry via Bragg phases (regist.topo: Burgers circuits and box-cycle Bragg-index change m); the maximum
# local shear gamma_max = max |du_y/dx| over the triangles. A slip = first record where the registry is lost;
# gamma* = gamma_max of the last intact record. A ring buffer of states before the slip is saved.
# Options: start = perfect + noise | file | scaled seed (seed.py), current sign (--usign), field sign (--bsign).
p = argparse.ArgumentParser()
p.add_argument('--nx', type=int, required=True); p.add_argument('--ny', type=int, required=True)
p.add_argument('--u', type=float, required=True); p.add_argument('--hold', type=float, required=True)
p.add_argument('--tag', required=True); p.add_argument('--out', required=True)
p.add_argument('--start', default=None); p.add_argument('--noise0', type=float, default=1e-3); p.add_argument('--seed', type=int, default=0)
p.add_argument('--usign', type=int, default=1); p.add_argument('--bsign', type=int, default=1)
p.add_argument('--every', type=float, default=200.0); p.add_argument('--dt', type=float, default=0.2)
p.add_argument('--after_slip', type=float, default=20000.0, help='keep running this long after a slip, then stop')
p.add_argument('--ring', type=int, default=20); p.add_argument('--ckpt', type=float, default=20000.0)
a_ = p.parse_args()
J, D = 1.0, 0.4; B = a_.bsign*0.5*D*D/J; a, b = 0.3, 0.6
nx, ny = a_.nx, a_.ny; Lx, Ly = 19*nx, 33*ny; HALF = 9.5
S0 = np.load(os.environ['SKX_CELL'])
if a_.start: S = np.load(a_.start).astype(float)
else:
    S = np.tile(S0, (nx, ny, 1)); S = S + a_.noise0*np.random.default_rng(3+a_.seed).standard_normal(S.shape)
if a_.bsign < 0: S = -S   # S -> -S with B -> -B: same energy, reversed skyrmion charge
S /= np.linalg.norm(S, axis=-1)[..., None]
assert S.shape[:2] == (Lx, Ly)
def grad(S): return (np.roll(S, -1, 0)-np.roll(S, 1, 0))/2, (np.roll(S, -1, 1)-np.roll(S, 1, 1))/2
def obs(S):
    Sp = S.copy()
    if a_.bsign < 0: Sp = -Sp                                           # analyse the mirrored texture with the B > 0 tools
    try:
        r = topo(Sp); intact = r['intact']; k = 2*np.pi/Lx; X, d = r['X'], r['u']
        st = tri_strain(X, d, Lx, Ly); F = lambda v: 2*abs(np.mean(v*np.exp(-1j*k*X[:, 0])))
        gam = float(np.abs(st[:, 6]).max()); sh = float(st[:, 4].max()); Ux, Uy = F(d[:, 0]), F(d[:, 1])
        sgn = float(np.sign(st[np.argmax(np.abs(st[:, 6])), 6])); um = float(np.abs(d).max()); nd = r['ndef']; m = np.array(r['m']).ravel().tolist()
    except Exception:
        intact, um, gam, sh, Ux, Uy, sgn, nd, m = False, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, -1, [np.nan]*4
    try: N, P6, _, fd, _ = order(Sp)
    except Exception: N, P6, fd = -1, np.nan, np.nan
    return [float(intact), um, gam, sh, Ux, Uy, sgn, N, P6, fd, nd, *m]
# record columns: 0 t, 1 intact (Burgers circuits all zero and Bragg-index change m = 0), 2 max|u|, 3 gamma_max = max|du_y/dx|,
# 4 max symmetric shear, 5 U_x(k1), 6 U_y(k1), 7 sign of du_y/dx at the max, 8 N_sk, 9 |psi6|, 10 coordination-defect fraction,
# 11 Burgers-defect triangles, 12-15 m (x-cycle a1, a2, y-cycle a1, a2), 16 Vx, 17 Vy, 18 energy
u = a_.usign*a_.u/(b/a-1)
ck = f"{a_.out}/ckpt_{a_.tag}.npz"; t = 0.0; rec = []; slip_t = None
os.makedirs(a_.out, exist_ok=True)
if os.path.exists(ck):
    z = np.load(ck); S = z['S']; rec = z['rec'].tolist(); t = float(z['t']); slip_t = float(z['slip_t']) if z['slip_t'] >= 0 else None
    print(f"resumed at t={t:.0f}", flush=True)
else:
    S = S + 1e-6*np.random.default_rng(int(a_.u*1e5)+7919*a_.seed).standard_normal(S.shape); S /= np.linalg.norm(S, axis=-1)[..., None]
I = Integrator(S, J, D, B, a, np.array([-u, 0.]), np.array([b*u, 0.]), 0.0, a_.dt, seed=a_.seed)
ring = collections.deque(maxlen=a_.ring); nsub = int(round(a_.every/a_.dt)); t0 = time.time()
while t < a_.hold-1e-9:
    S1 = I.S.copy(); I.run(5); dSdt = (I.S-S1)/(5*a_.dt)
    gx, gy = grad(I.S); M = np.array([[np.sum(gx*gx), np.sum(gx*gy)], [np.sum(gx*gy), np.sum(gy*gy)]])
    V = -np.linalg.solve(M, [np.sum(dSdt*gx), np.sum(dSdt*gy)])
    I.run(nsub-5); t += a_.every
    o = obs(I.S); rec.append([t, *o, V[0], V[1], energy(I.S, J, D, B)])
    if slip_t is None:
        ring.append((t, I.S.astype(np.float32).copy()))
        if o[0] == 0.0 and len(rec) > 1 and rec[-2][1] == 0.0:   # registry lost in two consecutive records
            slip_t = rec[-2][0]; np.savez_compressed(f"{a_.out}/slip_{a_.tag}.npz", t=np.array([x[0] for x in ring]), S=np.array([x[1] for x in ring]))
    if slip_t is not None and t >= slip_t+a_.after_slip: break
    if int(round(t/a_.every)) % int(round(a_.ckpt/a_.every)) == 0:
        np.savez(ck+'.tmp.npz', S=I.S, rec=np.array(rec), t=t, slip_t=-1.0 if slip_t is None else slip_t); os.replace(ck+'.tmp.npz', ck)
rec = np.array(rec)
np.save(f"{a_.out}/rec_{a_.tag}.npy", rec); np.save(f"{a_.out}/S_{a_.tag}.npy", I.S.astype(np.float32))
okm = (rec[:, 1] > 0.5) & (rec[:, 0] < (slip_t if slip_t is not None else np.inf))
gstar = float(rec[okm, 3][-1]) if (slip_t is not None and okm.any()) else None     # last intact record before the slip
q = rec[len(rec)*3//4:]
summ = dict(nx=nx, ny=ny, u=a_.u, usign=a_.usign, bsign=a_.bsign, t_end=float(t), slipped=slip_t is not None, t_slip=slip_t, gamma_star=gstar,
            gamma_end=float(rec[-1, 3]), gamma_lastQ=float(np.nanmean(q[:, 3])), gamma_lastQ_std=float(np.nanstd(q[:, 3])), intact_end=bool(rec[-1, 1] > 0.5),
            Ux=float(np.nanmean(q[:, 5])), Uy=float(np.nanmean(q[:, 6])), Nsk_end=int(rec[-1, 8]), fdef_end=float(rec[-1, 10]), psi6_end=float(rec[-1, 9]),
            ndef_end=int(rec[-1, 11]), m_end=rec[-1, 12:16].tolist(), Vx=float(np.mean(q[:, 16])), Vy=float(np.mean(q[:, 17])), wall=time.time()-t0)
json.dump(summ, open(f"{a_.out}/sum_{a_.tag}.json", 'w'), indent=1)
if os.path.exists(ck): os.remove(ck)
print(json.dumps(summ), flush=True)
