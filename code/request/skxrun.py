import sys, os, json, time, argparse, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'kt'))
from fastllg import Integrator
from orderpar import order, energy
# T = 0 (or T > 0) lab-frame driven SkX on an nx x ny tiling of the 19x33 cell (2 skyrmions per cell).
# A sequence of u_eff steps; each step re-injects 1e-6 noise and is held for its own time (hold list, or
# clip(10/|Gamma(u)|, tmin, tmax) from a Bloch growth table), with early stop once A is steady.
# Records per 'every' time units: t, A, Vx, Vy, Q, |psi6|, defect fraction, N_sk, energy, <S_y>.
p = argparse.ArgumentParser()
p.add_argument('--nx', type=int, required=True); p.add_argument('--ny', type=int, required=True)
p.add_argument('--us', required=True); p.add_argument('--tag', required=True)
p.add_argument('--holds', default=None, help='comma list, one per u')
p.add_argument('--gamma', default=None, help='json: list of [u_eff, Gamma] for automatic holds')
p.add_argument('--tmin', type=float, default=2e4); p.add_argument('--tmax', type=float, default=6e5)
p.add_argument('--start', default=None); p.add_argument('--noise0', type=float, default=1e-3)
p.add_argument('--every', type=float, default=50.0); p.add_argument('--dt', type=float, default=0.2)
p.add_argument('--T', type=float, default=0.0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--nostop', action='store_true'); p.add_argument('--out', default='.'); p.add_argument('--decay', type=float, default=1e-9)
p.add_argument('--ckpt', type=float, default=0, help='checkpoint every this many time units (resume from it on restart)')
args = p.parse_args()
J, D = 1.0, 0.4; B = 0.5*D*D/J; a, b = 0.3, 0.6
nx, ny = args.nx, args.ny; us = [float(x) for x in args.us.split(',')]
S0 = np.load(os.path.join(here, '..', '..', 'states', 'S_B0.5.npy'))
if args.start: S = np.load(args.start).astype(float)
else: S = np.tile(S0, (nx, ny, 1)) + args.noise0*np.random.default_rng(3+args.seed).standard_normal((19*nx, 33*ny, 3))
S /= np.linalg.norm(S, axis=-1)[..., None]
Lx, Ly = S.shape[:2]; assert (Lx, Ly) == (19*nx, 33*ny)
# Bragg points of the 19x33 cell in supercell FFT indices; reciprocal lattice of the cell = (kx%nx==0)&(ky%ny==0)
Gs = [(g0*nx, g1*ny) for g0, g1 in [(0, 2), (1, -1), (-1, -1), (0, -2), (-1, 1), (1, 1)]]
KX, KY = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing='ij'); offRL = ~((KX % nx == 0) & (KY % ny == 0))
def amp(S):
    P = np.abs(np.fft.fft2(S[..., 2]))**2
    return np.sqrt(P[offRL].sum()/sum(P[g0 % Lx, g1 % Ly] for g0, g1 in Gs))
def grad(S): return (np.roll(S, -1, 0)-np.roll(S, 1, 0))/2, (np.roll(S, -1, 1)-np.roll(S, 1, 1))/2
def topo(S):
    gx, gy = grad(S); return float(np.sum(S*np.cross(gx, gy))/(4*np.pi))
def ordp(S):
    try:
        N, P6, _, fd, _ = order(S); return N, P6, fd
    except Exception: return -1, np.nan, np.nan
if args.holds: holds = [float(x) for x in args.holds.split(',')]; holds = holds*len(us) if len(holds) == 1 else holds
elif args.gamma:
    tab = np.array(json.load(open(args.gamma))); tab = tab[np.argsort(tab[:, 0])]
    holds = [float(np.clip(10/max(abs(np.interp(u, tab[:, 0], tab[:, 1])), 1e-12), args.tmin, args.tmax)) for u in us]
else: holds = [args.tmin]*len(us)
os.makedirs(args.out, exist_ok=True)
nsub = int(round(args.every/args.dt)); t0 = time.time()
for u_eff, hold in zip(us, holds):
    u = u_eff/(b/a-1)
    S = S + 1e-6*np.random.default_rng(int(u_eff*1e5)+7919*args.seed).standard_normal(S.shape); S /= np.linalg.norm(S, axis=-1)[..., None]
    ck = f"{args.out}/ckpt_{args.tag}_{u_eff:.5f}.npz"; t = 0.0; rec = []; stop = ''
    if args.ckpt and os.path.exists(ck):
        z = np.load(ck); S = z['S']; rec = z['rec'].tolist(); t = float(z['t']); print(f"resumed {ck} at t={t:.0f}", flush=True)
    I = Integrator(S, J, D, B, a, np.array([-u, 0.]), np.array([b*u, 0.]), args.T, args.dt, seed=int(u_eff*1e5)+args.seed+int(t))
    while t < hold-1e-9:
        S1 = I.S.copy(); I.run(5); dSdt = (I.S-S1)/(5*args.dt)
        gx, gy = grad(I.S); M = np.array([[np.sum(gx*gx), np.sum(gx*gy)], [np.sum(gx*gy), np.sum(gy*gy)]])
        V = -np.linalg.solve(M, [np.sum(dSdt*gx), np.sum(dSdt*gy)])
        I.run(nsub-5); t += args.every
        N, P6, fd = ordp(I.S)
        rec.append([t, amp(I.S), V[0], V[1], topo(I.S), P6, fd, N, energy(I.S, J, D, B), float(I.S[..., 1].mean())])
        if args.ckpt and int(round(t/args.every)) % int(round(args.ckpt/args.every)) == 0:
            np.savez(ck+'.tmp.npz', S=I.S, rec=np.array(rec), t=t); os.replace(ck+'.tmp.npz', ck)
        if not args.nostop and t >= args.tmin and len(rec) >= 40:
            A = np.array(rec)[:, 1]; w = max(20, len(rec)//10); l1, l2 = A[-w:], A[-2*w:-w]
            if A[-1] < args.decay: stop = 'decayed'; break
            if A[-1] > 1e-3 and abs(l1.mean()/l2.mean()-1) < 5e-3 and l1.std() < 5e-3*l1.mean(): stop = 'steady'; break
    rec = np.array(rec); S = I.S.copy()
    if args.ckpt and os.path.exists(ck): os.remove(ck)
    np.save(f"{args.out}/rec_{args.tag}_{u_eff:.5f}.npy", rec); np.save(f"{args.out}/S_{args.tag}_{u_eff:.5f}.npy", S.astype(np.float32))
    np.save(f"{args.out}/S_{args.tag}_last.npy", S)
    q = rec[len(rec)*3//4:]
    print(f"u_eff={u_eff:.5f} hold={t:.0f}/{hold:.0f}{'('+stop+')' if stop else ''} A_end={rec[-1,1]:.3e} A_lastQ={q[:,1].mean():.3e} A_max={rec[:,1].max():.3e} "
          f"psi6={q[:,5].mean():.3f} fdef={q[:,6].mean():.3f} fdef_end={rec[-1,6]:.3f} Nsk={int(rec[-1,7])} Q={rec[-1,4]:.3f} "
          f"Vx={q[:,2].mean():.5f} Vy={q[:,3].mean():.5f} E={q[:,8].mean():.2f} wall={time.time()-t0:.0f}", flush=True)
