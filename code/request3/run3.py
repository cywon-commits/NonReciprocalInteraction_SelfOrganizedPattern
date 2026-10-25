import sys, os, json, time, argparse, collections, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
os.environ.setdefault('SKX_CELL', os.path.join(root, 'states', 'S_B0.5.npy'))
sys.path.insert(0, os.path.join(root, 'code', 'request2')); sys.path.insert(1, os.path.join(root, 'code', 'kt'))
from fastllg import Integrator
from orderpar import energy
from strain import tri_strain
from regist import topo
from scipy.spatial import cKDTree
# Request 3: quasi-static chain of u_eff steps for the driven SkX on an nx x ny tiling of the 19x33 cell (T = 0, lab frame,
# damping alpha, beta = 2 alpha so that u = u_eff). Restart safe: every finished step writes sum_<r>.json, the running
# step checkpoints. Per record (every 'every' time units):
#   registry (regist.topo): intact flag, number of Burgers-defect triangles, Bragg-index change m;
#   three strain measures from the triangle displacement gradients du_y/dx (reference = lattice) coordinates:
#     gmax  = max over triangles of |du_y/dx|,
#     gbar  = max over skyrmion sites of the disk average (radius R = 37.5 sites ~ soft-mode wavelength / 2) of du_y/dx,
#     gwave = max of the 1D profile of du_y/dx along x (Fourier harmonics 0..3 of 2 pi / Lx, least squares);
#   U_x, U_y of the k1 mode, x position of the gbar maximum, mean drift V.
# Step equilibrium: after hold, the means of gmax in the last two quarters must differ by < 1 %; otherwise the step is
# extended in chunks up to hold_max. The first record with a Burgers defect or m != 0 is the event; the event is permanent
# if the registry is still broken after_slip later. The chain stops after a permanent slip.
# Displacement trajectories (reference positions X and u per skyrmion) are kept every 'traj_every' time units, and for the
# last 'ring' records before an event.
p = argparse.ArgumentParser()
p.add_argument('--nx', type=int, required=True); p.add_argument('--ny', type=int, required=True)
p.add_argument('--uc', type=float, required=True); p.add_argument('--rs', required=True, help='comma list of u/u_c')
p.add_argument('--hold', type=float, required=True); p.add_argument('--hold_max', type=float, default=None)
p.add_argument('--alpha', type=float, default=0.3); p.add_argument('--start', required=True); p.add_argument('--noise_start', type=float, default=0.0)
p.add_argument('--seed', type=int, default=0); p.add_argument('--out', required=True); p.add_argument('--every', type=float, default=100.0)
p.add_argument('--traj_every', type=float, default=500.0); p.add_argument('--ring', type=int, default=40)
p.add_argument('--after_slip', type=float, default=20000.0); p.add_argument('--ckpt', type=float, default=10000.0); p.add_argument('--dt', type=float, default=0.2)
A = p.parse_args()
J, D = 1.0, 0.4; B = 0.5*D*D/J; a = A.alpha; b = 2*a
nx, ny = A.nx, A.ny; Lx, Ly = 19*nx, 33*ny; R = 37.5; hold_max = A.hold_max or A.hold
os.makedirs(A.out, exist_ok=True)
def grad(S): return (np.roll(S, -1, 0)-np.roll(S, 1, 0))/2, (np.roll(S, -1, 1)-np.roll(S, 1, 1))/2
def measures(S):
    r = topo(S); X, d = r['X'], r['u']; st = tri_strain(X, d, Lx, Ly); g = st[:, 6]
    # triangle centroids in reference coordinates (same construction as tri_strain)
    from scipy.spatial import Delaunay
    imgs = np.concatenate([X+[i*Lx, j*Ly] for i in (-1, 0, 1) for j in (-1, 0, 1)]); T = Delaunay(imgs).simplices
    c = imgs[T].mean(1); keep = (c[:, 0] >= 0) & (c[:, 0] < Lx) & (c[:, 1] >= 0) & (c[:, 1] < Ly); c = c[keep]
    cimg = np.concatenate([c+[i*Lx, j*Ly] for i in (-1, 0, 1) for j in (-1, 0, 1)]); gimg = np.tile(g, 9)
    tree = cKDTree(cimg); Xm = X % [Lx, Ly]
    gb = np.array([gimg[tree.query_ball_point(x, R)].mean() for x in Xm]); ib = int(np.argmax(np.abs(gb)))
    k = 2*np.pi/Lx; M = np.stack([np.ones(len(c))]+[f(n*k*c[:, 0]) for n in (1, 2, 3) for f in (np.cos, np.sin)], 1)
    co = np.linalg.lstsq(M, g, rcond=None)[0]; xs = np.linspace(0, Lx, 400, endpoint=False)
    prof = np.stack([np.ones(400)]+[f(n*k*xs) for n in (1, 2, 3) for f in (np.cos, np.sin)], 1)@co; iw = int(np.argmax(np.abs(prof)))
    F = lambda v: 2*abs(np.mean(v*np.exp(-1j*k*X[:, 0])))
    return r, X, d, dict(gmax=float(np.abs(g).max()), gbar=float(np.abs(gb[ib])), gwave=float(np.abs(prof[iw])), xbar=float(Xm[ib, 0]),
                          xwave=float(xs[iw]), Ux=F(d[:, 0]), Uy=F(d[:, 1]))
rs = [float(x) for x in A.rs.split(',')]
S = np.load(A.start).astype(float)
if A.noise_start > 0: S = S+A.noise_start*np.random.default_rng(100+A.seed).standard_normal(S.shape)
S /= np.linalg.norm(S, axis=-1)[..., None]; assert S.shape[:2] == (Lx, Ly)
nsub = int(round(A.every/A.dt)); COLS = ['t', 'intact', 'ndef', 'm1', 'm2', 'm3', 'm4', 'gmax', 'gbar', 'gwave', 'xbar', 'xwave', 'Ux', 'Uy', 'Vx', 'Vy', 'E']
for r_ in rs:
    tag = f'r{r_:.4f}'; fs = f'{A.out}/sum_{tag}.json'
    if os.path.exists(fs):
        d = json.load(open(fs)); S = np.load(f'{A.out}/S_{tag}.npy').astype(float)
        if d['permanent']: break
        continue
    u = r_*A.uc; ck = f'{A.out}/ckpt_{tag}.npz'; t = 0.0; rec = []; traj = []; ring = collections.deque(maxlen=A.ring); ev_t = None; target = A.hold
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True); S = z['S']; rec = z['rec'].tolist(); t = float(z['t']); target = float(z['target'])
        ev_t = None if z['ev_t'] < 0 else float(z['ev_t']); traj = list(z['traj']); print(f'{tag}: resumed at t={t:.0f}', flush=True)
    else:
        S = S+1e-6*np.random.default_rng(int(r_*1e5)+7919*A.seed).standard_normal(S.shape); S /= np.linalg.norm(S, axis=-1)[..., None]
    I = Integrator(S, J, D, B, a, np.array([-u, 0.]), np.array([b*u, 0.]), 0.0, A.dt, seed=A.seed+int(t)); t0 = time.time()
    while True:
        S1 = I.S.copy(); I.run(5); dSdt = (I.S-S1)/(5*A.dt); gx, gy = grad(I.S)
        Mm = np.array([[np.sum(gx*gx), np.sum(gx*gy)], [np.sum(gx*gy), np.sum(gy*gy)]]); V = -np.linalg.solve(Mm, [np.sum(dSdt*gx), np.sum(dSdt*gy)])
        I.run(nsub-5); t += A.every
        try:
            r, X, d, ms = measures(I.S); intact = r['intact']; nd = r['ndef']; m = list(np.array(r['m']).ravel())
        except Exception:
            intact, nd, m, X, d = False, -1, [np.nan]*4, None, None; ms = dict(gmax=np.nan, gbar=np.nan, gwave=np.nan, xbar=np.nan, xwave=np.nan, Ux=np.nan, Uy=np.nan)
        rec.append([t, float(intact), nd, *m, ms['gmax'], ms['gbar'], ms['gwave'], ms['xbar'], ms['xwave'], ms['Ux'], ms['Uy'], V[0], V[1], energy(I.S, J, D, B)])
        if X is not None:
            if ev_t is None: ring.append((t, X.astype(np.float32), d.astype(np.float32)))
            if int(round(t/A.every)) % int(round(A.traj_every/A.every)) == 0: traj.append((t, X.astype(np.float32), d.astype(np.float32)))
        if ev_t is None and not intact:
            ev_t = t; np.savez_compressed(f'{A.out}/slipbuf_{tag}.npz', t=np.array([x[0] for x in ring]), X=np.array([x[1] for x in ring], dtype=object), u=np.array([x[2] for x in ring], dtype=object), S=I.S.astype(np.float32))
        if ev_t is not None and t >= ev_t+A.after_slip: break
        if ev_t is None and t >= target-1e-9:
            g = np.array(rec)[:, 7]; n = len(g); q3, q4 = np.nanmean(g[n//2:3*n//4]), np.nanmean(g[3*n//4:])
            if abs(q4/q3-1) < 0.01 or target >= hold_max-1e-9: break
            target = min(target+A.hold/2, hold_max)
        if int(round(t/A.every)) % int(round(A.ckpt/A.every)) == 0:
            np.savez(ck+'.tmp.npz', S=I.S, rec=np.array(rec), t=t, target=target, ev_t=-1.0 if ev_t is None else ev_t, traj=np.array(traj, dtype=object))
            os.replace(ck+'.tmp.npz', ck)
    rec = np.array(rec); S = I.S.copy(); np.save(f'{A.out}/S_{tag}.npy', S.astype(np.float32)); np.save(f'{A.out}/rec_{tag}.npy', rec)
    np.savez_compressed(f'{A.out}/traj_{tag}.npz', t=np.array([x[0] for x in traj]), X=np.array([x[1] for x in traj], dtype=object), u=np.array([x[2] for x in traj], dtype=object))
    g = rec[:, 7]; n = len(g); pre = rec[rec[:, 0] < (ev_t if ev_t else np.inf)]
    eq = bool(ev_t is None and abs(np.nanmean(g[3*n//4:])/np.nanmean(g[n//2:3*n//4])-1) < 0.01)
    perm = bool(ev_t is not None and rec[-1, 1] < 0.5)
    star = {f'{k}_n{nn}': (float(np.nanmean(pre[-nn:, ci])) if ev_t is not None and len(pre) >= nn else None) for k, ci in (('gmax', 7), ('gbar', 8), ('gwave', 9)) for nn in (1, 3, 5)}
    q = pre[len(pre)*3//4:] if len(pre) > 8 else pre
    d = dict(nx=nx, ny=ny, alpha=a, r=r_, u=u, t_end=float(t), equilibrated=eq, event=ev_t is not None, t_event=ev_t, permanent=perm,
             transient=bool(ev_t is not None and not perm), m_end=rec[-1, 3:7].tolist(), ndef_end=int(rec[-1, 2]),
             sat={k: float(np.nanmean(q[:, ci])) for k, ci in (('gmax', 7), ('gbar', 8), ('gwave', 9), ('Ux', 12), ('Uy', 13), ('Vx', 14), ('Vy', 15))},
             star=star, wall=time.time()-t0, cols=COLS)
    json.dump(d, open(fs, 'w'), indent=1)
    if os.path.exists(ck): os.remove(ck)
    print(f"{tag}: t={t:.0f} eq={eq} event={d['event']} perm={perm} sat gmax={d['sat']['gmax']:.3f} gbar={d['sat']['gbar']:.3f} gwave={d['sat']['gwave']:.3f} star={ {k: (round(v, 3) if v else None) for k, v in star.items() if k.endswith('n5')} }", flush=True)
    if perm: break
