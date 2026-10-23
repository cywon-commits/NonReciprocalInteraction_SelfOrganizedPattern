import sys, os, json, time, argparse, collections, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'kt'))
from fastllg import Integrator
from orderpar import energy
# Chiral helix (cycloid in the xz plane, cone axis y) on an N x Ly lattice (Ly = 1: 1D chain; Ly > 1: strip),
# lab-frame LLG + Zhang-Li, B = 0, D = J tan(2 pi n0/N0) (n0/N0 = 3/200 by default, same pitch for all N).
# A sequence of u_eff steps (noise 1e-6 re-injected at each step), T >= 0.
# Records every 'every' time units: t, mean row winding, min/max row winding, max local pitch (raw, 5-site avg),
# <S_y>, max S_y, energy. On each winding change the preceding ring buffer of states is saved (slip anatomy).
p = argparse.ArgumentParser()
p.add_argument('--N', type=int, default=200); p.add_argument('--n0', type=int, default=3); p.add_argument('--Ly', type=int, default=1)
p.add_argument('--q0', type=float, default=2*np.pi*3/200)
p.add_argument('--us', required=True); p.add_argument('--hold', required=True, help='single value or comma list')
p.add_argument('--tag', required=True); p.add_argument('--start', default=None)
p.add_argument('--T', type=float, default=0.0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--every', type=float, default=10.0); p.add_argument('--dt', type=float, default=0.2)
p.add_argument('--ring', type=int, default=0, help='ring buffer length for slip snapshots (0 = off)')
p.add_argument('--stop_on_slip', action='store_true'); p.add_argument('--out', default='.')
p.add_argument('--norec', action='store_true')
args = p.parse_args()
J = 1.0; D = J*np.tan(args.q0); B = 0.0; a, b = 0.3, 0.6
N, Ly = args.N, args.Ly
if args.start: S = np.load(args.start).astype(float)
else:
    th = 2*np.pi*args.n0/N*np.arange(N); S = np.zeros((N, Ly, 3)); S[..., 0] = np.sin(th)[:, None]; S[..., 2] = np.cos(th)[:, None]
    S += 1e-3*np.random.default_rng(3+args.seed).standard_normal(S.shape)
S /= np.linalg.norm(S, axis=-1)[..., None]
def dphi(S):
    ph = np.arctan2(S[..., 0], S[..., 2]); return np.angle(np.exp(1j*(np.roll(ph, -1, 0)-ph)))
def winding(S): return dphi(S).sum(0)/(2*np.pi)
def pitch(S, w=5):
    d = dphi(S); r = np.hypot(S[..., 0], S[..., 2]); d = np.where((r > 0.05) & (np.roll(r, -1, 0) > 0.05), d, np.nan)
    if w > 1:
        k = np.ones(w)/w; dd = np.concatenate([d[-(w//2):], d, d[:w//2]], 0)
        sm = np.apply_along_axis(lambda c: np.convolve(c, k, 'valid'), 0, dd)
        return np.nanmax(d), np.nanmax(sm)
    return np.nanmax(d), np.nanmax(d)
us = [float(x) for x in args.us.split(',')]
hl = [float(x) for x in args.hold.split(',')]; holds = hl if len(hl) == len(us) else hl*len(us)
os.makedirs(args.out, exist_ok=True)
nsub = int(round(args.every/args.dt)); t0 = time.time(); summary = []
for u_eff, hold in zip(us, holds):
    u = u_eff/(b/a-1)
    S = S + 1e-6*np.random.default_rng(int(u_eff*1e5)+7919*args.seed).standard_normal(S.shape); S /= np.linalg.norm(S, axis=-1)[..., None]
    I = Integrator(S, J, D, B, a, np.array([-u, 0.]), np.array([b*u, 0.]), args.T, args.dt, seed=int(u_eff*1e5)+104729*args.seed)
    t = 0.0; rec = []; ring = collections.deque(maxlen=max(args.ring, 1)); wprev = winding(I.S).mean(); slips = []
    while t < hold-1e-9:
        I.run(nsub); t += args.every
        w = winding(I.S); pm, ps = pitch(I.S)
        rec.append([t, w.mean(), w.min(), w.max(), pm, ps, I.S[..., 1].mean(), I.S[..., 1].max(), energy(I.S, J, D, B)])
        if args.ring: ring.append((t, I.S.astype(np.float32).copy()))
        if abs(w.mean()-wprev) > 0.5/Ly:
            slips.append([t, wprev, w.mean()])
            if args.ring and len(slips) <= 3:
                np.savez_compressed(f"{args.out}/slip_{args.tag}_{u_eff:.5f}_{len(slips)}.npz", t=np.array([x[0] for x in ring]), S=np.array([x[1] for x in ring]))
            wprev = w.mean()
            if args.stop_on_slip: break
    rec = np.array(rec); S = I.S.copy()
    if not args.norec: np.save(f"{args.out}/rec_{args.tag}_{u_eff:.5f}.npy", rec)
    np.save(f"{args.out}/S_{args.tag}_last.npy", S)
    q = rec[len(rec)*3//4:]
    line = dict(u_eff=u_eff, t=t, n_end=float(rec[-1, 1]), n_min=float(rec[-1, 2]), n_max=float(rec[-1, 3]), slips=slips,
                My=float(q[:, 6].mean()), E=float(q[:, 8].mean()), qmax=float(q[:, 5].mean()))
    summary.append(line)
    print(f"u_eff={u_eff:.5f} t={t:.0f} n_end={rec[-1,1]:.3f} rows[{rec[-1,2]:.2f},{rec[-1,3]:.2f}] slips={[(round(s[0]),round(s[1],2),round(s[2],2)) for s in slips][:6]}{'...' if len(slips)>6 else ''} "
          f"My={q[:,6].mean():.4f} E={q[:,8].mean():.3f} wall={time.time()-t0:.0f}", flush=True)
json.dump(summary, open(f"{args.out}/sum_{args.tag}.json", 'w'))
