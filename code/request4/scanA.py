import sys, os, json, time, argparse, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..'); sys.path.insert(0, here); sys.path.insert(1, os.path.join(root, 'code', 'request2'))
from twist import TwistModel, relax, find_V, link_blocks, shear_guess, GD, shapes
from sparse_bloch import soft_eigs
# Task A: static stability of the uniformly deformed lattice H = gamma y x^T + eps_xx x x^T (u = 0 by default).
# Cell: nx x 1 tiling of the 19 x 33 cell, resampled along x to Lx (eps_xx = Lx/Lx0 - 1, Lx0 = 19 nx), twisted boundary
# S(x + Lx, y) = S(x, y - s), gamma = s/Lx0 (same conventions as request 2 scanS.py). For each s the state is relaxed and the
# Bloch Jacobian is scanned on an nk x (nk/2+1) grid of the cell Brillouin zone with sparse shift-invert (sparse_bloch.py);
# supercell wavevectors |k| < 0.025 (translation Goldstone and long-wavelength acoustic modes, lambda -> 0-) are excluded from
# the soft-branch maximum; the soft mode of request 2 has |k| = 0.083.
# The s range is extended automatically until the threshold is bracketed by two stable and two unstable values.
p = argparse.ArgumentParser()
p.add_argument('--nx', type=int, default=4); p.add_argument('--Lx', type=int, required=True); p.add_argument('--u', type=float, default=0.0)
p.add_argument('--nk', type=int, default=16); p.add_argument('--out', required=True); p.add_argument('--alpha', type=float, default=0.3)
p.add_argument('--s0', type=int, default=None)
A = p.parse_args()
J, D = 1.0, 0.4; B = 0.5*D*D/J; a = A.alpha; b = 2*a; Ly = 33; Lx0 = 19*A.nx; Lx = A.Lx; eps = Lx/Lx0-1
S0 = np.tile(np.load(os.path.join(root, 'states', 'S_B0.5.npy')), (A.nx, 1, 1))
if Lx != Lx0:
    xs = np.arange(Lx)*Lx0/Lx; x0 = np.floor(xs).astype(int); f = (xs-x0)[:, None, None]
    S0 = (1-f)*S0[x0 % Lx0]+f*S0[(x0+1) % Lx0]; S0 /= np.linalg.norm(S0, axis=-1)[..., None]
os.makedirs(A.out, exist_ok=True); tag = f'u{A.u:.3f}_nx{A.nx}_Lx{Lx}'; fn = f'{A.out}/A_{tag}.json'
res = json.load(open(fn)) if os.path.exists(fn) else {}
def point(s):
    key = str(s)
    if key in res: return res[key]
    t0 = time.time(); m = TwistModel(Lx, Ly, s, J, D, B)
    near = [int(k) for k in res if os.path.exists(f'{A.out}/state_{tag}_s{k}.npy')]
    if near:
        s1 = min(near, key=lambda k: abs(k-s)); S = shear_guess(np.load(f'{A.out}/state_{tag}_s{s1}.npy'), Lx, Ly, s-s1)
    else: S = shear_guess(S0.reshape(-1, 3), Lx, Ly, s)
    S, r = relax(m, S); w1 = np.zeros(2); w2 = np.zeros(2); V = [0., 0.]
    if A.u > 0: S, V, w1, w2, r = find_V(m, S, np.array([A.u, 0.]), a, b); V = list(map(float, V))
    np.save(f'{A.out}/state_{tag}_s{s}.npy', S); _, _, Q = GD(m, S)
    bl, _ = link_blocks(m, S, w1, w2, a); grid = []; mx = -np.inf; kmx = None; top = None
    for i in range(A.nk):
        for j in range(A.nk//2+1):
            if i == 0 and j == 0: continue
            kv = ((i/A.nk)-np.round(i/A.nk))*m.Bk[0]+(j/A.nk)*m.Bk[1]
            if np.hypot(*kv) < 0.025: continue        # long-wavelength acoustic modes (lambda ~ -k^2 -> 0) would mask the soft branch
            ev = soft_eigs(m, bl, kv)
            grid.append([i, j, float(ev.real.max())])
            if ev.real.max() > mx: mx = float(ev.real.max()); kmx = kv.tolist(); top = [[float(e.real), float(e.imag)] for e in ev[:3]]
    d = dict(s=s, gamma=s/Lx0, exx=eps, Lx=Lx, nx=A.nx, u=A.u, Q=float(Q), resid=float(r), V=V, shapes=shapes(m, S), maxRe=mx, k_max=kmx, top=top,
             grid=grid, wall=time.time()-t0)
    res[key] = d; json.dump(res, open(fn, 'w'))
    print(f'Lx={Lx} eps={eps:+.4f} s={s} gamma={s/Lx0:.4f} Q={Q:.3f} resid={r:.1e} maxRe={mx:+.3e} k={np.round(kmx, 4)} wall={d["wall"]:.0f}', flush=True)
    return d
s = A.s0 if A.s0 is not None else int(round(Lx0*(0.308+0.5*eps)))-2
point(s)
while True:
    st = sorted(int(k) for k, v in res.items() if v['maxRe'] < 0); allk = sorted(int(k) for k in res)
    un = [int(k) for k, v in res.items() if v['maxRe'] > 0 and int(k) > max(st, default=-10**9)]
    if not st: s = allk[0]-1
    elif len(un) < 2: s = allk[-1]+1
    elif len(st) < 2: s = allk[0]-1
    else: break
    point(s)
# threshold: linear interpolation of the soft-branch maximum between the last stable and first unstable s
ks = sorted(res, key=int); g = np.array([res[k]['gamma'] for k in ks]); l = np.array([res[k]['maxRe'] for k in ks])
i = int(np.where((l[:-1] < 0) & (l[1:] > 0))[0][-1]); gs = g[i]-l[i]*(g[i+1]-g[i])/(l[i+1]-l[i])
q = np.polyfit(g[max(i-1, 0):i+3], l[max(i-1, 0):i+3], 2) if len(g) >= 3 else None
gq = None
if q is not None:
    r_ = np.roots(q); r_ = r_[np.isreal(r_)].real; r_ = r_[(r_ > g[i]-1e-9) & (r_ < g[i+1]+1e-9)]; gq = float(r_[0]) if len(r_) else None
out = dict(Lx=Lx, nx=A.nx, exx=eps, u=A.u, gamma_s_lin=float(gs), gamma_s_quad=gq, bracket=[float(g[i]), float(g[i+1])], slope=float((l[i+1]-l[i])/(g[i+1]-g[i])))
json.dump(out, open(f'{A.out}/gs_{tag}.json', 'w'), indent=1); print('RESULT', json.dumps(out), flush=True)
