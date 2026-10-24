import sys, os, json, time, argparse, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
sys.path.insert(0, here)
from twist import TwistModel, relax, find_V, link_blocks, jac_k, GD
# Task A: local stability region S(u) of a uniformly sheared skyrmion lattice.
# Cell = nx x 1 tiling of the 19x33 cell (2 skyrmions per tile) with twisted boundary S(x+Lx, y) = S(x, y-s):
# simple shear gamma = du_y/dx = s/Lx (optionally du_x/dx = (Lx - Lx0)/Lx0 with --Lx). For each s the state is relaxed
# (u = 0) or brought to the comoving stationary state (find_V, u > 0), and the Bloch Jacobian is diagonalised on an
# n x n grid of the cell Brillouin zone (half of it: lambda(-k) = conj lambda(k)).
# Output per s: max Re lambda per k, number of eigenvalues with Re > tol per k, energy, Q, V, skyrmion shapes.
p = argparse.ArgumentParser()
p.add_argument('--u', type=float, default=0.0); p.add_argument('--nx', type=int, default=1); p.add_argument('--Lx', type=int, default=None)
p.add_argument('--smax', type=int, default=8); p.add_argument('--smin', type=int, default=0); p.add_argument('--nk', type=int, default=16)
p.add_argument('--out', required=True); p.add_argument('--tag', default=None); p.add_argument('--noeig', action='store_true')
A = p.parse_args()
J, D = 1.0, 0.4; B = 0.5*D*D/J; a, b = 0.3, 0.6
Lx0 = 19*A.nx; Lx = A.Lx or Lx0; Ly = 33
S0 = np.tile(np.load(os.path.join(root, 'states', 'S_B0.5.npy')), (A.nx, 1, 1))
if Lx != Lx0:   # uniaxial stretch of the reference state along x (bilinear resampling), then relaxed
    xs = np.arange(Lx)*Lx0/Lx; x0 = np.floor(xs).astype(int); f = (xs-x0)[:, None, None]
    S0 = (1-f)*S0[x0 % Lx0]+f*S0[(x0+1) % Lx0]; S0 /= np.linalg.norm(S0, axis=-1)[..., None]
tag = A.tag or f"u{A.u:.3f}_nx{A.nx}_Lx{Lx}"; os.makedirs(A.out, exist_ok=True)
fn = f"{A.out}/S_{tag}.json"; res = json.load(open(fn)) if os.path.exists(fn) else {}
def shear_guess(S, Lx, Ly, ds):
    # add a uniform increment ds/Lx of du_y/dx: column x is moved up by ds*x/Lx (periodic linear interpolation in y)
    S = S.reshape(Lx, Ly, 3).copy(); out = np.empty_like(S); y = np.arange(Ly)
    for x in range(Lx):
        yy = (y-ds*x/Lx) % Ly; y0 = np.floor(yy).astype(int); f = (yy-y0)[:, None]
        out[x] = (1-f)*S[x, y0]+f*S[x, (y0+1) % Ly]
    out /= np.linalg.norm(out, axis=-1)[..., None]; return out.reshape(-1, 3)
def shapes(m, S):
    # skyrmion cores (m_z < 0) on a 3x3 periodic image of the twisted cell: area, aspect ratio, orientation
    from scipy import ndimage
    Sz = S[:, 2].reshape(m.Lx, m.Ly)
    big = np.zeros((3*m.Lx, 3*m.Ly))
    for i in range(3):
        for j in range(3):
            big[i*m.Lx:(i+1)*m.Lx, j*m.Ly:(j+1)*m.Ly] = np.roll(Sz, (i-1)*m.sh, axis=1)   # images shifted by s per x-wrap
    lab, n = ndimage.label(big < 0); out = []
    for c in range(1, n+1):
        X, Y = np.nonzero(lab == c); cx, cy = X.mean(), Y.mean()
        if not (m.Lx <= cx < 2*m.Lx and m.Ly <= cy < 2*m.Ly): continue
        if X.min() == 0 or Y.min() == 0 or X.max() == 3*m.Lx-1 or Y.max() == 3*m.Ly-1: continue
        C = np.cov(np.stack([X, Y])); ev, V = np.linalg.eigh(C)
        out.append([len(X), float(np.sqrt(ev[1]/ev[0])), float(np.degrees(np.arctan2(V[1, 1], V[0, 1])))])
    return out
uvec = np.array([A.u, 0.]); S = None; prev_s = None
for s in range(A.smin, A.smax+1):
    key = str(s)
    if key in res and ('maxRe' in res[key] or A.noeig): S = np.load(f"{A.out}/state_{tag}_s{s}.npy"); prev_s = s; continue
    t0 = time.time(); m = TwistModel(Lx, Ly, s, J, D, B)
    if S is None:
        S = S0.reshape(-1, 3)
        if s: S = shear_guess(S, Lx, Ly, s)
    else: S = shear_guess(S, Lx, Ly, s-prev_s)
    S, r = relax(m, S)
    V = [0.0, 0.0]; w1 = np.zeros(2); w2 = np.zeros(2)
    if A.u > 0:
        S, V, w1, w2, r = find_V(m, S, uvec, a, b); V = list(map(float, V))
    np.save(f"{A.out}/state_{tag}_s{s}.npy", S); prev_s = s
    _, _, Q = GD(m, S)
    d = dict(s=s, gamma=s/Lx0, exx=Lx/Lx0-1, Lx=Lx, E=m.energy(S), area=Lx*Ly, Q=float(Q), resid=float(r), V=V, shapes=shapes(m, S))
    if not A.noeig:
        bl, col = link_blocks(m, S, w1, w2, a); nk = A.nk; mx = []; nun = []; top = []
        for i in range(nk):
            for j in range(nk//2+1):
                k = (i/nk)*m.Bk[0]+(j/nk)*m.Bk[1]; k = k-m.Bk[0]*np.round(i/nk)    # fold to i/nk in [-1/2, 1/2)
                ev = np.linalg.eigvals(jac_k(m, bl, k))
                mx.append(float(ev.real.max())); nun.append(int((ev.real > 1e-7).sum()))
                o = np.argsort(-ev.real)[:3]; top.append([[float(e.real), float(e.imag)] for e in ev[o]])
        d.update(nk=nk, kgrid=[[i, j] for i in range(nk) for j in range(nk//2+1)], maxRe=mx, n_unstable=nun, top=top)
    d['wall'] = time.time()-t0; res[key] = d; json.dump(res, open(fn, 'w'))
    print(f"s={s} gamma={s/Lx0:.4f} E={d['E']:.4f} Q={Q:.3f} resid={r:.1e} V={np.round(V, 5)} shapes={np.round(d['shapes'], 2).tolist()} "
          + (f"maxRe={max(d['maxRe']):.3e} n_unst_max={max(d['n_unstable'])}" if not A.noeig else '') + f" wall={d['wall']:.0f}", flush=True)
