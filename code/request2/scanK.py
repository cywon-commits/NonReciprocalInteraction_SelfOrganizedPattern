import sys, os, json, time, argparse, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..'); sys.path.insert(0, here)
from twist import TwistModel, relax, find_V, link_blocks, jac_k, GD
from twist import shear_guess, shapes
# Task A refinement: eigenvalues of the Bloch Jacobian of the sheared lattice (supercell nx x 1, twist s, gamma = s/(19 nx))
# at selected physical k only. k is given by indices (i, j) of an nk x nk grid of the 19x33 cell's Brillouin zone at the
# same gamma, so points are comparable across supercells.  usage: scanK.py --nx 3 --s 17 --u 0.0 --k 15,6 14,6 ...
p = argparse.ArgumentParser(); p.add_argument('--nx', type=int); p.add_argument('--s', type=int); p.add_argument('--u', type=float, default=0.0)
p.add_argument('--nk', type=int, default=16); p.add_argument('--k', nargs='+'); p.add_argument('--out', required=True)
A = p.parse_args(); J, D = 1.0, 0.4; B = 0.5*D*D/J; a, b = 0.3, 0.6; Lx = 19*A.nx; Ly = 33; gam = A.s/Lx
tag = f"K_u{A.u:.3f}_nx{A.nx}_s{A.s}"; fn = f"{A.out}/{tag}.json"
if os.path.exists(fn): print('done', fn); sys.exit()
t0 = time.time(); m = TwistModel(Lx, Ly, A.s, J, D, B)
S0 = np.tile(np.load(os.path.join(root, 'states', 'S_B0.5.npy')), (A.nx, 1, 1)).reshape(-1, 3)
S, r = relax(m, shear_guess(S0, Lx, Ly, A.s)); V = [0., 0.]; w1 = np.zeros(2); w2 = np.zeros(2)
if A.u > 0: S, V, w1, w2, r = find_V(m, S, np.array([A.u, 0.]), a, b); V = list(map(float, V))
_, _, Q = GD(m, S); bl, _ = link_blocks(m, S, w1, w2, a)
B19 = 2*np.pi*np.linalg.inv(np.array([[19.0, 19.0*gam], [0, 33.0]])).T
out = dict(nx=A.nx, s=A.s, gamma=gam, u=A.u, Q=float(Q), resid=float(r), V=V, shapes=shapes(m, S), k={})
for kk in A.k:
    i, j = map(int, kk.split(',')); kv = ((i/A.nk)-np.round(i/A.nk))*B19[0]+(j/A.nk)*B19[1]
    ev = np.linalg.eigvals(jac_k(m, bl, kv)); o = np.argsort(-ev.real)[:4]
    out['k'][kk] = dict(kvec=kv.tolist(), top=[[float(e.real), float(e.imag)] for e in ev[o]], n_unst=int((ev.real > 1e-7).sum()))
    print(kk, np.round(kv, 4), [(round(e.real, 7), round(e.imag, 6)) for e in ev[o][:2]], flush=True)
out['wall'] = time.time()-t0; json.dump(out, open(fn, 'w'), indent=1); print('wall', round(out['wall']))
