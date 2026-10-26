import sys, os, json, time, argparse, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
sys.path.insert(0, os.path.join(root, 'code', 'request2')); sys.path.insert(1, os.path.join(root, 'code', 'kt')); sys.path.insert(2, here)
from twist import TwistModel, link_blocks, basis
from fastllg import Integrator
from sparse_bloch import jac_sparse
# Task C2: energy criterion of the strain-wave state (client's construction).
# For snapshots of the saturated state the conservative Jacobian in the frame of the strain wave is built,
#   J_c = d/dS [ -S x H + (w1 . grad) S ],  alpha = 0, w2 = 0, w1 = V_w - u,
# in the tangent basis (e1, e2) of every spin (k = 0 of the whole box), and H' = Omega J_c with the per-site symplectic matrix
# Omega chosen such that H' is the (symmetric, positive semidefinite) energy Hessian for the relaxed static lattice
# (checked with --check). H' of a driven snapshot is not exactly symmetric (the snapshot is not a stationary point of the
# comoving functional); the asymmetry is reported and the lowest eigenvalues of the symmetric part are computed (shift-invert).
# V_w: lab velocity of the strain wave along x (12.2/Lx from request 3, measured here from the snapshots) and the lattice
# drift along y (the wave is uniform along y, so its y velocity is that of the lattice).
# For comparison the plain energy Hessian of every snapshot (w1 = 0) is also diagonalised.
p = argparse.ArgumentParser()
p.add_argument('--state'); p.add_argument('--uc', type=float); p.add_argument('--r', type=float); p.add_argument('--nsnap', type=int, default=12)
p.add_argument('--alpha', type=float, default=0.3); p.add_argument('--out'); p.add_argument('--check', action='store_true'); p.add_argument('--nev', type=int, default=8)
A = p.parse_args(); J, D = 1.0, 0.4; B = 0.5*D*D/J; z = np.zeros(2)
def omega(N, sign):
    I = sp.identity(N, format='csc'); Z = sp.csc_matrix((N, N))
    return sign*sp.bmat([[Z, I], [-I, Z]], format='csc')
def hess(m, S, w1, sign):
    bl, _ = link_blocks(m, S.reshape(-1, 3), w1, z, 0.0); Jc = jac_sparse(m, bl, (0., 0.)).real
    H = (omega(m.N, sign)@Jc).tocsc(); As = H-H.T; Hs = 0.5*(H+H.T)
    asym = float(sla.norm(As)/sla.norm(H)) if sla.norm(H) > 0 else 0.0
    return Hs.tocsc(), asym
def lowest(Hs, nev):
    ev = sla.eigsh(Hs, k=nev, sigma=-1e-4, which='LM', return_eigenvectors=False, tol=1e-10)
    return np.sort(ev)
if A.check:
    # static relaxed 4 x 1 lattice: which sign of Omega gives the symmetric PSD energy Hessian
    from twist import relax
    S0 = np.tile(np.load(os.path.join(root, 'states', 'S_B0.5.npy')), (4, 1, 1)); m = TwistModel(76, 33, 0, J, D, B); S, _ = relax(m, S0.reshape(-1, 3))
    for sign in (1, -1):
        Hs, asym = hess(m, S, z, sign); print('sign', sign, 'asymmetry', asym, 'lowest', lowest(Hs, 6))
    sys.exit()
sign = float(os.environ.get('C2_SIGN', '1'))
S = np.load(A.state).astype(float); Lx, Ly = S.shape[:2]; m = TwistModel(Lx, Ly, 0, J, D, B); u = A.r*A.uc; a = A.alpha; b = 2*a
# snapshots over one period of the lattice passing the wave (T = 19/v_rel, v_rel Lx = 2.7) and the wave velocity
T = 19*Lx/2.7; dt = 0.2; n = int(round(T/A.nsnap/dt))
sys.path.insert(3, os.path.join(root, 'code', 'request3'))
from regist import topo
from chain_tools import profile, critical
def wave_x(S_):
    tp = topo(S_); xs, pr = profile(tp['X'], tp['u'], Lx, Ly); return critical(xs, pr, Lx)['xpeak'], tp
I = Integrator(S.copy(), J, D, B, a, np.array([-u, 0.]), np.array([b*u, 0.]), 0.0, dt); snaps = []; xw = []; ym = []
for i in range(A.nsnap):
    x, tp = wave_x(I.S); xw.append(x); ym.append(tp['P'][:, 1].mean()); snaps.append(I.S.copy()); I.run(n)
tt = np.arange(A.nsnap)*n*dt; xu = np.unwrap(np.array(xw)*2*np.pi/Lx)*Lx/(2*np.pi); Vwx = float(np.polyfit(tt, xu, 1)[0])
Vy = float(np.polyfit(tt, np.unwrap(np.array(ym)*2*np.pi/33)*33/(2*np.pi), 1)[0]); w1 = np.array([Vwx-u, Vy])
out = []
for i, Sn in enumerate(snaps):
    t0 = time.time(); Hs, asym = hess(m, Sn, w1, sign); ev = lowest(Hs, A.nev)
    H0, asym0 = hess(m, Sn, z, sign); ev0 = lowest(H0, A.nev)          # plain energy Hessian (lab frame, w1 = 0)
    out.append(dict(i=i, asym=asym, ev=ev.tolist(), asym0=asym0, ev0=ev0.tolist()))
    print(f'r={A.r} snap {i} asym={asym:.3e} lowest={np.round(ev[:4], 7)} | w1=0: asym={asym0:.2e} lowest={np.round(ev0[:4], 7)} ({time.time()-t0:.0f}s)', flush=True)
lo = [o['ev'] for o in out]
json.dump(dict(state=A.state, r=A.r, u=u, Lx=Lx, Ly=Ly, V_w=[Vwx, Vy], w1=w1.tolist(), T=T, sign=sign, snaps=out,
               min_ev=float(np.min(lo)), min_ev_nonzero=float(np.min([e for row in lo for e in row if abs(e) > 1e-7] or [np.nan])), min_ev0=float(np.min([o['ev0'] for o in out]))), open(A.out, 'w'))
