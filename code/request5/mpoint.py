import sys, os, json, glob, re, numpy as np, scipy.sparse.linalg as sla
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
sys.path.insert(0, here); sys.path.insert(1, os.path.join(root, 'code', 'request4')); sys.path.insert(2, os.path.join(root, 'code', 'request2'))
from twist import TwistModel, link_blocks
from sparse_bloch import jac_sparse
# Task A addendum: the soft branch near the M point K_M = (0, 2 pi/33) of the triangular skyrmion lattice (b2/2), where the critical
# Floquet modes of the strain-wave states have their carrier (task C0: row sliding along x). For each relaxed uniform state of
# dispA.py: eigenvalues near 0 (nev = 40, shift-invert) at K = K_M + (q_x, q_y), q_x in [-0.06, 0.06], q_y in {-0.02, 0, 0.02};
# unfolding by the plane-wave purity of the skyrmion displacements (dispA.trans_basis); the branch value is the largest real part
# among eigenpairs with purity > 0.8 (NaN if none of the 40 belongs to K). Also the x-polarisation of that eigenvector.
from unfold import trans_basis
J, D = 1.0, 0.4; B = 0.5*D*D/J; z = np.zeros(2); KMp = np.array([0.0, 2*np.pi/33])
QX = np.round(np.arange(-0.06, 0.0601, 0.01), 4); QY = [-0.02, 0.0, 0.02]
for f in sys.argv[1:]:
    g = re.search(r'nx(\d+)_Lx(\d+)_s(\d+)', f); nx, Lx, s = map(int, g.groups()); alpha = float(os.environ.get('ALPHA', '0.3'))
    fo = f.replace('state_', 'mpoint_').replace('.npy', f'_a{alpha}.json')
    if os.path.exists(fo): continue
    m = TwistModel(Lx, 33, s, J, D, B); S = np.load(f); bl, _ = link_blocks(m, S, z, z, alpha); amps = trans_basis(m, S); res = []
    for qy in QY:
        for qx in QX:
            K = KMp+[qx, qy]; ev, V = sla.eigs(jac_sparse(m, bl, K), k=40, sigma=2e-4, which='LM', tol=1e-9, maxiter=8000)
            best = (np.nan, 0.0, np.nan)
            for i in range(len(ev)):
                d, R = amps(V[:, i], K); ph = np.exp(-1j*(R@K)); px = np.abs(ph@d[:, 0])**2; py = np.abs(ph@d[:, 1])**2
                pur = float((px+py)/(len(R)*np.sum(np.abs(d)**2)))
                if pur > 0.8 and not (ev[i].real <= best[0]): best = (float(ev[i].real), pur, float(px/(px+py)))
            res.append([qx, qy, *best, float(ev.real.max())])
    r = np.array(res); i = int(np.nanargmax(r[:, 2])) if np.isfinite(r[:, 2]).any() else 0
    json.dump(dict(nx=nx, Lx=Lx, s=s, gamma=s/(19*nx), exx=Lx/(19*nx)-1, alpha=alpha, cols=['qx', 'qy', 'lam_branch', 'purity', 'xpol', 'lam_max_all'], data=res), open(fo, 'w'))
    print(f'Lx={Lx} s={s} gamma={s/(19*nx):.4f} eps={Lx/(19*nx)-1:+.4f}: M-branch max {r[i, 2]:+.3e} at q=({r[i, 0]:+.2f},{r[i, 1]:+.2f}) purity {r[i, 3]:.2f} xpol {r[i, 4]:.2f}; at M: {r[(r[:, 0] == 0) & (r[:, 1] == 0)][0, 2]:+.3e}', flush=True)
