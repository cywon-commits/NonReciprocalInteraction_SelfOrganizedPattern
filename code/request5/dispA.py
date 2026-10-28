import sys, os, json, time, argparse, numpy as np, scipy.sparse.linalg as sla
from scipy import ndimage
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
sys.path.insert(0, os.path.join(root, 'code', 'request4')); sys.path.insert(1, os.path.join(root, 'code', 'request2'))
from twist import TwistModel, relax, link_blocks, shear_guess, basis
from sparse_bloch import jac_sparse
sys.path.insert(2, here)
from unfold import trans_basis
# Request 5, task A: full dispersion of the soft (elastic) branch of the uniformly deformed lattice
# H = gamma y x^T + eps_xx x x^T (u = 0): relaxed twisted cell nx x 1 (Lx0 = 19 nx) resampled to Lx (eps_xx = Lx/Lx0 - 1), gamma = s/Lx0.
# For every s the Bloch Jacobian is diagonalised by shift-invert near 0 (6 eigenpairs) on
#   (1) a polar grid |k| in [0.005, 0.20] (log, nkm points) x theta in [90, 150] deg (3 deg),
#   (2) cuts at the quantised k_y = 2 pi n/L_y of the boxes L_y = 0.434 L_x (L_x = 152 ... 608, n = 1, 2, 3), k_x in [-0.30, 0.12].
# Branch tracking and unfolding: the supercell (16 skyrmions) folds the elastic band; for every eigenpair the rigid displacement of
# each skyrmion is extracted (projection of the core part on exp(i k.(r - r_c)) dS/dx, dS/dy) and its purity as the plane wave
# exp(i k.R_j) is computed; the soft branch at k itself is the largest real part among the eigenpairs with purity > 0.8 (the two
# acoustic polarisations); NaN when none of the 12 eigenvalues nearest 0 belongs to k.
# Stored per k: (max Re of the 12 eigenvalues, soft-branch Re, the largest purity).
# --grid polarhi: continuation of the polar grid to |k| in [0.23, 0.40] (5 log points), stored separately in disphi_*.json (task B
# needs the branch beyond 0.2 at the larger margins of the small boxes).
p = argparse.ArgumentParser()
p.add_argument('--nx', type=int, default=8); p.add_argument('--Lx', type=int, required=True); p.add_argument('--s', required=True)
p.add_argument('--alpha', type=float, default=0.3); p.add_argument('--out', required=True); p.add_argument('--nkm', type=int, default=22)
p.add_argument('--grid', default='polar,cuts')
A = p.parse_args(); J, D = 1.0, 0.4; B = 0.5*D*D/J; Ly = 33; Lx0 = 19*A.nx; Lx = A.Lx; eps = Lx/Lx0-1; z = np.zeros(2)
os.makedirs(A.out, exist_ok=True)
S0 = np.tile(np.load(os.path.join(root, 'states', 'S_B0.5.npy')), (A.nx, 1, 1))
if Lx != Lx0:
    xs = np.arange(Lx)*Lx0/Lx; x0 = np.floor(xs).astype(int); f = (xs-x0)[:, None, None]
    S0 = (1-f)*S0[x0 % Lx0]+f*S0[(x0+1) % Lx0]; S0 /= np.linalg.norm(S0, axis=-1)[..., None]
KM = np.geomspace(0.005, 0.20, A.nkm); KMH = np.geomspace(0.20, 0.40, 6)[1:]; TH = np.arange(90, 151, 3.0)
KY = sorted(set(round(2*np.pi*n/(0.434*L), 6) for L in (152, 228, 304, 456, 608) for n in (1, 2, 3))); KX = np.round(np.arange(-0.30, 0.1201, 0.01), 4)
def soft(m, bl, amps, k, nev=12):
    # unfolding: the supercell spectrum at k contains the branches of all k + G (supercell reciprocal vectors); the eigenpair whose
    # skyrmion displacement field is the purest plane wave exp(i k.R_j) belongs to k itself. Returns (max Re overall, Re of the
    # purest eigenpair, its purity).
    Jm = jac_sparse(m, bl, k); ev, V = sla.eigs(Jm, k=nev, sigma=2e-4, which='LM', tol=1e-10, maxiter=5000)
    pur = []
    for i in range(len(ev)):
        d, R = amps(V[:, i], k); ph = np.exp(-1j*(R@np.asarray(k)))
        pur.append(float((np.abs(ph@d[:, 0])**2+np.abs(ph@d[:, 1])**2)/(len(R)*np.sum(np.abs(d)**2))))
    pur = np.array(pur); ok = pur > 0.8                    # both acoustic branches (two polarisations) are pure plane waves
    best = (float(pur[ok].max()), float(ev[ok].real.max())) if ok.any() else (float(pur.max()), np.nan)
    return float(ev.real.max()), best[1], best[0]
for s in [int(x) for x in A.s.split(',')]:
    fo = f"{A.out}/{'disphi' if A.grid == 'polarhi' else 'disp'}_nx{A.nx}_Lx{Lx}_s{s}_a{A.alpha}.json"
    if os.path.exists(fo): continue
    t0 = time.time(); m = TwistModel(Lx, Ly, s, J, D, B); fs = f'{A.out}/state_nx{A.nx}_Lx{Lx}_s{s}.npy'
    if os.path.exists(fs): S = np.load(fs)
    else:
        S, r = relax(m, shear_guess(S0.reshape(-1, 3), Lx, Ly, s)); np.save(fs, S)
    bl, _ = link_blocks(m, S, z, z, A.alpha); proj = trans_basis(m, S); out = dict(nx=A.nx, Lx=Lx, s=s, gamma=s/Lx0, exx=eps, alpha=A.alpha)
    if A.grid == 'polarhi':
        P = np.full((len(KMH), len(TH), 3), np.nan)
        for i, km in enumerate(KMH):
            for j, th in enumerate(TH):
                P[i, j] = soft(m, bl, proj, km*np.array([np.cos(np.radians(th)), np.sin(np.radians(th))]))
        out.update(km=KMH.tolist(), theta=TH.tolist(), polar=P.tolist()); json.dump(out, open(fo, 'w'))
        sb = P[..., 1]; print(f'Lx={Lx} s={s} hi: soft max {np.nanmax(sb) if np.isfinite(sb).any() else np.nan:+.3e} NaN {np.isnan(sb).sum()} wall={time.time()-t0:.0f}', flush=True); continue
    if 'polar' in A.grid.split(','):
        P = np.full((len(KM), len(TH), 3), np.nan)
        for i, km in enumerate(KM):
            for j, th in enumerate(TH):
                P[i, j] = soft(m, bl, proj, km*np.array([np.cos(np.radians(th)), np.sin(np.radians(th))]))
        out.update(km=KM.tolist(), theta=TH.tolist(), polar=P.tolist())
    if 'cuts' in A.grid:
        Cc = np.full((len(KY), len(KX), 3), np.nan)
        for i, ky in enumerate(KY):
            for j, kx in enumerate(KX): Cc[i, j] = soft(m, bl, proj, (kx, ky))
        out.update(ky=KY, kx=KX.tolist(), cuts=Cc.tolist())
    out['wall'] = time.time()-t0; json.dump(out, open(fo, 'w'))
    if 'polar' in A.grid.split(','):
        sb = np.array(out['polar'])[..., 1]; i, j = np.unravel_index(np.nanargmax(sb), sb.shape)
        print(f'Lx={Lx} eps={eps:+.4f} s={s} gamma={s/Lx0:.4f} alpha={A.alpha}: soft max {sb[i, j]:+.3e} at |k|={KM[i]:.4f} theta={TH[j]:.0f} wall={out["wall"]:.0f}', flush=True)
    else: print(f'Lx={Lx} s={s} done wall={out["wall"]:.0f}', flush=True)
