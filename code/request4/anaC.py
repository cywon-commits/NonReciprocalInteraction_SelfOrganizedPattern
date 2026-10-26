import sys, os, json, glob, numpy as np
from scipy import ndimage
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
os.environ.setdefault('SKX_CELL', os.path.join(root, 'states', 'S_B0.5.npy'))
sys.path.insert(0, os.path.join(root, 'code', 'request3')); sys.path.insert(1, os.path.join(root, 'code', 'request2')); sys.path.insert(2, os.path.join(root, 'code', 'request'))
from chain_tools import centroids, profile, critical
from strain import tri_strain
from regist import topo
from style import plt, C
# Task C summary and C3 eigenvector analysis.
#  - C1: leading Floquet exponents per state; threshold u/u_c and gamma_wave where the critical (oscillatory) pair crosses 0.
#  - C2: lowest eigenvalues of H' (comoving, w1 = V_w - u) and of the plain energy Hessian (w1 = 0) per state.
#  - C3: the leading Floquet vector q (spin perturbation) -> rigid displacement of each skyrmion d_j = -M^-1 b
#    (M_ab = sum_core dS/da . dS/db, b_a = sum_core q . dS/da over the core m_z < 0), then
#    (i) weight of the soft band (0.06 < |k| < 0.11, 0.045 < |k_y| < 0.1) in the displacement field (all torus wavevectors);
#    (ii) envelope |d|^2(x) (y average, 19-site box filter): position relative to the strain peak, width at half maximum;
#    (iii) width/(L sqrt(Delta)), Delta = gamma_wave - gamma_s(H_peak);
#    (iv) overlap with the local negative-stiffness window gamma(x) > gamma_s(eps_xx(x)) (task A map).
AS = json.load(open('results_request4/A_summary.json')); P2 = np.array(AS['fit_quad']); gsH = AS['self_consistent']['all']['gamma_s_Hpeak']
gs_of = lambda e: np.polyval(P2, e)
def disp_field(S, q):
    Lx, Ly = S.shape[:2]; gx = (np.roll(S, -1, 0)-np.roll(S, 1, 0))/2; gy = (np.roll(S, -1, 1)-np.roll(S, 1, 1))/2
    big = np.tile(S[..., 2] < 0, (3, 3)); lab, n = ndimage.label(big, structure=np.ones((3, 3)))
    cores = []
    for c in range(1, n+1):
        I, Jj = np.nonzero(lab == c); cx, cy = I.mean(), Jj.mean()
        if not (Lx <= cx < 2*Lx and Ly <= cy < 2*Ly): continue
        i = I % Lx; j = Jj % Ly
        M = np.array([[np.sum(gx[i, j]*gx[i, j]), np.sum(gx[i, j]*gy[i, j])], [np.sum(gx[i, j]*gy[i, j]), np.sum(gy[i, j]*gy[i, j])]])
        b = np.array([np.sum(q[i, j]*gx[i, j]), np.sum(q[i, j]*gy[i, j])]); cores.append((cx-Lx, cy-Ly, *(-np.linalg.solve(M, b))))
    return np.array(cores)
def analyse(fj, iq=0):
    d = json.load(open(fj)); S = np.load(fj.replace('.json', '_S.npy')).astype(float); q = np.load(fj.replace('.json', f'_q{iq}.npy')).astype(float)
    Lx, Ly = S.shape[:2]; c = disp_field(S, q); X, dd = c[:, :2], c[:, 2:]
    # (i) soft-band weight of the displacement field
    # wavevectors inside the Brillouin zone of the skyrmion lattice (row spacing 9.5 along x, 16.5 along y): no aliasing
    K = [(2*np.pi*m/Lx, 2*np.pi*n/Ly) for m in range(-Lx//19, Lx//19+1) for n in range(-Ly//33, Ly//33+1)]; K = np.array(K)
    ph = np.exp(-1j*(X@K.T)); Pw = np.abs(ph.T@dd[:, 0])**2+np.abs(ph.T@dd[:, 1])**2; kk = np.hypot(K[:, 0], K[:, 1])
    band = (kk > 0.06) & (kk < 0.11) & (np.abs(K[:, 1]) > 0.045); kpk = K[np.argmax(Pw)]
    # (ii) envelope along x
    xs = np.arange(Lx); env = np.zeros(Lx)
    for (x, y), v in zip(X, dd): env[int(np.floor(x)) % Lx] += v@v
    env = np.convolve(np.concatenate([env[-9:], env, env[:9]]), np.ones(19)/19, 'valid'); env /= env.max()
    xe = float(np.angle(np.sum(env*np.exp(2j*np.pi*xs/Lx)))*Lx/(2*np.pi)) % Lx; width = float((env > 0.5).sum())
    # strain profile and the local stiffness window
    tp = topo(S); xs4, pg = profile(tp['X'], tp['u'], Lx, Ly); cr = critical(xs4, pg, Lx); sg = np.sign(pg[np.argmax(abs(pg))])
    st = tri_strain(tp['X'], tp['u'], Lx, Ly); cc = centroids(tp['X'], Lx, Ly); k = 2*np.pi/Lx
    Mx = np.stack([np.ones(len(cc))]+[fn(n*k*cc[:, 0]) for n in (1, 2, 3) for fn in (np.cos, np.sin)], 1)
    Bm = np.stack([np.ones(400)]+[fn(n*k*xs4) for n in (1, 2, 3) for fn in (np.cos, np.sin)], 1)
    exx = Bm@np.linalg.lstsq(Mx, st[:, 7], rcond=None)[0]; gam = sg*pg; win = gam > gs_of(sg*exx)
    wwidth = float(win.sum()*Lx/400); env4 = np.interp(xs4, xs, env, period=Lx)
    overlap = float(np.sum(env4*win)/np.sum(env4))
    dx = ((xe-cr['xpeak'])+Lx/2) % Lx-Lx/2; Dl = cr['gpeak']-gsH
    return dict(file=fj, r=d['r'], alpha=d['alpha'], Lx=Lx, Gamma=d['Gamma'][:4], Gamma_err=d['Gamma_err'][:4], gamma_wave=cr['gpeak'], xpeak=cr['xpeak'],
                band_weight=float(Pw[band].sum()/Pw.sum()), k_peak=kpk.tolist(), env_center=xe, env_offset=float(dx), env_width=width,
                width_over_LsqrtD=float(width/(Lx*np.sqrt(Dl))) if Dl > 0 else None, window_width=wwidth, window_overlap=overlap, Delta_local=float(Dl),
                env=env.tolist(), gam=gam.tolist(), gs_local=gs_of(sg*exx).tolist(), xs=xs4.tolist())
if __name__ == '__main__':
    R = {}
    for fj in sorted(glob.glob('results_request4/C/C1/*/f_r*.json')):
        if fj.endswith('_S.json'): continue
        try: a = analyse(fj)
        except Exception as ex: print('fail', fj, ex); continue
        R[fj] = a; print(fj.split('/')[-2], a['r'], 'Gamma', np.round(a['Gamma'][:3], 6), 'gw %.4f' % a['gamma_wave'], 'band %.2f' % a['band_weight'], 'k', np.round(a['k_peak'], 3),
                         'env off %.0f width %.0f' % (a['env_offset'], a['env_width']), 'win %.0f overlap %.2f' % (a['window_width'], a['window_overlap']), flush=True)
    C2 = {}
    for fj in sorted(glob.glob('results_request4/C/C2/*/h_*.json')):
        d = json.load(open(fj)); C2[fj] = dict(r=d['r'], min_ev=d['min_ev'], min_ev0=d.get('min_ev0'), V_w=d['V_w'],
                                               asym=float(np.mean([s['asym'] for s in d['snaps']])), asym0=float(np.mean([s.get('asym0', np.nan) for s in d['snaps']])))
    json.dump(dict(C1=R, C2=C2), open('results_request4/C_summary.json', 'w'))
