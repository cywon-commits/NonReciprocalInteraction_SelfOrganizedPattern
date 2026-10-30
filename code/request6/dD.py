import sys, os, json, time, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..'); sys.path.insert(0, os.path.join(root, 'code', 'b1'))
from skx import Model, rk4, bloch_jac, GD
from scipy.sparse.linalg import eigs
# Request 6, task D: nonreciprocal DMI delta D, DL channel, on the Neel SkX (B = 0.5 D^2/J, alpha = 0.3, no current).
# Bond form (exchange-even, n(e) = n for both bond axes):
#   T_i = -dD sum_{e = x, y} S_i x [ S_i x ( n x (S_{i+e} + S_{i-e}) ) ]
# Expanding S_{i+e} + S_{i-e} = 2 S_i + d_e^2 S: the leading term is the field-like torque -2 dD S x n per axis (effective
# Zeeman field 2 dD n per axis, 4 dD n in total, axial, T-odd); the remainder is the second-order nonreciprocal stiffness
#   -dD sum_e S x [ S x ( n x d_e^2 S ) ].
# Models: 'local' = leading term only (-4 dD S x n); 'full' = bond form; 'stiff' = second-order term only.
# For each dD: stationary state (co-moving drift V solved as in code/b1/sot.py; field-like torques should not drive the lattice),
# static deformation against dD = 0 (skyrmion centre shift, core size, ellipticity), phonon pair at k = (+-k, 0), (0, +-k)
# (both branches: Re = damping / growth, Im = frequency), and the largest Re over a k grid (threshold, if any).
J, D = 1.0, 0.4; B = 0.5*D*D; a = 0.3
class DDModel(Model):
    def __init__(s, Lx, Ly, dD, n, form): super().__init__(Lx, Ly, J, D, B); s.dD, s.n, s.form = dD, np.array(n, float), form
    def rhs(s, S0, w1, w2, a, d=None, eps=0.0, k=(0., 0.)):
        S = S0 if d is None else S0+eps*d
        xp = s.nb(S0, d, eps, 1, 0, k); xm = s.nb(S0, d, eps, -1, 0, k); yp = s.nb(S0, d, eps, 1, 1, k); ym = s.nb(S0, d, eps, -1, 1, k)
        from skx import dx_, dy_, zh
        H = s.J*(xp+xm+yp+ym)+s.B*zh+s.D*(np.cross(xp, dx_)+np.cross(dx_, xm)+np.cross(yp, dy_)+np.cross(dy_, ym))
        gx = (xp-xm)/2; gy = (yp-ym)/2
        T = -np.cross(S, H)+w1[0]*gx+w1[1]*gy+np.cross(S, w2[0]*gx+w2[1]*gy)
        if s.dD:
            if s.form == 'full': src = xp+xm+yp+ym
            elif s.form == 'local': src = 4*S
            else: src = xp+xm+yp+ym-4*S                                      # 'stiff': sum_e d_e^2 S
            T = T-s.dD*np.cross(S, np.cross(S, np.cross(s.n, src)))
        T = T-np.sum(T*S, -1)[..., None]*S
        return (T+a*np.cross(S, T))/(1+a*a)
def find_drift(m, S, iters=12, n=1500):
    V = np.zeros(2)
    for _ in range(iters):
        f0 = m.rhs(S, V, -a*V, a).ravel()
        A = np.stack([(m.rhs(S, V+e, -a*(V+e), a).ravel()-f0)/1e-4 for e in (np.array([1e-4, 0]), np.array([0, 1e-4]))], 1)
        V = V+np.linalg.lstsq(A, -f0, rcond=None)[0]; S = rk4(m, S, V, -a*V, a, 0.1, n)
    for _ in range(40):                                                     # converge to max|dS/dt| < 1e-10 (request 6)
        r = np.abs(m.rhs(S, V, -a*V, a)).max()
        if r < 1e-10: break
        f0 = m.rhs(S, V, -a*V, a).ravel()
        A = np.stack([(m.rhs(S, V+e, -a*(V+e), a).ravel()-f0)/1e-4 for e in (np.array([1e-4, 0]), np.array([0, 1e-4]))], 1)
        V = V+np.linalg.lstsq(A, -f0, rcond=None)[0]; S = rk4(m, S, V, -a*V, a, 0.1, 2000)
    return S, V, float(np.abs(m.rhs(S, V, -a*V, a)).max())
def shape(S):
    # per skyrmion (m_z < 0 cores, periodic): centre, area, ellipticity (second moments) -> averages
    from scipy import ndimage
    mz = S[..., 2]; Lx, Ly = mz.shape; big = np.tile(mz < 0, (3, 3)); lab, nl = ndimage.label(big); out = []
    for c in range(1, nl+1):
        I, Jj = np.nonzero(lab == c); cx, cy = I.mean(), Jj.mean()
        if not (Lx <= cx < 2*Lx and Ly <= cy < 2*Ly): continue
        w = -np.tile(mz, (3, 3))[I, Jj]; X = np.stack([I, Jj], 1)-[cx, cy]; C = (X*w[:, None]).T@X/w.sum(); ev = np.linalg.eigvalsh(C)
        out.append([cx-Lx, cy-Ly, len(I), np.sqrt(ev[1]/ev[0]), np.degrees(np.arctan2(*np.linalg.eigh(C)[1][:, 1][::-1]))])
    return np.array(out)
def pair(m, S, V, k):
    lam = eigs(bloch_jac(m, S, V, -a*V, a, k), k=6, sigma=1e-9, return_eigenvectors=False); p = lam[np.argsort(np.abs(lam))[:2]]
    return sorted([(float(z.real), float(z.imag)) for z in p], key=lambda t: t[1])
if __name__ == '__main__':
    Lx = int(sys.argv[1]); form = sys.argv[2]; nvec = [float(x) for x in sys.argv[3].split(',')]; dDs = [float(x) for x in sys.argv[4].split(',')]
    S0 = np.tile(np.load(os.path.join(root, 'states', 'S_B0.5.npy')), (Lx//19, 1, 1)); m0 = DDModel(Lx, 33, 0.0, nvec, form)
    S0, V0, r0 = find_drift(m0, S0); sh0 = shape(S0); out = []
    fo = f"results_request6/D/dD_{form}_Lx{Lx}_n{sys.argv[3].replace(',', '_')}.json"
    ks = [2*np.pi/(19*20), 2*np.pi/(19*5), 2*np.pi/(19*2)] if Lx == 19 else [2*np.pi/Lx, 2*np.pi/(19*20)]
    for dD in dDs:
        t0 = time.time(); m = DDModel(Lx, 33, dD, nvec, form); S, V, r = find_drift(m, S0.copy()); sh = shape(S)
        L2 = np.array([Lx, 33.]); dd = sh[:, None, :2]-sh0[None, :, :2]; dd = (dd+L2/2) % L2-L2/2; jn = np.argmin(np.hypot(*dd.transpose(2, 0, 1)), 1)
        dsh = dict(shift=dd[np.arange(len(sh)), jn].mean(0).tolist(), area=float(sh[:, 2].mean()-sh0[:, 2].mean()), ellip=float(sh[:, 3].mean()), dS_max=float(np.abs(S-S0).max()),
                   mz_mean=float(S[..., 2].mean()-S0[..., 2].mean()), mn_mean=float((S[..., :]@np.array(nvec)).mean()))
        ph = {f'{lab}{kk:.4f}': pair(m, S, V, kv) for kk in ks for lab, kv in (('kx+', (kk, 0.)), ('kx-', (-kk, 0.)), ('ky+', (0., kk)), ('ky-', (0., -kk)))}
        out.append(dict(dD=dD, V=V.tolist(), resid=r, static=dsh, phonon=ph, wall=time.time()-t0))
        p = ph[f'kx+{ks[0]:.4f}']; q = ph[f'ky+{ks[0]:.4f}']
        print(f"{form} Lx={Lx} n={nvec} dD={dD}: V=({V[0]:+.2e},{V[1]:+.2e}) resid={r:.0e} shift={np.round(dsh['shift'], 5)} dArea={dsh['area']:+.2f} ellip={dsh['ellip']:.5f} "
              f"| k||x pair Re={p[0][0]:+.4e},{p[1][0]:+.4e} Im={p[0][1]:+.4e},{p[1][1]:+.4e} | k||y Re={q[0][0]:+.4e},{q[1][0]:+.4e} wall={time.time()-t0:.0f}", flush=True)
        json.dump(dict(Lx=Lx, form=form, n=nvec, ks=ks, ref=dict(V=V0.tolist(), resid=r0), rows=out), open(fo, 'w'), indent=1)
