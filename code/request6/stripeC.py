import sys, os, json, time, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
sys.path.insert(0, here); sys.path.insert(1, os.path.join(root, 'code', 'stripe'))
import nbrllg as NL
from conv import make, cycloid, steady, w1_of, growth, eh
# Request 6, task C1: finite-box threshold of the diagonal stripe under an oblique current.
# Geometry: sheared box (n = i + j across the stripes, m = j along them), Nn = P (one period), Nm sites along the stripe;
# one step in m at fixed n is the lattice vector (-1, 1), so the stripe length is L = sqrt(2) Nm and the smallest bending
# wavevector along e = (-1, 1)/sqrt2 is k1 = 2 pi/L. Frame co-moving with the stripe (w1 = u_eff (cos, sin)(45 deg + phi), w2 = 0,
# as code/stripe/conv.py), alpha = 0.3, beta = 0.6.
#  (1) exact linear threshold: converged 1D stationary state (residual < 1e-12) and its Bloch Jacobian at k = k1 e (bisection in u);
#  (2) dynamics: the 1D state copied along m, bending mode n -> n + eps cos(k1 m') seeded (eps = 0.005), Heun T = 0, dt = 0.1;
#      the stripe displacement per row, delta(m) = -P/(2 pi) arg sum_n m_z(n, m) e^{-2 pi i n/P}, is Fourier analysed in m;
#      the complex amplitude F1(t) of the k1 component (real field: F(-k1) = F(k1)*) is the sum of the two branches
#      exp(lambda+- t); lambda+- from a two-exponential Prony fit (samples above 1e-9 of the start, after the first 10 % of the run). u*(dyn) is the zero of max(Gamma+, Gamma-) interpolated linearly in u.
a = 0.3; z = np.zeros(2)
def lin_threshold(P, phi, L, m, S1, lo, hi, it=14):
    k1 = 2*np.pi/L; kv = tuple(k1*eh)
    def g(u):
        S, r = steady(m, S1.copy(), w1_of(u, phi), maxit=60)
        return (1.0 if r > 1e-8 else growth(m, S, w1_of(u, phi), kv)), S, r
    if g(hi)[0] < 0: return None
    for _ in range(it):
        mid = 0.5*(lo+hi)
        if g(mid)[0] > 0: hi = mid
        else: lo = mid
    u = 0.5*(lo+hi); _, S, r = g(u); M = float(S[:, :2].mean(0)@eh)
    return dict(u=u, resid=r, M=M)
def dyn_rate(P, phi, Nm, u, m, D, S1, tmax, dt=0.1, eps=0.005):
    w1 = w1_of(u, phi); S, r = steady(m, S1.copy(), w1, maxit=60)
    n = np.arange(P); mm = np.arange(Nm); th = 2*np.pi/P
    # seeded bending: interpolate the 1D state at n + eps cos(2 pi m/Nm)
    shift = eps*np.cos(2*np.pi*mm/Nm); X = (n[:, None]+shift[None, :]) % P; i0 = np.floor(X).astype(int); f = (X-i0)[..., None]
    S2 = (1-f)*S[i0 % P]+f*S[(i0+1) % P]; S2 /= np.linalg.norm(S2, axis=-1)[..., None]
    I = NL.Integrator(S2, NL.sheared(P, Nm), 1.0, D, 0.0, a, w1, z, 0.0, dt); rec = []; nrec = 400; step = max(1, int(round(tmax/dt/nrec)))
    for k in range(nrec):
        I.run(step); mz = I.S.reshape(P, Nm, 3)[..., 2]; ph = np.angle(np.exp(-1j*th*n)@mz); dl = -np.unwrap(ph)*P/(2*np.pi)
        F = np.fft.fft(dl-dl.mean())/Nm; rec.append([(k+1)*step*dt, F[1].real, F[1].imag])
    R = np.array(rec); x = R[:, 1]+1j*R[:, 2]; ok = (np.abs(x) > 1e-9*np.abs(x[0])) & (np.arange(len(x)) >= len(x)//10); x = x[ok]; tr = step*dt
    # Prony with two exponentials: x[j+2] = p1 x[j+1] + p2 x[j]  ->  z1, z2 = roots, lambda = log(z)/tr (the two branches)
    A = np.stack([x[1:-1], x[:-2]], 1); p1, p2 = np.linalg.lstsq(A, x[2:], rcond=None)[0]; zr = np.roots([1, -p1, -p2]); lam = np.log(zr.astype(complex))/tr
    lam = lam[np.argsort(lam.imag)]; gp, gm = lam[1].real, lam[0].real
    resid_fit = float(np.abs(A@np.array([p1, p2])-x[2:]).max()/np.abs(x).max())
    return dict(u=u, resid=r, Gp=float(gp), Gm=float(gm), lam=[[float(l.real), float(l.imag)] for l in lam], fit_resid=resid_fit, nfit=int(ok.sum()), rec=R[::10].tolist())
if __name__ == '__main__':
    P = int(sys.argv[1]); phi_deg = float(sys.argv[2]); Ls = [float(x) for x in sys.argv[3].split(',')]; mode = sys.argv[4]
    m, th, D = make(P); S1, r1 = steady(m, cycloid(P, th), z); phi = np.radians(phi_deg)
    c0 = {30: 0.0635, 60: 0.0165, 120: 0.00484}[P]; cb = {30: 0.404, 60: 0.502, 120: 0.523}[P]
    fo = f'results_request6/C/C1_P{P}_phi{phi_deg:.0f}_{mode}.json'; out = json.load(open(fo)) if os.path.exists(fo) else {}
    for L in Ls:
        Nm = int(round(L/np.sqrt(2))); Lr = np.sqrt(2)*Nm; key = f'{Nm}'
        if key in out and (mode == 'lin' or 'u_dyn' in out[key]): continue
        def pred(u):
            M = u*np.cos(phi)/D; return np.sqrt(c0**2*(1-M**2)+(2*np.pi*cb/Lr)**2)/np.sin(phi)
        up = pred(c0/np.sin(phi)); [up := pred(up) for _ in range(30)]           # self-consistent in M
        t0 = time.time()
        if mode == 'lin':
            res = lin_threshold(P, phi, Lr, m, S1, 0.5*up, min(2.0*up, 0.95*D/max(np.cos(phi), 1e-9)))
            out[key] = dict(L=Lr, Nm=Nm, u_pred=up, lin=res, wall=time.time()-t0)
            print(f'P={P} phi={phi_deg:.0f} L={Lr:.1f}: u*(Bloch k=2pi/L)={res["u"] if res else None} pred={up:.5f} wall={time.time()-t0:.0f}', flush=True)
        else:
            ul = out.get(key, {}).get('lin', {}) or {}; uc = (ul or {}).get('u', up)
            k1 = 2*np.pi/Lr; tmax = float(np.clip(1.5/(0.3*cb*k1**2), 3000, 60000))         # ~ 1.5 damping times alpha*c_b*k^2 of the mode
            rows = [dyn_rate(P, phi, Nm, uc*f, m, D, S1, tmax) for f in (0.9, 1.0, 1.1)]
            g = np.array([max(r['Gp'], r['Gm']) for r in rows]); us = np.array([r['u'] for r in rows]); pfit = np.polyfit(us, g, 1); ud = float(-pfit[1]/pfit[0])
            out.setdefault(key, dict(L=Lr, Nm=Nm, u_pred=up)); out[key].update(dyn=rows, u_dyn=ud, tmax=tmax, wall_dyn=time.time()-t0)
            print(f'P={P} phi={phi_deg:.0f} L={Lr:.1f}: u*(dynamics)={ud:.5f} (Bloch {uc:.5f}, pred {up:.5f}) rates {np.round(g*1e6, 3)} e-6 tmax={tmax:.0f} wall={time.time()-t0:.0f}', flush=True)
        json.dump(out, open(fo, 'w'), indent=1)
