import sys, os, json, glob, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..'); sys.path.insert(0, here)
from margin5 import margin, gs
# Task B: parameter-free prediction of the threshold from the uniform-cell dispersion (task A) and the measured strain-wave profile.
# Local dispersion model, taken directly from the polar grids of task A (no fit to the dynamics, no polynomial):
#   Q(k, theta; state) = lambda/k^2 of the unfolded soft branch on the grid |k| in [0.005, 0.2] (22 log points) x theta in [90, 150];
#   on every eps line Q is taken as a function of the margin m = gamma - gamma_s(eps) at each grid point, either linear
#   (MODEL_FORM=lin, fit over the states with |m| < 0.03: Q = A(k, theta; eps) m + B(k, theta; eps)) or piecewise linear through all
#   computed states (MODEL_FORM=pw; reproduces the independent k_x cuts at k_y = 0.127, 0.19 to 10-20 %, the linear form only to a
#   factor 3 at m = 0.04). Grid points without a soft-branch eigenpair (large k, outside the soft cone; strongly
#   stable) take the most negative Q of their k row. Bilinear interpolation in (log k, theta); |k| < 0.005 uses the k = 0.005 value
#   (lambda ~ k^2), beyond the largest |k| of the grid (0.2, or 0.4 with the disphi continuation) Q is held (stable); theta clamped to [90, 150] (plateau, stable);
#   eps interpolated linearly between the four lines and clamped to [-0.105, 0] (the margin always uses the full map gamma_s(eps));
#   lambda(-k) = lambda(k). The soft branch is the larger of the two acoustic branches; the other one is insensitive to m and sits on
#   the plateau Q ~ -0.23 seen at theta = 90 and 150 deg, so the linear form is floored there: Q = max(A m + B, F(k; eps)),
#   F = min over theta of B (without the floor the linear extrapolation to the strongly negative margins of the strain wave,
#   m down to -0.57, gives Q ~ -3 and a symbol so rough that the quantised operator has spurious positive eigenvalues).
# Operator for every allowed k_y = 2 pi n/L_y (n = 0 ... with k_y <= 0.2): Weyl quantisation in the plane-wave basis
#   k_x = 2 pi j/L_x, |k_x| <= kmax: <k|L|k'> = (1/L_x) int dx exp(-i(k-k')x) lambda((k+k')/2, k_y; H(x)).
# Alternative ordering (scheme dependence, 'sym'): <k|L|k'> = (1/2)[lambda~(k; q) + lambda~(k'; q)], lambda~(k; q) = Fourier
#   component q = k - k' of lambda(k, k_y; H(x)).
# Advection check: + i v_rel k_x on the diagonal (frame of the strain wave, v_rel = 2.7/L_x), largest real part of the eigenvalues.
TH = np.arange(90, 151, 3.0)
def build_model():
    disp = {}
    for f in glob.glob('results_request5/A/disp_nx8_Lx*_s*_a0.3.json'):
        d = json.load(open(f)); disp[(d['Lx'], d['s'])] = d
    lines = {}
    for (Lx, s), d in disp.items(): lines.setdefault(Lx, []).append(d)
    # continuation |k| in [0.23, 0.40] (disphi_*.json) appended when it exists for every state of every line
    hi = all(os.path.exists(f"results_request5/A/disphi_nx8_Lx{d['Lx']}_s{d['s']}_a0.3.json") for d in disp.values()) and os.environ.get('KHI', '1') == '1'
    M = {}
    for Lx, ds in lines.items():
        eps = ds[0]['exx']; km = np.array(ds[0]['km']); ms, Qs = [], []
        if hi: km = np.concatenate([km, json.load(open(f"results_request5/A/disphi_nx8_Lx{Lx}_s{ds[0]['s']}_a0.3.json"))['km']])
        for d in ds:
            P = np.array(d['polar'])[..., 1]
            if hi: P = np.concatenate([P, np.array(json.load(open(f"results_request5/A/disphi_nx8_Lx{Lx}_s{d['s']}_a0.3.json"))['polar'])[..., 1]])
            Q = P/km[:, None]**2
            for i in range(len(km)):
                bad = ~np.isfinite(Q[i])
                if bad.any(): Q[i, bad] = np.nanmin(Q[i]) if (~bad).any() else np.nan
            for i in range(1, len(km)):                                   # whole rows missing: carry the previous row
                if not np.isfinite(Q[i]).any(): Q[i] = Q[i-1]
            ms.append(d['gamma']-gs(eps)); Qs.append(Q)
        o = np.argsort(ms); ms = np.array(ms)[o]; Qs = np.array(Qs)[o]; near = np.abs(ms) < 0.03
        X = np.stack([ms[near], np.ones(near.sum())], 1); cf = np.linalg.lstsq(X, Qs[near].reshape(near.sum(), -1), rcond=None)[0]
        A_, B_ = cf[0].reshape(Qs.shape[1:]), cf[1].reshape(Qs.shape[1:])
        M[eps] = dict(A=A_, B=B_, F=B_.min(1), lk=np.log(km), ms=ms, Qs=Qs)
    return M
# MODEL_FORM = 'lin' (default): Q linear in m from the states with |m| < 0.03; 'pw': piecewise linear in m through all the states of the
# line (exact at every computed state), extrapolated beyond the computed range with the slope of the last two states at each end.
FORM = os.environ.get('MODEL_FORM', 'lin')
def q_of_m(M, G, m):
    # G: Q values of the states at one interpolated grid point (len(ms)); m: array over x
    if FORM == 'lin': return None
    ms = M['ms']; q = np.interp(m, ms, G)
    lo = m < ms[0]; hi = m > ms[-1]
    q[lo] = G[0]+(m[lo]-ms[0])*(G[1]-G[0])/(ms[1]-ms[0]); q[hi] = G[-1]+(m[hi]-ms[-1])*(G[-1]-G[-2])/(ms[-1]-ms[-2])
    return q
MODEL = build_model(); KHI = len(next(iter(MODEL.values()))['lk']) > 22; EPS = np.array(sorted(MODEL)); LK = MODEL[EPS[0]]['lk']
def lam_local(kx, ky, m, eps):
    # kx, ky scalars; m, eps arrays over x
    k = np.hypot(kx, ky)
    if k == 0: return np.zeros_like(m)
    th = np.clip(np.degrees(np.arctan2(ky, kx)) % 180, TH[0], TH[-1]); lk = np.clip(np.log(k), LK[0], LK[-1])
    i = int(np.clip(np.searchsorted(LK, lk)-1, 0, len(LK)-2)); u = (lk-LK[i])/(LK[i+1]-LK[i])
    j = int(np.clip(np.searchsorted(TH, th)-1, 0, len(TH)-2)); v = (th-TH[j])/(TH[j+1]-TH[j])
    def bil(G): return (1-u)*(1-v)*G[i, j]+u*(1-v)*G[i+1, j]+(1-u)*v*G[i, j+1]+u*v*G[i+1, j+1]
    fl = np.array([(1-u)*MODEL[e]['F'][i]+u*MODEL[e]['F'][i+1] for e in EPS])
    e = np.clip(eps, EPS[0], EPS[-1]); F_ = np.interp(e, EPS, fl)
    if FORM == 'lin':
        a = np.array([bil(MODEL[e_]['A']) for e_ in EPS]); b = np.array([bil(MODEL[e_]['B']) for e_ in EPS])
        Q = np.interp(e, EPS, a)*m+np.interp(e, EPS, b)
    else:
        # Q on each eps line at the local m, then linear in eps between the lines
        QL = np.array([q_of_m(MODEL[e_], bil(np.moveaxis(MODEL[e_]['Qs'], 0, -1)), m) for e_ in EPS])
        t = np.clip(np.searchsorted(EPS, e)-1, 0, len(EPS)-2); w = (e-EPS[t])/(EPS[t+1]-EPS[t]); ix = np.arange(len(m))
        Q = (1-w)*QL[t, ix]+w*QL[t+1, ix]
    return np.maximum(Q, F_)*k**2
def operator(xs, m, eps, Lx, ky, scheme='weyl', vrel=0.0, kmax=None):
    kmax = kmax or (0.35 if KHI else 0.2)
    J = int(kmax*Lx/(2*np.pi)); jj = np.arange(-J, J+1)
    if ky == 0: jj = jj[jj != 0]                                           # k = 0: uniform translation (Goldstone), not a mode
    ks = 2*np.pi*jj/Lx; n = len(ks); H = np.zeros((n, n), complex); N = len(xs)
    if scheme == 'weyl':
        cache = {}
        for a_ in range(n):
            for b_ in range(n):
                kb = 0.5*(ks[a_]+ks[b_]); key = round(kb, 10)
                if key not in cache: cache[key] = np.fft.fft(lam_local(kb, ky, m, eps))/N
                q = jj[a_]-jj[b_]; H[a_, b_] = cache[key][q % N]
    else:
        F = [np.fft.fft(lam_local(k, ky, m, eps))/N for k in ks]
        for a_ in range(n):
            for b_ in range(n):
                q = jj[a_]-jj[b_]; H[a_, b_] = 0.5*(F[a_][q % N]+F[b_][q % N])
    H += np.diag(1j*vrel*ks)
    return ks, H
def solve(xs, m, eps, Lx, Ly, scheme='weyl', vrel=0.0):
    best = None; per_n = []
    for nn in range(0, 40):
        ky = 2*np.pi*nn/Ly
        if ky > (0.35 if KHI else 0.2): break
        ks, H = operator(xs, m, eps, Lx, ky, scheme, vrel)
        ev, V = np.linalg.eig(H) if vrel else np.linalg.eigh(0.5*(H+H.conj().T))
        i = int(np.argmax(ev.real)); per_n.append((nn, float(ky), float(ev[i].real)))
        if best is None or ev[i].real > best[0]:
            v = V[:, i]; f = np.exp(1j*np.outer(xs, ks))@v; w = np.abs(f)**2; w /= w.sum()
            xc = np.angle(np.sum(w*np.exp(2j*np.pi*xs/Lx)))*Lx/(2*np.pi) % Lx; dx = (xs-xc+Lx/2) % Lx-Lx/2
            rms = float(np.sqrt(np.sum(w*dx**2))); kxc = float(np.sum(np.abs(v)**2*np.abs(ks))/np.sum(np.abs(v)**2))
            best = (float(ev[i].real), nn, float(ky), xc, rms, kxc, w)
    return best, per_n
if __name__ == '__main__':
    # consistency check (acceptance 2): uniform margin -> the uniform-cell value
    xs = np.arange(2000)*1520/2000; mm = np.full(len(xs), 0.0159); ee = np.zeros(len(xs))
    (lam0, n0, ky0, *_), _ = solve(xs, mm, ee, 1520, 1520.0)
    kk = np.linspace(0.005, 0.2, 400); grid = max(lam_local(k*np.cos(np.radians(t)), k*np.sin(np.radians(t)), np.array([0.0159]), np.array([0.0]))[0] for k in kk for t in TH)
    print('uniform check: operator max', lam0, 'model max over k', grid, 'uniform-cell grid (gamma 0.3092)', json.load(open('results_request5/A_summary.json'))['lines']['152']['kstar'][5])
    out = {}
    cases = []
    R3 = 'results_request3/T0'; R4 = 'results_request4/C/states'
    for size, nx in (('8x2', 8), ('12x3', 12), ('16x4', 16), ('24x6', 24)):
        fs = sorted(glob.glob(f'{R3}/{size}_s1/S_r*.npy'))+sorted(glob.glob(f'{R4}/{size}/S_r*.npy'))
        if size == '24x6': fs += sorted(glob.glob(f'results_request5/C1/states/24x6/S_r*.npy'))
        for f in fs: cases.append((size, nx, f))
    cases += [('32x8', 32, f) for f in sorted(glob.glob('results_request3/A32/chain/S_r*.npy'))]
    for size, nx, f in cases:
        jf = f.replace('.npy', '.json')
        if os.path.exists(jf) and not json.load(open(jf)).get('intact', True): continue
        try: mg = margin(np.load(f).astype(float))
        except Exception: continue
        Lx = 19*nx; Ly = 33*nx/4; xs = np.array(mg['xs']); m = np.array(mg['m']); e = np.array(mg['e'])
        sel = slice(0, None, 10)                                           # 400 x points
        (lam, n, ky, xc, rms, kxc, w), per_n = solve(xs[sel], m[sel], e[sel], Lx, Ly)
        (lams, *_), _ = solve(xs[sel], m[sel], e[sel], Lx, Ly, scheme='sym')
        (lamv, *_), _ = solve(xs[sel], m[sel], e[sel], Lx, Ly, vrel=2.7/Lx)
        off = ((xc-mg['x_peak'])+Lx/2) % Lx-Lx/2
        out[f] = dict(size=size, Lx=Lx, gamma_wave=mg['g_peak'], mmaxL=mg['mmaxL'], m2L2=mg['m2L2'], lam=lam, lam_sym=lams, lam_adv=lamv, n=n, ky=ky, kx=kxc,
                      rms=rms, offset=float(off), per_n=per_n)
        print(f"{size} {os.path.basename(f)} gw={mg['g_peak']:.4f} Lambda0={lam:+.3e} (sym {lams:+.3e}, adv {lamv:+.3e}) n={n} ky={ky:.4f} <|kx|>={kxc:.4f} rms={rms:.1f} offset={off:+.1f}", flush=True)
    json.dump(out, open(f'results_request5/B_weyl{"" if FORM == "lin" else "_"+FORM}.json', 'w'), indent=1)
