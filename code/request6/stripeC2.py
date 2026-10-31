import sys, os, json, time, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
sys.path.insert(0, here); sys.path.insert(1, os.path.join(root, 'code', 'stripe'))
import nbrllg as NL
from conv import make, cycloid, steady, w1_of
# Request 6, task C2: end state of the diagonal stripe above the finite-box threshold. Helical sheared box with Nn = 4P (four
# stripes across, so that dislocations can form) and Nm = L/sqrt2 along the stripes; frame co-moving with the stationary stripe
# (w1 = u_eff dir(45 deg + phi), w2 = 0), u_eff = fac x u*(L) (C1 Bloch value), alpha = 0.3, Heun dt = 0.1, T = 0.
# Start: the stationary 1D state (unstable above threshold) + noise (1e-3; optional 6th argument, e.g. 1e-2 for the slow
# P = 60, 1.5 x runs, tmax 4e5). Every `every` time units:
#   stripe phase theta = atan2(S.q, S_z) (q = (1,1,0)/sqrt2): per row m and stripe, displacement delta(m) from the phase of
#   sum_n exp(i theta - 2 pi i n/P) -> rms bending amplitude A_b and its leading along-stripe Fourier index; mean tilt
#   <d theta/d m>/(2 pi/P) (reorientation of q: zero for straight diagonal stripes); dislocations = plaquettes with phase
#   winding +-2 pi (count of |w| = 1); <m_z>, |<S_inplane>|.
a = 0.3; z = np.zeros(2)
P, phi_deg, L, fac, tmax = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]); every = 100.0
noise = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-3; suffix = '' if noise == 1e-3 else f'_n{noise:g}'
m, th, D = make(P); phi = np.radians(phi_deg); Nm = int(round(L/np.sqrt(2))); Nn = 4*P
ub = json.load(open(f'results_request6/C/C1_P{P}_phi{phi_deg:.0f}_lin.json'))[str(Nm)]['lin']['u']; u = fac*ub; w1 = w1_of(u, phi)
tag = f'P{P}_phi{phi_deg:.0f}_L{Nm}_f{fac}'+suffix; fo = f'results_request6/C/C2_{tag}.json'; ck = f'results_request6/C/ckpt_C2_{tag}.npz'
S1, _ = steady(m, cycloid(P, th), z); S1u, r1 = steady(m, S1.copy(), w1, maxit=60)
nb = NL.sheared(Nn, Nm); xp, xm, yp, ym = nb; corner4 = xm[yp]                 # (n, m+1) = -x neighbour of the +y neighbour
if os.path.exists(ck):
    zc = np.load(ck); S = zc['S'].astype(float); t = float(zc['t']); rec = json.load(open(fo))['rec']
else:
    S = np.tile(S1u, (4, 1))[:, None, :].repeat(Nm, 1).reshape(-1, 3)+noise*np.random.default_rng(1).standard_normal((Nn*Nm, 3)); S /= np.linalg.norm(S, axis=1)[:, None]; t = 0.0; rec = []
qh = np.array([1, 1, 0])/np.sqrt(2)
def wrap(x): return (x+np.pi) % (2*np.pi)-np.pi
def observe(S):
    thf = np.arctan2(S@qh, S[:, 2]); T2 = thf.reshape(Nn, Nm)
    w = wrap(thf[xp]-thf)+wrap(thf[yp]-thf[xp])+wrap(thf[corner4]-thf[yp])+wrap(thf-thf[corner4]); nd = int(np.sum(np.abs(np.round(w/(2*np.pi))) > 0))
    ph = np.angle(np.exp(1j*(T2-2*np.pi*np.arange(Nn)[:, None]/P)).sum(0)); dl = np.unwrap(ph)*P/(2*np.pi)
    tilt = float((dl[-1]-dl[0]+(ph[0]-ph[-1] if False else 0))/(Nm-1))
    F = np.abs(np.fft.rfft(dl-np.polyval(np.polyfit(np.arange(Nm), dl, 1), np.arange(Nm))))/Nm
    return [float(np.std(dl-np.polyval(np.polyfit(np.arange(Nm), dl, 1), np.arange(Nm)))), int(np.argmax(F[1:])+1), tilt, nd, float(S[:, 2].mean()), float(np.hypot(*S[:, :2].mean(0)))]
I = NL.Integrator(S, nb, 1.0, D, 0.0, a, w1, z, 0.0, 0.1); t0 = time.time()
while t < tmax-1e-9:
    I.run(int(round(every/0.1))); t += every; rec.append([t]+observe(I.S))
    if int(round(t)) % 10000 == 0 or t >= tmax-1e-9:
        json.dump(dict(P=P, phi=phi_deg, L=np.sqrt(2)*Nm, Nm=Nm, Nn=Nn, fac=fac, u=u, u_bloch=ub, resid1d=r1, cols=['t', 'A_b', 'mode', 'tilt', 'n_disloc', 'mz', 'm_inplane'], rec=rec), open(fo, 'w'))
        np.savez(ck+'.tmp.npz', S=I.S.astype(np.float32), t=t); os.replace(ck+'.tmp.npz', ck)
        print(f'{tag} t={t:.0f} A_b={rec[-1][1]:.3f} mode={rec[-1][2]} tilt={rec[-1][3]:+.4f} disloc={rec[-1][4]} wall={time.time()-t0:.0f}', flush=True)
np.save(f'results_request6/C/last_C2_{tag}.npy', I.S.reshape(Nn, Nm, 3).astype(np.float32)); print('done', flush=True)
