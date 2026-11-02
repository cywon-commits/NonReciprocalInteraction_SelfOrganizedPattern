"""Phenomenon data: diagonal stripes (P = 30) driven along the stripes (phi = 90 deg) at fac x u*(L), sheared box Nn = 4P across,
Nm = 181 along (L = 256), alpha = 0.3, co-moving frame of the stationary stripe, noise 1e-3 (same setup as code/request6/stripeC2.py).
Saves m_z snapshots (Nn x Nm, sheared labels n = i + j, m = j) to results/b1_figs/phen_stripe_f{fac}.npz."""
import sys, os, json, numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(R, 'code', 'request6')); sys.path.insert(0, os.path.join(R, 'code', 'stripe'))
os.chdir(R)
import nbrllg as NL
from conv import make, cycloid, steady, w1_of
P, phi_deg, Nm, fac, tmax, every = 30, 90.0, 181, float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
a = 0.3; z = np.zeros(2); m, th, D = make(P); Nn = 4*P
ub = json.load(open(f'results_request6/C/C1_P{P}_phi{phi_deg:.0f}_lin.json'))[str(Nm)]['lin']['u']; u = fac*ub; w1 = w1_of(u, np.radians(phi_deg))
S1, _ = steady(m, cycloid(P, th), z); S1u, r1 = steady(m, S1.copy(), w1, maxit=60)
nb = NL.sheared(Nn, Nm)
S = np.tile(S1u, (4, 1))[:, None, :].repeat(Nm, 1).reshape(-1, 3) + 1e-3*np.random.default_rng(1).standard_normal((Nn*Nm, 3)); S /= np.linalg.norm(S, axis=1)[:, None]
I = NL.Integrator(S, nb, 1.0, D, 0.0, a, w1, z, 0.0, 0.1); t = 0.0; ts, mz = [], []
while t <= tmax + 1e-9:
    ts.append(t); mz.append(I.S[:, 2].reshape(Nn, Nm).astype(np.float16)); print(f'fac={fac} t={t:.0f}', flush=True)
    I.run(int(round(every/0.1))); t += every
np.savez_compressed(os.path.join(R, 'results', 'b1_figs', f'phen_stripe_f{fac}.npz'), t=np.array(ts), mz=np.array(mz), u=u, ub=ub, resid1d=r1, P=P, Nn=Nn, Nm=Nm)
