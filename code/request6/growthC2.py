import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..'); sys.path.insert(1, os.path.join(root, 'code', 'stripe'))
import diag_stripe as ds
from conv import make, cycloid, steady, w1_of, eh
# Request 6, task C2 (Masell et al. comparison): linear growth of the bending instability of the infinite stripe above threshold.
# Converged stationary state at u, Bloch growth Re lambda(k e) on a log grid of k along the stripe; Gamma_max(u) and k*(u).
z = np.zeros(2); out = {}
for P in (30, 60):
    m, th, D = make(P); S1, _ = steady(m, cycloid(P, th), z); c0 = {30: 0.0635, 60: 0.0165}[P]
    for phi_deg in (90, 60):
        phi = np.radians(phi_deg); u0 = c0/np.sqrt(np.sin(phi)**2+(c0/D)**2*np.cos(phi)**2); ks = np.geomspace(0.003, 0.4, 40); rows = []
        for f in (1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0):
            u = f*u0
            if u*np.cos(phi) > 0.9*D: continue
            S, r = steady(m, S1.copy(), w1_of(u, phi), maxit=80)
            if r > 1e-8: rows.append(dict(f=f, u=u, resid=r)); continue
            g = []
            for k in ks:
                lam = ds.jac(m, S, w1_of(u, phi), z, 0.3, tuple(k*eh)); lam = lam[np.abs(lam) > 1e-9]; g.append(lam.real.max())
            g = np.array(g); i = int(np.argmax(g)); rows.append(dict(f=f, u=u, resid=r, k_star=float(ks[i]), G_max=float(g[i]), g=g.tolist()))
            print(f'P={P} phi={phi_deg} u/u*={f}: u={u:.4f} k*={ks[i]:.4f} Gamma_max={g[i]:.3e} resid={r:.0e}', flush=True)
        out[f'P{P}_phi{phi_deg}'] = dict(u0=u0, ks=ks.tolist(), rows=rows)
        json.dump(out, open('results_request6/C2_growth.json', 'w'), indent=1)
