import sys, json, numpy as np
sys.path.insert(0, 'code/request6'); sys.path.insert(1, 'code/stripe')
from stripeC import dyn_rate
import diag_stripe as ds
from conv import make, cycloid, steady, w1_of, eh
P = 60; phi = np.radians(30); m, th, D = make(P); z = np.zeros(2); S1, _ = steady(m, cycloid(P, th), z)
f = 'results_request6/C/C1_P60_phi30_dyn.json'; out = json.load(open(f))
for Nm, ub, facs in ((45, 0.08402, (1.0, 1.1, 1.2, 1.3, 1.45)), (91, 0.05778, (1.0, 1.05, 1.1, 1.15))):
    L = np.sqrt(2)*Nm; k1 = 2*np.pi/L
    # Bloch spectrum at the Bloch threshold: leading eigenvalues and the bending (displacement) content of the leading one
    S, r = steady(m, S1.copy(), w1_of(ub, phi)); lam = ds.jac(m, S, w1_of(ub, phi), z, 0.3, tuple(k1*eh)); lam = lam[np.argsort(-lam.real)][:4]
    rows = [dyn_rate(P, phi, Nm, ub*fa, m, D, S1, float(np.clip(1.5/(0.3*0.502*k1**2), 3000, 60000))) for fa in facs]
    g = np.array([max(r_['Gp'], r_['Gm']) for r_ in rows]); us = np.array([r_['u'] for r_ in rows]); i = np.where((g[:-1] <= 0) & (g[1:] > 0))[0]
    ud = float(us[i[0]]-g[i[0]]*(us[i[0]+1]-us[i[0]])/(g[i[0]+1]-g[i[0]])) if len(i) else float('nan')
    out[str(Nm)] = dict(L=L, Nm=Nm, dyn=rows, u_dyn=ud, bloch_top=[[float(x.real), float(x.imag)] for x in lam], note='wider bracket (Bloch top mode is not the bending mode here)')
    print(Nm, 'Bloch top eigenvalues at u_Bloch', np.round(lam, 5), '| dyn rates', np.round(g*1e6, 2), 'at', np.round(us, 5), '-> u_dyn', ud, flush=True)
json.dump(out, open(f, 'w'), indent=1)
