"""Data for Fig. 7(a): 1D chiral soliton lattice, Doppler overlap vs field against the homogenization formula
lambda = <1/M>/(<psi^-2><psi^2/M>) (code/csl). Writes results/b1_figs/fig7_csl.json."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'csl'))
from csl import relax, kappa, formula
from b1style import DATA
J, D, N = 1.0, 0.3, 40; k = 2*np.pi/(N*40); out = {}
for Kp, Bs in ((5.0, [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]), (0.2, [0.005, 0.01, 0.02, 0.035])):
    rows = []
    for B in Bs:
        m, S = relax(N, J, D, B, Kp); kt, _ = kappa(m, S, k)
        th = np.unwrap(np.arctan2(S[:, 1], S[:, 2])); tp = np.diff(np.concatenate([th, [th[0] + 2*np.pi]])); psi = (tp + np.roll(tp, 1))/2
        Mx = 2*Kp + B*np.cos(th) + 2*D*psi - 2*J*psi**2
        f1 = np.mean(1/Mx)/(np.mean(psi**-2)*np.mean(psi**2/Mx))
        rows.append(dict(B=B, num=float(abs(kt)), formula=float(f1))); print(Kp, B, abs(kt), f1, flush=True)
    out[str(Kp)] = rows
json.dump(out, open(os.path.join(DATA, 'fig7_csl.json'), 'w'), indent=1)
