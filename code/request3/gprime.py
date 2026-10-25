import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, '..', 'request2'))
from twist import TwistModel, link_blocks, jac_k
# Static soft-mode growth rate of the uniformly sheared lattice (u = 0) as a function of damping: max Re lambda of the
# Bloch Jacobian over a 16 x 16 k grid for the relaxed sheared states s = 5, 6, 7 (gamma = s/19) of request 2, task A.
# Gamma' = d lambda / d gamma from s = 6 and the gamma_s(0) = 0.308 crossing; the static states (and gamma_s) do not depend on alpha.
J, D = 1.0, 0.4; B = 0.5*D*D/J; z = np.zeros(2); nk = 16; out = {}
for s in (5, 6, 7):
    m = TwistModel(19, 33, s, J, D, B); S = np.load(f'results_request2/A_S/state_u0.000_nx1_Lx19_s{s}.npy')
    for a in (0.15, 0.3, 0.6):
        bl, _ = link_blocks(m, S, z, z, a); best = (-np.inf, None)
        for i in range(nk):
            for j in range(nk//2+1):
                kv = ((i/nk)-np.round(i/nk))*m.Bk[0]+(j/nk)*m.Bk[1]; ev = np.linalg.eigvals(jac_k(m, bl, kv))
                if ev.real.max() > best[0]: best = (float(ev.real.max()), [i, j])
        out[f's{s}_a{a}'] = dict(gamma=s/19, alpha=a, maxRe=best[0], k=best[1]); print(s, a, best, flush=True)
gs = 0.30776
for a in (0.15, 0.3, 0.6):
    l6 = out[f's6_a{a}']['maxRe']; out[f'Gprime_a{a}'] = l6/(6/19-gs)
json.dump(out, open('results_request3/gprime.json', 'w'), indent=1)
print({k: v for k, v in out.items() if k.startswith('G')})
