import sys, os, numpy as np
from scipy import ndimage
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, '..', 'request2'))
from twist import basis
# Unfolding helper shared by dispA.py and mpoint.py: rigid displacement amplitudes of the skyrmions in a Bloch eigenvector.
def trans_basis(m, S):
    # rigid-translation basis of every skyrmion core (tangent components of dS/dx, dS/dy times exp(i k.(r - r_c)))
    e1, e2 = basis(S); gx, gy = m.grads(S); N = m.N; cores = []
    lab, n = ndimage.label(S[:, 2].reshape(m.Lx, m.Ly) < 0)
    I, Jj = np.meshgrid(np.arange(m.Lx), np.arange(m.Ly), indexing='ij'); pos = np.stack([I.ravel(), Jj.ravel()], 1).astype(float)
    for c in range(1, n+1):
        idx = np.nonzero(lab.ravel() == c)[0]
        if len(idx) >= 5: cores.append((idx, pos[idx].mean(0)))
    T = [(np.concatenate([np.sum(g[idx]*e1[idx], 1), np.sum(g[idx]*e2[idx], 1)])) for idx, rc in cores for g in (gx, gy)]
    def amps(v, k):
        # displacement amplitude d_j (2 complex numbers) of every skyrmion: least squares of the core part of v on the
        # Bloch-phased translation vectors
        out = []; R = []
        for c, (idx, rc) in enumerate(cores):
            ph = np.exp(1j*((pos[idx]-rc)@np.asarray(k))); ph2 = np.concatenate([ph, ph])
            G = np.stack([T[2*c]*ph2, T[2*c+1]*ph2], 1); vv = np.concatenate([v[:N][idx], v[N:][idx]])
            out.append(np.linalg.lstsq(G, vv, rcond=None)[0]); R.append(rc)
        return np.array(out), np.array(R)
    return amps
