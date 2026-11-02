"""Bloch prediction for the driven skyrmion lattice of skx_dyn.py (u = 0.15, alpha = 0.3, beta = 0.6): growth rate of the
k = 2pi/76 mode on the co-moving stationary state and the skyrmion displacements of its eigenvector in the 76x33 strip.
Writes results/b1_figs/phen_skx_mode.json."""
import sys, os, json, numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.join(R, 'code', 'skx'))
from skx import Model, find_V, bloch_jac, basis
J, D, a, b, u = 1.0, 0.4, 0.3, 0.6, 0.15; Lx, Ly = 19, 33
S0 = np.load(os.path.join(R, 'states', 'S_B0.5.npy')); m = Model(Lx, Ly, J, D, 0.5*D*D/J)
Su, V, w1, w2, r = find_V(m, S0.copy(), np.array([u, 0.]), a, b, iters=10, n=1500)
out = dict(u=u, V=V.tolist(), resid=float(r), modes=[])
for nk in (4, 2):
    k = (2*np.pi/(Lx*nk), 0.)
    lam, vec = np.linalg.eig(bloch_jac(m, Su, w1, w2, a, k)); i = np.argmax(lam.real); v = vec[:, i]; N = Lx*Ly
    e1, e2 = basis(Su); psi = (v[:N, None]*e1.reshape(N, 3) + v[N:, None]*e2.reshape(N, 3)).reshape(Lx, Ly, 3)
    gx = (np.roll(Su, -1, 0) - np.roll(Su, 1, 0))/2; gy = (np.roll(Su, -1, 1) - np.roll(Su, 1, 1))/2
    X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing='ij'); dx = (X + Lx//2) % Lx - Lx//2; dy = (Y + Ly//2) % Ly - Ly//2
    disp = []
    for c in range(nk):                       # skyrmion at (0,0) and at (9.5,16.5) of each cell; displacement = -projection on grad S
        ph = np.exp(1j*k[0]*Lx*c)
        for (ci, cj) in ((0, 0), (9, 16)):
            win = ((X - ci + Lx//2) % Lx - Lx//2)**2 + ((Y - cj + Ly//2) % Ly - Ly//2)**2 < 25
            M = np.array([[np.sum(g1*g2*win[..., None]) for g2 in (gx, gy)] for g1 in (gx, gy)])
            f = np.array([np.sum(np.real(psi*ph)*g*win[..., None]) for g in (gx, gy)])
            disp.append([c*Lx + ci + (0.5 if cj else 0), cj + (0.5 if cj else 0)] + (-np.linalg.solve(M, f)).tolist())
    out['modes'].append(dict(nk=nk, k=k[0], growth=float(lam[i].real), omega=float(lam[i].imag), disp=disp))
    print(nk, 'growth', lam[i].real, 'omega', lam[i].imag, flush=True)
json.dump(out, open(os.path.join(R, 'results', 'b1_figs', 'phen_skx_mode.json'), 'w'), indent=1)
