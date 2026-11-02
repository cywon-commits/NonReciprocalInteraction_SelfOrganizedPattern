"""Phenomenon data for Fig. P3: real-space picture of lambda.
For a longitudinal strain wave (k = 2pi/38 along x, unit displacement along x), compare
  affine mode   Phi = -(c . grad) S0 e^{ikx}       (skyrmions deform with the lattice),
  relaxed mode  Psi_rel = H^-1 W K_rel             (static internal relaxation at the same displacement, as in code/b1/dD_ana.py),
and their difference (the counterflow carried by internal canting). Both have the same skyrmion displacement by construction.
Maps use a pure longitudinal displacement (column 0); the momentum density uses the phonon polarization of each subspace
(gyrotropic 2x2 problem, keys *_pol), so its average is the Doppler overlap of that mode. Also the local Doppler-momentum density of each mode (identity metric, kappa_P numerator). Two fields: B = 0.3 and 0.85.
Writes results/b1_figs/phen_relax_B{B}.npz."""
import sys, os, numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.join(R, 'code', 'skx'))
from skx import Model, bloch_jac, basis
J, D, Lx, Ly, nk = 1.0, 0.4, 19, 33, 2; z = np.zeros(2); N = Lx*Ly
for B in (0.3, 0.85):
    S = np.load(os.path.join(R, 'states', f'S_B{B}.npy')); m = Model(Lx, Ly, J, D, B*D*D/J); kx = 2*np.pi/(Lx*nk)
    Jm = bloch_jac(m, S, z, z, 0.0, (kx, 0.)); Om = np.block([[np.zeros((N, N)), np.eye(N)], [-np.eye(N), np.zeros((N, N))]])
    H = -Om @ Jm; H = (H + H.conj().T)/2
    e1, e2 = basis(S); E = [e1.reshape(N, 3), e2.reshape(N, 3)]
    gx = ((np.roll(S, -1, 0) - np.roll(S, 1, 0))/2).reshape(N, 3); gy = ((np.roll(S, -1, 1) - np.roll(S, 1, 1))/2).reshape(N, 3)
    X = (np.arange(Lx)[:, None]*np.ones((1, Ly))).reshape(N); ph = np.exp(1j*kx*X)
    Phi = np.zeros((2*N, 2), complex)
    for c, g in enumerate((gx, gy)): Phi[:N, c] = -np.sum(g*E[0], 1)*ph; Phi[N:, c] = -np.sum(g*E[1], 1)*ph   # -(c.grad)S0
    M = Phi.conj().T @ Om @ Phi; W = Om.T @ Phi @ np.linalg.inv(M).conj().T
    HiW = np.linalg.solve(H, W); Kr = np.linalg.inv(W.conj().T @ HiW); Pr = HiW @ Kr
    out = {}
    def mode(P):              # phonon polarization inside the subspace (as in code/b1/dD_ana.py); highest-frequency branch
        A_ = P.conj().T @ Om @ P; B_ = P.conj().T @ Om @ Jm @ P; lam, C = np.linalg.eig(np.linalg.solve(A_, B_)); i = np.argmax(lam.imag)
        c = C[:, i]; c = c/c[np.argmax(np.abs(c))]; return P @ c          # largest displacement component = 1 (real)
    for nm, v in (('affine', Phi[:, 0]), ('relaxed', Pr[:, 0]), ('affine_pol', mode(Phi)), ('relaxed_pol', mode(Pr))):
        cells = []
        for cidx in range(nk):
            vv = v*np.exp(1j*kx*Lx*cidx)
            d3 = (vv[:N, None]*E[0] + vv[N:, None]*E[1]).reshape(Lx, Ly, 3)      # complex tangent field
            cells.append(d3)
        d3 = np.concatenate(cells, 0); out[nm] = d3
        # momentum density of the mode: Re[psi^* (-i d_x) psi] in the identity metric, per site
        g = (np.roll(d3, -1, 0) - np.roll(d3, 1, 0))/2
        out[nm + '_p'] = np.real(np.sum(np.conj(d3)*(-1j)*g, -1)); out[nm + '_n'] = np.sum(np.abs(d3)**2, -1)
    S2 = np.tile(S, (nk, 1, 1))
    np.savez_compressed(os.path.join(R, 'results', 'b1_figs', f'phen_relax_B{B}.npz'), S=S2, kx=kx, **out)
    pa, pr = out['affine_pol_p'].sum(), out['relaxed_pol_p'].sum(); na, nr = out['affine_pol_n'].sum(), out['relaxed_pol_n'].sum()
    print(B, 'momentum/norm/k affine', pa/na/kx, ' relaxed', pr/nr/kx, flush=True)
