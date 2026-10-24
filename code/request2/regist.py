import numpy as np
from strain import R0, ideal, mind
from melt import sk_centers
# Robust version of strain.disp for a moving lattice: the global translation is first taken from the Bragg phases
# phi_a = arg <exp(i G_a . (r - R0[0]))> (two primitive reciprocal vectors), so the nearest-point assignment does not
# become ambiguous when the lattice has drifted by about half a lattice spacing. Otherwise identical to strain.disp.
A1 = np.array([19.0, 0.0]); A2 = R0[1]-R0[0]
Gp = 2*np.pi*np.linalg.inv(np.stack([A1, A2])).T          # rows: G1, G2 with G_i . A_j = 2 pi delta_ij
def disp(S, nx, ny):
    Lx, Ly = S.shape[:2]; L = np.array([Lx, Ly]); P = sk_centers(S)[:, :2]; I = ideal(nx, ny)
    ph = np.angle(np.exp(1j*((P-R0[0])@Gp.T)).mean(0))       # Bragg phases of the two primitive vectors
    sh = np.linalg.solve(Gp, ph)
    for it in range(6):
        D = mind((P[:, None, :]-sh)-I[None, :, :], L); j = np.argmin((D**2).sum(-1), 1)
        d = mind(P-sh-I[j], L); sh = sh+d.mean(0)
    ok = len(set(j)) == len(P) == len(I)
    d = mind(P-sh-I[j], L); return I[j], d, ok
