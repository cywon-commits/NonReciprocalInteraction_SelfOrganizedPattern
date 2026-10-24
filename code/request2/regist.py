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

# Topological registry without the |d| < a/2 limit. Bragg phase of each skyrmion psi_a = G_a.(r - R0[0]) mod 2 pi equals
# G_a.u mod 2 pi (G_a.R = 0 mod 2 pi on lattice points), so no assignment is needed. On the Delaunay graph of the periodic
# images, e_ij = wrap(psi_j - psi_i) = G_a.(u_j - u_i) as long as neighbours differ by less than pi.
#  - Burgers circuit: sum of e around a triangle = 2 pi b_a; nonzero = dislocation core.
#  - Registry: unwrapping psi on a spanning tree, the mismatch on the remaining edges is 2 pi (offset . m_a), where offset is
#    the net number of box periods the closed path winds; m_a != 0 = Bragg indices changed (rows added/removed).
#  - Displacement: u = Gp^-1 Phi from the unwrapped phases (single valued when m = 0 and no defects), any amplitude.
from scipy.spatial import Delaunay
from collections import deque
def wrap(x): return (x+np.pi) % (2*np.pi)-np.pi
def topo(S):
    Lx, Ly = S.shape[:2]; P = sk_centers(S)[:, :2]; n = len(P)
    offs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    imgs = np.concatenate([P+[a*Lx, b*Ly] for a, b in offs]); own = np.tile(np.arange(n), 9); off = np.repeat(np.array(offs), n, 0)
    T = Delaunay(imgs).simplices
    psi = ((P-R0[0])@Gp.T) % (2*np.pi)                         # (n, 2)
    E = {}
    for t in T:
        for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            for x, y in ((p, q), (q, p)):
                if tuple(off[x]) == (0, 0):
                    E[(own[x], own[y], *(off[y]))] = None
    edges = np.array(list(E.keys()))                              # i, j, ox, oy  (j's image offset relative to canonical i)
    e = wrap(psi[edges[:, 1]]-psi[edges[:, 0]])                   # (m, 2)
    # Burgers circuits on triangles whose first vertex is canonical
    burg = []
    for t in T:
        if not any(tuple(off[v]) == (0, 0) for v in t): continue
        s = 0
        for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])): s = s+wrap(psi[own[q]]-psi[own[p]])
        burg.append(np.round(s/(2*np.pi)))
    burg = np.array(burg); ndef = int((np.abs(burg).sum(1) > 0).sum())
    # spanning tree unwrap
    adj = [[] for _ in range(n)]
    for k, (i, j, ox, oy) in enumerate(edges): adj[i].append((j, k, ox, oy))
    Phi = np.full((n, 2), np.nan); Om = np.zeros((n, 2)); Phi[0] = psi[0]; dq = deque([0]); tree = set()
    while dq:
        i = dq.popleft()
        for j, k, ox, oy in adj[i]:
            if np.isnan(Phi[j, 0]): Phi[j] = Phi[i]+e[k]; Om[j] = Om[i]+[ox, oy]; tree.add(k); dq.append(j)
    rows = []; rhs = []
    for k, (i, j, ox, oy) in enumerate(edges):
        if k in tree: continue
        loop = Om[i]+[ox, oy]-Om[j]; dlt = (Phi[i]+e[k]-Phi[j])/(2*np.pi)
        rows.append(loop); rhs.append(dlt)
    rows = np.array(rows); rhs = np.array(rhs)
    m = np.linalg.lstsq(rows, rhs, rcond=None)[0] if len(rows) and np.abs(rows).sum() > 0 else np.zeros((2, 2))
    resid = float(np.abs(rows@m-rhs).max()) if len(rows) else 0.0
    mi = np.round(m); intact = bool(ndef == 0 and np.abs(mi).sum() == 0 and resid < 0.25)
    u = Phi@np.linalg.inv(Gp).T; u = u-u.mean(0)                  # displacement (unwrapped, mean removed)
    return dict(P=P, u=u, X=P-u, intact=intact, ndef=ndef, m=mi.tolist(), resid=resid, n=n)
