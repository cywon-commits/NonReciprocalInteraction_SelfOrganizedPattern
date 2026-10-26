import sys, os, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
os.environ.setdefault('SKX_CELL', os.path.join(root, 'states', 'S_B0.5.npy'))
sys.path.insert(0, os.path.join(root, 'code', 'request2'))
from regist import R0, Gp, wrap, Delaunay, deque
from melt import sk_centers
# Topological registry (request 2 regist.topo) for a box that is uniformly stretched along x by sx = Lx/Lx_ref (request 4,
# task B): skyrmion positions are mapped to the unstretched reference coordinates x' = x/sx (box length Lx/sx), where the
# reference lattice is the natural 19 x 33 lattice. Displacements u and the reference positions X are then in natural
# coordinates, so du_y/dX is the total simple shear gamma (as in the twisted cell, gamma = s/Lx0) and the total
# du_x/dx of the lattice is sx (1 + du_x/dX) - 1 = eps0 + sx du_x/dX.
def topo_s(S, sx=1.0):
    P = sk_centers(S)[:, :2].copy(); P[:, 0] /= sx; Lx, Ly = S.shape[0]/sx, S.shape[1]
    return topoP(P, Lx, Ly)
def topoP(P, Lx, Ly):
    n = len(P)
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
