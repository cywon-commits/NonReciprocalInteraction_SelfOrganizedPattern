import numpy as np
# Square-lattice Neel SkX model (same physics as code/skx/skx.py) on a cell with a twisted periodic boundary:
#   S(x + Lx, y) = S(x, y - s),  S(x, y + Ly) = S(x, y)
# i.e. lattice translations A1 = (Lx, s), A2 = (0, Ly). With the reference cell (Lx0, Ly) this is a uniform
# displacement gradient du_y/dx = s/Lx0 (simple shear gamma) and du_x/dx = Lx/Lx0 - 1.
# Spins are stored flat, shape (N, 3), site index i*Ly + j. Each neighbour link carries the lattice translation T
# between the unwrapped neighbour position and its canonical site; a Bloch perturbation d gets the phase exp(i k.T).
zh = np.array([0, 0, 1.]); xh = np.array([1., 0, 0]); yh = np.array([0, 1., 0])
class TwistModel:
    def __init__(s, Lx, Ly, sh, J, D, B):
        s.Lx, s.Ly, s.sh, s.J, s.D, s.B = Lx, Ly, sh, J, D, B; s.N = Lx*Ly
        I, Jj = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing='ij'); I = I.ravel(); Jj = Jj.ravel()
        s.idx = {}; s.T = {}
        for name, (di, dj) in {'xp': (1, 0), 'xm': (-1, 0), 'yp': (0, 1), 'ym': (0, -1)}.items():
            x, y = I+di, Jj+dj; Tx = np.zeros(s.N); Ty = np.zeros(s.N)
            hi = x == Lx; lo = x == -1
            x = np.where(hi, 0, np.where(lo, Lx-1, x)); y = np.where(hi, y-sh, np.where(lo, y+sh, y))
            Tx += np.where(hi, Lx, 0)-np.where(lo, Lx, 0); Ty += np.where(hi, sh, 0)-np.where(lo, sh, 0)
            w = np.floor_divide(y, Ly); y = y-w*Ly; Ty += w*Ly
            s.idx[name] = x*Ly+y; s.T[name] = np.stack([Tx, Ty], 1)
        s.A = np.array([[Lx, sh], [0, Ly]], float)                     # rows: A1, A2
        s.Bk = 2*np.pi*np.linalg.inv(s.A).T                            # rows: reciprocal vectors
    def nb(s, S0, d, eps, name, k):
        out = S0[s.idx[name]]
        if d is None: return out
        ph = np.exp(1j*(s.T[name]@np.asarray(k, float)))
        return out+eps*d[s.idx[name]]*ph[:, None]
    def rhs(s, S0, w1, w2, a, d=None, eps=0.0, k=(0., 0.)):
        S = S0 if d is None else S0+eps*d
        xp = s.nb(S0, d, eps, 'xp', k); xm = s.nb(S0, d, eps, 'xm', k); yp = s.nb(S0, d, eps, 'yp', k); ym = s.nb(S0, d, eps, 'ym', k)
        H = s.J*(xp+xm+yp+ym)+s.B*zh+s.D*(np.cross(xp, yh)+np.cross(yh, xm)+np.cross(yp, -xh)+np.cross(-xh, ym))
        gx = (xp-xm)/2; gy = (yp-ym)/2
        T = -np.cross(S, H)+w1[0]*gx+w1[1]*gy+np.cross(S, w2[0]*gx+w2[1]*gy)
        T = T-np.sum(T*S, -1)[..., None]*S
        return (T+a*np.cross(S, T))/(1+a*a)
    def energy(s, S):
        xp = S[s.idx['xp']]; yp = S[s.idx['yp']]
        return float(-s.J*np.sum(S*xp)-s.J*np.sum(S*yp)-s.D*np.sum(np.cross(S, xp)[:, 1])+s.D*np.sum(np.cross(S, yp)[:, 0])-s.B*np.sum(S[:, 2]))
    def grads(s, S): return (S[s.idx['xp']]-S[s.idx['xm']])/2, (S[s.idx['yp']]-S[s.idx['ym']])/2
def rk4(m, S, w1, w2, a, dt, n):
    for _ in range(n):
        k1 = m.rhs(S, w1, w2, a); k2 = m.rhs(S+dt/2*k1, w1, w2, a); k3 = m.rhs(S+dt/2*k2, w1, w2, a); k4 = m.rhs(S+dt*k3, w1, w2, a)
        S = S+dt/6*(k1+2*k2+2*k3+k4); S /= np.linalg.norm(S, axis=-1)[..., None]
    return S
def GD(m, S):
    gx, gy = m.grads(S); G = np.sum(S*np.cross(gx, gy))
    Dm = np.array([[np.sum(gx*gx), np.sum(gx*gy)], [np.sum(gx*gy), np.sum(gy*gy)]])
    return np.array([[0, G], [-G, 0]]), Dm, G/(4*np.pi)
def relax(m, S, tol=1e-10, a=0.5, dt=0.1, chunk=1000, maxit=60):
    z = np.zeros(2)
    for it in range(maxit):
        S = rk4(m, S, z, z, a, dt, chunk); r = np.abs(m.rhs(S, z, z, a)).max()
        if r < tol: break
    return S, r
def find_V(m, S, u, a, b, iters=12, dt=0.1, n=1500):
    for it in range(iters):
        Gm, Dm, _ = GD(m, S); V = np.linalg.solve(Gm-a*Dm, Gm@u-b*Dm@u); w1 = V-u; w2 = b*u-a*V
        S = rk4(m, S, w1, w2, a, dt, n)
    return S, V, w1, w2, np.abs(m.rhs(S, w1, w2, a)).max()
def basis(S):
    a = np.where((np.abs(S[..., 0]) < 0.9)[..., None], xh, yh)
    e1 = np.cross(S, a); e1 /= np.linalg.norm(e1, axis=-1)[..., None]; return e1, np.cross(S, e1)
def colors(m, r=3):
    # sites grouped so that sites of one group are > 2 links apart (their 5-point stencils do not overlap)
    col = -np.ones(m.N, int); nbs = [m.idx[k] for k in ('xp', 'xm', 'yp', 'ym')]
    near = [set() for _ in range(m.N)]
    for i in range(m.N):
        fr = {i}
        for _ in range(2):
            fr = fr | {int(n[j]) for j in fr for n in nbs}
        near[i] = fr
    for i in range(m.N):
        used = {col[j] for j in near[i] if col[j] >= 0}; c = 0
        while c in used: c += 1
        col[i] = c
    return col
def link_blocks(m, S, w1, w2, a, eps=1e-6, col=None):
    # real-space Jacobian blocks in the tangent basis: response of site i to site j through each link (and on-site),
    # computed with coloured finite differences (k = 0); J(k) is then assembled with the Bloch phases of the links.
    e1, e2 = basis(S); E = [e1, e2]; col = colors(m) if col is None else col
    ncol = col.max()+1; src = {'self': np.arange(m.N)}; src.update({k: m.idx[k] for k in ('xp', 'xm', 'yp', 'ym')})
    blocks = {k: np.zeros((m.N, 2, 2)) for k in src}
    for c in range(ncol):
        sel = col == c
        for cc in range(2):
            d = np.zeros((m.N, 3)); d[sel] = E[cc][sel]
            f = (m.rhs(S, w1, w2, a, d, eps, (0., 0.))-m.rhs(S, w1, w2, a, d, -eps, (0., 0.)))/(2*eps)
            fp = np.stack([np.sum(f*e1, 1), np.sum(f*e2, 1)], 1)       # response of every site
            for k, j in src.items():
                # site i responds to its source j = src[k][i] if j belongs to this colour; the stencil of j contains i
                hit = sel[j]
                blocks[k][hit, :, cc] = fp[hit].real
    return blocks, col
def jac_k(m, blocks, k):
    # J(k) in the tangent basis, ordering [e1 components of all sites, e2 components of all sites]
    N = m.N; Jm = np.zeros((2*N, 2*N), complex); ii = np.arange(N)
    for name, Bl in blocks.items():
        j = ii if name == 'self' else m.idx[name]
        ph = np.ones(N) if name == 'self' else np.exp(1j*(m.T[name]@np.asarray(k, float)))
        for r in range(2):
            for c in range(2):
                np.add.at(Jm, (r*N+ii, c*N+j), Bl[:, r, c]*ph)
    return Jm
