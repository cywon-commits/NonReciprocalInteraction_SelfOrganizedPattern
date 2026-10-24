import sys, os, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
os.environ.setdefault('SKX_CELL', os.path.join(root, 'states', 'S_B0.5.npy'))
sys.path.insert(0, here)
from regist import topo
# Scaled seed (scale invariance r -> s r, u -> s u): the displacement field of a small saturated state, d(x),
# is fitted by a Fourier series on its torus, stretched to d_big(r) = s d(r/s), and applied to the perfect
# big lattice: S(r) = S0(r - d_big(r)) (bilinear interpolation inside the periodic 19x33 cell).
# usage: python seed.py small.npy nx0 ny0 nx ny out.npy
def fit(X, d, Lx, Ly, mmax, nmax):
    ks = [(m, n) for m in range(-mmax, mmax+1) for n in range(-nmax, nmax+1)]
    ph = np.exp(2j*np.pi*(np.outer(X[:, 0], [m/Lx for m, n in ks])+np.outer(X[:, 1], [n/Ly for m, n in ks])))
    A = np.concatenate([ph.real, -ph.imag], 1)
    c = [np.linalg.lstsq(A, d[:, i], rcond=None)[0] for i in range(2)]
    def ev(x, y):
        P = np.exp(2j*np.pi*(np.multiply.outer(x, [m/Lx for m, n in ks])+np.multiply.outer(y, [n/Ly for m, n in ks])))
        return np.stack([np.concatenate([P.real, -P.imag], -1)@ci for ci in c], -1)
    return ev
def bilinear(S0, x, y):
    Lx, Ly = S0.shape[:2]; x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = (x-x0)[..., None]; fy = (y-y0)[..., None]
    g = lambda i, j: S0[i % Lx, j % Ly]
    S = (1-fx)*(1-fy)*g(x0, y0)+fx*(1-fy)*g(x0+1, y0)+(1-fx)*fy*g(x0, y0+1)+fx*fy*g(x0+1, y0+1)
    return S/np.linalg.norm(S, axis=-1)[..., None]
def make(Ssmall, nx0, ny0, nx, ny):
    s = nx/nx0; assert abs(ny/ny0-s) < 1e-12, 'aspect ratio must be kept'
    Lx0, Ly0 = 19*nx0, 33*ny0; r = topo(Ssmall); X, d, ok = r['X'], r['u'], r['intact']; assert ok, 'small state is not topologically intact'
    ev = fit(X, d, Lx0, Ly0, min(2, nx0//2), 0 if ny0 == 1 else 1)
    xs, ys = np.meshgrid(np.arange(19*nx, dtype=float), np.arange(33*ny, dtype=float), indexing='ij')
    db = s*ev(xs/s, ys/s); S0 = np.load(os.environ['SKX_CELL'])
    return bilinear(S0, xs-db[..., 0], ys-db[..., 1]), (X, d, ev)
if __name__ == '__main__':
    f, nx0, ny0, nx, ny, out = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6]
    S, (X, d, ev) = make(np.load(f).astype(float), nx0, ny0, nx, ny)
    r = d-ev(X[:, 0], X[:, 1]); print('fit residual rms', np.sqrt((r**2).mean()), 'max', np.abs(r).max(), '| d rms', np.sqrt((d**2).mean()))
    np.save(out, S)
