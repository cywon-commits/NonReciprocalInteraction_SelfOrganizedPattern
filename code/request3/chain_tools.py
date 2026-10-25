import sys, os, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
os.environ.setdefault('SKX_CELL', os.path.join(root, 'states', 'S_B0.5.npy'))
sys.path.insert(0, os.path.join(root, 'code', 'request2'))
from strain import tri_strain
from scipy.spatial import Delaunay
# Shared measurements for task B (request 3) on snapshots of the skyrmion registry (reference positions X, displacements u):
#   profile(X, u)      : 1D profile of du_y/dx along x (least-squares Fourier harmonics 0..3 of 2 pi/Lx over the triangles)
#   critical(prof)     : peak position/value and width of the region with profile > gamma_s(0)
#   softamp(X, u, x0)  : soft-mode amplitude in the strip |x - x0| < hw of the lattice: residual displacement after removing the
#                        smooth field (harmonics |m| <= 3 along x, |n| <= 1 along y), projected on plane waves of the band
#                        0.06 < |k| < 0.11 (wavelength 4a = 75 sites, |k| = 0.084) with 0.045 < |k_y| < 0.1 (soft mode k = (-0.043, 0.071); the strain wave itself has
#                        k_y = 0); the maximum over the band is the local amplitude a.
GS = 0.30776; NX = 400
def centroids(X, Lx, Ly):
    imgs = np.concatenate([X+[i*Lx, j*Ly] for i in (-1, 0, 1) for j in (-1, 0, 1)]); T = Delaunay(imgs).simplices
    c = imgs[T].mean(1); keep = (c[:, 0] >= 0) & (c[:, 0] < Lx) & (c[:, 1] >= 0) & (c[:, 1] < Ly); return c[keep]
def profile(X, u, Lx, Ly):
    g = tri_strain(X, u, Lx, Ly)[:, 6]; c = centroids(X, Lx, Ly); k = 2*np.pi/Lx
    M = np.stack([np.ones(len(c))]+[f(n*k*c[:, 0]) for n in (1, 2, 3) for f in (np.cos, np.sin)], 1)
    co = np.linalg.lstsq(M, g, rcond=None)[0]; xs = np.arange(NX)*Lx/NX
    return xs, np.stack([np.ones(NX)]+[f(n*k*xs) for n in (1, 2, 3) for f in (np.cos, np.sin)], 1)@co
def critical(xs, prof, Lx, sign=None):
    s = sign if sign is not None else np.sign(prof[np.argmax(np.abs(prof))]); p = s*prof; i = int(np.argmax(p))
    return dict(xpeak=float(xs[i]), gpeak=float(p[i]), w=float((p > GS).sum()*Lx/NX), sign=float(s))
def smooth_resid(X, u, Lx, Ly):
    ks = [(m, n) for m in range(-3, 4) for n in range(-1, 2)]
    ph = np.exp(2j*np.pi*(np.outer(X[:, 0], [m/Lx for m, n in ks])+np.outer(X[:, 1], [n/Ly for m, n in ks])))
    A = np.concatenate([ph.real, ph.imag], 1); return u-A@np.linalg.lstsq(A, u, rcond=None)[0]
def kband(Lx, Ly):
    K = [(2*np.pi*m/Lx, 2*np.pi*n/Ly) for m in range(-Lx//6, Lx//6+1) for n in range(0, Ly//6+1)]
    K = np.array([k for k in K if 0.06 < np.hypot(*k) < 0.11 and 0.045 < abs(k[1]) < 0.1]); return K
def softamp(X, u, Lx, Ly, x0, hw, K=None):
    K = kband(Lx, Ly) if K is None else K; r = smooth_resid(X, u, Lx, Ly)
    dx = (X[:, 0]-x0+Lx/2) % Lx-Lx/2; sel = np.abs(dx) < hw
    if sel.sum() < 4: return np.nan, None
    w = np.exp(-1j*(X[sel]@K.T))                          # (n_sel, n_k)
    amp = np.sqrt(np.abs(w.T@r[sel, 0])**2+np.abs(w.T@r[sel, 1])**2)/sel.sum()*2
    i = int(np.argmax(amp)); return float(amp[i]), K[i]
def unwrap_x(x, Lx): return np.unwrap(np.asarray(x)*2*np.pi/Lx)*Lx/(2*np.pi)
