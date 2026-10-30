import numpy as np
from numba import njit, prange
# Square-lattice Neel-DMI LLG (same equation and sign conventions as code/kt/fastllg.py) on an arbitrary periodic labelling of the
# sites: S is (N, 3); xp, xm, yp, ym are the indices of the +x, -x, +y, -y lattice neighbours. Used for the sheared stripe box
# (n = i + j, m = j): +x -> (n+1, m), -x -> (n-1, m), +y -> (n+1, m+1), -y -> (n-1, m-1), helical in n (Nn) and periodic in m (Nm).
@njit(parallel=True, fastmath=True, cache=True)
def rhs(S, xi, out, xp, xm, yp, ym, J, D, B, a, w1x, w1y, w2x, w2y):
    N = S.shape[0]; c = 1.0/(1+a*a)
    for s in prange(N):
        p, q, r, t = xp[s], xm[s], yp[s], ym[s]
        sx, sy, sz = S[s, 0], S[s, 1], S[s, 2]
        xpx, xpy, xpz = S[p, 0], S[p, 1], S[p, 2]; xmx, xmy, xmz = S[q, 0], S[q, 1], S[q, 2]
        ypx, ypy, ypz = S[r, 0], S[r, 1], S[r, 2]; ymx, ymy, ymz = S[t, 0], S[t, 1], S[t, 2]
        hx = J*(xpx+xmx+ypx+ymx)+xi[s, 0]; hy = J*(xpy+xmy+ypy+ymy)+xi[s, 1]; hz = J*(xpz+xmz+ypz+ymz)+B+xi[s, 2]
        hx += D*(-xpz+xmz); hz += D*(xpx-xmx); hy += D*(-ypz+ymz); hz += D*(ypy-ymy)
        gxx = (xpx-xmx)*0.5; gxy = (xpy-xmy)*0.5; gxz = (xpz-xmz)*0.5; gyx = (ypx-ymx)*0.5; gyy = (ypy-ymy)*0.5; gyz = (ypz-ymz)*0.5
        tx = -(sy*hz-sz*hy); ty = -(sz*hx-sx*hz); tz = -(sx*hy-sy*hx)
        tx += w1x*gxx+w1y*gyx; ty += w1x*gxy+w1y*gyy; tz += w1x*gxz+w1y*gyz
        vx = w2x*gxx+w2y*gyx; vy = w2x*gxy+w2y*gyy; vz = w2x*gxz+w2y*gyz
        tx += sy*vz-sz*vy; ty += sz*vx-sx*vz; tz += sx*vy-sy*vx
        d = tx*sx+ty*sy+tz*sz; tx -= d*sx; ty -= d*sy; tz -= d*sz
        out[s, 0] = c*(tx+a*(sy*tz-sz*ty)); out[s, 1] = c*(ty+a*(sz*tx-sx*tz)); out[s, 2] = c*(tz+a*(sx*ty-sy*tx))
@njit(parallel=True, fastmath=True, cache=True)
def axpy_norm(S, f, dt, out):
    for s in prange(S.shape[0]):
        x = S[s, 0]+dt*f[s, 0]; y = S[s, 1]+dt*f[s, 1]; z = S[s, 2]+dt*f[s, 2]; n = 1/np.sqrt(x*x+y*y+z*z)
        out[s, 0] = x*n; out[s, 1] = y*n; out[s, 2] = z*n
@njit(parallel=True, fastmath=True, cache=True)
def heun_finish(S, f1, f2, dt):
    for s in prange(S.shape[0]):
        x = S[s, 0]+0.5*dt*(f1[s, 0]+f2[s, 0]); y = S[s, 1]+0.5*dt*(f1[s, 1]+f2[s, 1]); z = S[s, 2]+0.5*dt*(f1[s, 2]+f2[s, 2])
        n = 1/np.sqrt(x*x+y*y+z*z); S[s, 0] = x*n; S[s, 1] = y*n; S[s, 2] = z*n
def sheared(Nn, Nm):
    # helical boundary: (n + Nn, m) is the site (n, m - Nn/2), i.e. the periodicity across the stripes is the lattice translation
    # (Nn/2, Nn/2) along q (Nn even, a multiple of the period); along the stripes (n, m + Nm) = (n, m), translation (-Nm, Nm).
    # The allowed wavevectors are then k = kappa e + (multiples of 2 pi sqrt2/Nn) q with kappa = 2 pi j/(sqrt2 Nm): pure bending allowed.
    n, m = np.meshgrid(np.arange(Nn), np.arange(Nm), indexing='ij')
    def idx(a, b):
        w = np.floor_divide(a, Nn); return ((a % Nn)*Nm+((b-w*(Nn//2)) % Nm)).ravel()
    return idx(n+1, m), idx(n-1, m), idx(n+1, m+1), idx(n-1, m-1)
class Integrator:
    def __init__(s, S, nb, J, D, B, a, w1, w2, T, dt, seed=0):
        s.S = np.ascontiguousarray(S.reshape(-1, 3), dtype=np.float64); s.nb = nb; s.p = (J, D, B, a, w1[0], w1[1], w2[0], w2[1])
        s.sig = np.sqrt(2*a*T/dt) if T > 0 else 0.0; s.dt = dt
        s.f1 = np.empty_like(s.S); s.f2 = np.empty_like(s.S); s.S1 = np.empty_like(s.S); s.xi = np.zeros_like(s.S); s.rng = np.random.default_rng(seed)
    def f(s, S):
        out = np.empty_like(S); rhs(S, s.xi*0, out, *s.nb, *s.p); return out
    def run(s, n):
        for _ in range(n):
            if s.sig > 0: s.xi[...] = s.sig*s.rng.standard_normal(s.S.shape)
            rhs(s.S, s.xi, s.f1, *s.nb, *s.p); axpy_norm(s.S, s.f1, s.dt, s.S1)
            rhs(s.S1, s.xi, s.f2, *s.nb, *s.p); heun_finish(s.S, s.f1, s.f2, s.dt)
