import sys, json
import numpy as np
import diag_stripe as ds
# Converged co-moving stationary stripes (residual < tol) and their bending/compression thresholds.
# Fixes: earlier scripts used 3000-4000 RK4 steps (residual ~1e-6), which produced a spurious bending gap.
a,b=0.3,0.6; fac=b/a-1; z=np.zeros(2)
def make(P,J=1.0):
    th=2*np.pi/P; D=np.sqrt(2)*J*np.tan(th); m=ds.DModel(P,J,D); return m,th,D
qh=np.array([1,1.])/np.sqrt(2); eh=np.array([-1,1.])/np.sqrt(2)
def cycloid(P,th):
    n=np.arange(P); S=np.zeros((P,3)); S[:,0]=qh[0]*np.sin(th*n); S[:,1]=qh[1]*np.sin(th*n); S[:,2]=np.cos(th*n); return S
def steady(m,S,w1,tol=1e-12,chunk=4000,maxit=200):
    for it in range(maxit):
        S=ds.rk4(m,S,w1,z,a,0.1,chunk); r=np.abs(m.rhs(S,w1,z,a)).max()
        if r<tol: break
    return S,r
def w1_of(ue,phi):            # phi = angle between u and q
    ang=np.pi/4+phi; return ue*np.array([np.cos(ang),np.sin(ang)])
def growth(m,S,w1,kvec):
    lam=ds.jac(m,S,w1,z,a,kvec); lam=lam[np.abs(lam)>1e-9]; return lam.real.max()
def low2(m,S,w1,kvec):
    lam=ds.jac(m,S,w1,z,0.0,kvec); return lam[np.argsort(np.abs(lam))[:2]]
