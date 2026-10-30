from skx import *
import sys
# B1 to-do 3.3: Landau threshold with a pinning gap.
# Pinning: field modulation that follows the skyrmion profile, B(r) = B + eps*B*(m_z0(r) - <m_z0>)
# (deepens the potential at the present skyrmion positions). Pinned lattice: v = 0, w1 = -u, w2 = beta u,
# so the LL drive W2 = w2 + alpha w1 = (beta - alpha) u is unchanged.
# Prediction for omega(k) = omega_pin + c2 k^2:  u_eff* = min_k omega(k)/(lambda k) = 2 sqrt(omega_pin c2)/lambda.
J=1.0; D=0.4; bb=0.5; Lx,Ly=19,33; z=np.zeros(2); ex=np.array([1.,0])
eps=float(sys.argv[1]) if len(sys.argv)>1 else 0.02
S0=np.load("../../states/S_B0.5.npy")
Bmap=bb*D*D/J*(1+eps*(S0[...,2]-S0[...,2].mean()))[...,None]
m=Model(Lx,Ly,J,D,Bmap)
S=rk4(m,S0.copy(),z,z,0.5,0.1,6000); print("pinned state resid",np.abs(m.rhs(S,z,z,0.5)).max(),flush=True)
def branch(Jm):
    lam=np.linalg.eigvals(Jm); p=lam[np.argsort(np.abs(lam))[:2]]; return p[np.argmax(p.imag)]
ns=[40,20,10,6,4]; ks=[2*np.pi/(Lx*n) for n in ns]; om=[]
for k in ks:
    om.append(branch(bloch_jac(m,S,z,z,0.0,(k,0.))).imag)
om=np.array(om); ks=np.array(ks)
A=np.stack([np.ones_like(ks),ks**2],1); (wp,c2)=np.linalg.lstsq(A,om,rcond=None)[0]
print("k:",np.round(ks,4)," omega:",np.round(om,7),f"\nfit omega_pin={wp:.3e} c2={c2:.4f}  k*=sqrt(wp/c2)={np.sqrt(wp/c2):.4f}",flush=True)
a,b=0.3,0.6; lam=0.505
print(f"prediction u_eff* = 2 sqrt(wp c2)/lambda = {2*np.sqrt(wp*c2)/lam:.5f}",flush=True)
def g(u,k): return np.linalg.eigvals(bloch_jac(m,S,-u*ex,b*u*ex,a,(k,0.))).real.max()
best=None
for k in ks[2:4]:
    u0=om[list(ks).index(k)]/(lam*k); lo,hi=0.5*u0,8*u0
    if g(hi,k)<0: print(f"k={k:.4f}: stable to {hi:.4f}",flush=True); continue
    for _ in range(9):
        mid=(lo+hi)/2
        if g(mid,k)>0: hi=mid
        else: lo=mid
    ue=(b/a-1)*(lo+hi)/2; uk=ue*np.array([1.,0]); print(f"k={k:.4f}: frozen-background u_eff*={ue:.5f}  omega/(lambda k)={om[list(ks).index(k)]/(lam*k):.5f}  ratio={ue/(om[list(ks).index(k)]/(lam*k)):.3f}",flush=True)
    best=ue if best is None else min(best,ue)
print("min over k:",best,flush=True)
