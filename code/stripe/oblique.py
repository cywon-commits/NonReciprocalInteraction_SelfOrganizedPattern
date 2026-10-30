import sys
from conv import *
# Redo of d3.py with converged stationary states: exact Bloch thresholds for oblique currents, P = 30.
# Prediction (lambda = 1): u_eff sin(phi) = omega0(k; M)/k on the driven background, omega0 at alpha = 0, w = 0.
P=int(sys.argv[1]) if len(sys.argv)>1 else 30; m,th,D=make(P)
S0=cycloid(P,th); S0,_=steady(m,S0,z)
cache={}
def state(ue,phi):
    key=(round(ue,6),phi)
    if key not in cache:
        w1=w1_of(ue,phi); S,r=steady(m,S0.copy(),w1,maxit=40); cache[key]=(S,w1,r)
    return cache[key]
def g(ue,phi,kvec):
    S,w1,r=state(ue,phi)
    return 1.0 if r>1e-8 else growth(m,S,w1,kvec)   # no stationary state = unstable
def pred(ue,phi,kk):
    S,w1,r=state(ue,phi); p=low2(m,S,z,tuple(kk*eh)); return abs(np.diff(p.imag)[0])/2/(kk*np.sin(phi))
for phi_deg in (60,30):
    phi=np.radians(phi_deg)
    for kk in (0.01,0.04):
        kvec=tuple(kk*eh); lo,hi=0.5*0.0635/np.sin(phi),min(2.5*0.0635/np.sin(phi),0.95*0.3006/max(np.cos(phi),1e-9))
        if g(hi,phi,kvec)<0: print(f"phi={phi_deg} k={kk}: stable to u_eff={hi}",flush=True); continue
        for _ in range(12):
            mid=(lo+hi)/2
            if g(mid,phi,kvec)>0: hi=mid
            else: lo=mid
        ue=(lo+hi)/2; S,w1,r=state(ue,phi); M=S[:,:2].mean(0)@eh
        print(f"phi={phi_deg} k={kk}: u_eff*={ue:.4f} resid={r:.0e} M={M:.4f}  omega0(k;M)/(k sin phi)={pred(ue,phi,kk):.4f}  c0 sqrt(1-M^2)/sin(phi)={0.06348*np.sqrt(1-M*M)/np.sin(phi):.4f}",flush=True)
