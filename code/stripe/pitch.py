import sys
from conv import *
# B1 to-do 3.1: bending dispersion vs pitch on CONVERGED stationary states.
# omega^2 = w_g^2 + c^2 k^2 + c_b^2 k^4, k along the stripes; current parallel to q builds the cone M.
P=int(sys.argv[1]); m,th,D=make(P); qn=np.sqrt(2)*th
ks=qn*np.array([0.01,0.02,0.04,0.08,0.16,0.24])
def fit(S,w1):
    w=np.array([abs(np.diff(low2(m,S,w1,tuple(k*eh)).imag)[0])/2 for k in ks])
    A=np.stack([np.ones_like(ks),ks**2,ks**4],1); return np.linalg.lstsq(A,w**2,rcond=None)[0],w
S=cycloid(P,th); S,r=steady(m,S,z); c,w=fit(S,z)
print(f"P={P} D={D:.5f}: u=0 resid={r:.0e} c0={np.sqrt(max(c[1],0)):.5f} c0/D={np.sqrt(max(c[1],0))/D:.4f} c_b={np.sqrt(max(c[2],0)):.4f}",flush=True)
for frac in (0.1,0.2,0.3):
    w1=w1_of(frac*D,0.0); S,r=steady(m,S,w1); M=S[:,:2].mean(0)@eh; c,w=fit(S,w1)
    print(f"P={P} u_eff/D={frac} resid={r:.0e} M={M:.4f} M*D/u_eff={M*D/(frac*D):.4f}: w_g^2={c[0]:+.2e} c={np.sqrt(max(c[1],0)):.5f} (c^2-c0^2)/M^2 below | c_b={np.sqrt(max(c[2],0)):.4f} omega/k={np.round(w/ks,5)}",flush=True)
