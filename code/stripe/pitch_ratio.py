import sys, json
from conv import *
# Fig. 6(a) data: k-resolved ratio omega(k; M)/omega(k; 0) of the bending branch on converged driven states (u parallel to q).
# The theta-M conjugate argument predicts sqrt(1 - M^2) for every k in the long-wave range.
P=int(sys.argv[1]); m,th,D=make(P); qn=np.sqrt(2)*th
ks=qn*np.array([0.0025,0.005,0.01,0.02,0.04])
def om(S,w1): return np.array([abs(np.diff(low2(m,S,w1,tuple(k*eh)).imag)[0])/2 for k in ks])
S=cycloid(P,th); S,r0=steady(m,S,z); w0=om(S,z); out=dict(P=P,D=D,k=ks.tolist(),omega0=w0.tolist(),rows=[])
for frac in (0.1,0.2,0.3,0.4):
    w1=w1_of(frac*D,0.0); S,r=steady(m,S,w1); M=float(S[:,:2].mean(0)@eh); w=om(S,w1)
    out['rows'].append(dict(ueff_over_D=frac,M=M,resid=float(r),ratio=(w/w0).tolist()))
    print(P,frac,f"M={M:.4f} resid={r:.0e} ratio={np.round(w/w0,4)} sqrt(1-M^2)={np.sqrt(1-M*M):.4f}",flush=True)
json.dump(out,open(f"../../results/b1_figs/pitch_ratio_P{P}.json","w"),indent=1)
