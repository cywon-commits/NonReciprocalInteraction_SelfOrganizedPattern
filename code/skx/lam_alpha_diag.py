from skx import *
import sys
J=1.0; D=0.4; bb=float(sys.argv[1]); a=float(sys.argv[2]); b=2*a
m=Model(19,33,J,D,bb*D*D/J); S0=np.load(f"../../states/S_B{bb}.npy")
k=(2*np.pi/(19*20),0.); ex=np.array([1.,0])
S=S0.copy()
for u in [float(x) for x in sys.argv[3:]]:
    S,V,w1,w2,r=find_V(m,S.copy(),-u*ex,a,b,iters=8,n=1500)
    lam=np.linalg.eigvals(bloch_jac(m,S,w1,w2,a,k))
    o=np.argsort(np.abs(lam))[:4]
    print(f"u={u:.4f} resid={r:.1e} V=({V[0]:+.4f},{V[1]:+.4f}) small:",
          " ".join(f"{l.real:+.3e}{l.imag:+.3e}i" for l in lam[o]), f" maxRe={lam.real.max():+.3e}",flush=True)
