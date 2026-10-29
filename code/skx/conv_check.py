from skx import *
import sys
J=1.0; D=0.4; bb=float(sys.argv[1]); a=float(sys.argv[2]); u=float(sys.argv[3]); b=2*a
m=Model(19,33,J,D,bb*D*D/J); S=np.load(f"../../states/S_B{bb}.npy"); k=(2*np.pi/380,0.); ex=np.array([1.,0])
for rnd in range(5):
    S,V,w1,w2,r=find_V(m,S,-u*ex,a,b,iters=6,n=3000)
    g=np.linalg.eigvals(bloch_jac(m,S,w1,w2,a,k)).real.max()
    print(f"round {rnd}: resid={r:.2e} V=({V[0]:.6f},{V[1]:.6f}) growth={g:+.4e}",flush=True)
