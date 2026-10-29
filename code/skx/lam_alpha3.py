from skx import *
import scipy.linalg as sla, sys, json
# Split (A - G): pair projection on the DRIVEN background (G_A) vs full Jacobian (A).
J=1.0; D=0.4; Lx,Ly=19,33; z=np.zeros(2); ex=np.array([1.,0])
bb=float(sys.argv[1]); a=float(sys.argv[2]); b=2*a; us=[float(x) for x in sys.argv[3:]]
m=Model(Lx,Ly,J,D,bb*D*D/J); S0=np.load(f"../../states/S_B{bb}.npy"); k=(2*np.pi/(Lx*20),0.)
J0=bloch_jac(m,S0,z,z,0.0,k); lam0,VL,VR=sla.eig(J0,left=True,right=True)
idx=np.argsort(np.abs(lam0))[:2]; R=VR[:,idx]; L=VL[:,idx]; L=L@np.linalg.inv(R.conj().T@L)
for u in us:
    S,V,w1,w2,r=find_V(m,S0.copy(),-u*ex,a,b,iters=8,n=1500)
    Jm=bloch_jac(m,S,w1,w2,a,k)
    # re-express the basis: bloch_jac uses a local basis of S; for the driven S the basis rotates slightly,
    # so also project with the pair of the driven background at alpha=0, w=0 (G_A2)
    J0u=bloch_jac(m,S,z,z,0.0,k); l,VLu,VRu=sla.eig(J0u,left=True,right=True); iu=np.argsort(np.abs(l))[:2]
    Ru=VRu[:,iu]; Lu=VLu[:,iu]; Lu=Lu@np.linalg.inv(Ru.conj().T@Lu)
    gA=np.linalg.eigvals(Jm).real.max(); gGA=np.linalg.eigvals(L.conj().T@Jm@R).real.max()
    gGA2=np.linalg.eigvals(Lu.conj().T@Jm@Ru).real.max()
    print(json.dumps(dict(u=u,gA=gA,gG_driven_u0pair=gGA,gG_driven_ownpair=gGA2,resid=r)),flush=True)
