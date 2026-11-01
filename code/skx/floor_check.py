import sys; sys.path.insert(0,'.')
from skx import *
J,D=1.0,0.4; a=0.3; b=0.6; ex=np.array([1.,0]); z=np.zeros(2)
for bb,u in ((0.7,0.0265),(0.5,0.0203)):
    m=Model(19,33,J,D,bb*D*D/J); S0=np.load(f"../../states/S_B{bb}.npy"); k=(2*np.pi/380,0.)
    S,V,w1,w2,r=find_V(m,S0.copy(),u*ex,a,b,iters=8,n=1500)
    R=m.rhs(S,w1,w2,a); gx=(np.roll(S,-1,0)-np.roll(S,1,0))/2; gy=(np.roll(S,-1,1)-np.roll(S,1,1))/2
    A=np.stack([gx.ravel(),gy.ravel()],1); c,*_=np.linalg.lstsq(A,R.ravel(),rcond=None); res2=R.ravel()-A@c
    g0=np.linalg.eigvals(bloch_jac(m,S,w1,w2,a,k)).real.max()
    # refine V by Newton on the residual (frame velocity as unknown), then relax
    Vr=V.copy(); Sr=S.copy()
    for it in range(30):
        W1=Vr-u*ex; W2=b*u*ex-a*Vr; f0=m.rhs(Sr,W1,W2,a).ravel()
        if np.abs(f0).max()<1e-12: break
        Jv=np.stack([(m.rhs(Sr,W1+e,W2-a*e,a).ravel()-f0)/1e-6 for e in (np.array([1e-6,0]),np.array([0,1e-6]))],1)
        Vr=Vr+np.linalg.lstsq(Jv,-f0,rcond=None)[0]; Sr=rk4(m,Sr,Vr-u*ex,b*u*ex-a*Vr,a,0.1,1500)
    W1=Vr-u*ex; W2=b*u*ex-a*Vr; r2=np.abs(m.rhs(Sr,W1,W2,a)).max()
    g1=np.linalg.eigvals(bloch_jac(m,Sr,W1,W2,a,k)).real.max()
    print(f"B={bb} u={u}: floor resid={r:.1e}, translation part explains {1-np.linalg.norm(res2)/np.linalg.norm(R):.4f} of it (dV={c}) | refined V-V={Vr-V} resid={r2:.1e} | growth before {g0:+.3e} after {g1:+.3e}",flush=True)
