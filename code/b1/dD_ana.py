import sys, os, json, numpy as np
here=os.path.dirname(os.path.abspath(__file__)); root=os.path.join(here,'..','..'); sys.path.insert(0,here); sys.path.insert(0,os.path.join(root,'code','request6'))
from skx import Model, rk4, bloch_jac, GD, basis
from dD import DDModel
# B1 to-do 7.1: second-order nonreciprocal stiffness of the exchange-even DL-channel coupling,
#   T2 = -dD sum_e S x [S x (n x d_e^2 S)] = dD P_perp(n x lap S).
# (1) Thiele force F_mu = sum_i d_mu S . (S x T2): exact lattice value, closed form, isotropic form; drift V from (G - a D) V = F.
# (2) Pumping: first-order Re shift of the +-omega branches from dJ/d(dD) on the exact, static-relaxed and affine modes (alpha = 0).
# (3) Strain-dependent force: F under a uniform affine strain of the texture.
J,D=1.0,0.4; B=0.5*D*D; a=0.3; Lx,Ly=19,33; N=Lx*Ly; z=np.zeros(2)
S=np.load(os.path.join(root,'states','S_B0.5.npy'))
def grads(S):
    gx=(np.roll(S,-1,0)-np.roll(S,1,0))/2; gy=(np.roll(S,-1,1)-np.roll(S,1,1))/2
    lap=np.roll(S,-1,0)+np.roll(S,1,0)+np.roll(S,-1,1)+np.roll(S,1,1)-4*S
    return gx,gy,lap
def T2(S,n):
    _,_,lap=grads(S); return -np.cross(S,np.cross(S,np.cross(n,lap)))
def forces(S,n):
    gx,gy,lap=grads(S); T=T2(S,n); X=np.cross(S,T)
    exact=np.array([np.sum(g*X) for g in (gx,gy)])
    g=[gx,gy]; sig=[[np.sum(g[m]*g[v],-1) for v in range(2)] for m in range(2)]; tr=sig[0][0]+sig[1][1]
    closed=np.array([np.sum(sum((g[v]@n)*sig[mu][v] for v in range(2))-1.5*(g[mu]@n)*tr) for mu in range(2)])
    iso=np.array([-np.sum(tr*(g[mu]@n)) for mu in range(2)])
    return exact,closed,iso
Gm,Dm,Q=GD(S)
out={}
print(f"Q={Q:.3f}  G={Gm[0,1]:.3f}  Dxx={Dm[0,0]:.3f} Dyy={Dm[1,1]:.3f}",flush=True)
for nm,n in (("y",np.array([0,1.,0])),("x",np.array([1.,0,0]))):
    ex,cl,iso=forces(S,n)
    V=np.linalg.solve(Gm-a*Dm,-ex)          # sign fixed against request-6 drift below
    V2=np.linalg.solve(Gm-a*Dm,ex)
    print(f"n={nm}: F/dD exact={np.round(ex,4)} closed={np.round(cl,4)} iso={np.round(iso,4)} | V/dD from Thiele (+F)={np.round(V2,4)} (-F)={np.round(V,4)}",flush=True)
    out[f"F_{nm}"]=dict(exact=ex.tolist(),closed=cl.tolist(),iso=iso.tolist(),V_plusF=V2.tolist(),V_minusF=V.tolist())
# (2) pumping generator on exact / relaxed / affine modes
n=np.array([0,1.,0]); dd=1e-4
m0=DDModel(Lx,Ly,0.0,n,'stiff'); m1=DDModel(Lx,Ly,dd,n,'stiff')
Om=np.block([[np.zeros((N,N)),np.eye(N)],[-np.eye(N),np.zeros((N,N))]])
e1,e2=basis(S); E=[e1.reshape(N,3),e2.reshape(N,3)]
gx,gy,_=grads(S); gx=gx.reshape(N,3); gy=gy.reshape(N,3)
rows=[]
for nk in (40,20,10,5):
    kx=2*np.pi/(Lx*nk)
    for kv in ((kx,0.),(0.,kx)):
        Jm=bloch_jac(m0,S,z,z,0.0,kv); dJ=(bloch_jac(m1,S,z,z,0.0,kv)-Jm)/dd
        H=-Om@Jm; H=(H+H.conj().T)/2
        X=(np.arange(Lx)[:,None]*np.ones((1,Ly))).reshape(N); Y=(np.ones((Lx,1))*np.arange(Ly)[None,:]).reshape(N)
        ph=np.exp(1j*(kv[0]*X+kv[1]*Y)); Phi=np.zeros((2*N,2),complex)
        for c,g in enumerate((gx,gy)): Phi[:N,c]=np.sum(g*E[0],1)*ph; Phi[N:,c]=np.sum(g*E[1],1)*ph
        M=Phi.conj().T@Om@Phi; W=Om.T@Phi@np.linalg.inv(M).conj().T; HiW=np.linalg.solve(H,W); Kr=np.linalg.inv(W.conj().T@HiW); Pr=HiW@Kr
        def modes(P):
            A=P.conj().T@Om@P; Bm=P.conj().T@Om@Jm@P; lam,C=np.linalg.eig(np.linalg.solve(A,Bm)); o=np.argsort(lam.imag); return [P@C[:,i] for i in o],lam[o]
        sh=lambda x:(x.conj()@Om@dJ@x)/(x.conj()@Om@x)
        ev,Xv=np.linalg.eig(Jm); idx=np.argsort(np.abs(ev))[:2]; idx=idx[np.argsort(ev[idx].imag)]
        exs=[sh(Xv[:,i]).real for i in idx]; rel=[sh(x).real for x in modes(Pr)[0]]; aff=[sh(x).real for x in modes(Phi)[0]]
        w=abs(ev[idx[0]].imag); kk=np.hypot(*kv)
        r=dict(k=kk,dir='x' if kv[1]==0 else 'y',omega=w,exact=exs,relaxed=rel,affine=aff)
        rows.append(r)
        print(f"k={kk:.4f}||{r['dir']}: Re dlam/d(dD) (-,+): exact {np.round(exs,5)} relaxed {np.round(rel,5)} affine {np.round(aff,5)} | half-split/k exact={abs(exs[1]-exs[0])/2/kk:.4f} relaxed={abs(rel[1]-rel[0])/2/kk:.4f}",flush=True)
out['pump']=rows
# (3) strain-dependent force: affinely strained texture (resampled by interpolation in a strained cell is awkward on a lattice);
# instead use the Bloch-mode statement: the k-linear part of the force on the relaxed mode = Fourier component of dF/d(strain).
json.dump(out,open(os.path.join(root,'results','b1_todo','dD_ana.json'),'w'),indent=1)
