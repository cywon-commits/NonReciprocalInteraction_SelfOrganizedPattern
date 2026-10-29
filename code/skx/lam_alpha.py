from skx import *
import scipy.linalg as sla
import sys, json
# Decomposition of the alpha dependence of the SkX Doppler threshold (B1, handoff item 2).
#   (C) first-order overlap formula     g1(u) = u (w2.K_P + a P w1.K_tau) - a P w_n
#   (G) two-mode Galerkin reduction of the full Jacobian on the frozen u=0 background (exact in a, u)
#   (F) full Bloch Jacobian on the frozen u=0 background with exact Thiele w1, w2
#   (A) full Bloch Jacobian on the driven co-moving background S(u)          [= the exact threshold]
# beta = 2 alpha everywhere, so u_eff = u.
J=1.0; D=0.4; Lx,Ly=19,33; z=np.zeros(2)
bb=float(sys.argv[1]); alphas=[float(x) for x in sys.argv[2].split(',')]
nk=int(sys.argv[3]) if len(sys.argv)>3 else 20
do_driven = (len(sys.argv)<=4 or sys.argv[4]!='nodrive')
m=Model(Lx,Ly,J,D,bb*D*D/J); S0=np.load(f"../../states/S_B{bb}.npy")
kx=2*np.pi/(Lx*nk); k=(kx,0.)
ex=np.array([1.,0]); ey=np.array([0,1.])

def eig2(Jm):
    lam=np.linalg.eigvals(Jm); return lam
def growth(Jm):
    lam=np.linalg.eigvals(Jm); return lam.real.max()

# --- reference phonon pair at alpha=0, u=0 and its left/right vectors
J0=bloch_jac(m,S0,z,z,0.0,k)
lam0,VL,VR=sla.eig(J0,left=True,right=True)
idx=np.argsort(np.abs(lam0))[:2]
R=VR[:,idx]; L=VL[:,idx]; L=L@np.linalg.inv(R.conj().T@L)   # L^H R = I
def gal(Jm):
    return np.linalg.eigvals(L.conj().T@Jm@R).real.max()

# --- first-order overlaps (same definitions as lam_full.py)
def mode(w1,w2,a):
    lam=np.linalg.eigvals(bloch_jac(m,S0,w1,w2,a,k)); p=lam[np.argsort(np.abs(lam))[:2]]
    return p[np.argmax(p.imag)]
e=1e-4
wn=mode(z,z,0.0).imag
Kt=np.array([-(mode(e*v,z,0).imag-mode(-e*v,z,0).imag)/(2*e) for v in (ex,ey)])
KP=np.array([(mode(z,e*v,0).real-mode(z,-e*v,0).real)/(2*e) for v in (ex,ey)])
P=-(mode(z,z,e).real-mode(z,z,-e).real)/(2*e)/wn
Gm,Dm,Q=GD(S0)
print(f"B={bb} k={kx:.5f} wn/k={wn/kx:.5f} P={P:.4f} kap_tau=({Kt[0]/kx:+.4f},{Kt[1]/kx:+.4f}) kap_P=({KP[0]/(P*kx):+.4f},{KP[1]/(P*kx):+.4f}) D/G={Dm[0,0]/Gm[0,1]:+.4f}",flush=True)

def secant(f,u0,u1,tol=1e-6,it=30):
    f0=f(u0); f1=f(u1)
    for _ in range(it):
        u2=u1-f1*(u1-u0)/(f1-f0)
        u0,f0=u1,f1; u1=u2; f1=f(u1)
        if abs(u1-u0)<tol*abs(u1): break
    return u1

out={}
for a in alphas:
    b=2*a
    V=np.linalg.solve(Gm-a*Dm, Gm@ex-b*Dm@ex); c1=V-ex; c2=b*ex-a*V      # w per unit u (frozen Thiele)
    s=c2@KP + a*P*(c1@Kt); uC=a*P*wn/s
    f=(a*P*c1@Kt)/s
    uF=secant(lambda u: growth(bloch_jac(m,S0,u*c1,u*c2,a,k)), 0.9*uC, 1.1*uC)
    uG=secant(lambda u: gal(bloch_jac(m,S0,u*c1,u*c2,a,k)), 0.9*uC, 1.1*uC)
    res=dict(alpha=a,f=f,uC=uC,uG=uG,uF=uF)
    if do_driven:
        cache={}
        def gA(u):
            Su,Vv,w1,w2,r=find_V(m,S0.copy(),u*ex,a,b,iters=8,n=1500)
            cache[u]=(Vv,r); return growth(bloch_jac(m,Su,w1,w2,a,k))
        uA=secant(gA,0.98*uF,1.02*uF,tol=2e-5,it=8); res['uA']=uA
    print(json.dumps({kk:(round(v,6) if isinstance(v,float) else v) for kk,v in res.items()}),flush=True)
    out[a]=res
json.dump(dict(B=bb,k=kx,wn=wn,P=P,Kt=list(Kt),KP=list(KP),res=out),open(f"lam_alpha_B{bb}_nk{nk}.json","w"),indent=1)
