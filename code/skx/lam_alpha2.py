from skx import *
import sys, json
# Linear-response decomposition of the SkX Doppler threshold at finite alpha (beta = 2 alpha, u_eff = u).
# growth of the positive-frequency phonon branch, g(u) = g0 + u (s_w + s_S) + O(u^2):
#   g0   exact damping rate at u = 0                         [first-order theory: -a P w_n]
#   s_w  slope from the explicit w1, w2 terms (frozen S0)    [first-order theory: c2.K_P + a P c1.K_tau]
#   s_S  slope from the deformation of the co-moving background S(u)
# u_lin = -g0/(s_w+s_S) is compared with the exact threshold u_A from lam_alpha.py.
J=1.0; D=0.4; Lx,Ly=19,33; z=np.zeros(2); ex=np.array([1.,0])
bb=float(sys.argv[1]); alphas=[float(x) for x in sys.argv[2].split(',')]; nk=20
m=Model(Lx,Ly,J,D,bb*D*D/J); S0=np.load(f"../../states/S_B{bb}.npy")
kx=2*np.pi/(Lx*nk); k=(kx,0.)
ref=None
def branch(S,w1,w2,a):
    lam=np.linalg.eigvals(bloch_jac(m,S,w1,w2,a,k)); p=lam[np.argsort(np.abs(lam))[:2]]
    return p[np.argmax(p.imag)]
Gm,Dm,_=GD(S0)
du=float(sys.argv[3]) if len(sys.argv)>3 else 0.002
res={}
for a in alphas:
    b=2*a
    V=np.linalg.solve(Gm-a*Dm, Gm@ex-b*Dm@ex); c1=V-ex; c2=b*ex-a*V
    l0=branch(S0,z,z,a); g0=l0.real
    gFp=branch(S0, du*c1, du*c2,a).real; gFm=branch(S0,-du*c1,-du*c2,a).real
    s_w=(gFp-gFm)/(2*du)
    gA={}
    for sgn in (+1,-1):
        Su,Vv,w1,w2,r=find_V(m,S0.copy(),sgn*du*ex,a,b,iters=8,n=1500)
        gA[sgn]=branch(Su,w1,w2,a).real
    s_tot=(gA[1]-gA[-1])/(2*du); curv=(gA[1]+gA[-1]-2*g0)/du**2
    r=dict(alpha=a, g0=g0, w0=l0.imag, s_w=s_w, s_S=s_tot-s_w, s_tot=s_tot, curv=curv,
           u_lin_frozen=-g0/s_w, u_lin=-g0/s_tot)
    print(json.dumps({kk:float(f"{v:.6g}") for kk,v in r.items()}),flush=True)
    res[a]=r
json.dump(res,open(f"lam_alpha2_B{bb}.json","w"),indent=1)
