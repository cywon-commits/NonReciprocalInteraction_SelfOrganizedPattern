from skx import *
import sys, json
# B1 to-do 3.5: pinned SkX under current WITH its static deformation.
# Pinning dB = eps*B*(m_z0 - <m_z0>) (as pinning.py). Lab frame, v = 0: w1 = -u, w2 = beta u (alpha = 0.3, beta = 0.6, u_eff = u).
# For each u: relax to a stationary state (residual < 1e-10) or detect depinning (residual stalls, lattice phase drifts);
# if stationary, scan the exact Bloch growth rate on the deformed state over k.
J=1.0; D=0.4; bb=0.5; Lx,Ly=19,33; a,b=0.3,0.6; ex=np.array([1.,0])
eps=float(sys.argv[1]); us=[float(x) for x in sys.argv[2].split(',')]; maxit=int(sys.argv[3]) if len(sys.argv)>3 else 60; tag=sys.argv[4] if len(sys.argv)>4 else ''
S0=np.load("../../states/S_B0.5.npy")
Bmap=bb*D*D/J*(1+eps*(S0[...,2]-S0[...,2].mean()))[...,None]
m=Model(Lx,Ly,J,D,Bmap); z=np.zeros(2)
Sp=S0.copy()
for _ in range(200):
    Sp=rk4(m,Sp,z,z,0.5,0.1,4000)
    if np.abs(m.rhs(Sp,z,z,0.5)).max()<1e-11: break
X=np.arange(Lx)[:,None]*np.ones((1,Ly)); Y=np.ones((Lx,1))*np.arange(Ly)[None,:]
def phase(S):   # lattice displacement from the phase of the first Bragg components of m_z
    f1=np.sum(S[...,2]*np.exp(-2j*np.pi*X/Lx)); f2=np.sum(S[...,2]*np.exp(-2j*np.pi*2*Y/Ly))
    return np.array([-np.angle(f1)*Lx/(2*np.pi), -np.angle(f2)*Ly/(2*np.pi*2)])
ks=[(k,0.) for k in (2*np.pi/380,2*np.pi/190,2*np.pi/95,2*np.pi/57)]+[(0.,k) for k in (2*np.pi/380,2*np.pi/95)]
out=[]; S=Sp.copy(); p0=phase(Sp)
for u in us:
    w1=-u*ex; w2=b*u*ex; hist=[]
    for it in range(maxit):
        S=rk4(m,S,w1,w2,a,0.1,2000); r=np.abs(m.rhs(S,w1,w2,a)).max(); ph=phase(S); hist.append((r,ph.tolist()))
        if r<1e-10: break
    drift=(np.array(hist[-1][1])-np.array(hist[max(0,len(hist)-6)][1]))
    stat=r<1e-8
    row=dict(u=u,resid=r,steps=2000*len(hist),shift=(phase(S)-p0).tolist(),drift_last_6chunks=drift.tolist(),stationary=bool(stat))
    if stat:
        g=[]
        for kv in ks:
            lam=np.linalg.eigvals(bloch_jac(m,S,w1,w2,a,kv)); lam=lam[np.abs(lam)>1e-9]; g.append(float(lam.real.max()))
        row['growth']=g; row['maxgrowth']=max(g)
    out.append(row)
    print(json.dumps({k:(np.round(v,8).tolist() if isinstance(v,(list,np.ndarray)) else v) for k,v in row.items()}),flush=True)
    json.dump(out,open(f"../../results/b1_todo/pinned_drive_eps{eps}{tag}.json","w"),indent=1)
    if not stat: break
