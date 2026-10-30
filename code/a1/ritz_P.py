from ritz import *
import sys
# B1 to-do 1.2: two-field (displacement + conjugate canting) Ritz theory for BOTH overlaps.
# The Ritz L and T modes span the relaxed subspace; the phonon is the 2x2 Galerkin eigenvector in it.
ex=np.array([1.,0])
def two_field_modes(d,P,conj=True):
    kx,gx,gy,to_t=d['kx'],d['gx'],d['gy'],d['to_t']; L_,Ly=d['Lx'],d['Ly']; Xg=d['X']; Yg=np.ones((L_,1))*np.arange(Ly)[None,:]
    paff=np.tile(np.exp(1j*kx*Xg).reshape(-1),2); Om=d['Om']
    F=[np.exp(2j*np.pi*(p*Xg/L_+q*Yg/Ly)) for p in range(-P,P+1) for q in range(-2*P,2*P+1)]
    cols=[]
    for pol,phi in (('L',gx),('T',gy)):
        base=paff*to_t(phi)
        V=[paff*to_t(f[...,None]*g) for f in F for g in (gx,gy) if not (f is F[len(F)//2] and g is phi)]
        if conj: V+=[paff*(Om@to_t(f[...,None]*g)) for f in F for g in (gx,gy)]
        Dm=np.stack(V,1)*(1j*kx); K=Dm.conj().T@d['H']@Dm; Cv=Dm.conj().T@d['H']@base
        s=np.linalg.lstsq(K,-Cv,rcond=1e-12)[0]; cols.append(base+Dm@s)
    return np.stack(cols,1)
def overlaps_on(d,Basis,ops):
    Om,Jm,kx=d['Om'],d['Jm'],d['kx']
    A=Basis.conj().T@Om@Basis; Bm=Basis.conj().T@Om@Jm@Basis; lam,C=np.linalg.eig(np.linalg.solve(A,Bm))
    i=np.argmax(lam.imag); Psi=Basis@C[:,i]; w=lam[i].imag; nrm=Psi.conj()@Om@Psi
    # on-shell forms (exact identities dJ/dw2 = -Om dJ/dw1, dJ/dalpha = -Om J): no division by omega
    kt=abs(((Psi.conj()@Om@ops['t']@Psi)/nrm).imag)/kx                 # symplectic-metric momentum
    kP=abs((Psi.conj()@ops['t']@Psi).imag)/((Psi.conj()@Psi).real*kx)  # identity-metric momentum
    return kt,kP
if __name__=="__main__":
    keys=sys.argv[1:] or ["19_0.3","19_0.5","19_0.7","19_0.85","13_0.5","15_0.5","23_0.6"]
    for key in keys:
        Lx,bb=key.split('_'); S=np.load(f"../../states/Sgrid/Sgrid_{Lx}_{bb}.npy"); d=setup(S,float(bb))
        m=Model(d['Lx'],d['Ly'],J,D,float(bb)*D*D/J); k=(d['kx'],0.); e=1e-4
        ops=dict(t=(bloch_jac(m,S,e*ex,z,0.0,k)-bloch_jac(m,S,-e*ex,z,0.0,k))/(2*e))
        res=[]
        for P_,cj in ((1,False),(2,False),(0,True),(1,True),(2,True)):
            res.append(overlaps_on(d,two_field_modes(d,P_,cj),ops))
        lab=["disp P=1","disp P=2","2f P=0","2f P=1","2f P=2"]
        print(f"{key}: "+" | ".join(f"{l}: tau={r[0]:.3f} P={r[1]:.3f}" for l,r in zip(lab,res)),flush=True)
