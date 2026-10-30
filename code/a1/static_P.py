from skx import *
import re, json, sys
# B1 to-do 1.1: does the static theorem hold for the DISSIPATIVE overlap kappa_P?
# For a type-B phonon the O(k) Bloch corrector is the static relaxed (internal-strain) mode Psi_rel.
# First-order shifts on Psi (Hamiltonian left vector Om^T Psi):
#   reactive  O_tau = dJ/dw1x  -> kappa_tau = |Im shift|/k
#   dissipative O_P = dJ/dw2x  -> K_P = Re shift
#   damping   O_a  = dJ/dalpha -> P = -Re shift / omega
#   kappa_P = K_P / (P k)
J,D=1.0,0.4; z=np.zeros(2); ex=np.array([1.,0])
def overlaps(S,bb,nk=20):
    Lx,Ly=S.shape[:2]; m=Model(Lx,Ly,J,D,bb*D*D/J); N=Lx*Ly; kx=2*np.pi/(Lx*nk); k=(kx,0.); e=1e-4
    Jm=bloch_jac(m,S,z,z,0.0,k)
    Ot=(bloch_jac(m,S,e*ex,z,0.0,k)-bloch_jac(m,S,-e*ex,z,0.0,k))/(2*e)
    OP=(bloch_jac(m,S,z,e*ex,0.0,k)-bloch_jac(m,S,z,-e*ex,0.0,k))/(2*e)
    Oa=(bloch_jac(m,S,z,z,e,k)-bloch_jac(m,S,z,z,-e,k))/(2*e)
    Om=np.block([[np.zeros((N,N)),np.eye(N)],[-np.eye(N),np.zeros((N,N))]])
    H=-Om@Jm; H=(H+H.conj().T)/2
    e1,e2=basis(S); E=[e1.reshape(N,3),e2.reshape(N,3)]
    gx=((np.roll(S,-1,0)-np.roll(S,1,0))/2).reshape(N,3); gy=((np.roll(S,-1,1)-np.roll(S,1,1))/2).reshape(N,3)
    ph=np.exp(1j*kx*(np.arange(Lx)[:,None]*np.ones((1,Ly))).reshape(N))
    Phi=np.zeros((2*N,2),complex)
    for c,g in enumerate((gx,gy)): Phi[:N,c]=np.sum(g*E[0],1)*ph; Phi[N:,c]=np.sum(g*E[1],1)*ph
    M=Phi.conj().T@Om@Phi; W=Om.T@Phi@np.linalg.inv(M).conj().T
    HiW=np.linalg.solve(H,W); Krel=np.linalg.inv(W.conj().T@HiW); Prel=HiW@Krel
    def on_mode(Basis):
        A=Basis.conj().T@Om@Basis; Bm=Basis.conj().T@Om@Jm@Basis; lam,C=np.linalg.eig(np.linalg.solve(A,Bm))
        i=np.argmax(lam.imag); Psi=Basis@C[:,i]; w=lam[i].imag; nrm=Psi.conj()@Om@Psi
        sh=lambda O:(Psi.conj()@Om@O@Psi)/nrm
        kt=abs(sh(Ot).imag)/kx; P=-sh(Oa).real/w; kP=abs(sh(OP).real)/(P*kx)
        return kt,kP,P
    st=on_mode(Prel); af=on_mode(Phi)
    # dynamic reference (exact eigenvalue derivatives)
    lam0=np.linalg.eigvals(Jm); pos=lam0[lam0.imag>0]; l0=pos[np.argmin(np.abs(pos))]
    def dshift(O):
        lp=np.linalg.eigvals(Jm+e*O); lm=np.linalg.eigvals(Jm-e*O)
        return (lp[np.argmin(np.abs(lp-l0))]-lm[np.argmin(np.abs(lm-l0))])/(2*e)
    Pd=-dshift(Oa).real/l0.imag
    dyn=(abs(dshift(Ot).imag)/kx, abs(dshift(OP).real)/(Pd*kx), Pd)
    return dict(dyn=dyn, static=st, affine=af)
if __name__=="__main__":
    cases=[tuple(a.split(':')) for a in sys.argv[1:]] or [('19','0.3'),('19','0.5'),('19','0.7'),('19','0.85'),('13','0.5'),('23','0.6')]
    out={}
    for Lx,bb in cases:
        r=overlaps(np.load(f"../../states/Sgrid/Sgrid_{Lx}_{bb}.npy"),float(bb)); out[f"{Lx}_{bb}"]=r
        d,s,a=r['dyn'],r['static'],r['affine']
        print(f"a={Lx} B={bb}: kappa_tau dyn/static/aff = {d[0]:.4f}/{s[0]:.4f}/{a[0]:.4f} | kappa_P dyn/static/aff = {d[1]:.4f}/{s[1]:.4f}/{a[1]:.4f} | P dyn/static = {d[2]:.3f}/{s[2]:.3f}",flush=True)
    json.dump(out,open(f"static_P_{'_'.join(c[0]+'-'+c[1] for c in cases)}.json","w"),indent=1)
