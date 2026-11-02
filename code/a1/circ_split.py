import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))+'/../skx')
from skx import *
# B1 to-do 2.8: split the Bloch mode into co-/counter-rotating tangent-plane components (eigenvectors of Omega)
# and measure their norms n+-, momenta per norm pi+-, and the cross term; check kappa_P, kappa_tau formulas.
J,D=1.0,0.4; z=np.zeros(2); ex=np.array([1.,0])
for bb in (0.3,0.5,0.7,0.85):
    S=np.load(f"../../states/S_B{bb}.npy"); m=Model(19,33,J,D,bb*D*D/J); N=19*33; k=(2*np.pi/380,0.); e=1e-4
    Jm=bloch_jac(m,S,z,z,0.0,k); O=(bloch_jac(m,S,e*ex,z,0.0,k)-bloch_jac(m,S,-e*ex,z,0.0,k))/(2*e)
    lam,V=np.linalg.eig(Jm); pos=np.where(lam.imag>0)[0]; i=pos[np.argmin(np.abs(lam[pos]))]; psi=V[:,i]
    a,b=psi[:N],psi[N:]
    # Omega = [[0,I],[-I,0]] : eigenvectors (1, +i)/sqrt2 (eigenvalue +i) and (1,-i)/sqrt2 (eigenvalue -i)
    cp=(a-1j*b)/np.sqrt(2); cm=(a+1j*b)/np.sqrt(2)          # components along (1,i)/sqrt2 and (1,-i)/sqrt2
    vp=np.concatenate([cp,1j*cp])/np.sqrt(2); vm=np.concatenate([cm,-1j*cm])/np.sqrt(2)
    P_=-1j*O                                                   # momentum generator (Hermitian in identity metric up to lattice error)
    n_p=np.vdot(vp,vp).real; n_m=np.vdot(vm,vm).real
    p_p=np.vdot(vp,P_@vp).real; p_m=np.vdot(vm,P_@vm).real; cross=abs(np.vdot(vp,P_@vm))
    kk=k[0]; pi_p=p_p/(n_p*kk); pi_m=p_m/(n_m*kk)
    if n_m>n_p: n_p,n_m,pi_p,pi_m=n_m,n_p,pi_m,pi_p          # '+' = dominant (co-rotating) component
    r=n_m/n_p; P=(n_p+n_m)/(n_p-n_m)
    kP=(pi_p+r*pi_m)/(1+r); kt=(pi_p-r*pi_m)/(1-r)
    print(f"B={bb}: P={P:.3f} r={r:.3f} | pi+={abs(pi_p):.4f} pi-={abs(pi_m):.4f} | from split: kappa_P={abs(kP):.4f} kappa_tau={abs(kt):.4f} | cross/diag={cross/abs(p_p):.1e}",flush=True)
