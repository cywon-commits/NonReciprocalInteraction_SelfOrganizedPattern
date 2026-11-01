import sys, os, numpy as np
here=os.path.dirname(os.path.abspath(__file__)); root=os.path.join(here,'..','..'); sys.path.insert(0,here); sys.path.insert(0,os.path.join(root,'code','request6'))
from skx import bloch_jac, basis
from dD import DDModel
# Decompose the first-order Re shift of the +-omega pair under T2 at finite alpha:
#   dJ/d(dD) at alpha = [O2 + alpha (S x O2)]/(1+alpha^2)  ->  Re shift = intrinsic(O2) + alpha * Gilbert-rotated part
# Also the Im (frequency) shift of O2 (reactive nonreciprocity omega(k) != omega(-k)).
S=np.load(os.path.join(root,'states','S_B0.5.npy')); Lx,Ly=19,33; N=Lx*Ly; z=np.zeros(2); n=np.array([0,1.,0]); dd=1e-4
Om=np.block([[np.zeros((N,N)),np.eye(N)],[-np.eye(N),np.zeros((N,N))]])
for kk in (2*np.pi/380,2*np.pi/190):
    for kv,lab in (((kk,0.),'x'),((0.,kk),'y')):
        J0=bloch_jac(DDModel(Lx,Ly,0.0,n,'stiff'),S,z,z,0.0,kv)
        ev,X=np.linalg.eig(J0); idx=np.argsort(np.abs(ev))[:2]; idx=idx[np.argsort(ev[idx].imag)]
        sh=lambda dJ:[(X[:,i].conj()@Om@dJ@X[:,i])/(X[:,i].conj()@Om@X[:,i]) for i in idx]
        res={}
        for a in (0.0,0.3):
            Ja=bloch_jac(DDModel(Lx,Ly,0.0,n,'stiff'),S,z,z,a,kv); Jb=bloch_jac(DDModel(Lx,Ly,dd,n,'stiff'),S,z,z,a,kv)
            res[a]=sh((Jb-Ja)/dd)
        s0=res[0.0]; s3=res[0.3]
        print(f"k={kk:.4f}||{lab}: alpha=0: Re(-,+)={np.round([x.real for x in s0],5)} Im(-,+)={np.round([x.imag for x in s0],5)} | alpha=0.3: Re(-,+)={np.round([x.real for x in s3],5)} | half-split/k a=0: {abs(s0[1].real-s0[0].real)/2/kk:.4f} a=0.3: {abs(s3[1].real-s3[0].real)/2/kk:.4f} | Im-shift/k (common) {np.mean([x.imag for x in s0])/kk:+.4f}",flush=True)
