import sys, os, numpy as np
root=os.environ.get('REPO_ROOT','.')   # set REPO_ROOT to the repository root
os.environ['SKX_CELL']=root+'/states/S_B0.5.npy'
sys.path.insert(0,root+'/code/request3'); sys.path.insert(0,root+'/code/request2')
from chain_tools import centroids
from strain import tri_strain
from regist import topo
gs=lambda e: 0.2932+0.437*e-0.74*e**2
cases=[('T0/8x2_s1/S_r1.0850.npy',0.362),('T0/12x3_s1/S_r1.0775.npy',0.347),('T0/16x4_s1/S_r1.0775.npy',0.333),('T0/24x6_s1/S_r1.0700.npy',0.316),('A32/chain/S_r1.0650.npy',0.305)]
for f,gc in cases:
    S=np.load(f'{root}/results_request3/{f}').astype(float); Lx,Ly=S.shape[:2]; t=topo(S)
    X,u=t['X'],t['u']; st=tri_strain(X,u,Lx,Ly); c=centroids(X,Lx,Ly); k=2*np.pi/Lx
    M=np.stack([np.ones(len(c))]+[fn(n*k*c[:,0]) for n in (1,2,3) for fn in (np.cos,np.sin)],1)
    xs=np.arange(4000)*Lx/4000; B=np.stack([np.ones(4000)]+[fn(n*k*xs) for n in (1,2,3) for fn in (np.cos,np.sin)],1)
    g=B@np.linalg.lstsq(M,st[:,6],rcond=None)[0]; e=B@np.linalg.lstsq(M,st[:,7],rcond=None)[0]
    s=np.sign(g[np.argmax(abs(g))]); g=s*g; e=e  # eps sign unaffected by shear sign? keep as is
    ip=np.argmax(g); m=g-gs(e); im=np.argmax(m)
    d=((xs[im]-xs[ip]+Lx/2)%Lx)-Lx/2
    # curvature of m at max
    h=Lx/4000; m2=-(np.roll(m,-40)[im]-2*m[im]+np.roll(m,40)[im])/(40*h)**2
    # also find downstream direction: sign convention unknown -> report signed offset
    print(f"{f.split('/')[1] if f.startswith('T0') else '32x8'} L={Lx}: g_peak={g[ip]:.3f} eps_peak={e[ip]:+.3f} m_peak={m[ip]:+.4f} | m_max={m[im]:+.4f} at offset {d:+.1f} ({d/Lx:+.3f} L), eps there={e[im]:+.3f} | m_max*L={m[im]*Lx:.1f}  m2*L^2={m2*Lx**2:.2f}  harmonic sqrt(m2)*L={np.sqrt(abs(m2))*Lx:.3f}")
