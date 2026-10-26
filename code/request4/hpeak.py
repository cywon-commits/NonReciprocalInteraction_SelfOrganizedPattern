import sys, os, numpy as np, glob, json
root=os.environ.get('REPO_ROOT','.')   # set REPO_ROOT to the repository root
os.environ['SKX_CELL']=root+'/states/S_B0.5.npy'
sys.path.insert(0,root+'/code/request3'); sys.path.insert(0,root+'/code/request2')
from chain_tools import centroids
from strain import tri_strain
from regist import topo
cases=[('T0/8x2_s1',8,2),('T0/12x3_s1',12,3),('T0/16x4_s1',16,4),('T0/24x6_s1',24,6),('A32/chain',32,8)]
for d,nx,ny in cases:
    fs=sorted(glob.glob(f'{root}/results_request3/{d}/S_r*.npy'))
    for f in fs[-3:]:
        S=np.load(f).astype(float); Lx,Ly=S.shape[:2]; t=topo(S)
        if not t['intact']: print(d,os.path.basename(f),'broken'); continue
        X,u=t['X'],t['u']; st=tri_strain(X,u,Lx,Ly); c=centroids(X,Lx,Ly); k=2*np.pi/Lx
        comp={'dyux(=gamma)':st[:,6],'dxux':st[:,7],'dyuy':st[:,1],'dxuy_minus':2*st[:,2]-st[:,6]}
        M=np.stack([np.ones(len(c))]+[fn(n*k*c[:,0]) for n in (1,2,3) for fn in (np.cos,np.sin)],1)
        xs=np.arange(400)*Lx/400; B=np.stack([np.ones(400)]+[fn(n*k*xs) for n in (1,2,3) for fn in (np.cos,np.sin)],1)
        prof={kk:B@np.linalg.lstsq(M,v,rcond=None)[0] for kk,v in comp.items()}
        g=prof['dyux(=gamma)']; s=np.sign(g[np.argmax(abs(g))]); i=np.argmax(s*g)
        print(f"{d} {os.path.basename(f)}: gamma_wave={s*g[i]:.3f} | at peak: du_x/dx={s*prof['dxux'][i]:+.3f} du_y/dy={s*prof['dyuy'][i]:+.3f} du_x/dy={s*prof['dxuy_minus'][i]:+.3f} | max|du_x/dx| over x={abs(prof['dxux']).max():.3f}")
