import numpy as np, sys
from melt import sk_centers
import os
R0=sk_centers(np.load(os.environ.get('SKX_CELL','states/S_B0.5.npy')))[:,:2]   # 2 skyrmions in the 19x33 cell; set SKX_CELL to its path
def ideal(nx,ny):
    return np.array([r+[19*i,33*j] for i in range(nx) for j in range(ny) for r in R0])
def mind(d,L): return d-L*np.round(d/L)
def disp(S,nx,ny):
    Lx,Ly=S.shape[:2]; L=np.array([Lx,Ly]); P=sk_centers(S)[:,:2]; I=ideal(nx,ny)
    sh=np.zeros(2)
    for it in range(6):
        D=mind((P[:,None,:]-sh)-I[None,:,:],L); j=np.argmin((D**2).sum(-1),1)
        d=mind(P-sh-I[j],L); sh=sh+d.mean(0)
    ok=len(set(j))==len(P)==len(I)
    d=mind(P-sh-I[j],L); return I[j],d,ok
def tri_strain(X,d,Lx,Ly):
    # deformation gradient on each triangle of the ideal lattice (periodic images)
    from scipy.spatial import Delaunay
    N=len(X); imgs=np.concatenate([X+[a*Lx,b*Ly] for a in (-1,0,1) for b in (-1,0,1)]); dd=np.tile(d,(9,1))
    T=Delaunay(imgs).simplices; c=imgs[T].mean(1); keep=(c[:,0]>=0)&(c[:,0]<Lx)&(c[:,1]>=0)&(c[:,1]<Ly); T=T[keep]
    out=[]
    for t in T:
        A=np.stack([imgs[t[1]]-imgs[t[0]],imgs[t[2]]-imgs[t[0]]],1); B=np.stack([dd[t[1]]-dd[t[0]],dd[t[2]]-dd[t[0]]],1)
        Gd=B@np.linalg.inv(A)            # displacement gradient du_i/dx_j
        e=0.5*(Gd+Gd.T); w=0.5*(Gd[1,0]-Gd[0,1])
        ev=np.linalg.eigvalsh(e); out.append([e[0,0],e[1,1],e[0,1],w,(ev[1]-ev[0])/2,ev.sum(),Gd[1,0],Gd[0,0]])
    return np.array(out)
if __name__=="__main__":
    for f,nx,ny in [(a,int(b),int(c)) for a,b,c in zip(sys.argv[1::3],sys.argv[2::3],sys.argv[3::3])]:
        S=np.load(f).astype(float); Lx,Ly=S.shape[:2]; X,d,ok=disp(S,nx,ny)
        st=tri_strain(X,d,Lx,Ly); k=2*np.pi/Lx
        F=lambda v: 2*abs(np.mean(v*np.exp(-1j*k*X[:,0])))
        print(f"{f.split('/')[-2]}/{f.split('/')[-1]} assign_ok={ok} |d|max={np.abs(d).max():.2f}  U_x(k1)={F(d[:,0]):.2f} U_y(k1)={F(d[:,1]):.2f} -> k*U_x={k*F(d[:,0]):.3f} k*U_y={k*F(d[:,1]):.3f} | tri: max shear={st[:,4].max():.3f} max|dil|={np.abs(st[:,5]).max():.3f} max|du_y/dx|={np.abs(st[:,6]).max():.3f} max|du_x/dx|={np.abs(st[:,7]).max():.3f} max|rot|={np.abs(st[:,3]).max():.3f}")
