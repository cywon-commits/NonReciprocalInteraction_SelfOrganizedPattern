import sys, os, json, glob, re, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here); sys.path.insert(1, os.path.join(here, '..', 'request2'))
from twist import TwistModel, link_blocks
from sparse_bloch import soft_eigs
# Long-wave limit of the soft branch: c(theta) = lambda_max(k e_theta)/k^2 at k = 0.01, theta = 0..175 deg (refined around the
# maximum to 1 deg). c_max > 0 means the uniformly deformed lattice is unstable to a long-wave mode (elastic, Born-type).
# usage: python ctheta.py state_files...   (states of scanA.py: state_u{u}_nx{nx}_Lx{Lx}_s{s}.npy in the same directory)
J, D = 1.0, 0.4; B = 0.5*D*D/J; k0 = 0.01; z = np.zeros(2)
for f in sys.argv[1:]:
    g = re.search(r'nx(\d+)_Lx(\d+)_s(\d+)', f); nx, Lx, s = map(int, g.groups()); fo = f.replace('state_', 'ctheta_').replace('.npy', '.json')
    if os.path.exists(fo): continue
    m = TwistModel(Lx, 33, s, J, D, B); S = np.load(f); bl, _ = link_blocks(m, S, z, z, 0.3)
    c = lambda th: float(soft_eigs(m, bl, k0*np.array([np.cos(np.radians(th)), np.sin(np.radians(th))]), nev=6, sigma=1e-6).real.max())/k0**2
    th = np.arange(0, 180, 15.0); cs = [c(t) for t in th]; t0 = th[int(np.argmax(cs))]
    th2 = np.arange(t0-14, t0+15, 2.0); cs2 = [c(t) for t in th2]; t1 = th2[int(np.argmax(cs2))]
    th3 = np.arange(t1-1.5, t1+1.6, 0.5); cs3 = [c(t) for t in th3]; i = int(np.argmax(cs3))
    d = dict(nx=nx, Lx=Lx, s=s, gamma=s/(19*nx), exx=Lx/(19*nx)-1, k0=k0, theta=float(th3[i]), c_max=float(cs3[i]), coarse=[th.tolist(), cs])
    json.dump(d, open(fo, 'w')); print(f'Lx={Lx} s={s} gamma={s/(19*nx):.4f} eps={Lx/(19*nx)-1:+.4f} theta_max={th3[i]:.1f} c_max={cs3[i]:+.4f}', flush=True)
