"""Data for Fig. 2(b,c): the first-order distortion of a driven Neel skyrmion lattice is set by w1, not by u.
Co-moving stationary states (exact Thiele projection), u = 0.02, beta = 2 alpha, alpha = 0.1, 0.3, 0.6, current at 0, 30, 90 deg.
The change dS = S(u) - S0 is cleaned of its uniform translation (least squares on dS0/dx, dS0/dy); the distortion axis d is the
first moment of the cleaned dm_z inside r < 5 around the on-site skyrmion. Writes results/b1_figs/fig2_distortion.json and
fig2_map.npy (cleaned dm_z for alpha = 0.3, current at 0 deg)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'skx'))
from skx import Model, find_V
from b1style import DATA, ROOT
J, D, Lx, Ly = 1.0, 0.4, 19, 33
S0 = np.load(os.path.join(ROOT, 'states', 'S_B0.5.npy')); m = Model(Lx, Ly, J, D, 0.5*D*D/J)
X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing='ij'); dx = (X + Lx//2) % Lx - Lx//2; dy = (Y + Ly//2) % Ly - Ly//2
gx = (np.roll(S0, -1, 0) - np.roll(S0, 1, 0))/2; gy = (np.roll(S0, -1, 1) - np.roll(S0, 1, 1))/2; A = np.stack([gx.ravel(), gy.ravel()], 1)
def clean(Su):
    dS = Su - S0; c, *_ = np.linalg.lstsq(A, dS.ravel(), rcond=None); return dS - c[0]*gx - c[1]*gy, c
rows = []
for al in (0.1, 0.3, 0.6):
    for th in (0, 30, 90):
        u = 0.02*np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
        Su, V, w1, w2, r = find_V(m, S0.copy(), u, al, 2*al, iters=10, n=1500)
        dP, tr = clean(Su); win = dx**2 + dy**2 < 25
        d = np.array([np.sum(dP[..., 2]*dx*win), np.sum(dP[..., 2]*dy*win)])
        if al == 0.3 and th == 0: np.save(os.path.join(DATA, 'fig2_map.npy'), dP[..., 2]); mapw = dict(w1=w1.tolist(), w2=w2.tolist(), u=u.tolist())
        rows.append(dict(alpha=al, theta_u=th, w1=w1.tolist(), w2=w2.tolist(), resid=float(r), translation=tr.tolist(), d=d.tolist(),
                         ang_d=float(np.degrees(np.arctan2(d[1], d[0]))), ang_w1=float(np.degrees(np.arctan2(w1[1], w1[0])))))
        print(al, th, 'resid %.1e' % r, 'd-w1 %.2f' % (rows[-1]['ang_d'] - rows[-1]['ang_w1']), '|d| %.4f' % np.linalg.norm(d), flush=True)
json.dump(dict(rows=rows, map=mapw), open(os.path.join(DATA, 'fig2_distortion.json'), 'w'), indent=1)
