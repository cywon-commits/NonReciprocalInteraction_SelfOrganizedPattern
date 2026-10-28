import sys, os, json, glob, numpy as np
from scipy import ndimage
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..')
os.environ.setdefault('SKX_CELL', os.path.join(root, 'states', 'S_B0.5.npy'))
sys.path.insert(0, os.path.join(root, 'code', 'request3')); sys.path.insert(1, os.path.join(root, 'code', 'request2'))
from chain_tools import profile, critical
from regist import topo
# Task C0: re-analysis of the request-4 Floquet vectors (results_request4/C/C1/<size>/f_r*_q*.npy).
# Each spin perturbation q is turned into the rigid displacement d_j of every skyrmion (projection of q on dS/dx, dS/dy over the
# core m_z < 0), assigned to the skyrmion's ideal lattice position X_j (topo). The Fourier power P(K) = sum_a |sum_j d_ja e^{-iK.X_j}|^2
# is then periodic in the reciprocal lattice of the triangular skyrmion lattice (b1 = (2pi/19, -2pi/33), b2 = (0, 4pi/33)), so the
# carrier wavevector is read in the first Brillouin zone (box-allowed K = (2pi m/Lx, 2pi n/Ly)). Envelope: |d_j|^2 against x,
# rms width and half-weight width about the circular centre, centre relative to the shear peak. The critical vector is the
# saved vector (q0..q3) with the largest growth among the soft pair (Gamma of the pair, see request 4 anaC.py).
b1 = np.array([2*np.pi/19, -2*np.pi/33]); b2 = np.array([0., 4*np.pi/33])
def reduce_bz(K):
    best = K.copy()
    for n1 in range(-20, 21):
        for n2 in range(-20, 21):
            c = K+n1*b1+n2*b2
            best = np.where((np.hypot(*c.T) < np.hypot(*best.T))[:, None], c, best)
    return best
def analyse(fj, iq):
    d = json.load(open(fj)); S = np.load(fj.replace('.json', '_S.npy')).astype(float); q = np.load(fj.replace('.json', f'_q{iq}.npy')).astype(float)
    Lx, Ly = S.shape[:2]; gx = (np.roll(S, -1, 0)-np.roll(S, 1, 0))/2; gy = (np.roll(S, -1, 1)-np.roll(S, 1, 1))/2
    tp = topo(S); P, Xid = tp['P'], tp['X']
    big = np.tile(S[..., 2] < 0, (3, 3)); lab, n = ndimage.label(big, structure=np.ones((3, 3))); D_ = []; Xc = []
    for c in range(1, n+1):
        I, Jj = np.nonzero(lab == c); cx, cy = I.mean(), Jj.mean()
        if not (Lx <= cx < 2*Lx and Ly <= cy < 2*Ly): continue
        i = I % Lx; j = Jj % Ly
        M = np.array([[np.sum(gx[i, j]*gx[i, j]), np.sum(gx[i, j]*gy[i, j])], [np.sum(gx[i, j]*gy[i, j]), np.sum(gy[i, j]*gy[i, j])]])
        b = np.array([np.sum(q[i, j]*gx[i, j]), np.sum(q[i, j]*gy[i, j])]); D_.append(-np.linalg.solve(M, b))
        dd = (P-[cx-Lx, cy-Ly]+[Lx/2, Ly/2]) % [Lx, Ly]-[Lx/2, Ly/2]; Xc.append(Xid[np.argmin(np.hypot(*dd.T))])
    D_ = np.array(D_); Xc = np.array(Xc)
    K = np.array([(2*np.pi*m/Lx, 2*np.pi*nn/Ly) for m in range(-Lx//2, Lx//2) for nn in range(-Ly//2, Ly//2)])
    K = np.unique(np.round(reduce_bz(K), 9), axis=0)
    ph = np.exp(-1j*(Xc@K.T)); Pw = np.abs(ph.T@D_[:, 0])**2+np.abs(ph.T@D_[:, 1])**2; kpk = K[np.argmax(Pw)]
    w2 = (D_**2).sum(1); xs = Xc[:, 0] % Lx
    xe = float(np.angle(np.sum(w2*np.exp(2j*np.pi*xs/Lx)))*Lx/(2*np.pi)) % Lx
    dxe = (xs-xe+Lx/2) % Lx-Lx/2; rms = float(np.sqrt(np.sum(w2*dxe**2)/w2.sum()))
    o = np.argsort(np.abs(dxe)); cw = np.cumsum(w2[o])/w2.sum(); half = float(2*np.abs(dxe[o])[np.searchsorted(cw, 0.5)])
    xs4, pg = profile(tp['X'], tp['u'], Lx, Ly); cr = critical(xs4, pg, Lx); off = float(((xe-cr['xpeak'])+Lx/2) % Lx-Lx/2)
    # weight of the carrier: fraction of the power within |K - kpk| < 2 pi/Lx (neighbouring box wavevectors)
    near = np.hypot(*(K-kpk).T) < 2.1*np.pi/Lx*1.01
    return dict(file=fj, iq=iq, r=d['r'], alpha=d['alpha'], Lx=Lx, Ly=Ly, Gamma=d['Gamma'][:4], gamma_wave=cr['gpeak'], k=kpk.tolist(), kabs=float(np.hypot(*kpk)),
                theta=float(np.degrees(np.arctan2(kpk[1], kpk[0])) % 180), carrier_weight=float(Pw[near].sum()/Pw.sum()), rms=rms, half=half, offset=off,
                sigma_k=rms*float(np.hypot(*kpk)), env_x=(((xs-cr['xpeak'])+Lx/2) % Lx-Lx/2).tolist(), env_w=(w2/w2.max()).tolist())
if __name__ == '__main__':
    out = {}
    for fj in sorted(glob.glob('results_request4/C/C1/*/f_r*.json')):
        d = json.load(open(fj)); G = d['Gamma']
        # the critical pair: below threshold the leading vector is a slow y-uniform mode (k_y = 0); take every saved vector
        res = [analyse(fj, iq) for iq in range(4) if os.path.exists(fj.replace('.json', f'_q{iq}.npy'))]
        if not res: continue
        cand = [x for x in res if abs(x['k'][1]) > 1e-9] or res          # the soft pair has k_y != 0
        a = max(cand, key=lambda x: x['carrier_weight']); a['all'] = [(x['iq'], x['k'], round(x['carrier_weight'], 2)) for x in res]
        out[fj] = a
        print(fj.split('/')[-2], a['r'], 'iq', a['iq'], 'Gamma', np.round(G[:3], 6), 'gw %.4f' % a['gamma_wave'], 'k', np.round(a['k'], 4), '|k| %.4f th %.0f w %.2f' % (a['kabs'], a['theta'], a['carrier_weight']),
              'rms %.1f half %.0f off %+.0f sigma*k %.2f' % (a['rms'], a['half'], a['offset'], a['sigma_k']), flush=True)
    json.dump(out, open('results_request5/C0_summary.json', 'w'), indent=1)
