import sys, os, json, glob, numpy as np
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
# Request 6, task A analysis of the centre trajectories (hexA.py).
# Steady window t >= 16000 (u_eff = 0.15 melts from the initial crystal at t = 12000-14000), every 4th stored snapshot
# (200 time units), and the two halves (16000-28000, 28000-40000) separately. Per snapshot: periodic Delaunay -> coordination z_i, bond-orientational psi6_i;
#   defects: clusters of z != 6 sites connected by Delaunay bonds; charge q = sum(6 - z): q != 0 -> free disclination(s) (|q| counted),
#   q = 0 and 2 sites -> free dislocation (isolated 5-7 pair), q = 0 and >= 4 sites -> bound dislocation pairs / grain-boundary bits;
#   structure factor S(q) of the centres on a polar grid around |G0| = 4 pi/(sqrt3 a0) -> three Bragg vectors G1..3 (one per 60 deg sector);
#   g6(r) = Re<psi6_i psi6_j*>/<|psi6|^2>, gG(r) = <cos(G.(r_i - r_j))> averaged over the three G (minimum image, r < L/2).
# Correlation lengths: exponential fit of the envelope (upper hull of the binned function) over r in [a0, L/2] while above the
# noise floor 2/sqrt(N); xi = L/2 is reported as a lower bound (">") when the envelope does not decay.
def snapshot(P, Lx, Ly):
    N = len(P); imgs = np.concatenate([P+[dx*Lx, dy*Ly] for dx in (-1, 0, 1) for dy in (-1, 0, 1)]); own = np.tile(np.arange(N), 9)
    tri = Delaunay(imgs); ip, ind = tri.vertex_neighbor_vertices; base = 4*N
    z = np.zeros(N, int); psi = np.zeros(N, complex); rows, cols = [], []
    for i in range(N):
        nb = ind[ip[base+i]:ip[base+i+1]]; d = imgs[nb]-P[i]; z[i] = len(nb); psi[i] = np.mean(np.exp(6j*np.arctan2(d[:, 1], d[:, 0])))
        rows += [i]*len(nb); cols += list(own[nb])
    return z, psi, np.array(rows), np.array(cols)
def defects(z, rows, cols):
    N = len(z); bad = z != 6; m = bad[rows] & bad[cols]
    G = coo_matrix((np.ones(m.sum()), (rows[m], cols[m])), shape=(N, N)); nc, lab = connected_components(G, directed=False)
    free_disc = free_disl = bound = 0
    for c in np.unique(lab[bad]):
        idx = np.where((lab == c) & bad)[0]; q = int(np.sum(6-z[idx]))
        if q != 0: free_disc += abs(q)
        elif len(idx) == 2: free_disl += 1
        else: bound += len(idx)//2
    return dict(f_def=float(bad.mean()), disc=free_disc/N, disl=free_disl/N, bound=bound/N)
def bragg(P, Lx, Ly, a0, psi):
    # Bragg vectors without per-snapshot maximisation bias: directions from the global bond-orientational phase
    # (theta6 = arg(sum psi6)/6, reciprocal vectors at theta6 + 30 + 60 k deg), magnitudes from the S(q) peak along each of these
    # directions fixed at |G| = G0 = 4 pi/(sqrt3 a0) from the density (no search, so no bias; a strained lattice decays faster).
    G0 = 4*np.pi/(np.sqrt(3)*a0); t6 = np.angle(psi.sum())/6; out = []
    for k in range(3):
        th = t6+np.radians(30+60*k); out.append(G0*np.array([np.cos(th), np.sin(th)]))
    return np.array(out), 0.0
def corr(P, psi, Gs, Lx, Ly, dr=2.0):
    d = P[:, None, :]-P[None, :, :]; d[..., 0] -= Lx*np.round(d[..., 0]/Lx); d[..., 1] -= Ly*np.round(d[..., 1]/Ly); r = np.hypot(d[..., 0], d[..., 1])
    iu = np.triu_indices(len(P), 1); r = r[iu]; rmax = min(Lx, Ly)/2; sel = r < rmax; r = r[sel]; b = (r/dr).astype(int); nb = int(rmax/dr)+1
    cnt = np.bincount(b, minlength=nb)
    g6 = np.bincount(b, weights=np.real(psi[:, None]*psi[None, :].conj())[iu][sel], minlength=nb)
    gG = sum(np.bincount(b, weights=np.cos(d[..., 0]*G[0]+d[..., 1]*G[1])[iu][sel], minlength=nb) for G in Gs)/len(Gs)
    br = (r/1.0).astype(int); cr = np.bincount(br, minlength=int(rmax)+1)[:int(rmax)]          # complete rings only
    return cnt, g6, gG, cr
def xi_fit(rc, g, N, a0, rmax, level=0.1):
    # envelope = running maximum from the right; exponential fit from a0 to where the envelope first drops below `level`;
    # if it never does within L/2 the length is a lower bound (flag True)
    ok = (rc > a0) & np.isfinite(g); rr, gg = rc[ok], g[ok]; env = np.maximum.accumulate(gg[::-1])[::-1]
    below = np.where(env < level)[0]; end = below[0] if len(below) else len(env)
    use = np.arange(len(env)) < max(end, 3); use &= env > 0
    if use.sum() < 3: return float('nan'), False
    p = np.polyfit(rr[use], np.log(env[use]), 1); xi = -1/p[0] if p[0] < 0 else float('inf')
    return float(min(xi, 1e4)), bool(len(below) == 0)
def analyse(f, t_ss=16000, every=4, t_end=1e9):
    z_ = np.load(f); T = z_['times']; C = z_['C']; Lx, Ly = float(z_['Lx']), float(z_['Ly']); sel = np.where((T >= t_ss) & (T < t_end))[0][::every]
    acc = None; rows_d = []; Ns = []; psi6 = []
    for k in sel:
        P = C[k][np.isfinite(C[k][:, 0])]; N = len(P); a0 = np.sqrt(2*Lx*Ly/(np.sqrt(3)*N)); Ns.append(N)
        zz, psi, ro, co = snapshot(P, Lx, Ly); rows_d.append(defects(zz, ro, co)); psi6.append(abs(psi.mean()))
        Gs, _ = bragg(P, Lx, Ly, a0, psi); cnt, g6, gG, cr = corr(P, psi, Gs, Lx, Ly)
        g6 = g6/np.mean(np.abs(psi)**2); cr = cr/(N*(N-1)/2/(Lx*Ly)*2*np.pi*(np.arange(len(cr))+0.5))      # g(r), dr = 1
        acc = [cnt, g6, gG, cr] if acc is None else [acc[0]+cnt, acc[1]+g6, acc[2]+gG, acc[3]+cr]
    cnt, g6, gG, gr = acc; gr = gr/len(sel); rc = (np.arange(len(cnt))+0.5)*2.0; with_ = cnt > 0; g6 = np.where(with_, g6/np.maximum(cnt, 1), np.nan); gG = np.where(with_, gG/np.maximum(cnt, 1), np.nan)
    N = np.mean(Ns); a0 = np.sqrt(2*Lx*Ly/(np.sqrt(3)*N)); rmax = min(Lx, Ly)/2
    x6, b6 = xi_fit(rc, g6, N, a0, rmax); xG, bG = xi_fit(rc, gG, N, a0, rmax)
    # translational length from the radial distribution: envelope of |g(r) - 1| (2-site bins), fit down to level 0.1 (unbiased)
    n2 = len(gr)//2; g2 = gr[:2*n2].reshape(n2, 2).mean(1); r2 = 2*np.arange(n2)+1.0
    xT, bT = xi_fit(r2, np.abs(g2-1), N, a0, rmax, level=0.1)
    D = {k: float(np.mean([r[k] for r in rows_d])) for k in rows_d[0]}
    return dict(file=f, Lx=Lx, Ly=Ly, N=N, a0=a0, ue=float(z_['ue']), nsnap=len(sel), Psi6=float(np.mean(psi6)), xi6=x6, xi6_lower=b6, xiT=xT, xiT_lower=bT,
                xi6_a0=x6/a0, xiT_a0=xT/a0, xiG=xG, xiG_lower=bG, gr=gr.tolist(), ratio=x6/xT if xT > 0 else float('nan'), defects=D, rc=rc.tolist(), g6=np.nan_to_num(g6).tolist(), gG=np.nan_to_num(gG).tolist())
if __name__ == '__main__':
    out = []
    for f in sorted(glob.glob('results_request6/A/centres_*.npz')):
        r = analyse(f); h1 = analyse(f, 16000, 4, 28000); h2 = analyse(f, 28000, 4)
        r['halves'] = [dict(xi6=h['xi6'], xiT=h['xiT'], Psi6=h['Psi6'], f_def=h['defects']['f_def']) for h in (h1, h2)]; out.append(r)
        print(f"{os.path.basename(f)}: N={r['N']:.0f} snaps={r['nsnap']} Psi6={r['Psi6']:.3f} xi6={r['xi6']:.1f}{'(>)' if r['xi6_lower'] else ''} ({r['xi6_a0']:.2f} a0) "
              f"xiT={r['xiT']:.1f}{'(>)' if r['xiT_lower'] else ''} ({r['xiT_a0']:.2f} a0) ratio={r['ratio']:.2f} | halves xi6 {r['halves'][0]['xi6']:.1f}/{r['halves'][1]['xi6']:.1f} xiT {r['halves'][0]['xiT']:.1f}/{r['halves'][1]['xiT']:.1f} | "
              + ' '.join(f'{k}={v:.4f}' for k, v in r['defects'].items()), flush=True)
    json.dump(out, open('results_request6/A_summary.json', 'w'))
