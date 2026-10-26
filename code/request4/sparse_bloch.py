import sys, os, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, '..', 'request2'))
from twist import TwistModel, link_blocks, jac_k
# Sparse Bloch Jacobian of the twisted cell (same blocks as twist.jac_k, assembled as a CSC matrix) and the eigenvalues
# with the largest real part near zero by shift-invert. The soft (elastic) branch lives near lambda = 0, the spin-wave
# branches are far away (|lambda| ~ 0.1-1), so the eigenvalues closest to a small positive shift contain the soft mode.
def jac_sparse(m, blocks, k):
    N = m.N; ii = np.arange(N); rows, cols, vals = [], [], []
    for name, Bl in blocks.items():
        j = ii if name == 'self' else m.idx[name]
        ph = np.ones(N) if name == 'self' else np.exp(1j*(m.T[name]@np.asarray(k, float)))
        for r in range(2):
            for c in range(2):
                rows.append(r*N+ii); cols.append(c*N+j); vals.append(Bl[:, r, c]*ph)
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2*N, 2*N))
def soft_eigs(m, blocks, k, nev=8, sigma=2e-4):
    Jm = jac_sparse(m, blocks, k)
    ev = sla.eigs(Jm, k=nev, sigma=sigma, which='LM', return_eigenvectors=False, tol=1e-10, maxiter=5000)
    return ev[np.argsort(-ev.real)]
def soft_vec(m, blocks, k, sigma=2e-4):
    Jm = jac_sparse(m, blocks, k); ev, V = sla.eigs(Jm, k=8, sigma=sigma, which='LM', tol=1e-10, maxiter=5000)
    o = int(np.argmax(ev.real)); return ev[o], V[:, o]
if __name__ == '__main__':
    # validation against the dense Bloch Jacobian on a request-2 state (19 x 33 cell)
    J, D = 1.0, 0.4; B = 0.5*D*D/J; s = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    m = TwistModel(19, 33, s, J, D, B); S = np.load(f'results_request2/A_S/state_u0.000_nx1_Lx19_s{s}.npy'); z = np.zeros(2)
    bl, _ = link_blocks(m, S, z, z, 0.3)
    for (i, j) in [(15, 6), (15, 5), (1, 5), (3, 2)]:
        kv = ((i/16)-np.round(i/16))*m.Bk[0]+(j/16)*m.Bk[1]
        ed = np.linalg.eigvals(jac_k(m, bl, kv)); es = soft_eigs(m, bl, kv)
        print((i, j), 'dense max', ed.real.max(), 'sparse max', es.real.max(), 'sparse top3', np.round(es[:3], 7))
