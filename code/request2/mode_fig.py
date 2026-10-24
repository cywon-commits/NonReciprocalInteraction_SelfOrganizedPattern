import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here); sys.path.insert(1, os.path.join(here, '..', 'request'))
from twist import TwistModel, link_blocks, jac_k, basis
from style import plt, C
# Task A, figure 5: the soft mode of the uniformly sheared lattice (u = 0) at the first unstable shear.
# The eigenvector (tangent components per spin) is converted to rigid displacements of the two skyrmions of the cell,
# dr = -M^-1 b with M_ab = sum_core dS/da . dS/db and b_a = sum_core Re(dS) . dS/da, then repeated over many cells with the
# Bloch phase exp(i k.R). The arrows show which rows slide against which: the nucleation pattern of the dislocation pair.
O = 'results_request2/A_S'; s = int(sys.argv[1]) if len(sys.argv) > 1 else 6; ksel = sys.argv[2] if len(sys.argv) > 2 else '15,6'
J, D = 1.0, 0.4; B = 0.5*D*D/J; a = 0.3; z = np.zeros(2); nk = 16
m = TwistModel(19, 33, s, J, D, B); S = np.load(f'{O}/state_u0.000_nx1_Lx19_s{s}.npy'); bl, _ = link_blocks(m, S, z, z, a)
i, j = map(int, ksel.split(',')); kv = ((i/nk)-np.round(i/nk))*m.Bk[0]+(j/nk)*m.Bk[1]
ev, V = np.linalg.eig(jac_k(m, bl, kv)); o = int(np.argmax(ev.real)); v = V[:, o]; N = m.N
e1, e2 = basis(S); dS = v[:N, None]*e1+v[N:, None]*e2                   # complex spin perturbation of the cell
gx, gy = m.grads(S); Lx, Ly, sh = 19, 33, s
from scipy import ndimage
Sz = S[:, 2].reshape(Lx, Ly); big = np.zeros((3*Lx, 3*Ly))
for a_ in range(3):
    for b_ in range(3): big[a_*Lx:(a_+1)*Lx, b_*Ly:(b_+1)*Ly] = np.roll(Sz, (a_-1)*sh, axis=1)
lab, nc = ndimage.label(big < 0); cores = []
for c in range(1, nc+1):
    I, Jb = np.nonzero(lab == c); cx, cy = I.mean(), Jb.mean()
    if not (Lx <= cx < 2*Lx and Ly <= cy < 2*Ly): continue
    ix = I//Lx; idx = (I % Lx)*Ly+((Jb % Ly)-(ix-1)*sh) % Ly            # canonical site of every core pixel
    M = np.array([[np.sum(gx[idx]*gx[idx]), np.sum(gx[idx]*gy[idx])], [np.sum(gy[idx]*gx[idx]), np.sum(gy[idx]*gy[idx])]])
    b = np.array([np.sum(dS[idx]*gx[idx]), np.sum(dS[idx]*gy[idx])])
    cores.append((np.array([cx-Lx, cy-Ly]), -np.linalg.solve(M, b)))
# repeat over cells: R = n1 A1 + n2 A2 with A1 = (19, s), A2 = (0, 33)
fig, ax = plt.subplots(figsize=(8, 8)); P = []; Dp = []
for n1 in range(-6, 7):
    for n2 in range(-3, 4):
        R = n1*np.array([19., s])+n2*np.array([0., 33.])
        for r0, dr in cores:
            P.append(r0+R); Dp.append((dr*np.exp(1j*kv@R)).real)
P = np.array(P); Dp = np.array(Dp); Dp /= np.abs(Dp).max()
ax.scatter(P[:, 0], P[:, 1], s=12, color='#52514e')
ax.quiver(P[:, 0], P[:, 1], Dp[:, 0], Dp[:, 1], color=C[1], angles='xy', scale_units='xy', scale=0.12, width=0.003)
ax.set_aspect('equal'); ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_xlabel('x (sites)'); ax.set_ylabel('y (sites)')
ax.set_title(f'soft mode at gamma = {s/19:.3f} (u = 0), k = {np.round(kv, 3)}, lambda = {ev[o].real:.2e}', fontsize=9)
fig.savefig(f'{O}/fig_A_softmode.png')
json.dump(dict(s=s, gamma=s/19, k=kv.tolist(), eig=[float(ev[o].real), float(ev[o].imag)], core_disp=[[r0.tolist(), [complex(x).real for x in dr], [complex(x).imag for x in dr]] for r0, dr in cores]),
          open(f'{O}/softmode.json', 'w'), indent=1)
print('eig', ev[o], 'cores', len(cores), 'disp', [np.round(dr, 3) for _, dr in cores])
pol = [np.degrees(np.arctan2(dr.real[1], dr.real[0])) % 180 for _, dr in cores]; print('polarisation angles (deg, mod 180):', np.round(pol, 1), ' k direction (deg):', round(np.degrees(np.arctan2(kv[1], kv[0])) % 180, 1), ' |k| =', round(np.linalg.norm(kv), 4), ' wavelength', round(2*np.pi/np.linalg.norm(kv), 1))
