import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from twist import TwistModel, link_blocks, jac_k
# Geometric protection: a box of nx x ny cells only admits Bloch vectors k = (m/nx) B1 + (n/ny) B2 of the (sheared)
# 19x33 cell. For each relaxed uniformly sheared state (u = 0, results_request2/A_S/state_u0.000_nx1_Lx19_s<s>.npy) the
# largest real part over the box-allowed k gives the static shear threshold gamma_s(box).
import argparse
pa = argparse.ArgumentParser(); pa.add_argument('--nx', type=int, default=1); pa.add_argument('--s', type=int, nargs='*', default=list(range(4, 11)))
A = pa.parse_args()
O = 'results_request2/A_S'; boxes = [(4, 1), (4, 2), (4, 3), (4, 4), (8, 1), (8, 2), (12, 3), (16, 4)]
J, D = 1.0, 0.4; B = 0.5*D*D/J; a = 0.3; z = np.zeros(2)
fn = f'{O}/box_static.json' if A.nx == 1 else f'{O}/box_static_nx{A.nx}.json'; res = json.load(open(fn)) if os.path.exists(fn) else {}
from twist import relax, shear_guess
for s in A.s:
    Lx = 19*A.nx; f = f'{O}/state_u0.000_nx{A.nx}_Lx{Lx}_s{s}.npy'
    if str(s) in res: continue
    m = TwistModel(Lx, 33, s, J, D, B)
    if os.path.exists(f): S = np.load(f)
    else:
        S0 = np.tile(np.load(os.path.join(here, '..', '..', 'states', 'S_B0.5.npy')), (A.nx, 1, 1)).reshape(-1, 3)
        S, r = relax(m, shear_guess(S0, Lx, 33, s)); np.save(f, S)
    bl, _ = link_blocks(m, S, z, z, a); cache = {}
    B19 = 2*np.pi*np.linalg.inv(np.array([[19.0, s/A.nx], [0, 33.0]])).T     # 19x33-cell reciprocal vectors at this gamma
    out = {}
    for nx, ny in boxes:
        mx = -1.0; arg = None
        for i in range(nx):
            for j in range(ny):
                key = (i/nx, j/ny)
                if key not in cache:
                    kv = (key[0]-np.round(key[0]))*B19[0]+key[1]*B19[1]; cache[key] = float(np.linalg.eigvals(jac_k(m, bl, kv)).real.max())
                if cache[key] > mx: mx, arg = cache[key], [i, j]
        out[f'{nx}x{ny}'] = dict(maxRe=mx, k=arg)
    res[str(s)] = dict(gamma=s/Lx, boxes=out); json.dump(res, open(fn, 'w'), indent=1)
    print(s, round(s/Lx, 3), {b: f"{v['maxRe']:.1e}" for b, v in out.items()}, flush=True)
