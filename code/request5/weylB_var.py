import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
os.environ.setdefault('MODEL_FORM', 'pw')
from margin5 import margin
from weylB import solve, FORM
# Task B variants on one near-threshold state per size (primary model): threshold shift Delta* (margin units, as weylB_shift.py) for
#   base: as in B_shift (no advection, k_y = 2 pi n/L_y),
#   adv:  + i v_rel k_x on the diagonal, v_rel = 2.7/L_x (frame of the strain wave),
#   ky:   k_y quantisation removed (L_y -> 8 L_y, practically continuous k_y),
# converted to gamma_c with the measured d m_max/d gamma_wave of each size (B_shift_pw.json). Both orderings.
def root(xs, m, e, Lx, Ly, scheme, vrel):
    f = lambda D: solve(xs, m-D, e, Lx, Ly, scheme, vrel)[0][0]
    lo, hi = (0.0, float(m.max())+0.05) if f(0.0) > 0 else (-0.06, 0.0)
    if f(lo) <= 0 or f(hi) > 0: return np.nan
    for _ in range(14):
        mid = 0.5*(lo+hi); lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5*(lo+hi)
CASES = [('8x2', 8, 'results_request4/C/states/8x2/S_r1.0825.npy'), ('12x3', 12, 'results_request4/C/states/12x3/S_r1.0775.npy'),
         ('16x4', 16, 'results_request4/C/states/16x4/S_r1.0765.npy'), ('24x6', 24, 'results_request5/C1/states/24x6/S_r1.0720.npy'),
         ('32x8', 32, 'results_request3/A32/chain/S_r1.0650.npy')]
BS = json.load(open(f'results_request5/B_shift{"" if FORM == "lin" else "_"+FORM}.json')); out = {}
for sz, nx, f in CASES:
    rows = BS[sz]; g = np.array([r['gamma_wave'] for r in rows]); mm = np.array([r['m_max'] for r in rows]); sl = np.polyfit(g, mm, 1)[0]
    mg = margin(np.load(f).astype(float)); Lx = 19*nx; Ly = 33*nx/4; xs = np.array(mg['xs'])[::10]; m = np.array(mg['m'])[::10]; e = np.array(mg['e'])[::10]
    r = dict(gamma_wave=mg['g_peak'], slope=sl)
    for sc in ('sym', 'weyl'):
        for var, (ly, v) in (('base', (Ly, 0.0)), ('adv', (Ly, 2.7/Lx)), ('ky', (8*Ly, 0.0))):
            D = root(xs, m, e, Lx, ly, sc, v); r[f'{sc}_{var}'] = mg['g_peak']-D/sl
    out[sz] = r
    print(sz, ' '.join(f'{k} {v:.4f}' for k, v in r.items()), flush=True)
json.dump(out, open(f'results_request5/B_variants{"" if FORM == "lin" else "_"+FORM}.json', 'w'), indent=1)
