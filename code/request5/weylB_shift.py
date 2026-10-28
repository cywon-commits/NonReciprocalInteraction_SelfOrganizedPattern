import sys, os, json, glob, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from margin5 import margin
from weylB import solve
# Task B, threshold from the local theory: Lambda0 > 0 in every measured strain-wave state (B_weyl.json), so the predicted threshold
# is read as the margin the local operator tolerates. For each state: the uniform downward shift Delta* of the measured margin
# profile, m(x) -> m(x) - Delta, at which the largest Weyl (and 'sym') eigenvalue crosses zero (bisection). The local theory predicts
# stability for m_max < m_max - Delta* = m_crit, i.e. m_crit L is the tolerated margin, against the measured m_max L of the
# Floquet / dynamical threshold states (request 4/5 task D: 22.5 (12x3), 26.0 (16x4)). Predicted gamma_c: the state sequence of
# each size gives d m_max/d gamma_wave; gamma_c^pred = gamma_wave - Delta*/(d m_max/d gamma_wave) from the state nearest threshold.
def root(xs, m, e, Lx, Ly, scheme):
    # Delta* > 0: the margin must be lowered to reach Lambda0 = 0 (state predicted unstable); < 0: raised (predicted stable)
    f = lambda D: solve(xs, m-D, e, Lx, Ly, scheme)[0][0]
    if f(0.0) > 0: lo, hi = 0.0, float(m.max())+0.05
    else: lo, hi = -0.06, 0.0
    if f(lo) <= 0 or f(hi) > 0: return np.nan
    for _ in range(14):
        mid = 0.5*(lo+hi)
        if f(mid) > 0: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)
SETS = {'8x2': ('results_request4/C/states/8x2', 8), '12x3': ('results_request4/C/states/12x3', 12), '16x4': ('results_request4/C/states/16x4', 16),
        '24x6': ('results_request5/C1/states/24x6', 24), '32x8': ('results_request3/A32/chain', 32)}
out = {}
for size, (d, nx) in SETS.items():
    rows = []
    for f in sorted(glob.glob(f'{d}/S_r*.npy')):
        jf = f.replace('.npy', '.json')
        if os.path.exists(jf) and not json.load(open(jf)).get('intact', True): continue
        mg = margin(np.load(f).astype(float)); Lx = 19*nx; Ly = 33*nx/4
        if mg['g_peak'] > 0.4: continue                                    # wave already broken up (past the dynamical threshold)
        xs = np.array(mg['xs'])[::10]; m = np.array(mg['m'])[::10]; e = np.array(mg['e'])[::10]
        Dw = root(xs, m, e, Lx, Ly, 'weyl'); Ds = root(xs, m, e, Lx, Ly, 'sym')
        rows.append(dict(file=f, gamma_wave=mg['g_peak'], m_max=mg['m_max'], mmaxL=mg['mmaxL'], m2L2=mg['m2L2'], D_weyl=Dw, D_sym=Ds,
                         mcritL_weyl=(mg['m_max']-Dw)*Lx, mcritL_sym=(mg['m_max']-Ds)*Lx))
        print(size, os.path.basename(f), 'gw %.4f m_max*L %.1f | Delta* weyl %.4f sym %.4f -> m_crit*L weyl %.1f sym %.1f' % (mg['g_peak'], mg['mmaxL'], Dw, Ds, rows[-1]['mcritL_weyl'], rows[-1]['mcritL_sym']), flush=True)
    if len(rows) >= 2:
        g = np.array([r['gamma_wave'] for r in rows]); mm = np.array([r['m_max'] for r in rows]); sl = np.polyfit(g, mm, 1)[0]
        for r in rows: r['gc_pred_weyl'] = r['gamma_wave']-r['D_weyl']/sl; r['gc_pred_sym'] = r['gamma_wave']-r['D_sym']/sl
        print(size, 'd m_max/d gamma_wave = %.2f' % sl, 'gamma_c^pred weyl', np.round([r['gc_pred_weyl'] for r in rows], 4), 'sym', np.round([r['gc_pred_sym'] for r in rows], 4), flush=True)
    out[size] = rows
json.dump(out, open('results_request5/B_shift.json', 'w'), indent=1)
