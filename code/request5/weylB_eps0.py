import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
os.environ.setdefault('MODEL_FORM', 'pw')
from margin5 import margin
from weylB_var import root
# Task D addendum: local-theory threshold under external strain eps0 (the task-D states just below threshold, margin with eps0
# included, primary dispersion model). gamma_c^pred = gamma_wave - Delta*/(d m_max/d gamma_wave), slope from the eps0 = 0 chains (1.33).
D = json.load(open('results_request5/D_margin.json')); out = []; seen = set()
for r in D:
    if (r['size'], r['eps0']) in seen: continue
    seen.add((r['size'], r['eps0'])); nx = {'12x3': 12, '16x4': 16}[r['size']]; Lx = 19*nx; Ly = 33*nx/4
    mg = margin(np.load(r['file']).astype(float), r['sx']); xs = np.array(mg['xs'])[::10]; m = np.array(mg['m'])[::10]; e = np.array(mg['e'])[::10]
    res = dict(size=r['size'], eps0=r['eps0'], gamma_wave=mg['g_peak'])
    for sc in ('sym', 'weyl'): res[sc] = mg['g_peak']-root(xs, m, e, Lx, Ly, sc, 0.0)/1.33
    out.append(res); print(res, flush=True)
json.dump(out, open('results_request5/B_eps0.json', 'w'), indent=1)
