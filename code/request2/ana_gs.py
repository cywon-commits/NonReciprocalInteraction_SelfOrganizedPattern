import json, glob, os, numpy as np
# gamma_s(u): soft (largest real part) eigenvalue at the k points near the soft mode versus gamma; linear interpolation of
# the max over k (Doppler zone |k|^2 < 2.5 u |k_x| excluded) between the last stable and first unstable gamma.
O = 'results_request2/A_S'; out = {}
for u in [0.0, 0.02, 0.05, 0.1]:
    pts = {}
    for f in glob.glob(f'{O}/K_u{u:.3f}_nx3_s*.json'):
        d = json.load(open(f))
        for kk, v in d['k'].items():
            kv = np.array(v['kvec'])
            if (kv**2).sum() < 2.5*u*abs(kv[0]) or np.linalg.norm(kv) < 0.02: continue   # Doppler zone and the long-wave acoustic point
            pts.setdefault(round(d['gamma'], 4), {})[kk] = v['top'][0][0]
    S = json.load(open(f'{O}/S_u{u:.3f}_nx1_Lx19.json'))
    for sk in ['5', '6']:
        x = S.get(sk)
        if not x or 'top' not in x: continue
        for kk in sorted({k for v in pts.values() for k in v}):
            i, j = map(int, kk.split(','))
            if [i, j] in x['kgrid']: pts.setdefault(round(x['gamma'], 4), {})[kk] = x['top'][x['kgrid'].index([i, j])][0][0]
    g = sorted(pts); mx = [max(pts[x].values()) for x in g]; arg = [max(pts[x], key=pts[x].get) for x in g]
    gs = None
    for a, b, ma, mb in zip(g, g[1:], mx, mx[1:]):
        if ma < 0 <= mb: gs = a+(b-a)*(-ma)/(mb-ma); break
    out[f'{u:g}'] = dict(gamma=g, maxRe=mx, k_at_max=arg, gamma_s=gs)
    print(f'u={u}: ' + ', '.join(f'{x:.3f}: {m:+.2e} ({k})' for x, m, k in zip(g, mx, arg)) + f'  -> gamma_s = {gs}')
json.dump(out, open(f'{O}/gamma_s.json', 'w'), indent=1)
