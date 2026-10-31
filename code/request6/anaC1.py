import json, glob, numpy as np
# Request 6, task C1 summary: Bloch (exact linear) threshold at k = 2 pi/L, dynamics threshold (zero of max(Gamma+, Gamma-) by
# linear interpolation between the two bracketing runs at 0.9, 1.0, 1.1 x the Bloch value), and the closed form
# u* sin(phi) = sqrt(c0^2 (1 - M^2) + (2 pi c_b/L)^2), M = u cos(phi)/D (self-consistent).
out = []
for P in (30, 60):
    for phi in (90, 60, 30):
        L = json.load(open(f'results_request6/C/C1_P{P}_phi{phi}_lin.json')); Dy = json.load(open(f'results_request6/C/C1_P{P}_phi{phi}_dyn.json'))
        for k in sorted(L, key=int):
            l = L[k]; d = Dy[k]; us = np.array([r['u'] for r in d['dyn']]); g = np.array([max(r['Gp'], r['Gm']) for r in d['dyn']])
            i = np.where((g[:-1] <= 0) & (g[1:] > 0))[0]; ud = float(us[i[0]]-g[i[0]]*(us[i[0]+1]-us[i[0]])/(g[i[0]+1]-g[i[0]])) if len(i) else float('nan')
            ub = l['lin']['u']; out.append(dict(P=P, phi=phi, L=l['L'], u_bloch=ub, u_dyn=ud, u_pred=l['u_pred'], M=l['lin']['M'], rates=g.tolist(), us=us.tolist()))
            print(f"P={P} phi={phi:3d} L={l['L']:6.1f}: Bloch {ub:.5f} dynamics {ud:.5f} ({(ud/ub-1)*100:+.1f} %) closed form {l['u_pred']:.5f} ({(l['u_pred']/ub-1)*100:+.1f} %) M={l['lin']['M']:.3f}")
json.dump(out, open('results_request6/C1_summary.json', 'w'), indent=1)
