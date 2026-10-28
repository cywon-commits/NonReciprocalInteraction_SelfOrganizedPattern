import sys, os, json, glob, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from c0 import analyse
# Task C1: Floquet threshold of the 24x6 strain wave (request-4 floquet.py on the request-5 states) and the C0 analysis of its vectors.
# gamma_c: linear interpolation of the leading exponent Gamma_1 against gamma_wave between the last negative and first positive state.
out = {}
for sz in ('24x6', '32x8'):
    rows = []
    for fj in sorted(glob.glob(f'results_request5/C1/{sz}/f_r*.json')):
        d = json.load(open(fj)); res = [analyse(fj, iq) for iq in range(4) if os.path.exists(fj.replace('.json', f'_q{iq}.npy'))]
        cand = [x for x in res if abs(x['k'][1]) > 1e-9] or res; a = max(cand, key=lambda x: x['carrier_weight'])
        a['all'] = [(x['iq'], x['k'], round(x['carrier_weight'], 2)) for x in res]; a['Gamma_err'] = d['Gamma_err'][:4]; rows.append(a)
        print(sz, d['r'], 'Gamma', np.round(d['Gamma'][:3], 7), '+-', np.round(d['Gamma_err'][:2], 7), 'gw %.4f' % a['gamma_wave'], 'k', np.round(a['k'], 4),
              'w %.2f rms %.1f half %.0f off %+.0f sigma*k %.2f' % (a['carrier_weight'], a['rms'], a['half'], a['offset'], a['sigma_k']), flush=True)
    if not rows: continue
    rows.sort(key=lambda a: a['gamma_wave']); g = np.array([a['gamma_wave'] for a in rows]); G = np.array([a['Gamma'][0] for a in rows])
    i = np.where((G[:-1] < 0) & (G[1:] > 0))[0]; gc = float(g[i[0]]-G[i[0]]*(g[i[0]+1]-g[i[0]])/(G[i[0]+1]-G[i[0]])) if len(i) else None
    up = [a for a in rows if a['Gamma'][0] > 0]
    out[sz] = dict(gamma_c=gc, rows=rows, pick=dict(state=up[0]['file'].replace('C1/'+sz+'/f_', 'C1/states/'+sz+'/S_').replace('.json', '.npy'), key=f"C1/{sz}/f_r{up[0]['r']:.4f}") if up else None)
    print(sz, 'gamma_c (Floquet) =', gc, flush=True)
s = out.get('24x6', {}); json.dump(dict(gamma_c=s.get('gamma_c'), pick=s.get('pick'), c0={a['file']: a for v in out.values() for a in v['rows']}, sizes=out),
                                      open('results_request5/C1_summary.json', 'w'), indent=1)
