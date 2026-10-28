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
        if res and res[0]['gamma_wave'] > 0.4:
            # the base trajectory itself broke up (above threshold the wave is unstable; r = 1.073 after window 24): Gamma from the
            # windows before the break-up (10 ... 22), no mode analysis
            h = np.array(d['hist']); d['Gamma'] = h[10:23].mean(0).tolist(); d['Gamma_err'] = (h[10:23].std(0)/np.sqrt(13)).tolist()
            gw0 = json.load(open(d['start'].replace('.npy', '.json')))['gamma_wave']
            rows.append(dict(file=fj, r=d['r'], Gamma=d['Gamma'][:4], Gamma_err=d['Gamma_err'][:4], gamma_wave=gw0, broken=True))
            print(sz, d['r'], 'base broke up; Gamma (windows 10-22)', np.round(d['Gamma'][:3], 7), 'gw(start) %.4f' % gw0, flush=True); continue
        cand = [x for x in res if abs(x['k'][1]) > 1e-9] or res; a = max(cand, key=lambda x: x['carrier_weight'])
        a['all'] = [(x['iq'], x['k'], round(x['carrier_weight'], 2)) for x in res]; a['Gamma_err'] = d['Gamma_err'][:4]
        a['gamma_wave_end'] = a['gamma_wave']; a['gamma_wave'] = json.load(open(d['start'].replace('.npy', '.json')))['gamma_wave']; rows.append(a)
        print(sz, d['r'], 'Gamma', np.round(d['Gamma'][:3], 7), '+-', np.round(d['Gamma_err'][:2], 7), 'gw %.4f' % a['gamma_wave'], 'k', np.round(a['k'], 4),
              'w %.2f rms %.1f half %.0f off %+.0f sigma*k %.2f' % (a['carrier_weight'], a['rms'], a['half'], a['offset'], a['sigma_k']), flush=True)
    if not rows: continue
    rows.sort(key=lambda a: a['gamma_wave']); g = np.array([a['gamma_wave'] for a in rows]); G = np.array([a['Gamma'][0] for a in rows])
    i = np.where((G[:-1] < 0) & (G[1:] > 0))[0]; gc = float(g[i[0]]-G[i[0]]*(g[i[0]+1]-g[i[0]])/(G[i[0]+1]-G[i[0]])) if len(i) else None
    up = [a for a in rows if a['Gamma'][0] > 0 and not a.get('broken')]
    out[sz] = dict(gamma_c=gc, rows=rows, pick=dict(state=up[0]['file'].replace('C1/'+sz+'/f_', 'C1/states/'+sz+'/S_').replace('.json', '.npy'), key=f"C1/{sz}/f_r{up[0]['r']:.4f}") if up else None)
    # gamma_wave of the prepared state (start of the Floquet run, as in task B and request 3); the base drifts by +0.001 over the
    # 30 windows (gamma_wave_end), which gives the alternative value
    ge = np.array([a.get('gamma_wave_end', a['gamma_wave']) for a in rows]); gce = float(ge[i[0]]-G[i[0]]*(ge[i[0]+1]-ge[i[0]])/(G[i[0]+1]-G[i[0]])) if len(i) else None
    out[sz]['gamma_c_end'] = gce
    print(sz, 'gamma_c (Floquet) =', gc, '(end of the base trajectory:', gce, ')', flush=True)
s = out.get('24x6', {}); json.dump(dict(gamma_c=s.get('gamma_c'), gamma_c_end=s.get('gamma_c_end'), pick=s.get('pick'), c0={a['file']: a for v in out.values() for a in v['rows'] if not a.get('broken')}, sizes=out),
                                      open('results_request5/C1_summary.json', 'w'), indent=1)
