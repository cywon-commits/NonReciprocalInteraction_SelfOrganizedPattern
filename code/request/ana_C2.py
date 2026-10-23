import sys, os, glob, json, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, C
# Task C2: winding staircase n(u) for N = 200, 600, 1200 (T = 0, hold 1e5 per 0.0025 step) vs linear thresholds u_L(n).
O = 'results_request/C_helix'
def nseq(tag):
    out = []
    for f in sorted(glob.glob(f'{O}/C2/rec_{tag}_*.npy')):
        u = float(re.search(r'_([\d.]+)\.npy', f).group(1)); r = np.load(f)
        n = 0 if r[-1, 6] > 0.9999 else int(round(r[-1, 1])); out.append((u, n))
    return out
res = {}
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for i, N in enumerate([200, 600, 1200]):
    up, dn = nseq(f'up{N}'), nseq(f'dn{N}')
    thr = json.load(open(f'{O}/thr_{N}.json')) if os.path.exists(f'{O}/thr_{N}.json') else {}
    # linear prediction: largest n whose cone is linearly stable at u (u < u_L(n), or stable to unwinding and u < u_unw(n))
    def pred(u):
        ok = [int(n) for n, d in thr.items() if (d['u_L'] if d['u_L'] else d['u_unw']) > u]
        return max(ok) if ok else 0
    jumps = [(up[j][0], up[j-1][1], up[j][1]) for j in range(1, len(up)) if up[j][1] != up[j-1][1]]
    res[N] = dict(up=up, dn=dn, jumps=jumps, u_L={n: d['u_L'] for n, d in thr.items()}, u_unw={n: d['u_unw'] for n, d in thr.items()},
                  pred_at_jump=[(u, pred(u)) for u, _, _ in jumps] if thr else None)
    a = ax[i]
    if up: a.step(*zip(*up), where='post', color=C[0], label='up')
    if dn: a.step(*zip(*dn), where='post', color=C[1], ls='--', label='down')
    if thr:
        uu = np.linspace(0.05, 0.2, 400); a.plot(uu, [pred(x) for x in uu], color='#52514e', lw=1, ls=':', label='largest linearly stable n')
    a.set_title(f'N = {N}', fontsize=10); a.set_xlabel(r'$u_{\rm eff}$'); a.set_ylabel('winding n'); a.legend(fontsize=8)
fig.savefig(f'{O}/fig_C2_stairs.png'); json.dump(res, open(f'{O}/C2_summary.json', 'w'), indent=1)
for N, d in res.items(): print(N, 'jumps (u, from, to):', d['jumps'], ' linear prediction at those u:', d['pred_at_jump'])
