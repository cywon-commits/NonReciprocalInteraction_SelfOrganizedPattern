import sys, os, glob, json, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, C
# Task C3: helix strips N = 200 x Ly. Staircase vs the 1D chain, and the anatomy of a slip at u = 0.075:
# does the winding change row by row (a phase-slip line = dislocation gliding across the strip) or all rows at once?
O = 'results_request/C_helix'; res = {}
def stair(tag):
    out = []
    for f in sorted(glob.glob(f'{O}/C3/rec_{tag}_*.npy')) if tag != '1d' else sorted(glob.glob('results/hyst/rec/rec_st_up_*.npy')):
        u = float(re.search(r'_([\d.]+)\.npy', f).group(1)); r = np.load(f)
        if tag == '1d': out.append((u, 0 if r[-1, 5] > 0.9999 else int(round(r[-1, 4]))))
        else: out.append((u, 0 if r[-1, 6] > 0.9999 else int(round(r[-1, 1]))))
    return out
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].step(*zip(*stair('1d')), where='post', color='#52514e', lw=1.2, ls='--', label='1D chain (bundle, T=0)')
for i, Ly in enumerate([8, 16, 32]):
    st = stair(f'strip{Ly}')
    if not st: continue
    res[f'stairs_Ly{Ly}'] = st; ax[0].step(*zip(*st), where='post', color=C[i], label=f'strip Ly={Ly}')
    f = f'{O}/C3/rec_s75_{Ly}_0.07500.npy'
    if os.path.exists(f):
        r = np.load(f); t, wm, wmin, wmax = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        split = np.where(wmax-wmin > 0.5)[0]
        d = dict(t_first_split=float(t[split[0]]) if len(split) else None, t_last_split=float(t[split[-1]]) if len(split) else None,
                 split_duration=float(t[split[-1]]-t[split[0]]) if len(split) else 0.0, n_end=float(wm[-1]), t_end=float(t[-1]))
        res[f'slip_Ly{Ly}'] = d
        if len(split):
            m = (t > t[split[0]]-300) & (t < t[split[-1]]+300)
            ax[1].plot(t[m]-t[split[0]], wm[m], color=C[i], label=f'Ly={Ly}: mean row winding')
            ax[1].fill_between(t[m]-t[split[0]], wmin[m], wmax[m], color=C[i], alpha=0.2)
ax[0].set_xlabel(r'$u_{\rm eff}$'); ax[0].set_ylabel('winding n'); ax[0].legend(fontsize=8); ax[0].set_title('C3: staircase, strips vs 1D (hold 1e5/step)', fontsize=10)
ax[1].set_xlabel('t − (first time rows disagree)'); ax[1].set_ylabel('row winding (band = min..max over rows)'); ax[1].legend(fontsize=8)
ax[1].set_title('slip at u = 0.075: row-by-row (phase-slip line) or all at once', fontsize=10)
fig.savefig(f'{O}/fig_C3_strip.png'); json.dump(res, open(f'{O}/C3_summary.json', 'w'), indent=1)
for k, v in res.items(): print(k, v)
