import sys, os, glob, json, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, C
# Task B2: healing of the frozen defect state of the 8x2 SkX (down-chain end, u = 0) at finite T.
O = 'results_request/B_temp/skx_8x2'; res = []
for f in sorted(glob.glob(f'{O}/rec_T*_0.00000.npy')):
    m = re.search(r'rec_T([\d.]+?)(?:_s(\d+))?_0\.00000\.npy', f); T = float(m.group(1)); s = int(m.group(2) or 1)
    r = np.load(f); t, fd, p6 = r[:, 0], r[:, 6], r[:, 5]; k = 50
    sm = np.convolve(fd, np.ones(k)/k, 'valid'); ts = t[k-1:]; ok = sm < 0.01
    res.append(dict(T=T, seed=s, t_heal=float(ts[np.argmax(ok)]) if ok.any() else None, t_max=float(t[-1]),
                    fdef_last=float(np.nanmean(fd[-300:])), psi6_last=float(np.nanmean(p6[-300:])), Q_last=float(r[-1, 4])))
res.sort(key=lambda d: (d['T'], d['seed'])); json.dump(res, open(f'{O}/B2_summary.json', 'w'), indent=1)
fig, ax = plt.subplots(figsize=(6, 4))
for d in res:
    ax.plot(d['T'], d['t_heal'] if d['t_heal'] else d['t_max'], 'o' if d['t_heal'] else '^', color=C[0] if d['t_heal'] else C[1], mfc=C[0] if d['seed'] == 1 else 'none')
ax.set_yscale('log'); ax.set_xlabel('T'); ax.set_ylabel('healing time (smoothed defect fraction < 0.01)')
ax.set_title('8x2 SkX, u = 0, start = frozen defect state (▲: not healed within run)', fontsize=9)
fig.savefig(f'{O}/fig_B2_heal.png')
for d in res: print(d)
