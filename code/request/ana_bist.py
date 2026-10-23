import sys, os, glob, json, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bragg import registry, bragg_weight
# Task A fixed-u tests: end state from the perfect lattice vs from the strongly deformed (2 u_c up-chain end) state.
O = 'results_request/A_size'; uc = {'4x1': 0.09942, '8x2': 0.04983, '12x3': 0.03327, '16x4': 0.02492}
res = {}
for s, u0 in uc.items():
    nx, ny = map(int, s.split('x')); reg0 = tuple(sorted([(0, 2*ny), (nx, -ny), (nx, ny)]))
    for f in sorted(glob.glob(f'{O}/bist_{s}/rec_*.npy')):
        st, u = re.search(r'rec_(perf|def)_([\d.]+)\.npy', f).groups(); u = float(u); r = np.load(f); q = r[len(r)*3//4:]
        S = np.load(f.replace('/rec_', '/S_')).astype(float); reg = registry(S); w = bragg_weight(S, reg)
        d = dict(start=st, t=float(r[-1, 0]), A=float(q[:, 1].mean()), A_trend=float(r[-1, 1]/q[0, 1]), psi6=float(np.nanmean(q[:, 5])),
                 fdef_end=float(r[-1, 6]), fdef_max=float(np.nanmax(r[:, 6])), registry=reg, registry_changed=reg != reg0, A_reg=float(np.sqrt((1-w)/w)),
                 Vx_over_u=float(q[:, 2].mean()/u))
        res.setdefault(s, {}).setdefault(f'{u/u0:.2f}', {})[st] = d
json.dump(res, open(f'{O}/bist_summary.json', 'w'), indent=1)
for s, D in res.items():
    for r_, x in sorted(D.items()):
        print(s, 'u/uc=' + r_, ' | '.join(f"{k}: t={v['t']:.0f} A={v['A']:.3g} (x{v['A_trend']:.2f}) psi6={v['psi6']:.2f} fdef={v['fdef_end']:.3f}/{v['fdef_max']:.3f} reg={'chg' if v['registry_changed'] else 'orig'} A_reg={v['A_reg']:.2f}" for k, v in x.items()))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, C
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, s in enumerate(res):
    for r_, x in res[s].items():
        for st, mk in [('perf', 'o'), ('def', 's')]:
            if st in x:
                v = x[st]; off = -0.004 if st == 'perf' else 0.004
                ax[0].plot(float(r_)+off+0.002*i, v['psi6'], mk, color=C[i], mfc=C[i] if st == 'perf' else 'none', label=f'{s}' if (st == 'perf' and r_ == sorted(res[s])[0]) else None)
                ax[1].plot(float(r_)+off+0.002*i, v['fdef_end'], mk, color=C[i], mfc=C[i] if st == 'perf' else 'none')
ax[0].set_ylabel(r'$|\langle\psi_6\rangle|$ (end state)'); ax[1].set_ylabel('defect fraction (end state)')
for a in ax: a.set_xlabel(r'$u_{\rm eff}/u_c$')
ax[0].legend(fontsize=8, title='filled ○: perfect start, open □: deformed start', title_fontsize=8)
fig.savefig(f'{O}/fig_A_fixed_u.png')
