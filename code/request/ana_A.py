import sys, os, glob, json, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, C
from bragg import registry, bragg_weight
# Task A summary: per size and chain step, end-state A, |psi6|, defect fraction, N_sk, V; u_d and hysteresis; figures.
O = 'results_request/A_size'
ucB = json.load(open(f'{O}/uc_bloch.json')); ucD = json.load(open(f'{O}/uc_dyn.json')) if os.path.exists(f'{O}/uc_dyn.json') else {}
sizes = [s for s in ['4x1', '8x1', '4x2', '8x2', '12x3', '16x4'] if os.path.isdir(f'{O}/chain_{s}')]
def uc_of(s):
    if s in ucD: return ucD[s]['u_c_dyn'], 'dyn'
    base = {'4x2': '4x1', '8x2': '8x1'}.get(s, s)
    if base in ucD: return ucD[base]['u_c_dyn'], 'dyn(' + base + ')'
    return 0.983*ucB[base]['u_c_bloch'], 'bloch*0.983 (mean lab/Bloch ratio of 4x1, 8x1, 12x3)'
summ = {}
for s in sizes:
    uc, how = uc_of(s); rows = {}
    for br in ['up', 'dn']:
        L = []
        for f in sorted(glob.glob(f'{O}/chain_{s}/rec_{br}_*.npy')):
            u = float(re.search(r'_([\d.]+)\.npy', f).group(1)); r = np.load(f); q = r[len(r)*3//4:]
            Sf = f.replace('/rec_', '/S_'); reg, w = None, np.nan
            if os.path.exists(Sf): Sn = np.load(Sf).astype(float); reg = registry(Sn); w = bragg_weight(Sn, reg)
            L.append(dict(u=u, r=u/uc, t=float(r[-1, 0]), A=float(q[:, 1].mean()), A_end=float(r[-1, 1]), psi6=float(np.nanmean(q[:, 5])),
                          fdef=float(np.nanmean(q[:, 6])), fdef_end=float(r[-1, 6]), fdef_max=float(np.nanmax(r[:, 6])), Nsk=int(r[-1, 7]),
                          Q=float(r[-1, 4]), Vx=float(q[:, 2].mean()), Vy=float(q[:, 3].mean()), E=float(q[:, 8].mean()), registry=reg, bragg_w=w, A_reg=float(np.sqrt(max(1-w, 0)/w)) if w == w else None))
        rows[br] = sorted(L, key=lambda d: d['u'], reverse=(br == 'dn'))
    up = rows['up']; dn = rows['dn']
    nx_, ny_ = map(int, s.split('x')); reg0 = tuple(sorted([(0, 2*ny_), (nx_, -ny_), (nx_, ny_)]))
    for d in up+dn: d['registry_changed'] = d['registry'] is not None and tuple(map(tuple, d['registry'])) != reg0 and d['A_reg'] is not None and d['A_reg'] < 0.5
    ud = next((d['u'] for d in up if d['fdef_max'] > 0), None)
    uslip = next((d['u'] for d in up if d['registry_changed']), None)
    dn_reg = [d['u'] for d in dn if d['registry_changed']]
    dn_def = [d['u'] for d in dn if d['fdef_end'] > 0]
    summ[s] = dict(u_c=uc, u_c_source=how, u_c_bloch=ucB[{'4x2': '4x1', '8x2': '8x1'}.get(s, s)]['u_c_bloch'], Lx=19*int(s.split('x')[0]),
                   Ly=33*int(s.split('x')[1]), u_d=ud, u_d_over_uc=None if ud is None else ud/uc,
                   defects_persist_down_to=min(dn_def) if dn_def else None, u_slip=uslip, u_slip_over_uc=None if uslip is None else uslip/uc, registry_changed_down_to=min(dn_reg) if dn_reg else None, up=up, dn=dn)
json.dump(summ, open(f'{O}/A_summary.json', 'w'), indent=1)
for s, d in summ.items():
    print(f"{s}: u_c={d['u_c']:.5f} ({d['u_c_source']}) u_d={d['u_d']} u_d/u_c={d['u_d_over_uc']} defects persist down to {d['defects_persist_down_to']} | u_slip={d['u_slip']} (/uc={d['u_slip_over_uc']}) registry changed down to {d['registry_changed_down_to']}")
    for br in ['up', 'dn']:
        for x in d[br]: print(f"  {br} u/uc={x['r']:.3f} t={x['t']:.0f} A={x['A']:.3g} psi6={x['psi6']:.3f} fdef={x['fdef']:.3f}/{x['fdef_max']:.3f} Nsk={x['Nsk']} Vx/u={x['Vx']/max(x['u'],1e-9):.4f} Vy/Vx={x['Vy']/max(x['Vx'],1e-9):.3f} reg={x['registry']} A_reg={x['A_reg']}")
# figure A1: A(u/u_c) and defect fraction per size
fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
for i, s in enumerate(sizes):
    d = summ[s]
    for br, ls, mk in [('up', '-', 'o'), ('dn', '--', 's')]:
        x = [p['r'] for p in d[br]]; ax[0].plot(x, [max(p['A'], 1e-9) for p in d[br]], ls, marker=mk, ms=4, color=C[i], label=f'{s} {br}' if br == 'up' else None)
        ax[1].plot(x, [p['fdef'] for p in d[br]], ls, marker=mk, ms=4, color=C[i])
ax[0].set_yscale('log'); ax[0].set_ylim(1e-6, 30); ax[0].set_ylabel('deformation amplitude A (last quarter)'); ax[0].axvline(1, color='#52514e', lw=1, ls=':')
ax[0].legend(fontsize=8, ncol=3, title='solid: up, dashed: down', title_fontsize=8)
ax[1].set_ylabel('defect fraction (coordination ≠ 6)'); ax[1].set_xlabel(r'$u_{\rm eff}/u_c(L)$'); ax[1].axvline(1, color='#52514e', lw=1, ls=':')
fig.savefig(f'{O}/fig_A_bifurcation.png')
# figure A2: u_c and u_d vs 1/Lx
fig, ax = plt.subplots(figsize=(6, 4.2))
iL = np.array([1/summ[s]['Lx'] for s in sizes])
ax.plot(iL, [summ[s]['u_c'] for s in sizes], 'o', color=C[0], label=r'$u_c$ (linear threshold)')
xx = np.linspace(0, iL.max()*1.1, 10); ax.plot(xx, xx*np.mean([summ[s]['u_c']*summ[s]['Lx'] for s in sizes]), color=C[0], lw=1, ls='--')
ud = [(1/summ[s]['Lx'], summ[s]['u_d']) for s in sizes if summ[s]['u_d']]
if ud: ax.plot(*zip(*ud), 's', color=C[1], label=r'$u_d$ (first defects, up chain)')
nd = [(1/summ[s]['Lx'], max(p['u'] for p in summ[s]['up'])) for s in sizes if not summ[s]['u_d'] and summ[s]['up']]
if nd: ax.plot(*zip(*nd), '^', mfc='none', color=C[1], label=r'no defects up to (lower bound on $u_d$)')
for s in sizes: ax.annotate(s, (1/summ[s]['Lx'], summ[s]['u_c']), textcoords='offset points', xytext=(5, -10), fontsize=8, color='#52514e')
ax.set_xlabel(r'$1/L_x$'); ax.set_ylabel(r'$u_{\rm eff}$'); ax.set_xlim(0, iL.max()*1.15); ax.set_ylim(0, None); ax.legend(fontsize=8)
fig.savefig(f'{O}/fig_A_uc_ud.png')
