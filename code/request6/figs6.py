import sys, os, json, glob, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '..', '..'); sys.path.insert(0, os.path.join(root, 'code', 'request'))
from style import plt, C
# Request 6 figures. check(): overlap test of all text artists (titles, labels, tick labels, legends, annotations) of a figure
# by their rendered bounding boxes (stand-in for the B2 style.py check(), which is not in this repository); a figure is saved
# only after the test, and the overlaps found are printed.
O = 'results_request6'
def check(fig, name):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); boxes = []
    for ax in fig.axes:
        x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim()); eps = 1e-12
        tx = [t for t in ax.get_xticklabels() if t.get_text() and x0-eps*abs(x0) <= t.get_position()[0] <= x1+eps*abs(x1)]
        ty = [t for t in ax.get_yticklabels() if t.get_text() and y0-eps*abs(y0) <= t.get_position()[1] <= y1+eps*abs(y1)]
        arts = [ax.title, ax.xaxis.label, ax.yaxis.label]+tx+ty+list(ax.texts)
        lg = ax.get_legend()
        if lg: arts.append(lg)
        boxes += [(a, a.get_window_extent(r)) for a in arts if a.get_visible() and (not hasattr(a, 'get_text') or a.get_text())]
    bad = []
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.width > 0 and b.width > 0 and a.overlaps(b):
                ov = max(0, min(a.x1, b.x1)-max(a.x0, b.x0))*max(0, min(a.y1, b.y1)-max(a.y0, b.y0))
                if ov > 4: bad.append((str(boxes[i][0])[:40], str(boxes[j][0])[:40]))
    print(f'{name}: overlap check', 'OK' if not bad else f'{len(bad)} overlaps: {bad[:4]}', flush=True)
    fig.savefig(f'{O}/{name}.png'); plt.close(fig); return bad
# ---- A: correlation lengths and defects against N
A = json.load(open(f'{O}/A_summary.json')); fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for i, ue in enumerate((0.15, 0.2, 0.3)):
    rs = sorted([r for r in A if abs(r['ue']-ue) < 1e-9], key=lambda r: r['N']); N = [r['N'] for r in rs]
    ax[0].plot(N, [r['xi6_a0'] for r in rs], 'o-', color=C[i], label=f'u = {ue}: xi6'); ax[0].plot(N, [r['xiT_a0'] for r in rs], 's--', color=C[i], mfc='none', label=f'u = {ue}: xiT')
    ax[1].plot(N, [r['ratio'] for r in rs], 'o-', color=C[i], label=f'u_eff = {ue}')
    ax[2].plot(N, [r['defects']['disc'] for r in rs], 'o-', color=C[i], label=f'u = {ue}: disclinations'); ax[2].plot(N, [r['defects']['disl'] for r in rs], 's--', color=C[i], mfc='none', label=f'u = {ue}: dislocations')
for a_ in ax: a_.set_xscale('log'); a_.set_xticks([100, 400, 900]); a_.set_xticklabels(['100', '400', '900']); a_.set_xlabel('N (skyrmions)')
ax[0].set_ylabel('correlation length / a0'); ax[1].set_ylabel('xi6 / xiT'); ax[2].set_ylabel('density per skyrmion'); ax[1].set_ylim(0, 3)
ax[0].legend(fontsize=7, ncol=2); ax[1].legend(fontsize=8); ax[2].legend(fontsize=7, ncol=2)
check(fig, 'figA_hexatic')
# ---- B: Psi6(T) on heating (seed mean) and T_m(u)/T_m(0) against the long-wave prediction
Bs = json.load(open(f'{O}/B_summary.json')); fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for j, Lx in enumerate((152, 228)):
    for i, ue in enumerate((0.0, 0.01, 0.02, 0.03)):
        cur = {}
        for f in glob.glob(f'{O}/B/melt_u{ue}_{"8x16" if Lx == 152 else "12x11"}_s*.json'):
            for s in json.load(open(f))['steps']:
                if s['leg'] == 'heat' and 0.155 < s['T'] < 0.245: cur.setdefault(round(s['T'], 3), []).append(s['Psi6'])
        T = sorted(cur); ax[j].plot(T, [np.mean(cur[t]) for t in T], 'o-', color=C[i], label=f'u_eff = {ue}')
    ax[j].axhline(0.4, color='#9a9893', lw=1, ls=':'); ax[j].set_xlabel('T'); ax[j].set_ylabel('Psi6 (heating, mean of 3 seeds)'); ax[j].set_title(f'Lx = {Lx}', fontsize=10); ax[j].legend(fontsize=8)
hs = Bs['heating_seeds']
for j, (Lx, mk) in enumerate(((152, 'o'), (228, 's'))):
    rs = sorted([r for r in hs if r['Lx'] == Lx], key=lambda r: r['ue']); u = [r['ue'] for r in rs]
    ax[2].errorbar(u, [r['ratio'] for r in rs], yerr=[r['ratio_err'] for r in rs], fmt=mk+'-', color=C[j], capsize=3, label=f'measured, Lx = {Lx}')
    ax[2].plot(u, [r['pred']['four'] for r in rs], mk+':', color=C[j], mfc='none', label=f'long-wave, 4 modes, Lx = {Lx}')
ax[2].set_xlabel('u_eff'); ax[2].set_ylabel('T_m(u)/T_m(0)'); ax[2].set_ylim(0.5, 1.1); ax[2].legend(fontsize=7)
check(fig, 'figB_melting')
# ---- C1: finite-box thresholds
C1 = json.load(open(f'{O}/C1_summary.json')); fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for j, P in enumerate((30, 60)):
    for i, phi in enumerate((90, 60, 30)):
        rs = sorted([r for r in C1 if r['P'] == P and r['phi'] == phi], key=lambda r: r['L']); L = [r['L'] for r in rs]
        ax[j].plot(L, [r['u_dyn'] for r in rs], 'o', color=C[i], ms=8, label=f'phi = {phi}: dynamics')
        ax[j].plot(L, [r['u_bloch'] for r in rs], 'x', color=C[i], ms=8, label=f'phi = {phi}: Bloch')
        ax[j].plot(L, [r['u_pred'] for r in rs], '-', color=C[i], lw=1, label=f'phi = {phi}: closed form')
    ax[j].set_xscale('log'); ax[j].set_yscale('log'); ax[j].set_xlabel('L (along the stripe)'); ax[j].set_ylabel('u_eff*'); ax[j].set_title(f'P = {P}', fontsize=10); ax[j].legend(fontsize=7, ncol=3)
check(fig, 'figC1_thresholds')
# ---- C2: linear growth above threshold (Bloch) and end states
G = json.load(open(f'{O}/C2_growth.json')); fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for i, key in enumerate(G):
    rs = [r for r in G[key]['rows'] if 'G_max' in r]; f = np.array([r['f'] for r in rs])
    ax[0].loglog(f-1, [r['k_star'] for r in rs], 'o-', color=C[i], label=key); ax[1].loglog(f-1, [r['G_max'] for r in rs], 'o-', color=C[i], label=key)
x = np.geomspace(0.1, 5, 10); ax[0].loglog(x, 0.04*x**0.5, ':', color='#52514e', lw=1, label='slope 1/2'); ax[1].loglog(x, 2e-4*x**2, ':', color='#52514e', lw=1, label='slope 2')
ax[0].set_xlabel('u/u* - 1'); ax[0].set_ylabel('k* (fastest bending wavevector)'); ax[1].set_xlabel('u/u* - 1'); ax[1].set_ylabel('Gamma_max'); ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
for f_ in sorted(glob.glob(f'{O}/C/C2_*.json')):
    d = json.load(open(f_)); R = np.array(d['rec']); lab = os.path.basename(f_)[3:-5]
    if 'n0.01' in lab or d['fac'] == 3.0: ax[2].semilogy(R[:, 0], np.maximum(R[:, 1], 1e-4), lw=1, label=lab)
ax[2].set_xlabel('t'); ax[2].set_ylabel('rms bending amplitude (sites)'); ax[2].set_xscale('log'); ax[2].legend(fontsize=5, ncol=2)
check(fig, 'figC2_growth_endstates')
# ---- D: delta D channel
Dd = json.load(open(f'{O}/D_summary.json')); fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for i, form in enumerate(('local', 'full', 'stiff')):
    r = [x for x in Dd if x['form'] == form and x['Lx'] == 19 and x['n'] == [0.0, 1.0, 0.0]][0]; raw = np.array(r['ks']['kx+0.0165']['raw'])
    dD = [s['dD'] for s in r['static']]
    ax[0].plot(dD, raw[:, 0]*1e5, 'o-', color=C[i], label=f'{form}: branch -omega'); ax[0].plot(dD, raw[:, 1]*1e5, 's--', color=C[i], mfc='none', label=f'{form}: branch +omega')
r = [x for x in Dd if x['form'] == 'stiff' and x['Lx'] == 19][0]; ks = [0.0165, 0.0661, 0.1653]
for j, ax_ in enumerate(('kx', 'ky')):
    kk = ks[:2] if ax_ == 'kx' else ks                                   # k = 0.1653 || x is the zone boundary 2 pi/38 (split = 0 by symmetry)
    sl = [abs(r['ks'][f'{ax_}+{k:.4f}']['dsplit_ddD']) for k in kk]; ax[1].loglog(kk, sl, 'o-', color=C[j], label=f'stiffness term, k || {ax_[1]}')
ax[1].loglog(ks, 0.12*np.array(ks), ':', color='#52514e', lw=1, label='slope 1')
r = [x for x in Dd if x['form'] == 'full' and x['Lx'] == 19 and x['n'] == [0.0, 1.0, 0.0]][0]
V = [0.2386*d for d in dD]; ax[2].plot(dD, V, 'o-', color=C[0], label='|V_x| (full)'); ax[2].plot(dD, [0.1043*d for d in dD], 's-', color=C[1], label='|V_y| (full)')
ax[0].set_xlabel('dD'); ax[0].set_ylabel('Re lambda (1e-5), k = 0.0165 || x'); ax[0].legend(fontsize=6, ncol=2); ax[1].set_xlabel('k'); ax[1].set_ylabel('|d(split)/d dD|'); ax[1].legend(fontsize=7)
from matplotlib.ticker import NullFormatter; ax[1].xaxis.set_minor_formatter(NullFormatter())
ax[2].set_xlabel('dD'); ax[2].set_ylabel('drift velocity'); ax[2].legend(fontsize=8)
check(fig, 'figD_deltaD')
