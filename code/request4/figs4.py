import sys, os, json, glob, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, '..', 'request'))
from style import plt, C
# Request 4 figures 2-5 (figure 1 = task A map, anaA.py).
#  2. gamma_c(L) with gamma_inf fixed at gamma_s(H_peak) and alternatives;  3. gamma_c against the external strain eps0 (and the
#  local strain) with the prediction lines;  4. Floquet exponents and lowest H' eigenvalues against u/u_c;  5. envelope of the critical
#  Floquet vector against the local negative-stiffness window.
O = 'results_request4'; AS = json.load(open(f'{O}/A_summary.json')); GC = json.load(open('results_request3/final_summary.json'))['T0']
P2 = np.array(AS['fit_quad']); gsH = AS['self_consistent']['all']['gamma_s_Hpeak']
# ---- 2 ----
S_ = ('8x2', '12x3', '16x4', '24x6', '32x8'); L = np.array([GC[f'{s}_main']['Lx'] for s in S_], float)
g = np.array([GC[f'{s}_main']['gamma_c']['est'] for s in S_]); e = np.array([GC[f'{s}_main']['gamma_c']['err'] for s in S_])
fig, ax = plt.subplots(figsize=(6.5, 4.4)); ax.errorbar(1/L, g, yerr=e, fmt='D', color=C[3], capsize=3, label=r'$\gamma_c(L)$ (request 3)')
x = np.linspace(0, 1/140, 60)
for i, (name, F) in enumerate(AS['fits_gamma_c'].items()):
    A2 = F['A_L_B_L2']; ax.plot(x, F['gamma_inf']+A2['A']*x+A2['B']*x**2, '-', color=C[i], lw=1, label=f"$\\gamma_\\infty$ = {F['gamma_inf']:.3f} ({name}): A/L+B/L$^2$ $\\chi^2$ = {A2['chi2']:.1f}/3; A/L $\\chi^2$ = {F['A_over_L']['chi2']:.0f}/4")
ax.set_xlim(0, None); ax.set_xlabel(r'$1/L_x$'); ax.set_ylabel(r'$\gamma_c$'); ax.legend(fontsize=6); fig.savefig(f'{O}/fig2_gamma_c_fit.png')
# ---- 3 ----
B = json.load(open(f'{O}/B_summary.json')); LA = json.load(open(f'{O}/B_localavg.json'))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for ci, (sz, g0k) in enumerate([('16x4', '16x4_main'), ('12x3', '12x3_main')]):
    pts = {}
    for dn, d in B.items():
        if d['gamma_c'] is None or f'/{sz}/' not in dn: continue
        pts.setdefault(round(d['steps'][0]['eps0'], 3), []).append(d['gamma_c'])
    pts[0.0] = [dict(lo=GC[g0k]['gamma_c']['lo'], hi=GC[g0k]['gamma_c']['hi'], est=GC[g0k]['gamma_c']['est'])]
    for k, v in sorted(pts.items()):
        b_ = min(v, key=lambda q: q['hi']-q['lo'])
        ax[0].errorbar(k, b_['est'], yerr=[[b_['est']-b_['lo']], [b_['hi']-b_['est']]], fmt='o', color=C[ci], capsize=4, label=sz if k == 0 else None)
    ex = np.linspace(-0.035, 0.035, 10); ax[0].plot(ex, GC[g0k]['gamma_c']['est']+0.46*ex, '--', color=C[ci], lw=0.8)
ax[0].set_xlabel(r'external strain $\varepsilon_0$'); ax[0].set_ylabel(r'$\gamma_c$ (growth onset)'); ax[0].legend(fontsize=8)
ax[0].text(-0.034, 0.36, r'dashed: $\gamma_c(0) + (d\gamma_s/d\varepsilon_{xx})\,\varepsilon_0$', fontsize=7)
for i, (kk, lab) in enumerate([('exx_peak', 'at the peak'), ('exx_mode', 'mode window (0..30 downstream)'), ('exx_strip', 'strip |dx| < 37.5')]):
    sz = [(int(n.split()[1].split('x')[0])*19, LA[n]['DcL_'+kk]) for n in LA if n.startswith('size')] + [(304, LA['eps0=0']['DcL_'+kk])]
    sz.sort(); ax[1].plot(*zip(*sz), 'o-', color=C[i], label=f'eps0 = 0, local strain {lab}')
    for n, mk in (('eps0=-0.03', 'v'), ('eps0=+0.03', '^'), ('12x3 eps0=+0.03', '^'), ('12x3 eps0=-0.03', 'v')):
        if n in LA: ax[1].plot(228 if n.startswith('12x3') else 304, LA[n]['DcL_'+kk], mk, color=C[i], ms=8, mfc='none')
ax[1].set_xlabel(r'$L_x$'); ax[1].set_ylabel(r'$\Delta_c L_x = (\gamma_c - \gamma_s(\varepsilon_{xx}^{loc}))\,L_x$'); ax[1].set_ylim(0, None); ax[1].legend(fontsize=7, loc='lower right')
ax[1].text(150, 33, 'open triangles: external strain -0.03 (v) / +0.03 (^)', fontsize=7)
fig.savefig(f'{O}/fig3_external_strain.png')
# ---- 4 ----
CS = json.load(open(f'{O}/C_summary.json'))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for i, sz in enumerate(['8x2', '12x3', '16x4', '12x3_a0.15', '12x3_a0.6']):
    rows = sorted([v for k, v in CS['C1'].items() if f'/C1/{sz}/' in k and v['gamma_wave'] > 0.25], key=lambda v: v['r'])   # alpha 0.15, 1.096: base state broke down
    if not rows: continue
    # critical exponent: below threshold the leading exponent belongs to the slow y-uniform mode, the critical pair is next
    ax[0].plot([v['gamma_wave'] for v in rows], np.array([max(v['Gamma'][1:3]) if v['Gamma'][0] < 0 else v['Gamma'][0] for v in rows])*1e4, 'o-', color=C[i], label=sz)
for i, sz in enumerate(['8x2', '12x3', '16x4', '12x3_a0.15', '12x3_a0.6']):
    rows = sorted([v for k, v in CS['C2'].items() if f'/C2/{sz}/h_r' in k], key=lambda v: v['r'])
    if rows:
        ax[1].plot([v['r'] for v in rows], [v['min_ev']*1e4 for v in rows], 'o-', color=C[i], label=f'{sz}: H$^\\prime$ (comoving)')
        ax[1].plot([v['r'] for v in rows], [(v['min_ev0'] or np.nan)*1e4 for v in rows], 's--', mfc='none', color=C[i], label=f'{sz}: energy Hessian (w$_1$=0)')
fl = [v for k, v in CS['C2'].items() if 'flat' in k]
if fl: ax[1].axhline(fl[0]['min_ev']*1e4, color='#52514e', ls=':', lw=1); ax[1].text(1.076, fl[0]['min_ev']*1e4+0.2, "H' of the flat driven lattice (0.98 u_c)", fontsize=7)
ax[0].axhline(0, color='#52514e', lw=0.6); ax[0].set_xlabel(r'$\gamma_{wave}$'); ax[0].set_ylabel(r'critical Floquet exponent ($10^{-4}$)'); ax[0].legend(fontsize=7)
ax[1].set_xlabel(r'$u/u_c$'); ax[1].set_ylabel(r'lowest eigenvalue ($10^{-4}$)'); ax[1].legend(fontsize=5)
fig.savefig(f'{O}/fig4_floquet_energy.png')
# ---- 5 ----
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for i, (sz, r) in enumerate([('8x2', 1.085), ('12x3', 1.0775), ('16x4', 1.0765)]):
    v = [v for k, v in CS['C1'].items() if f'/C1/{sz}/' in k and abs(v['r']-r) < 1e-6]
    if not v: continue
    v = v[0]; xs = np.array(v['xs']); Lx = v['Lx']; xp = v['xpeak']; dx = (xs-xp+Lx/2) % Lx-Lx/2; o = np.argsort(dx)
    gam = np.array(v['gam']); gl = np.array(v['gs_local']); env = np.array(v['env']); xe = (np.arange(Lx)-xp+Lx/2) % Lx-Lx/2; oe = np.argsort(xe)
    ax[i].plot(dx[o], gam[o], '-', color=C[0], label=r'$\gamma(x)$'); ax[i].plot(dx[o], gl[o], '-', color=C[1], label=r'$\gamma_s(\varepsilon_{xx}(x))$')
    ax[i].fill_between(dx[o], gl[o], gam[o], where=gam[o] > gl[o], color=C[1], alpha=0.2, label='negative local stiffness')
    a2 = ax[i].twinx(); a2.plot(xe[oe], env[oe], '-', color=C[3], label='|critical Floquet vector|$^2$ envelope'); a2.set_ylim(0, 1.1)
    ax[i].set_title(f"{sz}, u/u_c = {r}, width {v['env_width']:.0f}, overlap {v['window_overlap']:.2f}", fontsize=9); ax[i].set_xlabel('x - x_peak (sites)')
    if i == 0: ax[i].legend(fontsize=7, loc='lower left'); a2.legend(fontsize=7, loc='upper right')
fig.savefig(f'{O}/fig5_mode_envelope.png')
