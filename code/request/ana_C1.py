import sys, os, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, C
from scipy.optimize import brentq
# Task C1: local pitch just before a phase slip vs the unwinding criterion u > u_unw(q_loc).
J = 1.0; D = np.tan(2*np.pi*3/200)
uunw = lambda q: 2*(J*np.cos(q)+D*np.sin(q)-J)/np.sin(q)
O = 'results_request/C_helix'; res = []
for f in sorted(glob.glob(f'{O}/C1/rec_*.npy')):
    u = float(f.split('_')[-1][:-4]); n0 = 3 if 'n3' in f else 2; r = np.load(f); t, q5, sy = r[:, 0], r[:, 5], r[:, 7]
    qs = brentq(lambda q: uunw(q)-u, 1e-3, 1.0); q0 = 2*np.pi*n0/200
    d = dict(u_eff=u, n0=n0, q0=q0, q_star=qs, slipped=bool((sy > 0.99).any()))
    if d['slipped'] and q0 < qs:
        ic = int(np.argmax(sy > 0.99)); iq = int(np.argmax(q5 > qs))
        d.update(t_cross=float(t[iq]), t_onset=float(t[ic]), t_slip=float(t[-1]), q_max_onset=float(np.nanmax(q5[:ic])), Sy_at_cross=float(sy[iq]),
                 lead=float(t[ic]-t[iq]))
        dq = q5-q0
        def rate(a, b):
            m = (t > a) & (t < b) & (dq > 0); return float(np.polyfit(t[m], np.log(dq[m]), 1)[0])
        # growth rate of the local compression before vs after it crosses q*: acceleration = onset of the runaway
        d.update(rate_before=rate(t[iq]-30000, t[iq]-10000), rate_after=rate(t[iq], t[iq]+0.8*(t[ic]-t[iq])))
    res.append(d)
json.dump(res, open(f'{O}/C1_summary.json', 'w'), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
qq = np.linspace(0.03, 0.3, 300); ax[0].plot(qq, uunw(qq), color='#52514e', lw=1.5, label=r'$u_{\rm unw}(q)$ (analytic)')
for n0, c, mk in [(3, C[0], 'o'), (2, C[1], 's')]:
    s = [d for d in res if d['n0'] == n0 and 'q_max_onset' in d]
    ax[0].plot([d['q_star'] for d in s], [d['u_eff'] for d in s], mk, mfc='none', color=c, label=f'n={n0}: $q^*$ where $u_{{unw}}(q^*)=u$')
    ax[0].plot([d['q_max_onset'] for d in s], [d['u_eff'] for d in s], mk, color=c, label=f'n={n0}: measured max local pitch at onset')
    ax[0].axvline(2*np.pi*n0/200, color=c, lw=1, ls=':')
ax[0].set_xlim(0.04, 0.3); ax[0].set_ylim(0.05, 0.17); ax[0].set_xlabel('local pitch q (5-site average)'); ax[0].set_ylabel(r'$u_{\rm eff}$'); ax[0].legend(fontsize=8); ax[0].set_title('C1: local pitch before slip vs unwinding criterion', fontsize=10)
for i, u in enumerate([0.0725, 0.08, 0.09]):
    r = np.load(f'{O}/C1/rec_c1n3_{u:.5f}.npy'); d = [x for x in res if x['n0'] == 3 and abs(x['u_eff']-u) < 1e-9][0]
    ax[1].plot(r[:, 0]-d['t_onset'], r[:, 5], color=C[i], lw=1.5, label=f'u={u}')
    ax[1].axhline(d['q_star'], color=C[i], lw=1, ls='--')
ax[1].set_xlim(-15000, 500); ax[1].set_xlabel(r'$t - t_{\rm onset}$  (onset: max $S_y$ > 0.99)'); ax[1].set_ylabel('max local pitch')
ax[1].legend(fontsize=8); ax[1].set_title('n=3 start: dashed = $q^*(u)$', fontsize=10)
fig.savefig(f'{O}/fig_C1_pitch.png')
for d in res: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})
