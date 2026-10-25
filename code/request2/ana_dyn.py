import sys, os, json, glob, re, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'request'))
from style import plt, C
# Tasks B-D summary from run2.py outputs.
# Per run: gamma_sat = time average of gamma_max over the last quarter (if no topological event), first topological event
# (registry lost in two consecutive records: Burgers circuit or Bragg-index change), gamma* = gamma_max of the last intact
# record before it, and whether the change is permanent (end state not intact) or a transient dipole burst.
O = 'results_request2'; uc = {4: 0.0994, 8: 0.0498, 12: 0.0333, 16: 0.0249, 24: 7.57/456}
def load(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f)); d['file'] = f; d['r'] = round(d['u']/uc[d['nx']], 3)
        r = np.load(f.replace('/sum_', '/rec_').replace('.json', '.npy'))
        ev = d['slipped']; t_ev = d['t_slip']
        pre = r[r[:, 0] < (t_ev if ev else np.inf)]; q = pre[len(pre)*3//4:] if len(pre) > 8 else pre
        d['gamma_sat'] = float(np.nanmean(q[:, 3])) if len(q) else None
        d['gamma_sat_std'] = float(np.nanstd(q[:, 3])) if len(q) else None
        d['permanent'] = bool(ev and not d['intact_end'])
        w = pre[-5:, 3] if ev and len(pre) else np.array([np.nan])
        d['gamma_star_mean5'] = float(np.nanmean(w)); d['gamma_star_max5'] = float(np.nanmax(w))
        d['n_events'] = int(np.sum(np.diff((r[:, 1] < 0.5).astype(int)) == 1))
        d['fdef_max'] = float(np.nanmax(r[:, 10]))
        out.append(d)
    return out
res = {}
# B: universal curve
B = {}
for size in ['4x1', '8x2', '12x3', '16x4', '24x6']:
    runs = load(f'{O}/B_univ/{size}/sum_*.json')
    if runs: B[size] = runs
res['B'] = {s: [{k: d[k] for k in ('file', 'r', 'slipped', 't_slip', 'gamma_star', 'gamma_star_mean5', 'gamma_star_max5', 'gamma_sat', 'gamma_sat_std', 'permanent', 'n_events', 'intact_end', 'm_end', 'ndef_end', 't_end')} for d in v] for s, v in B.items()}
S = {}
if os.path.exists(f'{O}/A_S/gamma_s.json'): S['gamma_s0'] = json.load(open(f'{O}/A_S/gamma_s.json'))['0']['gamma_s']
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
CR = {}
for ny in [1]:
    CR['4x1'] = load(f'{O}/C_rows/4x1/sum_*.json')                       # 4x1 at 1.10, 1.15 (events)
for i, (s, runs) in enumerate(B.items()):
    ok = sorted([d for d in runs if d['gamma_sat'] is not None and not d['slipped'] and d['r'] <= 1.081], key=lambda d: d['r'])
    if ok:
        x = np.array([d['r'] for d in ok]); g = np.array([d['gamma_sat'] for d in ok])
        ax[0].plot(x, g**2/(x-1), 'o', color=C[i], label=s); ax[1].plot(x, g, 'o', color=C[i], label=s)
    for d in [d for d in runs if d['slipped'] and 'chain' in d['file']]+[d for d in CR.get(s, []) if d['slipped']]:
        ax[1].plot(d['r'], d['gamma_star_mean5'], 's' if d['permanent'] else 'x', color=C[i], ms=8)
ax[0].set_xlabel(r'$u/u_c$'); ax[0].set_ylabel(r'$\gamma_{\max}^2/(u/u_c-1)$'); ax[0].legend(fontsize=8); ax[0].set_title('B: universal curve (intact saturated states)', fontsize=10)
ax[1].set_xlabel(r'$u/u_c$'); ax[1].set_ylabel(r'$\gamma_{\max}$  (o saturated, ■ $\gamma^*$ permanent slip, x transient)'); ax[1].legend(fontsize=8)
if 'gamma_s0' in S: ax[1].axhline(S['gamma_s0'], color='#52514e', ls='--', lw=1); ax[1].text(1.005, S['gamma_s0']+0.005, r'$\gamma_s(0)$ (task A)', fontsize=8)
# u_s/u_c and gamma* vs 1/L from the quasi-static chains (seeded runs above u_s overshoot and are not used);
# 4x1: chain intact to 1.08, first events at 1.10 (C_rows). 24x6: chain if finished, else the seeded event as an upper bound.
us = {}
for s, runs in B.items():
    ch = [d for d in runs if 'chain' in d['file'] or (s == '4x1' and 'seed' not in d['file'])] + CR.get(s, [])
    ch += [d for d in runs if 'seed' in d['file'] and not d['slipped']]
    ev = sorted([d for d in ch if d['slipped']], key=lambda d: d['r'])
    if s == '4x1': ev = sorted([d for d in CR.get(s, []) if d['slipped']], key=lambda d: d['r']) or ev   # 4x1: gamma* from the long fixed-u runs (C_rows)
    if s == '24x6' and not ev: ev = sorted([d for d in runs if d['slipped'] and (d['gamma_star'] is not None)], key=lambda d: d['r'])
    lo = max([d['r'] for d in ch if not d['slipped'] and (not ev or d['r'] < ev[0]['r'])], default=None)
    us[s] = dict(Lx=19*int(s.split('x')[0]), u_s_upper=ev[0]['r'] if ev else None, u_s_lower=lo, gamma_star=ev[0]['gamma_star_mean5'] if ev else None,
                 gamma_star_last=ev[0]['gamma_star'] if ev else None, permanent=ev[0]['permanent'] if ev else None, source=ev[0]['file'] if ev else None)
res['u_s'] = us
for i, (s, v) in enumerate(us.items()):
    if v['u_s_upper']: ax[2].errorbar(1/v['Lx'], (v['u_s_upper']+(v['u_s_lower'] or v['u_s_upper']))/2, yerr=(v['u_s_upper']-(v['u_s_lower'] or v['u_s_upper']))/2, fmt='s', color=C[i], label=s)
if 'gamma_s0' in S:
    # bulk prediction: universal curve reaches gamma_s(0)
    allok = [(d['r'], d['gamma_sat']) for runs in B.values() for d in runs if d['gamma_sat'] and not d['slipped'] and d['r'] <= 1.081]
    x, g = np.array(allok).T; p = np.polyfit(x-1, g**2, 2); rr = np.linspace(0, 0.1, 1001); r_pred = 1+rr[np.argmin(abs(np.polyval(p, rr)-S['gamma_s0']**2))]
    ax[2].axhline(r_pred, color='#52514e', ls='--', lw=1); ax[2].text(0.0002, r_pred+0.002, r'universal curve reaches $\gamma_s(0)$', fontsize=8); res['u_s_pred_bulk'] = float(r_pred)
ax[2].set_xlabel(r'$1/L_x$'); ax[2].set_ylabel(r'$u_s/u_c$ (quasi-static)'); ax[2].set_xlim(0, None); ax[2].legend(fontsize=7)
fig.savefig(f'{O}/fig_B_universal.png')
fig, ax = plt.subplots(figsize=(6, 4.2))
for i, (s, v) in enumerate(us.items()):
    if v['gamma_star']: ax.plot(1/v['Lx'], v['gamma_star'], 's' if v['permanent'] else 'x', color=C[i], ms=8, label=s)
if 'gamma_s0' in S: ax.axhline(S['gamma_s0'], color='#52514e', ls='--', lw=1, label=r'$\gamma_s(0)$, task A')
ax.axhline(1/(2*np.sqrt(3)), color=C[5], ls=':', lw=1, label='Frenkel b/4d')
ax.set_xlabel(r'$1/L_x$'); ax.set_ylabel(r'$\gamma^*$ (mean of 5 records before the slip)'); ax.set_xlim(0, None); ax.legend(fontsize=8)
fig.savefig(f'{O}/fig_B_gammastar.png')
# C: rows
Cr = {}
for ny in [1, 2, 3, 4]:
    runs = load(f'{O}/C_rows/4x{ny}/sum_*.json')
    if runs: Cr[ny] = runs
res['C'] = {ny: [{k: d[k] for k in ('r', 'slipped', 't_slip', 'gamma_star', 'gamma_star_mean5', 'gamma_star_max5', 'gamma_sat', 'permanent', 'n_events', 'intact_end', 'm_end', 'ndef_end', 'fdef_max')} for d in v] for ny, v in Cr.items()}
fig, ax = plt.subplots(figsize=(6, 4.2))
for i, r_ in enumerate(['1.06', '1.1', '1.15']):
    pts = [(2*ny, d['gamma_star_mean5']) for ny, v in Cr.items() for d in v if abs(d['r']-float(r_)) < 1e-3 and d['slipped']]
    sat = [(2*ny, d['gamma_sat']) for ny, v in Cr.items() for d in v if abs(d['r']-float(r_)) < 1e-3 and not d['slipped']]
    if pts: ax.plot(*zip(*sorted(pts)), 'o-', color=C[i], label=f'{r_} u_c: gamma*')
    if sat: ax.plot(*zip(*sorted(sat)), 'o', mfc='none', color=C[i], label=f'{r_} u_c: intact, gamma_sat')
if 'gamma_s0' in S: ax.axhline(S['gamma_s0'], color='#52514e', ls='--', lw=1, label='bulk gamma_s(0)')
ax.set_xlabel('rows of skyrmions along y (2 n_y)'); ax.set_ylabel(r'$\gamma^*$'); ax.legend(fontsize=7); fig.savefig(f'{O}/fig_C_rows.png')
# D: direction
Dd = {}
for s in ['8x1', '8x2']:
    runs = load(f'{O}/D_dir/{s}/sum_*.json')
    if runs: Dd[s] = [dict(tag=os.path.basename(d['file'])[4:-5], slipped=d['slipped'], permanent=d['permanent'], m_end=d['m_end'], ndef_end=d['ndef_end'], gamma_star=d['gamma_star'], usign=d['usign'], bsign=d['bsign']) for d in runs]
res['D'] = Dd
json.dump(res, open(f'{O}/dyn_summary.json', 'w'), indent=1)
for s, v in res['B'].items():
    print(s, [(d['r'], 'seed' if 'seed' in d['file'] else 'noise' if 'noise' in d['file'] else '', round(d['gamma_sat'], 3) if d['gamma_sat'] else None, ('EV g*=%.3f %s' % (d['gamma_star'] or np.nan, 'perm' if d['permanent'] else 'trans')) if d['slipped'] else '') for d in v])
print('u_s', us)
for ny, v in res['C'].items(): print('C 4x%d' % ny, [(d['r'], round(d['gamma_sat'], 3) if d['gamma_sat'] else None, ('EV g*=%.3f mean5=%.3f max5=%.3f n_ev=%d %s' % (d['gamma_star'] or np.nan, d['gamma_star_mean5'], d['gamma_star_max5'], d['n_events'], 'perm' if d['permanent'] else 'trans')) if d['slipped'] else 'intact') for d in v])
for s, v in Dd.items(): print('D', s, [(d['tag'], d['m_end'], 'perm' if d['permanent'] else '') for d in v])
