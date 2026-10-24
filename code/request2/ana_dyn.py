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
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for i, (s, runs) in enumerate(B.items()):
    ok = [d for d in runs if d['gamma_sat'] is not None and not d['slipped'] and 'seed' in d['file'] or (d['gamma_sat'] is not None and not d['slipped'] and s == '4x1')]
    ok = sorted(ok, key=lambda d: d['r'])
    if ok:
        x = np.array([d['r'] for d in ok]); g = np.array([d['gamma_sat'] for d in ok])
        ax[0].plot(x, g**2/(x-1), 'o-', color=C[i], label=s); ax[1].plot(x, g, 'o-', color=C[i], label=s)
    ev = [d for d in runs if d['slipped']]
    for d in ev: ax[1].plot(d['r'], d['gamma_star'] or np.nan, 'x' if not d['permanent'] else 's', color=C[i], ms=8)
ax[0].set_xlabel(r'$u/u_c$'); ax[0].set_ylabel(r'$\gamma_{\max}^2/(u/u_c-1)$'); ax[0].legend(fontsize=8); ax[0].set_title('B: universal curve (intact saturated states)', fontsize=10)
ax[1].set_xlabel(r'$u/u_c$'); ax[1].set_ylabel(r'$\gamma_{\max}$ (o: saturated, x: $\gamma^*$ transient, ■: $\gamma^*$ permanent)'); ax[1].legend(fontsize=8)
S = {}
if os.path.exists(f'{O}/A_S/gamma_s.json'): S['gamma_s0'] = json.load(open(f'{O}/A_S/gamma_s.json'))['0']['gamma_s']
if 'gamma_s0' in S: ax[1].axhline(S['gamma_s0'], color='#52514e', ls='--', lw=1); ax[1].text(1.005, S['gamma_s0']+0.005, r'$\gamma_s(0)$ bulk', fontsize=8)
# u_s/u_c vs 1/L: smallest r with a topological event (any start)
us = {}
for s, runs in B.items():
    ev = sorted(d['r'] for d in runs if d['slipped']); nev = sorted(d['r'] for d in runs if not d['slipped'])
    us[s] = dict(first_event=ev[0] if ev else None, no_event_up_to=max([x for x in nev if not ev or x < ev[0]], default=None))
res['u_s'] = us
for i, (s, v) in enumerate(us.items()):
    Lx = 19*int(s.split('x')[0])
    if v['first_event']: ax[2].plot(1/Lx, v['first_event'], 's', color=C[i], label=f'{s}: first event')
    if v['no_event_up_to']: ax[2].plot(1/Lx, v['no_event_up_to'], 'o', mfc='none', color=C[i])
ax[2].set_xlabel(r'$1/L_x$'); ax[2].set_ylabel(r'$u_s/u_c$ (■ first event, ○ largest intact)'); ax[2].set_xlim(0, None); ax[2].legend(fontsize=7)
fig.savefig(f'{O}/fig_B_universal.png')
# C: rows
Cr = {}
for ny in [1, 2, 3, 4]:
    runs = load(f'{O}/C_rows/4x{ny}/sum_*.json')
    if runs: Cr[ny] = runs
res['C'] = {ny: [{k: d[k] for k in ('r', 'slipped', 't_slip', 'gamma_star', 'gamma_star_mean5', 'gamma_star_max5', 'gamma_sat', 'permanent', 'n_events', 'intact_end', 'm_end', 'ndef_end', 'fdef_max')} for d in v] for ny, v in Cr.items()}
fig, ax = plt.subplots(figsize=(6, 4.2))
for i, r_ in enumerate(['1.06', '1.1', '1.15']):
    pts = [(2*ny, d['gamma_star']) for ny, v in Cr.items() for d in v if abs(d['r']-float(r_)) < 1e-3 and d['slipped']]
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
