"""Fig. 8: below threshold the current puts the lattice phonons in a Gibbs ensemble moving at lambda u_eff: the two branches carry
T/(1 -+ x); heating sits in the longest modes along the current, so the melting point moves only by the zone average <1/(1-x^2)> - 1.
Data: results/teff/fitamp.json (76x33, k1 = 2pi/76; x = u/u*, u* = 0.098 as in code/teff/fig_teff.py),
results/kt/t1b_final.txt (152x66, u_eff = 0.03), results_request6/B_summary.json (heating ladders),
results/b1_lambda_alpha/lam_alpha_B0.5_nk20.json (lambda = kappa_P, c2)."""
import json, os, re, numpy as np, matplotlib.pyplot as plt
from b1style import *
fig = plt.figure(figsize=(W1, 3.0))
gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.5)
a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, :])

# (a) branch temperatures at k1
amp = json.load(open(os.path.join(ROOT, 'results', 'teff', 'fitamp.json'))); ust = 0.098
x = np.linspace(0, 0.8, 100)
a.plot(x, 1/(1 - x), color=VE, label=r'$1/(1-x)$'); a.plot(x, 1/(1 + x), color=BL, ls='--', label=r'$1/(1+x)$')
c0, e0 = np.array(amp['0.0'][0]), np.array(amp['0.0'][1])
for i, col, m in ((0, VE, 'o'), (1, BL, 's')):
    xs, rs, es = [], [], []
    for u in ('0.0', '0.05', '0.07'):
        cc, ee = np.array(amp[u][0]), np.array(amp[u][1]); r = cc[i]/c0[i]
        xs.append(float(u)/ust); rs.append(r); es.append(r*np.hypot(ee[i]/cc[i], e0[i]/c0[i]))
    pts(a, xs, rs, yerr=es, c=col, m=m)
a.set_xlabel(r'$x=\lambda u_{\rm eff}k/\omega_k$'); a.set_ylabel(r'$T_{\rm eff}/T$'); a.set_xlim(-0.03, 0.8); a.set_ylim(0, 5)
a.legend(loc='upper left', fontsize=5.5)

# (b) k-resolved variance ratio vs 1/(1 - x^2)
rows = []
for l in open(os.path.join(ROOT, 'results', 'kt', 't1b_final.txt')):
    m = re.match(r'\s*([+-][\d.]+)\s+([+-][\d.]+)\s+\|\s+([\d.]+) ± ([\d.]+)\s+\|\s+([\d.]+)', l)
    if m: rows.append([float(v) for v in m.groups()])
R = np.array(rows); along = np.abs(R[:, 1]) < 1e-9
b.plot([0.9, 1.8], [0.9, 1.8], color='0.6', lw=0.6, zorder=0)
pts(b, R[along, 4], R[along, 2], yerr=R[along, 3], c=VE, m='o', label=r'$\mathbf{k}\parallel\mathbf{u}$')
pts(b, R[~along, 4], R[~along, 2], yerr=R[~along, 3], c=BL, m='s', label='transverse, oblique')
b.set_xlabel(r'predicted $1/(1-x^2)$'); b.set_ylabel('variance ratio')
b.set_xlim(0.9, 1.8); b.set_ylim(0.85, 1.9); b.legend(loc='upper left', fontsize=5.3)

# (c) melting point: zone average vs long-wave criterion
d0 = json.load(open(os.path.join(ROOT, 'results', 'b1_lambda_alpha', 'lam_alpha_B0.5_nk20.json')))
lam = abs(d0['KP'][0])/(d0['P']*d0['k']); c2 = d0['wn']/d0['k']**2
Gs = [(h, g) for h in range(-3, 4) for g in range(-3, 4) if (h + g) % 2 == 0]
def zone(Lx, Ly, u):          # <1/(1-x^2)> - 1 over the primitive Brillouin zone of the centred 19x33 two-skyrmion cell
    ks = []
    for n in range(2*(Lx//19)):
        for m in range(Ly//33):
            k = np.array([2*np.pi*n/Lx, 2*np.pi*m/Ly])
            ks.append(min((k - 2*np.pi*np.array([h/19, g/33]) for h, g in Gs), key=lambda v: v @ v))
    ks = np.array(ks); k2 = (ks**2).sum(1); ok = k2 > 1e-12; xx = lam*u*ks[ok, 0]/(c2*k2[ok])
    return np.mean(1/(1 - xx**2)) - 1
HS = json.load(open(os.path.join(ROOT, 'results_request6', 'B_summary.json')))['heating_seeds']
uu = np.linspace(0, 0.031, 32)
for j, (Lx, Ly, col, m) in enumerate(((152, 528, BL, 'o'), (228, 363, VE, 's'))):
    hs = sorted([h for h in HS if h['Lx'] == Lx], key=lambda h: h['ue'])
    c.plot(uu, [1/(1 + zone(Lx, Ly, u)) for u in uu], color=col, label=f'zone average, $L_x={Lx}$')
    c.plot([h['ue'] for h in hs], [h['pred']['four'] for h in hs], color=col, ls=':', lw=0.9, label=f'long-wave, $L_x={Lx}$')
    pts(c, np.array([h['ue'] for h in hs]) + (j - 0.5)*0.0007, [h['ratio'] for h in hs], yerr=[h['ratio_err'] for h in hs], c=col, m=m)
c.set_xlabel(r'$u_{\rm eff}$'); c.set_ylabel(r'$T_m(u)/T_m(0)$'); c.set_xlim(-0.001, 0.0335); c.set_ylim(0.4, 1.08); c.set_yticks([0.6, 0.8, 1.0])
c.legend(loc='lower left', fontsize=5.8, ncol=2, columnspacing=0.8)
label(a, 'a', x=-0.33, y=1.0); label(b, 'b', x=-0.36, y=1.0); label(c, 'c', x=-0.1, y=1.0)
check(fig, 'fig8_gibbs')
print('zone shift at u=0.03:', round(zone(152, 528, 0.03), 4), round(zone(228, 363, 0.03), 4))
