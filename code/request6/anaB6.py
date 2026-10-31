import json, glob, numpy as np
# Request 6, task B summary. T_m per leg: temperature where Psi6 crosses 0.4 (heating: downward crossing; cooling: upward
# crossing on the way down), linear interpolation; also where the defect fraction crosses 0.2. T_m = mean of the two legs,
# hysteresis = heating - cooling. Long-wave prediction (request section B): x(k) = lambda u_eff k_x/omega_k with the T = 0
# phonon of the 19 x 33 cell (omega = c2 k^2, c2 = omega_n/k^2 at k = 0.01653: 0.01018/0.01653 = 0.616, lambda = 0.502 at
# alpha = 0.3), over the box modes k = 2 pi (i/Lx, j/Ly); T_m(u)/T_m(0) = 1 - <x^2> over (a) the single longest k_x mode,
# (b) the four longest modes (1,0), (0,1), (1,+-1), (c) all modes with |k| <= 2 k_min.
c2, lam = 0.01018/0.01653, 0.502
def cross(T, y, level, down):
    for i in range(len(T)-1):
        a, b = y[i]-level, y[i+1]-level
        if (down and a >= 0 > b) or (not down and a < 0 <= b): return float(T[i]+(T[i+1]-T[i])*a/(a-b))
    return float('nan')
def pred(u, Lx, Ly):
    ks = [(2*np.pi*i/Lx, 2*np.pi*j/Ly) for i in range(-4, 5) for j in range(-4, 5) if (i, j) != (0, 0)]
    x = lambda k: lam*u*abs(k[0])/(c2*(k[0]**2+k[1]**2))
    kmin = min(np.hypot(*k) for k in ks)
    a = x((2*np.pi/Lx, 0.)); b = np.mean([x(k)**2 for k in [(2*np.pi/Lx, 0.), (0., 2*np.pi/Ly), (2*np.pi/Lx, 2*np.pi/Ly), (2*np.pi/Lx, -2*np.pi/Ly)]])
    c = np.mean([x(k)**2 for k in ks if np.hypot(*k) <= 2*kmin*1.0001])
    return dict(x_max=a, one=1-a**2, four=1-b, near=1-c)
out = []
for f in sorted(glob.glob('results_request6/B/melt_*.json')):
    d = json.load(open(f)); st = d['steps']; H = [s for s in st if s['leg'] == 'heat']; C = [s for s in st if s['leg'] == 'cool']
    Th = np.array([s['T'] for s in H]); Tc = np.array([s['T'] for s in C])[::-1]
    r = dict(ue=d['ue'], Lx=d['Lx'], Ly=d['Ly'], N=float(np.mean([s['N'] for s in st])),
             Th_psi=cross(Th, [s['Psi6'] for s in H], 0.4, True), Tc_psi=cross(Tc, [s['Psi6'] for s in C][::-1], 0.4, True),
             Th_def=cross(Th, [s['f_def'] for s in H], 0.2, False), Tc_def=cross(Tc, [s['f_def'] for s in C][::-1], 0.2, False))
    r['Tm'] = 0.5*(r['Th_psi']+r['Tc_psi']); r['hyst'] = r['Th_psi']-r['Tc_psi']; r['pred'] = pred(d['ue'], d['Lx'], d['Ly']); out.append(r)
for r in out:
    r0 = [q for q in out if q['Lx'] == r['Lx'] and q['ue'] == 0.0][0]
    r['ratio_heat'] = r['Th_psi']/r0['Th_psi']; r['ratio_cool'] = r['Tc_psi']/r0['Tc_psi']; r['ratio_mid'] = r['Tm']/r0['Tm']
    print(f"Lx={r['Lx']} u={r['ue']:.2f} N={r['N']:.0f}: T_heat={r['Th_psi']:.4f} T_cool={r['Tc_psi']:.4f} (defects {r['Th_def']:.4f}/{r['Tc_def']:.4f}) T_m={r['Tm']:.4f} hyst={r['hyst']:.3f} | "
          f"ratio heat {r['ratio_heat']:.3f} cool {r['ratio_cool']:.3f} mid {r['ratio_mid']:.3f} | pred x_max={r['pred']['x_max']:.2f}: 1-x^2 {r['pred']['one']:.3f}, 4 modes {r['pred']['four']:.3f}, |k|<=2kmin {r['pred']['near']:.3f}")
json.dump(out, open('results_request6/B_summary.json', 'w'), indent=1)
