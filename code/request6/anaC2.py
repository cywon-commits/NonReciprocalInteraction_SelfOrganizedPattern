import json, glob, os, numpy as np
# Request 6, task C2 summary of the end states (stripeC2.py): averages over the last quarter of each run.
# Classification: 'straight' (A_b < 0.05: instability not grown within the run), 'zigzag' (saturated bending, no dislocations,
# no tilt), 'reoriented' (|tilt| > 0.05 with or without dislocations), 'dislocations' (n_disloc > 0 at the end).
# Tilt -> rotation of q: q' = q - (s/2) e, angle atan(s/2) toward the stripe direction e (towards the current for phi = 90).
out = []
for f in sorted(glob.glob('results_request6/C/C2_*.json')):
    d = json.load(open(f)); R = np.array(d['rec']); h = R[3*len(R)//4:]; A = h[:, 1].mean(); tilt = h[:, 3].mean(); nd = h[:, 4].mean()
    cls = 'straight' if A < 0.05 else ('dislocations' if nd > 0.5 else ('reoriented' if abs(tilt) > 0.05 else 'zigzag'))
    r = dict(name=os.path.basename(f)[3:-5], P=d['P'], phi=d['phi'], L=d['L'], fac=d['fac'], u=d['u'], t_end=float(R[-1, 0]), A_b=float(A), mode=int(np.bincount(h[:, 2].astype(int)).argmax()),
             tilt=float(tilt), angle_deg=float(np.degrees(np.arctan(abs(tilt)/2))), n_disloc=float(nd), n_disloc_max=float(R[:, 4].max()), mz=float(h[:, 5].mean()), cls=cls)
    out.append(r)
    print(f"{r['name']:28s} t={r['t_end']:.0f} A_b={A:6.3f} mode={r['mode']} q-rotation={r['angle_deg']:4.1f} deg disloc={nd:6.1f} (max {r['n_disloc_max']:.0f}) <mz>={r['mz']:+.3f} -> {cls}")
json.dump(out, open('results_request6/C2_summary.json', 'w'), indent=1)
