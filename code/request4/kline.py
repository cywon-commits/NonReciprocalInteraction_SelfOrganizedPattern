import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here); sys.path.insert(1, os.path.join(here, '..', 'request2'))
from twist import TwistModel, link_blocks
from sparse_bloch import soft_eigs
# Soft branch along a line through k = 0: lambda_max(k) for k = t * (direction) on a relaxed state of scanA.py.
# usage: python kline.py Lx s [angle_deg] -> prints |k|, lambda_max (the k -> 0 end shows whether the instability is long-wave)
J, D = 1.0, 0.4; B = 0.5*D*D/J; Lx, s = int(sys.argv[1]), int(sys.argv[2]); ang = float(sys.argv[3]) if len(sys.argv) > 3 else np.degrees(np.arctan2(0.071, -0.043))
m = TwistModel(Lx, 33, s, J, D, B); S = np.load(f'results_request4/A/state_u0.000_nx4_Lx{Lx}_s{s}.npy'); z = np.zeros(2)
bl, _ = link_blocks(m, S, z, z, 0.3); e = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]); out = []
for t in [0.004, 0.008, 0.012, 0.016, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11]:
    ev = soft_eigs(m, bl, t*e, nev=8, sigma=1e-5); out.append((t, float(ev.real.max())))
    print(f'Lx={Lx} s={s} gamma={s/76:.4f} ang={ang:.0f} |k|={t:.3f} lam_max={ev.real.max():+.3e}', flush=True)
json.dump(dict(Lx=Lx, s=s, angle=ang, line=out), open(f'results_request4/A/kline_Lx{Lx}_s{s}_a{int(round(ang))}.json', 'w'))
