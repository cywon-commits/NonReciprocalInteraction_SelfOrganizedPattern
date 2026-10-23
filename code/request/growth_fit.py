import sys, glob, json, re, os, numpy as np
# lab-frame growth rate Gamma(u) from fixed-u runs (fit of ln A over the second half), linear fit -> u_c(dynamic)
fn = 'results_request/A_size/uc_dyn.json'; out = json.load(open(fn)) if os.path.exists(fn) else {}
for size in sys.argv[1:]:
    pts = []
    for f in sorted(glob.glob(f'results_request/A_size/growth/rec_g{size}_*.npy')):
        u = float(re.search(r'_([\d.]+)\.npy', f).group(1)); r = np.load(f); t, A = r[:, 0], r[:, 1]
        m = t > t[-1]/2; pts.append([u, float(np.polyfit(t[m], np.log(A[m]), 1)[0])])
    pts = np.array(pts); p = np.polyfit(pts[:, 0], pts[:, 1], 1); uc = -p[1]/p[0]
    out[size] = dict(points=pts.tolist(), u_c_dyn=uc, slope=p[0]); print(size, f"u_c(dyn)={uc:.5f}", np.round(pts, 8).tolist())
json.dump(out, open(fn, 'w'), indent=1)
