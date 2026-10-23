import re, sys, json, numpy as np
# u_c from a bloch_uc log: zero crossing of max growth, linear interpolation between the closest bracketing points.
out = {}
for log in sys.argv[1:]:
    pts = sorted({(float(m.group(1)), float(m.group(2))) for m in re.finditer(r"u_eff=([\d.]+) maxRe=([-+\deE.]+)", open(log).read())})
    d = dict(pts); u = np.array(sorted(d)); g = np.array([d[x] for x in u])
    neg = u[g < 0].max(); pos = u[g > 0].min(); gn = d[neg]; gp = d[pos]
    uc = neg+(pos-neg)*(-gn)/(gp-gn); size = re.search(r"uc_(\d+x\d+)", log).group(1)
    out[size] = dict(u_c_bloch=uc, bracket=[neg, pos], points=pts); print(size, f"u_c(Bloch)={uc:.5f}")
json.dump(out, open('results_request/A_size/uc_bloch.json', 'w'), indent=1)
