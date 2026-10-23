import re, sys, json
# Bloch growth table [u_eff, Gamma] from a bloch_uc log, u rescaled by the measured lab-frame/Bloch threshold ratio.
log, out, f = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
pts = sorted({(round(float(m.group(1))*f, 7), float(m.group(2))) for m in re.finditer(r"u_eff=([\d.]+) maxRe=([-+\deE.]+)", open(log).read())})
json.dump(pts, open(out, 'w'))
