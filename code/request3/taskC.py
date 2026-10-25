import sys, os, json, glob, time, subprocess, numpy as np
# Task C driver for one damping alpha (beta = 2 alpha), restart safe (every stage is skipped when its output exists):
#  1. Bloch threshold of the 12x3 tiling (bloch_uc3.py) -> u_B; u_c L_x is constant, so the 4x1 Bloch threshold is 3 u_B.
#  2. 4x1 chain from the perfect lattice at u/u_B(4x1) = 0.99 ... 1.08 (run3.py).
#  3. lab threshold u_c(4x1): linear fit of the saturated U_y^2 (k1 mode amplitude) against u over the intact states with
#     U_y above the noise, extrapolated to zero; u_c(12x3) = u_c(4x1)/3.
#  4. 4x1 state at u = 1.06 u_c(4x1) -> scaled 12x3 seed (seed.py) -> chains as in task 0, seeds 1 and 2 in parallel.
# usage: python taskC.py alpha
a = float(sys.argv[1]); O = f'results_request3/C/a{a}'; os.makedirs(O, exist_ok=True)
RS = '1.06,1.065,1.07,1.0725,1.075,1.0775,1.08,1.0825,1.085,1.09,1.095,1.10'
def sh(c): print(time.strftime('%H:%M'), c, flush=True); subprocess.run(c, shell=True, check=True)
fB = f'results_request3/uc/uc_12x3_a{a}.json'
while not os.path.exists(fB) and subprocess.run(f'ps -eo args | grep -v grep | grep -q "bloch_uc3.py 12 3 {a} "', shell=True).returncode == 0: time.sleep(60)
if not os.path.exists(fB): sh(f'python3 code/request3/bloch_uc3.py 12 3 {a} {fB} > {fB[:-5]}.log 2>&1')
uB = json.load(open(fB))['u_c_bloch']; uB4 = 3*uB
rs4 = '0.99,1.0,1.01,1.02,1.03,1.04,1.05,1.06,1.08'
if not os.path.exists(f'{O}/4x1/DONE'):
    sh(f'python3 code/request3/run3.py --nx 4 --ny 1 --uc {uB4} --rs {rs4} --hold 2e5 --hold_max 4e5 --alpha {a} --start perfect --out {O}/4x1 > {O}/log_4x1.txt 2>&1')
    open(f'{O}/4x1/DONE', 'w').close()
fu = f'{O}/uc.json'
if not os.path.exists(fu):
    pts = []
    for f in sorted(glob.glob(f'{O}/4x1/sum_*.json')):
        d = json.load(open(f))
        if not d['event']: pts.append((d['u'], d['sat']['Uy'], d['sat']['gmax'], d['equilibrated']))
    P = np.array(pts); ok = (P[:, 1] > 0.2) & (P[:, 3] > 0)                       # nonzero wave (noise level ~ 1e-5)
    c = np.polyfit(P[ok, 0], P[ok, 1]**2, 1); uc4 = -c[1]/c[0]
    json.dump(dict(alpha=a, u_bloch_12x3=uB, u_bloch_4x1=uB4, u_c_4x1=uc4, u_c_12x3=uc4/3, ratio_lab_bloch=uc4/uB4, fit=c.tolist(), points=pts), open(fu, 'w'), indent=1)
uc4 = json.load(open(fu))['u_c_4x1']
if not os.path.exists(f'{O}/4x1_r1.06/sum_r1.0600.json'):
    sh(f'python3 code/request3/run3.py --nx 4 --ny 1 --uc {uc4} --rs 1.06 --hold 3e5 --hold_max 6e5 --alpha {a} --start {O}/4x1/S_r1.0600.npy --out {O}/4x1_r1.06 > {O}/log_4x1_r1.06.txt 2>&1'
       if os.path.exists(f'{O}/4x1/S_r1.0600.npy') else
       f'python3 code/request3/run3.py --nx 4 --ny 1 --uc {uc4} --rs 1.06 --hold 3e5 --hold_max 6e5 --alpha {a} --start perfect --out {O}/4x1_r1.06 > {O}/log_4x1_r1.06.txt 2>&1')
seed = f'{O}/S_seed12x3_r1.06.npy'
if not os.path.exists(seed): sh(f'python3 code/request2/seed.py {O}/4x1_r1.06/S_r1.0600.npy 4 1 12 3 {seed}')
ps = []
for s in (1, 2):
    if not os.path.exists(f'{O}/12x3_s{s}/DONE'):
        c = (f'python3 code/request3/run3.py --nx 12 --ny 3 --uc {uc4/3} --rs {RS} --hold 1.5e5 --hold_max 3e5 --alpha {a} --start {seed} '
             f'--noise_start 1e-3 --seed {s} --out {O}/12x3_s{s} > {O}/log_12x3_s{s}.txt 2>&1 && touch {O}/12x3_s{s}/DONE')
        print(time.strftime('%H:%M'), c, flush=True); ps.append(subprocess.Popen(c, shell=True))
for p in ps: p.wait()
print(time.strftime('%H:%M'), 'task C done', a, flush=True)
