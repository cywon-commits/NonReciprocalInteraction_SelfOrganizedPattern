#!/bin/sh
# Tasks B and C (1D helix and strip); run from repo root. Each line is independent and single-threaded.
R=code/request; H="env NUMBA_NUM_THREADS=1 python3 $R/helixrun.py"
UP=$(python3 -c "print(','.join(f'{0.06+0.005*i:.4f}' for i in range(29)))")
DN=$(python3 -c "print(','.join(f'{0.195-0.005*i:.4f}' for i in range(40)))")
mkdir -p results_request/C_helix results_request/B_temp
# C1: slip anatomy at fixed u (T = 0), n = 3 and n = 2 starts
for u in 0.0715 0.0725 0.0750 0.0800 0.0850 0.0900; do $H --n0 3 --us $u --hold 400000 --every 2 --ring 80 --stop_on_slip --tag c1n3 --out results_request/C_helix/C1; done > results_request/C_helix/C1.log 2>&1
for u in 0.1100 0.1125 0.1150 0.1200 0.1300; do $H --n0 2 --us $u --hold 400000 --every 2 --ring 80 --stop_on_slip --tag c1n2 --out results_request/C_helix/C1; done >> results_request/C_helix/C1.log 2>&1
# C2: longer rings, stair up / down (T = 0)
UPF=$(python3 -c "print(','.join(f'{0.06+0.0025*i:.4f}' for i in range(57)))")
DNF=$(python3 -c "print(','.join(f'{0.1975-0.0025*i:.4f}' for i in range(80)))")
for NN in 200:3 600:9 1200:18; do N=${NN%:*}; n=${NN#*:}
  $H --N $N --n0 $n --us $UPF --hold 100000 --tag up$N --out results_request/C_helix/C2 > results_request/C_helix/C2_up$N.log 2>&1
  $H --N $N --n0 $n --us $DNF --hold 100000 --tag dn$N --start results_request/C_helix/C2/S_up${N}_last.npy --out results_request/C_helix/C2 > results_request/C_helix/C2_dn$N.log 2>&1
done
