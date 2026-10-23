#!/bin/sh
# Task B (1D helix N = 200 at finite T): stair up/down per T and seed, then first-passage times at fixed u.
R=code/request; H="env NUMBA_NUM_THREADS=1 python3 $R/helixrun.py"; O=results_request/B_temp
UP=$(python3 -c "print(','.join(f'{0.06+0.005*i:.4f}' for i in range(29)))")
DN=$(python3 -c "print(','.join(f'{0.195-0.005*i:.4f}' for i in range(40)))")
mkdir -p $O/stairs $O/wait
for T in 0.002 0.005 0.01 0.02; do for s in 1 2 3 4 5; do
  $H --T $T --seed $s --us $UP --hold 100000 --every 50 --tag up_T${T}_s$s --out $O/stairs > $O/stairs/up_T${T}_s$s.log 2>&1
  $H --T $T --seed $s --us $DN --hold 100000 --every 50 --tag dn_T${T}_s$s --start $O/stairs/S_up_T${T}_s${s}_last.npy --out $O/stairs > $O/stairs/dn_T${T}_s$s.log 2>&1
done; done
