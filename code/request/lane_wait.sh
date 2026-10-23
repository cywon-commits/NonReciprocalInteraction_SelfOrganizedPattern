#!/bin/sh
# Task B: first-passage (waiting) times for re-winding n=1 -> 2 and slipping n=3 -> 2 at fixed u and T, 8 seeds.
R=code/request; H="env NUMBA_NUM_THREADS=1 python3 $R/helixrun.py"; O=results_request/B_temp/wait
mkdir -p $O
for T in 0.004 0.005 0.006 0.008 0.01 0.0125 0.015; do for s in 1 2 3 4 5 6 7 8; do
  for u in 0.0000 0.0300 0.0600; do $H --n0 1 --T $T --seed $s --us $u --hold 1000000 --every 10 --stop_on_slip --norec --tag w1_T${T}_u${u}_s$s --out $O; done
  for u in 0.0400 0.0550 0.0650; do $H --n0 3 --T $T --seed $s --us $u --hold 1000000 --every 10 --stop_on_slip --norec --tag w3_T${T}_u${u}_s$s --out $O; done
done; done > $O/wait.log 2>&1
