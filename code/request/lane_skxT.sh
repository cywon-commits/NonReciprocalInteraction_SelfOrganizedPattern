#!/bin/sh
# Task B2: does the frozen defect state of a hysteretic SkX size heal at finite T (u = 0)? 8x2, start = down-chain end (u = 0).
R=code/request; O=results_request/B_temp/skx_8x2; mkdir -p $O
C=results_request/A_size/chain_8x2/S_dn_0.00000.npy
while [ ! -f $C ]; do sleep 120; done
for T in 0.05 0.1 0.13 0.16 0.2; do
  env NUMBA_NUM_THREADS=1 python3 $R/skxrun.py --nx 8 --ny 2 --us 0.0 --holds 300000 --T $T --seed 1 --start $C --nostop --every 100 --tag T$T --out $O >> $O/skxT.log 2>&1
done
