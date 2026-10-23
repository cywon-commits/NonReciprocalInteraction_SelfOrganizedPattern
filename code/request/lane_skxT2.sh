#!/bin/sh
# Task B2 extra points: healing of the frozen 8x2 defect state at u = 0, intermediate T and a second seed.
R=code/request; O=results_request/B_temp/skx_8x2; C=results_request/A_size/chain_8x2/S_dn_0.00000.npy
for ts in 0.07:1 0.085:1 0.085:2 0.1:2 0.13:2; do T=${ts%:*}; s=${ts#*:}
  env NUMBA_NUM_THREADS=1 python3 $R/skxrun.py --nx 8 --ny 2 --us 0.0 --holds 300000 --T $T --seed $s --start $C --nostop --every 100 --tag T${T}_s$s --out $O >> $O/skxT2.log 2>&1
done
