#!/bin/sh
# usage: growth_lane.sh nx ny uc_est hold   -> fixed-u runs from the perfect lattice (+1e-3 noise) at 0.97..1.03 uc_est
nx=$1; ny=$2; uc=$3; hold=$4; R=$(dirname $0)
for r in 0.97 0.99 1.01 1.03; do
  u=$(python3 -c "print(f'{$r*$uc:.5f}')")
  NUMBA_NUM_THREADS=${NT:-1} python3 $R/skxrun.py --nx $nx --ny $ny --us $u --holds $hold --tag g${nx}x${ny} --nostop --out growth
done
