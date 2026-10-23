#!/bin/sh
# Task A fixed-u bistability: at u/u_c in a list, (i) perfect lattice + noise, (ii) strongly deformed start
# (end state of the up chain at 2 u_c). Long holds with steady/decay early stop. Runs after the chain of that size exists.
R=code/request
run() { nx=$1; ny=$2; uc=$3; tperf=$4; tdef=$5; noise=$6; shift 6; O=results_request/A_size/bist_${nx}x${ny}; mkdir -p $O
  C=results_request/A_size/chain_${nx}x${ny}/S_upend.npy
  while [ ! -f $C ]; do sleep 120; done
  for r in "$@"; do u=$(python3 -c "print(f'{$r*$uc:.5f}')")
    P="env NUMBA_NUM_THREADS=1 python3 $R/skxrun.py --nx $nx --ny $ny --us $u --tmin 100000 --decay 1e-8 --out $O"
    $P --tag perf --holds $tperf --noise0 $noise >> $O/bist.log 2>&1
    $P --tag def --holds $tdef --start $C >> $O/bist.log 2>&1
  done; }
run 4 1 0.0994 600000 600000 1e-3 1.03 1.06 1.1
run 8 2 0.0498 800000 600000 1e-3 1.03 1.06 1.1
run 4 2 0.0994 600000 600000 1e-3 1.03 1.06 1.1
