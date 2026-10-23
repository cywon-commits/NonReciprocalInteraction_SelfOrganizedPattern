#!/bin/sh
# Task A fixed-u tests for the large sizes, one (size, u, start) per invocation so they run in parallel lanes.
# usage: lane_bist_big.sh nx ny uc r start(perf|def) hold
R=code/request; nx=$1; ny=$2; uc=$3; r=$4; st=$5; hold=$6; O=results_request/A_size/bist_${nx}x${ny}; mkdir -p $O
u=$(python3 -c "print(f'{$r*$uc:.5f}')")
P="env NUMBA_NUM_THREADS=1 python3 $R/skxrun.py --nx $nx --ny $ny --us $u --holds $hold --tmin 100000 --decay 1e-8 --out $O"
if [ $st = perf ]; then $P --tag perf --noise0 1e-2 >> $O/bist_${st}_$r.log 2>&1
else $P --tag def --start results_request/A_size/chain_${nx}x${ny}/S_upend.npy >> $O/bist_${st}_$r.log 2>&1; fi
