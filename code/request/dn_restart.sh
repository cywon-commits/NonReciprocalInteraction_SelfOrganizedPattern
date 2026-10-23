#!/bin/sh
# Remaining down-chain steps for 12x3 and 16x4 with a fixed hold of 1e5 (the Gamma-based hold is meaningless for a frozen defect state).
# usage: dn_restart.sh nx ny uc nthreads r1 r2 ...   (continues from S_dn_last.npy)
R=code/request; nx=$1; ny=$2; uc=$3; nt=$4; shift 4; O=results_request/A_size/chain_${nx}x${ny}
DN=$(python3 -c "print(','.join(f'{r*$uc:.5f}' for r in [$(echo $@ | tr ' ' ',')]))")
env NUMBA_NUM_THREADS=$nt python3 $R/skxrun.py --nx $nx --ny $ny --gamma $O/gamma.json --us $DN --holds 100000 --tmin 50000 --decay 1e-8 --tag dn --start $O/S_dn_last.npy --out $O >> $O/dn.log 2>&1
echo "exit $? $(date)" >> $O/dn_exit.log
