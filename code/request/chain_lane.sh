#!/bin/sh
# Task A up/down chains for one size. usage (from repo root): chain_lane.sh nx ny uc_dyn tmin tmax
# u grid in units of the lab-frame threshold uc_dyn; holds = clip(10/|Gamma_Bloch(u)|, tmin, tmax), Bloch u rescaled by 0.98.
nx=$1; ny=$2; uc=$3; tmin=$4; tmax=$5; R=code/request; O=results_request/A_size/chain_${nx}x${ny}
mkdir -p $O; python3 $R/mkgamma.py results_request/A_size/uc_${nx}x${ny}.log $O/gamma.json 0.98
UP=$(python3 -c "print(','.join(f'{r*$uc:.5f}' for r in [0.8,0.9,0.95,1.0,1.03,1.06,1.1,1.15,1.2,1.3,1.4,1.5,1.65,1.8,2.0]))")
DN=$(python3 -c "print(','.join(f'{r*$uc:.5f}' for r in [1.8,1.65,1.5,1.4,1.3,1.2,1.15,1.1,1.06,1.03,1.0,0.95,0.9,0.8,0.6,0.4,0.2,0.0]))")
P="env NUMBA_NUM_THREADS=${NT:-1} python3 $R/skxrun.py --nx $nx --ny $ny --gamma $O/gamma.json --tmin $tmin --tmax $tmax --out $O"
$P --us $UP --tag up > $O/up.log 2>&1
cp $O/S_up_last.npy $O/S_upend.npy
$P --us $DN --tag dn --start $O/S_upend.npy > $O/dn.log 2>&1
