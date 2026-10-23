#!/bin/sh
# 16x4 chain continued from the saved 0.8 u_c state (first step done), decayed threshold 1e-8, tmax 3e5.
R=code/request; O=results_request/A_size/chain_16x4; uc=0.0249
UP=$(python3 -c "print(','.join(f'{r*$uc:.5f}' for r in [0.95,1.0,1.03,1.06,1.1,1.15,1.2,1.3,1.4,1.5,1.65,1.8,2.0]))")
DN=$(python3 -c "print(','.join(f'{r*$uc:.5f}' for r in [1.8,1.65,1.5,1.4,1.3,1.2,1.15,1.1,1.06,1.03,1.0,0.95,0.9,0.8,0.6,0.4,0.2,0.0]))")
P="env NUMBA_NUM_THREADS=2 python3 $R/skxrun.py --nx 16 --ny 4 --gamma $O/gamma.json --tmin 50000 --tmax 300000 --decay 1e-8 --out $O"
$P --us $UP --tag up --start $O/S_up_last.npy >> $O/up.log 2>&1
cp $O/S_up_last.npy $O/S_upend.npy
$P --us $DN --tag dn --start $O/S_upend.npy > $O/dn.log 2>&1
