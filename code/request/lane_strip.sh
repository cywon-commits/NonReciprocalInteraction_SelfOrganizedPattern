#!/bin/sh
# C3: helix strip N = 200 x Ly, stair up (T = 0) + slip anatomy at u = 0.075
R=code/request; H="env NUMBA_NUM_THREADS=1 python3 $R/helixrun.py"
UP=$(python3 -c "print(','.join(f'{0.06+0.005*i:.4f}' for i in range(29)))")
for Ly in 8 16 32; do
  $H --Ly $Ly --us $UP --hold 100000 --tag strip$Ly --out results_request/C_helix/C3 > results_request/C_helix/C3_up$Ly.log 2>&1
  $H --Ly $Ly --us 0.0750 --hold 400000 --every 5 --ring 120 --stop_on_slip --tag s75_$Ly --out results_request/C_helix/C3 > results_request/C_helix/C3_fix$Ly.log 2>&1
done
