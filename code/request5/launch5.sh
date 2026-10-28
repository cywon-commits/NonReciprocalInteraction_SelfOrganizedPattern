#!/bin/sh
# Restart-safe launcher for request 5 (run again after a worker restart). Every job skips finished outputs.
cd "$(dirname "$0")/../.."; O=results_request5
running() { ps -eo args | grep -v grep | grep -q -- "$1"; }
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for spec in "152 42,43,44,45,46,47,48,49,50,51,52" "144 38,39,40,41,42,43,44,45,46,47,48" "140 36,37,38,39,40,41,42,43,44,45,46" "136 34,35,36,37,38,39,40,41,42,43,44"; do
  set -- $spec
  running "dispA.py --Lx $1 --s" || setsid nohup python3 code/request5/dispA.py --Lx $1 --s $2 --out $O/A >> $O/A/log_Lx$1.txt 2>&1 < /dev/null &
done
[ -f $O/C1/states/24x6/S_r1.0720.json ] || running "states/24x6" || setsid nohup sh -c "python3 code/request4/prep_c.py --start results_request3/T0/24x6_s1/S_r1.0650.npy --uc 0.016601 --rs 1.0700 --hold 3e5 --out $O/C1/states/24x6 && python3 code/request4/prep_c.py --start $O/C1/states/24x6/S_r1.0700.npy --uc 0.016601 --rs 1.0710,1.0720 --hold 5e4 --out $O/C1/states/24x6" >> $O/C1/states/log_24x6.txt 2>&1 < /dev/null &
