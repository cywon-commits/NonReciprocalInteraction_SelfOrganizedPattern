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
running "mpoint_loop.sh" || setsid nohup sh code/request5/mpoint_loop.sh > /dev/null 2>&1 < /dev/null &
running "autocommit.sh" || setsid nohup sh code/request3/autocommit.sh > /dev/null 2>&1 < /dev/null &
running "dispA.py --Lx 152 --s 45,46,47 --alpha 0.15" || setsid nohup python3 code/request5/dispA.py --Lx 152 --s 45,46,47 --alpha 0.15 --grid polar --out $O/A >> $O/A/log_Lx152_a015.txt 2>&1 < /dev/null &
running "dispA.py --Lx 136 --s 37,38,39 --alpha 0.15" || setsid nohup python3 code/request5/dispA.py --Lx 136 --s 37,38,39 --alpha 0.15 --grid polar --out $O/A >> $O/A/log_Lx136_a015.txt 2>&1 < /dev/null &
running "dispA.py --Lx 144 --s 41,42,43 --alpha 0.15" || setsid nohup python3 code/request5/dispA.py --Lx 144 --s 41,42,43 --alpha 0.15 --grid polar --out $O/A >> $O/A/log_Lx144_a015.txt 2>&1 < /dev/null &
for r in 1.0700 1.0710 1.0720; do
  running "run_c1.sh 24x6 0.016601 $r" || setsid nohup sh code/request5/run_c1.sh 24x6 0.016601 $r > /dev/null 2>&1 < /dev/null &
done
