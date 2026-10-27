#!/bin/sh
# usage: run_c.sh SIZE UC "r1 r2 ..." [ALPHA DT]: C1 (floquet.py) and C2 (c2.py) on the prepared states of a size, in order,
# waiting for each state file (prep_c.py) to appear. One core.
cd "$(dirname "$0")/../.."
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SZ=$1; UC=$2; RS=$3; AL=${4:-0.3}; DT=${5:-0.2}; ST=results_request4/C/states/$SZ; mkdir -p results_request4/C/C1/$SZ results_request4/C/C2/$SZ
for r in $RS; do
  while [ ! -f $ST/S_r$r.json ]; do sleep 120; done
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$ST/S_r$r.json'))['intact'] else 1)" || continue
  [ -f results_request4/C/C1/$SZ/f_r$r.json ] || python3 code/request4/floquet.py --start $ST/S_r$r.npy --uc $UC --r $r --alpha $AL --dt $DT --K 6 --iters 30 --out results_request4/C/C1/$SZ/f_r$r.json > results_request4/C/C1/$SZ/log_r$r.txt 2>&1
  [ -f results_request4/C/C2/$SZ/h_r$r.json ] || python3 code/request4/c2.py --state $ST/S_r$r.npy --uc $UC --r $r --alpha $AL --dt $DT --out results_request4/C/C2/$SZ/h_r$r.json > results_request4/C/C2/$SZ/log_r$r.txt 2>&1
done
