#!/bin/sh
# usage: run_c1.sh SIZE UC R [ALPHA DT]: task C1 Floquet (request-4 floquet.py, unchanged) on one prepared state of request 5,
# waiting for the state file (prep_c.py) to appear. One core.
cd "$(dirname "$0")/../.."
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SZ=$1; UC=$2; r=$3; AL=${4:-0.3}; DT=${5:-0.2}; ST=results_request5/C1/states/$SZ; O=results_request5/C1/$SZ; mkdir -p $O
while [ ! -f $ST/S_r$r.json ]; do sleep 120; done
python3 -c "import json,sys; sys.exit(0 if json.load(open('$ST/S_r$r.json'))['intact'] else 1)" || exit 0
[ -f $O/f_r$r.json ] || python3 code/request4/floquet.py --start $ST/S_r$r.npy --uc $UC --r $r --alpha $AL --dt $DT --K 6 --iters 30 --out $O/f_r$r.json > $O/log_r$r.txt 2>&1
