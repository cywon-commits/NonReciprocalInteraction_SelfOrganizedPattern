#!/bin/sh
cd "$(dirname "$0")/../.."
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for r in 1.0775 1.0800 1.0825 1.0850; do
  [ -f results_request4/C/C2/8x2/h_r$r.json ] || python3 code/request4/c2.py --state results_request4/C/states/8x2/S_r$r.npy --uc 0.0498 --r $r --out results_request4/C/C2/8x2/h_r$r.json > results_request4/C/C2/8x2/log_r$r.txt 2>&1
done
