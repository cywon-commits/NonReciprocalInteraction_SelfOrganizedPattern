#!/bin/sh
# C1 and C2 on the 8x2 clean states (sequential, one core each)
cd "$(dirname "$0")/../.."
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for r in 1.0775 1.0800 1.0825 1.0850; do
  [ -f results_request4/C/C1/8x2/f_r$r.json ] || python3 code/request4/floquet.py --start results_request4/C/states/8x2/S_r$r.npy --uc 0.0498 --r $r --K 6 --iters 30 --out results_request4/C/C1/8x2/f_r$r.json > results_request4/C/C1/8x2/log_r$r.txt 2>&1
done
