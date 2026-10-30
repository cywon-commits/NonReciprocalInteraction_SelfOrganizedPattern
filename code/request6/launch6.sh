#!/bin/sh
# Restart-safe launcher for request 6 (rerun after a worker restart). Jobs resume from their checkpoints / skip finished outputs.
cd "$(dirname "$0")/../.."; O=results_request6
running() { ps -eo args | grep -v grep | grep -q -- "$1"; }
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# task A: one single-thread queue per u_eff (sizes small to large)
for ue in 0.15 0.2 0.3; do
  running "hexA_queue_$ue" || setsid nohup sh -c "for sz in '10 5' '20 10' '30 15'; do set -- \$sz; [ -f results_request6/A/last_u${ue}_\$1x\$2.npy ] || python3 code/request6/hexA.py $ue \$1 \$2 40000 results_request6/A >> results_request6/A/log_u${ue}_\$1x\$2.txt 2>&1; done # hexA_queue_$ue" > /dev/null 2>&1 < /dev/null &
done
