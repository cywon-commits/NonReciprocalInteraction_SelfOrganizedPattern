#!/bin/sh
# Restart-safe launcher for request 6 (rerun after a worker restart). Jobs resume from their checkpoints / skip finished outputs.
cd "$(dirname "$0")/../.."; O=results_request6
running() { ps -eo args | grep -v grep | grep -q -- "$1"; }
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# task A: one single-thread queue per u_eff (sizes small to large)
for ue in 0.15 0.2 0.3; do
  running "hexA_queue_$ue" || setsid nohup sh -c "for sz in '10 5' '20 10' '30 15'; do set -- \$sz; [ -f results_request6/A/last_u${ue}_\$1x\$2.npy ] || python3 code/request6/hexA.py $ue \$1 \$2 40000 results_request6/A >> results_request6/A/log_u${ue}_\$1x\$2.txt 2>&1; done # hexA_queue_$ue" > /dev/null 2>&1 < /dev/null &
done
running "request6/autocommit.sh" || setsid nohup sh code/request6/autocommit.sh > /dev/null 2>&1 < /dev/null &
# task D: delta D scans (one single-thread queue)
running "dD_queue" || setsid nohup sh -c 'for spec in "19 local 0,1,0" "19 full 0,1,0" "19 stiff 0,1,0" "19 full 1,0,0" "19 full 0,0,1" "76 full 0,1,0" "76 local 0,1,0"; do set -- $spec; f=results_request6/D/dD_$2_Lx$1_n$(echo $3 | tr , _).json; [ -f $f.done ] || { python3 code/request6/dD.py $1 $2 $3 0.0,0.0005,0.001,0.002,0.004,0.008 >> results_request6/D/log_dD.txt 2>&1 && touch $f.done; }; done # dD_queue' > /dev/null 2>&1 < /dev/null &
