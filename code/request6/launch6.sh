#!/bin/sh
# Restart-safe launcher for request 6 (rerun after a worker restart). Jobs resume from their checkpoints / skip finished outputs.
cd "$(dirname "$0")/../.."; O=results_request6
running() { ps -eo args | awk -v pat="$1" 'index($0, pat) && !/awk -v pat/' | head -1 | grep -q .; }
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# task A: one single-thread queue per u_eff (sizes small to large)
for ue in 0.15 0.2 0.3; do
  running "hexA_queue_$ue" || setsid nohup sh -c "for sz in '10 5' '20 10' '30 15'; do set -- \$sz; [ -f results_request6/A/last_u${ue}_\$1x\$2.npy ] || python3 code/request6/hexA.py $ue \$1 \$2 40000 results_request6/A >> results_request6/A/log_u${ue}_\$1x\$2.txt 2>&1; done # hexA_queue_$ue" > /dev/null 2>&1 < /dev/null &
done
running "request6/autocommit.sh" || setsid nohup sh code/request6/autocommit.sh > /dev/null 2>&1 < /dev/null &
# task D: delta D scans (one single-thread queue)
running "dD_queue" || setsid nohup sh -c 'for spec in "19 local 0,1,0" "19 full 0,1,0" "19 stiff 0,1,0" "19 full 1,0,0" "19 full 0,0,1" "76 full 0,1,0" "76 local 0,1,0"; do set -- $spec; f=results_request6/D/dD_$2_Lx$1_n$(echo $3 | tr , _).json; [ -f $f.done ] || { python3 code/request6/dD.py $1 $2 $3 0.0,0.0005,0.001,0.002,0.004,0.008 >> results_request6/D/log_dD.txt 2>&1 && touch $f.done; }; done # dD_queue' > /dev/null 2>&1 < /dev/null &
# task C1: stripe finite-box thresholds (Bloch first, then dynamics), one single-thread queue
[ -f results_request6/C1_summary.json ] || running "stripeC_queue" || setsid nohup sh -c 'for mode in lin dyn; do for P in 60 30; do for phi in 90 60 30; do python3 code/request6/stripeC.py $P $phi 64,128,256,512 $mode >> results_request6/C/log_C1.txt 2>&1; done; done; done # stripeC_queue' > /dev/null 2>&1 < /dev/null &
# task B: melting ramps, three single-thread queues; each waits for its task-A queue to finish (its 30x15 run done) first
i=0
for list in "0.0:8:16 0.03:8:16 0.02:12:11" "0.01:8:16 0.0:12:11 0.03:12:11" "0.02:8:16 0.01:12:11"; do
  i=$((i+1)); qa=$(echo "0.15 0.2 0.3" | cut -d' ' -f$i)
  running "meltB_queue_$i" || setsid nohup sh -c "while [ ! -f results_request6/A/last_u${qa}_30x15.npy ]; do sleep 120; done; for spec in $list; do u=\$(echo \$spec | cut -d: -f1); nx=\$(echo \$spec | cut -d: -f2); ny=\$(echo \$spec | cut -d: -f3); [ -f results_request6/B/done_u\${u}_\${nx}x\${ny} ] || { python3 code/request6/meltB.py \$u \$nx \$ny 0.12 0.28 0.01 5000 1 results_request6/B >> results_request6/B/log_u\${u}_\${nx}x\${ny}.txt 2>&1 && touch results_request6/B/done_u\${u}_\${nx}x\${ny}; }; done # meltB_queue_$i" > /dev/null 2>&1 < /dev/null &
done
# task C2: stripe end states above threshold (one single-thread queue, after C1)
running "stripeC2_queue" || setsid nohup sh -c 'for L in 256 512; do for P in 60 30; do for phi in 90 60 30; do for f in 1.5 3.0; do t=40000; [ $L = 512 ] && t=60000; Nm=$(python3 -c "print(round($L/2**0.5))"); [ -f results_request6/C/last_C2_P${P}_phi${phi}_L${Nm}_f${f}.npy ] || python3 code/request6/stripeC2.py $P $phi $L $f $t >> results_request6/C/log_C2.txt 2>&1; done; done; done; done # stripeC2_queue' > /dev/null 2>&1 < /dev/null &
