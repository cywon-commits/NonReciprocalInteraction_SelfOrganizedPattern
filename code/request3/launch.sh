#!/bin/sh
# Idempotent launcher for request 3 (run again after a worker restart): job queues + the 32x8 chain.
cd "$(dirname "$0")/../.."; O=results_request3; RS=1.06,1.065,1.07,1.0725,1.075,1.0775,1.08,1.0825,1.085,1.09,1.095,1.10
running() { ps -eo args | grep -v grep | grep -q -- "$1"; }
NQ=${NQ:-3}
running "jobqueue.py $O/jobs/all.txt" || { python3 -c "import glob,os; [os.remove(f) for f in glob.glob('$O/**/*.lock', recursive=True)]"; setsid nohup python3 code/request2/jobqueue.py $O/jobs/all.txt $NQ > $O/jobs/queue.log 2>&1 < /dev/null & }
if [ ! -f $O/A32/DONE ] && ! running "out $O/A32/chain"; then
  NUMBA_NUM_THREADS=${T32:-1} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 setsid nohup sh -c "python3 code/request3/run3.py --nx 32 --ny 8 --uc 0.012451 --rs $RS --hold 4e5 --hold_max 8e5 --every 200 --start $O/A32/S_seed32x8_r1.06.npy --seed 1 --out $O/A32/chain && touch $O/A32/DONE" >> $O/A32/log_chain.txt 2>&1 < /dev/null &
fi
for a in 0.15 0.6; do
  [ -f $O/C/a$a/DONE ] || running "taskC.py $a" || { mkdir -p $O/C/a$a; NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 setsid nohup sh -c "python3 code/request3/taskC.py $a && touch $O/C/a$a/DONE" >> $O/C/a$a/driver.log 2>&1 < /dev/null & }
done
# 24x6 restarted from the intact request-2 chain state (the request-2 seed file at 1.06 is a post-slip state)
running "jobqueue.py $O/jobs/extra.txt" || setsid nohup python3 code/request2/jobqueue.py $O/jobs/extra.txt 1 > $O/jobs/queue_extra.log 2>&1 < /dev/null &
