#!/bin/sh
# Restart-safe launcher for request 4 (run again after a worker restart). Every job resumes from its files.
cd "$(dirname "$0")/../.."; O=results_request4
running() { ps -eo args | grep -v grep | grep -q -- "$1"; }
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
running "jobqueue.py" || python3 -c "import glob,os; [os.remove(f) for f in glob.glob('$O/**/*.lock', recursive=True)]"
for q in p m; do running "jobqueue.py $O/B/jobs_$q.txt" || setsid nohup python3 code/request2/jobqueue.py $O/B/jobs_$q.txt 2 > $O/B/queue_$q.log 2>&1 < /dev/null & done
# task C: clean states (prep_c.py resumes at the first missing u/u_c)
running "states/16x4 " || running "out $O/C/states/16x4" || setsid nohup python3 code/request4/prep_c.py --start results_request3/T0/16x4_s1/S_r1.0700.npy --uc 0.0249 --rs 1.0725,1.075,1.0765,1.0775,1.0785 --hold 2.5e5 --out $O/C/states/16x4 >> $O/C/states/log_16x4.txt 2>&1 < /dev/null &
running "out $O/C/states/12x3_a0.15" || setsid nohup python3 code/request4/prep_c.py --start results_request3/C/a0.15/12x3_s1/S_r1.0900.npy --uc 0.032543734989315044 --alpha 0.15 --dt 0.1 --rs 1.0925,1.0950,1.0975,1.1000 --hold 1.5e5 --out $O/C/states/12x3_a0.15 >> $O/C/states/log_12x3_a0.15.txt 2>&1 < /dev/null &
# task C: C1 + C2 per size
running "run_c.sh 8x2 " || setsid nohup sh code/request4/run_c.sh 8x2 0.0498 "1.0775 1.0800 1.0825 1.0850" > /dev/null 2>&1 < /dev/null &
running "run_c.sh 12x3 " || setsid nohup sh code/request4/run_c.sh 12x3 0.0333 "1.0725 1.0750 1.0775 1.0800 1.0825 1.0850" > /dev/null 2>&1 < /dev/null &
running "run_c.sh 16x4 " || setsid nohup sh code/request4/run_c.sh 16x4 0.0249 "1.0725 1.0750 1.0765 1.0775 1.0785" > /dev/null 2>&1 < /dev/null &
running "run_c.sh 12x3_a0.15 " || setsid nohup sh code/request4/run_c.sh 12x3_a0.15 0.032543734989315044 "1.0925 1.0950 1.0975 1.1000" 0.15 0.1 > /dev/null 2>&1 < /dev/null &
running "run_c.sh 12x3_a0.6 " || setsid nohup sh code/request4/run_c.sh 12x3_a0.6 0.03385975610194408 "1.0875 1.0900 1.0925 1.0950" 0.6 0.2 > /dev/null 2>&1 < /dev/null &
running "autocommit.sh" || setsid nohup sh code/request3/autocommit.sh > /dev/null 2>&1 < /dev/null &
# short-hold states just above threshold (the long-hold preparations slipped there)
[ -f $O/C/states/12x3/S_r1.0795.json ] || running "rs 1.0785,1.0795" || setsid nohup python3 code/request4/prep_c.py --start $O/C/states/12x3/S_r1.0775.npy --uc 0.0333 --rs 1.0785,1.0795 --hold 4e4 --out $O/C/states/12x3 >> $O/C/states/log_12x3.txt 2>&1 < /dev/null &
[ -f $O/C/states/16x4/S_r1.0770.json ] || running "rs 1.0770 " || setsid nohup python3 code/request4/prep_c.py --start $O/C/states/16x4/S_r1.0765.npy --uc 0.0249 --rs 1.0770 --hold 5e4 --out $O/C/states/16x4 >> $O/C/states/log_16x4.txt 2>&1 < /dev/null &
[ -f $O/C/states/12x3_a0.15/S_r1.0960.json ] || running "rs 1.0960" || setsid nohup python3 code/request4/prep_c.py --start $O/C/states/12x3_a0.15/S_r1.0950.npy --uc 0.032543734989315044 --alpha 0.15 --dt 0.1 --rs 1.0960 --hold 4e4 --out $O/C/states/12x3_a0.15 >> $O/C/states/log_12x3_a0.15.txt 2>&1 < /dev/null &
running "run_c.sh 12x3 0.0333 1.0785" || setsid nohup sh code/request4/run_c.sh 12x3 0.0333 "1.0785 1.0795" > /dev/null 2>&1 < /dev/null &
running "run_c.sh 16x4 0.0249 1.0770" || setsid nohup sh code/request4/run_c.sh 16x4 0.0249 "1.0770" > /dev/null 2>&1 < /dev/null &
running "1.0960 0.15" || setsid nohup sh code/request4/run_c.sh 12x3_a0.15 0.032543734989315044 "1.0960" 0.15 0.1 > /dev/null 2>&1 < /dev/null &
running "jobqueue.py $O/B/jobs_pf.txt" || setsid nohup python3 code/request2/jobqueue.py $O/B/jobs_pf.txt 2 > $O/B/queue_pf.log 2>&1 < /dev/null &
