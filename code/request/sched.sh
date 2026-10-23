#!/bin/sh
# Keep the 4 cores filled without oversubscription: SkX chain jobs (skxrun --gamma) have priority;
# other compute jobs (helixrun, thr_N, growth/bistability skxrun) are SIGSTOPped/SIGCONTed to fill the remaining cores.
while true; do
  hi=0
  for p in $(pgrep -f "^python3 [^ ]*skxrun.py .*--gamma"); do
    if grep -q "NUMBA_NUM_THREADS=2" /proc/$p/environ 2>/dev/null; then hi=$((hi+2)); else hi=$((hi+1)); fi
  done
  free=$((4-hi)); n=0
  his=" $(pgrep -f "^python3 [^ ]*skxrun.py .*--gamma" | tr '\n' ' ') "
  for p in $(pgrep -f "^python3 [^ ]*(helixrun.py|thr_N.py|skxrun.py .*--(nostop|holds))" | sort -n); do
    case "$his" in *" $p "*) continue;; esac
    if [ $n -lt $free ]; then kill -CONT $p 2>/dev/null; n=$((n+1)); else kill -STOP $p 2>/dev/null; fi
  done
  sleep 20
done
