#!/bin/sh
# Run mpoint.py on every relaxed state of task A as it appears (restart safe: finished states are skipped).
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1
while true; do
  for f in results_request5/A/state_nx8_*.npy; do
    [ -f "$f" ] || continue
    out=$(echo $f | sed 's/state_/mpoint_/; s/.npy$/_a0.3.json/'); [ -f "$out" ] || python3 code/request5/mpoint.py "$f" >> results_request5/A/log_mpoint.txt 2>&1
  done
  [ -f results_request5/A/MPOINT_STOP ] && break; sleep 300
done
