#!/bin/sh
# usage: mkseed.sh nx ny r out   -- scaled seed for size nx x ny at u/u_c = r: from the 8x2 end state at the same r if it is
# topologically intact, otherwise from the 4x1 end state (waits until the source run has finished).
nx=$1; ny=$2; r=$3; out=$4; R=code/request2; B=results_request2/B_univ
while [ ! -f $B/4x1/sum_r$r.json ]; do sleep 60; done
if [ $nx -ne 8 ]; then while [ ! -f $B/8x2/sum_seed_r$r.json ]; do sleep 60; done; fi
if [ $nx -ne 8 ] && python3 $R/seed.py $B/8x2/S_seed_r$r.npy 8 2 $nx $ny $out 2>/dev/null; then echo "seed from 8x2" > $out.src
else python3 $R/seed.py $B/4x1/S_r$r.npy 4 1 $nx $ny $out && echo "seed from 4x1" > $out.src; fi
