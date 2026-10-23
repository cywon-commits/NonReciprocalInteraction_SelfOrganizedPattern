import sys, os, subprocess, time, shlex
# Idempotent job queue (restart-safe): each line of the job file is "marker<TAB>command". A job is skipped if its
# marker file exists; the command is run with N parallel workers (single-threaded jobs). Run from the repo root.
jobs = [l.rstrip('\n').split('\t', 1) for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
nw = int(sys.argv[2]) if len(sys.argv) > 2 else 4
todo = [(m, c) for m, c in jobs if not os.path.exists(m)]; run = []
env = dict(os.environ, NUMBA_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
print(f"{len(todo)} of {len(jobs)} jobs to run", flush=True)
while todo or run:
    run = [(m, p) for m, p in run if p.poll() is None]
    while todo and len(run) < nw:
        m, c = todo.pop(0); print(time.strftime('%H:%M'), 'start', m, flush=True)
        run.append((m, subprocess.Popen(c, shell=True, env=env)))
    time.sleep(10)
print(time.strftime('%H:%M'), 'queue done', flush=True)
