import glob, os, numpy as np
# Downsample the raw high-rate time series (git-ignored rec_*.npy) into one compressed ts_<folder>.npz per folder (<= 2000 rows per run).
for d in ['results_request/C_helix/C1', 'results_request/C_helix/C2', 'results_request/C_helix/C3', 'results_request/B_temp/stairs']:
    fs = sorted(glob.glob(f'{d}/rec_*.npy'))
    if not fs: continue
    out = {}
    for f in fs:
        r = np.load(f); k = max(1, len(r)//2000); out[os.path.basename(f)[4:-4]] = r[::k].astype(np.float32)
    np.savez_compressed(f'{d}/ts_{os.path.basename(d)}.npz', **out); print(d, len(out), 'runs')
