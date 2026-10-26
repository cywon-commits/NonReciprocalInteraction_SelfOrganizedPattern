import sys, numpy as np
# Resample a spin state along x to a new box length (bilinear, as in scanS.py): python resample.py in.npy Lx_new out.npy
S = np.load(sys.argv[1]).astype(float); L0 = S.shape[0]; L = int(sys.argv[2])
xs = np.arange(L)*L0/L; x0 = np.floor(xs).astype(int); f = (xs-x0)[:, None, None]
S = (1-f)*S[x0 % L0]+f*S[(x0+1) % L0]; S /= np.linalg.norm(S, axis=-1)[..., None]; np.save(sys.argv[3], S)
