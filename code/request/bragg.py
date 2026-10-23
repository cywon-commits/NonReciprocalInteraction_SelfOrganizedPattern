import numpy as np
# Registry of the skyrmion lattice on the periodic box: the six strongest S_z Fourier peaks (k != 0), as signed integer
# indices (kx, ky) in units of 2 pi/(Lx, Ly), reduced to the three with kx > 0 or (kx == 0, ky > 0).
# For the perfect nx x ny tiling these are (nx, ny), (nx, -ny), (0, 2ny); a change is a lattice 'phase slip'
# (rows added/removed across the box), the SkX analogue of a helix winding change.
def registry(S, npk=6):
    F = np.abs(np.fft.fft2(S[..., 2]-S[..., 2].mean()))**2; Lx, Ly = F.shape
    idx = np.argsort(F.ravel())[::-1][:npk]
    pk = []
    for i in idx:
        kx, ky = np.unravel_index(i, F.shape); kx = kx-Lx if kx > Lx//2 else kx; ky = ky-Ly if ky > Ly//2 else ky
        if kx > 0 or (kx == 0 and ky > 0): pk.append((int(kx), int(ky)))
    return tuple(sorted(pk))
def bragg_weight(S, reg):
    # fraction of off-zero power in the registry peaks (1 = perfect single-registry lattice)
    F = np.abs(np.fft.fft2(S[..., 2]-S[..., 2].mean()))**2
    return float(2*sum(F[kx, ky] for kx, ky in reg)/F.sum())
