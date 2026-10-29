"""Residual magnetic space groups of driven chiral patterns (paper B1, Sec. III).

A commensurate realization of each pattern is decorated, site by site, with two markers:
  w2 (T-even polar vector): a nonmagnetic atom displaced along w2;
  w1 (T-odd polar vector):  a toroidal vortex of four moments m_i = w1 x r_i at r_i = +-d e_a, +-d e_b (e_a, e_b ⊥ w1).
Markers sit on every site, so all translations of the pattern survive and the decorated structure has exactly the
symmetry (pattern group) ∩ (bias group). spglib then identifies the magnetic space group (BNS, OG, Litvin numbers).
3D realizations: 1D patterns are extruded along y and z (orthorhombic host b=1, c=1.3), SkX along z (hexagonal host).
"""
import numpy as np, spglib, json, sys

def classify(L, pos_cart, types, mag):
    Linv = np.linalg.inv(L)
    frac = (pos_cart @ Linv) % 1.0
    ds = spglib.get_magnetic_symmetry_dataset((L, frac, types, mag), symprec=1e-4, mag_symprec=1e-4)
    t = spglib.get_magnetic_spacegroup_type(ds.uni_number)
    # magnetic point group: distinct Cartesian rotations with their time-reversal flags
    pg = {}
    for R, tr in zip(ds.rotations, ds.time_reversals):
        Rc = np.round(L.T @ R @ np.linalg.inv(L.T), 6)
        pg[name(Rc) + ("'" if tr else "")] = 1
    return dict(bns=t.bns_number, og=t.og_number, litvin=t.litvin_number, type=t.type,
                n_ops=len(ds.rotations), point_ops=sorted(pg))

def name(R):
    d = np.linalg.det(R); tr = np.trace(R)
    if np.allclose(R, np.eye(3)): return "1"
    if np.allclose(R, -np.eye(3)): return "-1"
    P = R * d                     # proper part
    ang = np.degrees(np.arccos(np.clip((np.trace(P) - 1) / 2, -1, 1)))
    w, v = np.linalg.eig(P); ax = np.real(v[:, np.argmin(abs(w - 1))]); ax /= ax[np.argmax(abs(ax))]
    n = int(round(360 / ang)) if ang > 1 else 1
    axs = "[" + ",".join(f"{a:.2g}" for a in ax) + "]"
    if d > 0: return f"{n}{axs}"
    if n == 2: return f"m{axs}"     # mirror with normal ax
    return f"-{n}{axs}"

def decorate(L, pos, types, mag, w1=None, w2=None, d1=0.11, d2=0.07):
    P, T, M = [pos], [types], [mag]
    if w2 is not None:
        w2 = np.asarray(w2, float); w2 /= np.linalg.norm(w2)
        P.append(pos + d2 * w2); T.append(np.full(len(pos), 2)); M.append(np.zeros_like(mag))
    if w1 is not None:
        w1 = np.asarray(w1, float); w1 /= np.linalg.norm(w1)
        ea = np.array([0., 0, 1]); eb = np.cross(ea, w1)          # w1 is in plane; vortex axes along z and z x w1
        eb /= np.linalg.norm(eb)
        for r in (ea, -ea, eb, -eb):
            P.append(pos + d1 * r); T.append(np.full(len(pos), 3)); M.append(np.tile(np.cross(w1, r), (len(pos), 1)))
    return np.vstack(P), np.concatenate(T), np.vstack(M)

def chain(kind, N=8, b=1.0, c=1.3):
    x = np.arange(N); q = 2 * np.pi / N
    pos = np.stack([x, 0 * x, 0 * x], 1).astype(float)
    if kind == "neel":   mag = np.stack([np.sin(q * x), 0 * x, np.cos(q * x)], 1)   # cycloid in xz, q || x
    else:                mag = np.stack([0 * x, np.sin(q * x), np.cos(q * x)], 1)   # Bloch helix, q || x
    return np.diag([N, b, c]).astype(float), pos, np.ones(N, int), mag

def skx(kind, n=6, M0=1.0, c=1.0):
    a1 = np.array([1., 0, 0]); a2 = np.array([0.5, np.sqrt(3) / 2, 0])
    L = np.array([n * a1, n * a2, [0, 0, c]])
    pos = np.array([i * a1 + j * a2 for i in range(n) for j in range(n)])
    # reciprocal vectors of the skyrmion lattice
    B = 2 * np.pi * np.linalg.inv(L).T
    Qs = [B[0], B[1], -(B[0] + B[1])]
    m = np.tile([0, 0, M0], (len(pos), 1)).astype(float)
    for Q in Qs:
        ph = pos @ Q; qh = Q / np.linalg.norm(Q)
        s = qh if kind == "neel" else np.cross([0, 0, 1], qh)
        m += -np.outer(np.cos(ph), [0, 0, 1]) + np.outer(np.sin(ph), s)
    m /= np.linalg.norm(m, axis=1)[:, None]
    return L, pos, np.ones(len(pos), int), m

X, Y, Z = np.eye(3)
cases = [
    ("Neel cycloid q||x, unbiased",       chain("neel"),  None, None),
    ("Neel cycloid, w1 || q (u||q)",      chain("neel"),  X,    None),
    ("Neel cycloid, w1 ⊥ q (u||y)",       chain("neel"),  Y,    None),
    ("Bloch helix q||x, unbiased",        chain("bloch"), None, None),
    ("Bloch helix, w1 || q",              chain("bloch"), X,    None),
    ("Bloch helix, w1 ⊥ q (y)",           chain("bloch"), Y,    None),
    ("Neel SkX, unbiased",                skx("neel"),    None, None),
    ("Neel SkX, w1||x (⊥ m_x), w2||y",    skx("neel"),    X,    Y),
    ("Neel SkX, w1||y (in m_x), w2||x",   skx("neel"),    Y,    X),
    ("Neel SkX, w1 generic, w2 ⊥ w1",     skx("neel"),    [np.cos(.3), np.sin(.3), 0], [-np.sin(.3), np.cos(.3), 0]),
    ("Neel SkX, w1 only ||x (w2=0)",      skx("neel"),    X,    None),
    ("Bloch SkX, unbiased",               skx("bloch"),   None, None),
    ("Bloch SkX, w1||x, w2||y",           skx("bloch"),   X,    Y),
    ("Bloch SkX, w1||y, w2||x",           skx("bloch"),   Y,    X),
    ("Bloch SkX, w1 generic, w2 ⊥ w1",    skx("bloch"),   [np.cos(.3), np.sin(.3), 0], [-np.sin(.3), np.cos(.3), 0]),
    ("Bloch SkX, w1 only ||x (w2=0)",     skx("bloch"),   X,    None),
]
out = []
for lab, (L, pos, ty, mag), w1, w2 in cases:
    P, T, M = decorate(L, pos, ty, mag, w1, w2)
    r = classify(L, P, T, M); r["case"] = lab; out.append(r)
    print(f"{lab:36s} BNS {r['bns']:>9s}  OG {r['og']:>11s}  Litvin {r['litvin']:5d}  type {r['type']}  "
          f"point ops: {' '.join(r['point_ops'])}", flush=True)
json.dump(out, open("residual_bns.json", "w"), indent=1, ensure_ascii=False)
