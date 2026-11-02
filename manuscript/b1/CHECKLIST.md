# Paper B1 manuscript — claim evidence and pre-submission checks

Build: `cd manuscript/b1 && latexmk -pdf main.tex` (REVTeX 4-2, prb). Figures are copies of `figs_b1/out/*.pdf`; regenerate with `cd figs_b1 && python3 figN.py` (one script per figure; data in `results/b1_figs/`).

## Evidence grade of each claim

[증명] = derived exactly (symmetry or identity); [수치확인] = derived and confirmed numerically; [추측] = plausible, not shown.

| Claim (section) | Grade | Basis |
|---|---|---|
| Bias enters only through w₁ (T-odd), w₂ (T-even); LL drive W₂ = (β − α)u (II) | [증명] | Eqs. (2), (4); exact algebra |
| Thresholds depend on β only through u_eff (II) | [증명] + [수치확인] | algebra; Fig. 5 (five (α, β) to 0.1%) |
| Current = nearest-neighbour (κ_f, κ_r) coupling, Eq. (5) (II) | [증명] | central-difference matching |
| Residual groups and BNS labels (III) | [수치확인] | spglib on decorated cells (`results/b1_residual_group`) |
| Nonsymmorphic protection by T t_{λ/2} (III) | [증명] | group elements listed; superspace argument |
| Cone M = s u_eff/D (III) | [증명] + [수치확인] | conjugate-variable argument; lattice cone exact (residual 10⁻¹⁵) |
| d ⊥ w₁ for skyrmion lattices (III) | [증명] + [수치확인] | linear response of 6m′m′; Fig. 3(c) within 1.5° |
| Lowest orientational torque ∝ |w₁|⁶ sin 6(θ − φ_w) (III) | [증명] | invariant counting |
| No precessional Walker breakdown (IV) | [증명] | quadratic cone energy |
| Landau criterion with λ = κ_P (IV) | [증명] at O(α⁰) + [수치확인] | App. A; Table II, Fig. 6 |
| u_L = D/√2 s < u_unw; with anisotropy u_L < u_unw (IV) | [증명] + [수치확인] | Eq. (10); Fig. 5 |
| SkX threshold = c₂k/κ_P (V) | [수치확인] | four k, 0.4–0.9% |
| α² correction, c = −0.08, 0.10, 0.33, 0.54 (V) | [수치확인] | fit for α ≤ 0.3; mechanism decomposition at B = 0.7 |
| Real-time growth at the Bloch rate (V) | [수치확인] | Fig. 7(b), no fitted slope |
| Stripe bending factor √(1 − ηM²), k-independent, η = 1.40 (V) | [수치확인] | k-resolved ratios, P = 30, 60 |
| η ≠ 1 comes from the anisotropic origin of c₀ (V, App. C) | [추측] | not derived; P = 120 unresolved |
| Finite-box collapse u* sin φ = √(c_M² + (2πc_b/L)²) (V) | [수치확인] | request-6 boxes, Fig. 8(c) |
| Static theorem for type-B modes (VI) | [증명] + [수치확인] | k·p argument; Table III |
| 1D homogenization formula (VI) | [증명] (slaved limit) + [수치확인] | Fig. 9(a) within 2% |
| λ = 1 − counterflow; two-field theory ±0.01 (VI) | [수치확인] | Table III, Figs. 9(b), 10 |
| κ_P > κ_τ ⇔ π₋ > π₊ (VI) | [증명] + [수치확인] | App. G; Table IV |
| Moving Gibbs ensemble T/(1 ∓ x) (VII) | [증명] (linear Langevin) + [수치확인] | App. B; Fig. 11(a, b) |
| Melting shift = zone average; bound → 0 for large L (VII) | [수치확인] for two boxes; bound [증명] | Fig. 11(c), Eq. (18) |
| Exchange-parity rule; SOT pumping through relaxed mode (VIII) | [증명] (parity) + [수치확인] | Fig. 12(a, b) |
| δD stiffness force and zero-threshold pumping (VIII) | [수치확인] | App. F; Fig. 12(c) (two k per direction) |
| j_L formula, Eq. (19) (IX) | [증명] (unit conversion) | magnitude for real materials not evaluated |

## Points that differ from the working draft (rev 4)

- Stripe cone factor: η = 1.40 for both P = 30 and 60 (k-resolved), replacing η = 1.27, 1.03 → 1; Eq. (12) and App. C carry η.
- SkX thresholds 0.0101, 0.0203, 0.0405, 0.081 (new runs), replacing older scan values.
- α² coefficients from a fit over α ≤ 0.3: −0.08, 0.10, 0.33, 0.54.
- Zone-average melting shift at L_x = 228: 6.6% (draft 6.3%).
- Figures renumbered (OUTLINE.md); three real-space figures added (1, 7, 10).

## Open before submission

- [ ] Guide step 7: every entry of `refs.bib` is unverified. Check each against its DOI; in particular Jiang 2020 (MnP), Masell 2020 author list, Kishine 2015 volume, spglib reference.
- [ ] Cite the foundational paper on nonreciprocal spin couplings once it has an arXiv number; B2 cross-reference likewise.
- [ ] Decide whether P = 120 stripe data (unresolved small-k eigenvalues) are needed.
- [ ] Journal choice (PRB regular as drafted).
- [ ] Optional: a numerical value of j_L for a named material (needs D, P, β/α from literature, step 7).
