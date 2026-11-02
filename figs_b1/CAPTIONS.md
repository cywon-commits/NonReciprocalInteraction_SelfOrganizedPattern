# Paper B1 figures: captions and sources

Each figure has one script `figN.py` that reads only repository data and writes `out/figN_*.pdf` and `.png` (3.4 in, Okabe–Ito colours). `b1style.py` holds the style and the render check (text overlaps, column width, legends over data). Data that did not exist yet are produced by `data_fig2.py`, `data_fig4.py`, `data_fig7.py`, `code/stripe/pitch_ratio.py` and the reruns listed under each figure; their output is in `results/b1_figs/`.

## Fig. 1 — `fig1.py`
**A current is a fixed nonreciprocal bond bias, and in the co-moving frame it enters only through two vectors of opposite time-reversal parity.**
(a) On a bond i → j the adiabatic term is a receiver-dependent coupling κ_r and the β term a field coupling κ_f, with u = −2aκ_r and βu = −2aκ_f.
(b) Slot vectors of the driven skyrmion lattice (B = 0.5D²/J, α = 0.3, β = 0.6) from the exact Thiele projection. w₁ = v − u (T-odd) and w₂ = βu − αv (T-even) are perpendicular, and the Landau–Lifshitz drive W₂ = w₂ + αw₁ = (β − α)u points along u whatever the Hall angle of v.
Data: Thiele tensors in `results/b1_todo/dD_ana.log`.

## Fig. 2 — `fig2.py`, data `data_fig2.py`
**The residual magnetic group of the driven pattern fixes its first-order response, and the vector that matters is w₁, not u.**
(a) Group–subgroup steps with BNS numbers (spglib on minimal commensurate cells). Cycloid: u ∥ q leaves Pm′a2′ and allows a uniform cone M ∥ ẑ×u; u ⊥ q leaves Pm′m2′ and the cycloid stays exact; the out-of-plane cone is forbidden by the half-period antitranslation T t_{λ/2}. Skyrmion lattice: linear response of 6m′m′ gives a dipolar distortion d = a ẑ×w₁ + b w₂, perpendicular to w₁.
(b) Change of m_z of a driven Néel skyrmion (α = 0.3, u = 0.02 along x) after the uniform translation is removed. Black contour: m_z = 0 of the static skyrmion. Green arrow: first moment d inside r < 5.
(c) Angle of d measured from w₁ (filled) and from u (open) for α = 0.1, 0.3, 0.6 and current directions 0°, 30°, 90° (circle, triangle, square). d stays perpendicular to w₁ within 1.5°; its angle to u follows the Hall rotation.
Data: `results/b1_residual_group/residual_bns.json`, `results/b1_figs/fig2_distortion.json`, `fig2_map.npy`.

## Fig. 3 — `fig3.py`
**A pattern has no precessional Walker breakdown; its static branch ends in a Landau instability before it can unwind.**
(a) The conjugate variable of a domain wall, the tilt φ, has a bounded restoring force. A drive above F_max (dotted) has no static root and the wall precesses.
(b) The conjugate variable of a pattern, the cone M, has a quadratic energy, so a static root exists up to |M| = 1.
(c) Helix with exchange and DMI: the compression velocity on the driven cone, c = (D/s)√(1 − M²) with M = su_eff/D, meets the Doppler line λu_eff (λ = 1) at u_L = D/(√2 s), below the unwinding point u_unw = D/s.
(d) With easy-axis anisotropy K, u_L = Λ/√(1 + κ²) stays below u_unw = Λ/κ for every K (Λ = √(D² + 2AK)/s, κ = D/√(D² + 2AK)). Point: exact Bloch threshold of an N = 800 chain.

## Fig. 4 — `fig4.py`, data `data_fig4.py`
**The driven helix breaks at u_L = D/(√2 s) for every (α, β): the threshold depends on β only through u_eff = (β/α − 1)u.**
(a) Largest Bloch growth rate divided by α against u_eff for five (α, β), including β = 0 and β < 0 (N = 200; u_L includes the finite-chain correction √((D² + k_min²)/2)). Inset: the same thresholds in the bare current u (spread over a factor of two and both signs) and in u_eff (equal to 0.1%).
(b) Finite chains: exact thresholds (points) against √((D² + k_min²)/2) (line), converging to D/√2 (dashed).

## Fig. 5 — `fig5.py`, reruns `code/skx/lam_alpha.py 0.5 0.3 {40,10,5}`
**The skyrmion-lattice threshold is u_eff* = c₂k/λ with λ = κ_P, the dissipative overlap; damping enters only at order α².**
(a) Exact Bloch thresholds (B = 0.5D²/J, α = 0.3, β = 0.6) at k = 2π/(19n), n = 40, 20, 10, 5, against c₂k/κ_P with c₂ and κ_P from the static lattice (line; agreement within 0.4–0.9%) and the affine estimate λ = 1 (dashed). Top axis: box length L = 2π/k, so the threshold falls as 1/L.
(b) λ from exact thresholds at α = 0.05 and 0.3 lies on κ_P (solid), not on κ_τ (dashed), at four fields.
(c) Relative change of λ from α₀ = 0.05, Δλ/λ = −c(α² − α₀²), fitted for α ≤ 0.3 (filled, dashed lines). Open symbols: α = 0.6, near critical damping. Shaded band: α ≤ 0.1, where λ = κ_P within 0.6%.

## Fig. 6 — `fig6.py`, data `code/stripe/pitch_ratio.py`
**Stripes have a quadratic Goldstone mode: a current component along q only scales the bending velocity, and every finite-box threshold collapses onto u_eff* sin φ = √(c_M² + (2πc_b/L)²).**
(a) k-resolved bending velocity on converged driven states (u ∥ q, residual < 10⁻¹²; error bars: spread over k = 0.0025–0.04 q). 1 − c(M)²/c₀² = ηM² with η ≈ 1.4, the same for every k and for P = 30 and 60, steeper than the exchange value η = 1 (solid).
(b) Threshold against the current angle (P = 30): exact Bloch thresholds at k = 0.01 (points) and the k → 0 closed form u* = c₀/√(sin²φ + η(c₀/D)²cos²φ) (line).
(c) Dynamical thresholds in sheared boxes of length L = 64–512 (request 6) for P = 30 (filled) and 60 (open) and φ = 90°, 60°, 30° collapse onto Y = √(1 + X²), with c_M = c₀√(1 − ηM²). The P = 60, L = 64 points leave the curve where the box wavenumber approaches the stripe wavenumber.

## Fig. 7 — `fig7.py`, data `data_fig7.py`
**λ is a static property of the pattern: one minus the counterflow momentum carried by its internal relaxation.**
(a) 1D chiral soliton lattice (D = 0.3, period 40): exact Doppler overlap against the homogenization formula ⟨1/M⟩/(⟨ψ₀⁻²⟩⟨ψ₀²/M⟩) with a slaved conjugate field (K⊥ = 5). With K⊥ = 0.2 the conjugate field is not slaved and the formula (dotted) overestimates λ.
(b) Seven skyrmion lattices: predictions against the exact dynamic overlaps κ_P (filled) and κ_τ (open). The static relaxed mode and the two-field Ritz theory (displacement plus canting, no fitted parameter) fall on the diagonal; displacement-only theory and the affine mode do not.
(c) Circular decomposition of the Bloch mode (19×33): the counterflow cancels mainly the co-rotating momentum π₊; the counter-rotating π₋ keeps more, so κ_P > κ_τ once π₋ > π₊.

## Fig. 8 — `fig8.py`
**Below threshold the current puts the lattice phonons in a Gibbs ensemble moving at λu_eff, and the melting point moves only by the zone average of that heating.**
(a) Effective temperatures of the two branches at k₁ = 2π/76 (stochastic LLG, 76×33, k_BT = 0.01) against 1/(1 ∓ x).
(b) Variance ratio of displacement modes in a 152×66 box at u_eff = 0.03 against 1/(1 − x²). Modes along the current heat; transverse and oblique modes stay at the bath temperature.
(c) Melting temperature from heating ladders (256–264 skyrmions, three seeds) against the zone average 1/⟨1/(1 − x²)⟩_BZ (solid) and the rejected long-wave criterion (four longest modes, dotted), for two box shapes.

## Fig. 9 — `fig9.py`, reruns `code/b1/b1_k.py`, `code/b1/b1_static.py`
**Exchange-even couplings pump the lattice phonons with zero threshold through the static relaxed mode, the dual of the Zhang–Li Doppler term.**
(a) Damping-like SOT (c = 0.004, n = ŷ, α = 0.3): pumping ratio x = ΔΓ/Γ₀ against k for three directions of k. x·k is nearly constant (dashed: 1/k), so the threshold vanishes in an infinite system.
(b) First-order generator Re ∂λ/∂c on the exact mode, the static relaxed mode and the affine mode (α = 0). The relaxed mode carries all the pumping; the affine mode carries none.
(c) Second-order nonreciprocal stiffness of δD (n = ŷ, α = 0.3): threshold δD* = Γ₀/(half-split per δD) for k ⊥ n (Doppler shift converted by damping) and k ∥ n (intrinsic splitting). Both are ∝ k.
