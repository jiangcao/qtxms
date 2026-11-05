# CLAUDE.md — Mode space as the coarse level of a multigrid transport solver

## Goal
Make mode-space (basis-reduced) quantum transport *exact and cheap* for rough Si nanosheet/nanowire FETs by
treating the mode-space basis U as the coarse level of a multigrid / deflated-Krylov solver for the
open-boundary Schrödinger (QTBM) equation, instead of using it as a stand-alone approximation.
Target: ab initio / sp3d5s*-type tight binding, where RGF costs O(S · n_slab³) per energy and plain mode
space suffers from slow convergence on rough devices and from spurious bands.

Context: this continues the user's mode-space study (effective-mass toy, 3×5 nm rough Si sheet, ~39k points),
where a "two-level basis" (local level 1 with residual enrichment, then slab density-matrix level 2) reached
0.037% max current error at N = 112 with a 270 s build. Results are reported to Mathieu Luisier.

## Naming (user definition, 2026-11-04)
Two ways to partition the device:
- **Slab**: one block of the block-tridiagonal partition along transport (x); covers the whole cross section.
- **Tile**: a piece of a slab; N tiles form a slab (transverse bisection, optionally also cut in x, `--tile-xsplit`).
A basis is block-diagonal over the chosen partition (per tile or per slab).
Partition and level are independent: either level can use either partition.
Both levels are density matrices integrated over a complex contour covering the energy window.
- **Level 1 = local density matrix**: per block (tile or slab), the ρ of the block embedding closed by a CAP or by
  CAP + the exact lead Σ on the lead faces ("CAP+OBC", `--tile-obc`). Basis U⁽¹⁾ = its dominant eigenvectors per
  block. MS1 = mode space in U⁽¹⁾. (Current runs: tiles.)
- **Level 2 = global density matrix**: ρ of the whole open device (exact lead OBC), computed by RGF; the basis is
  the dominant eigenvectors of its diagonal blocks (tile or slab). Two variants: *level 2 on level 1* (RGF in the
  level-1 basis, U = U⁽¹⁾V, nested; MS2) and *direct level 2* (RGF on the fine device, U⁽¹⁾ = I). (Current runs:
  slabs.)
Use these names in code comments, logs, notes and reports: "local ρ" = level 1, "global ρ" = level 2; "tile" and
"slab" name only the partition.
All of these are density-matrix mode space (DMMS); lead CMS is the other family. Label a basis by partition, level and
cut, e.g. "slab level 1, cut 1e-8", "slab level 2, cut 5e-14" (the direct level-2 n400 basis: top 400 per slab,
λ_400 = 1e-14…6e-14, + end-slab lead RMS, N 1636), never just "DMMS". Global ρ decays much faster than local ρ:
slab 1 level 2 keeps 322 vectors at 1e-10 (level 1: 1159), 220 at 1e-6 (level 1: 352).

## Method and key equations
Per energy E:
- Fine operator: A(E) = E·S − H − Σ_L(E) − Σ_R(E)   (S = I in effective mass / orthogonal TB)
- Solve A ψ = b with QTBM injection b = Γ_L^{1/2} columns (propagating channels only);
  T(E) = Σ_cols ψ_N† Γ_R ψ_N, ψ_N = last slab.
- Coarse space: block-diagonal per-slab basis U = diag(U_1 … U_S) (orthonormal columns per slab).
  Prolongation P = U, restriction R = U†, Galerkin A_c = U† A U (includes U†ΣU automatically).
- P_c = U A_c⁻¹ U†;  coarse-grid projector π = P_c A (π² = π, verified ~1e-14).
  Identity: π A⁻¹ = U A_c⁻¹ U† = G_MS, so mode-space error = (I − π) ψ.
- Balancing two-level preconditioner:  M⁻¹ = P_c + (I − P_c A) S (I − A P_c)
  S = smoother:
    * Chebyshev for (−A)⁻¹ on [E_b − E, λ_max]: valid only when the complement is definite
      (effective mass with basis window starting at the band bottom).
    * k steps of GMRES on A z = r (default for TB): handles the indefinite complement
      (valence states below E, high conduction states above E).
- Outer solver: right-preconditioned *flexible* GMRES (stores Z = M⁻¹V, so nonlinear smoothers are OK),
  x0 = P_c b = mode-space solution. Iteration 0 is plain mode space.
- Nested (not yet implemented) form of the user's two-level basis: U = U⁽¹⁾U⁽²⁾,
  A₁ = U⁽¹⁾†AU⁽¹⁾, A₂ = U⁽²⁾†A₁U⁽²⁾ = U†AU → three-level V-cycle; range(U) ⊆ range(U⁽¹⁾) explains why
  level 2 cannot beat its parent.

## Decisions and why
- Mode space as preconditioner/deflation, not stand-alone: converges to RGF accuracy; the basis only needs to
  be good enough for fast convergence (a 1.2%-error basis converges to 1e-11 in 7 iterations).
- Galerkin coarse operator: guarantees consistency and the projector identities; no separate coarse Σ needed.
- Exact lead Σ on the fine level, only the coarse solve sees U†ΣU: lead-compression error becomes something
  the iteration corrects rather than an error floor (the 0.17% lead-compression issue in the user's study).
- Level-1 enrichment from local Ritz problems: per slab s a closed cluster of slabs s−w…s+w (default w = 1)
  in the level-1 basis, generalized eigh in a window around Ec, fine residual r = (H − λS)U⁽¹⁾c kept only on
  the centre slab s (the outer cluster slabs carry the artificial cut: closed-end states, missing coupling),
  flag |r_s| > rthr, SVD of the residual per block of slab s, append. Cost S·(2w+1)³n₁³ instead of (S n₁)³ for
  the global H1 of qtmode.enrich (w = −1 reproduces the global variant), linear in device length, and the
  cluster problems are independent (parallel). Cuts between tiles of slab s and towards s±1 are interior to
  the cluster, so the spurious cut states that enrichment targets are seen. Second variant `--enrich-mode tile`:
  cluster = one tile + every tile coupled to it (block pairs with any |H|+|S| coupling, `--enrich-tol` to drop
  weak ones), residual and enrichment on that tile only; smaller clusters when slabs hold many tiles.
- Wave-function (QTBM) formulation: few RHS (one per propagating channel); density via G diag not done.
- Basis built in an energy window [mid-gap, Ec + E_b]: E_b sets where the complement spectrum starts, hence the
  smoother's condition number; this matters more than N. (CMS iteration counts barely change with M.)
- Flexible GMRES outer: allows GMRES smoother; plain GMRES with Chebyshev also works.
- Slab partition by binning the transport coordinate at width = coupling range (from nnz of H and S), bin edges
  half a plane spacing before the first plane; user can override with --slab-width (= lead cell length).
- Injection channels: eigenvalues of Γ_L above 1e-4 eV (absolute); relative tolerances picked up
  η-broadened evanescent modes. Count is cross-checked against lead band crossings.
- Level 1 of the user's two-level basis (qtmode): local tiles + bulk-Bloch cores for the Si sheet. For general
  ab initio TB there is NO tile / core separation: level 1 is built from local tiles over all atoms (cores need
  a bulk unit cell, which ab initio devices do not provide). Level 2 = transport slabs (contour slab ρ).
- All models go through one generic input set (qtx_ms/prep/inputs.py: xyz, norb, blocks, H, S, lead cell,
  meta.json), so solvers and bases never call a TB builder directly; model-specific extras (Si structure.npz)
  are optional side files.
- Per-slab "slab" (CMS-like) basis in TB is selected by count (M lowest above mid-gap), not by window,
  because isolated-slab energies sit above the device band edge.
- QuaTrEx devices with oxide (MoS2/HfO2): basis = direct level 2 (top-K eigenvectors of the contour slab ρ of
  the open device, U⁽¹⁾ = I), not lead CMS — the lead-CMS basis cannot represent oxide / displaced atoms (−17…−39%).
  Count K per slab (not an occupation threshold); end-slab spurious lead bands removed with RMS (generalized S).
- H/S truncation only as block-tridiagonal trim (drop couplings beyond the next block), never a magnitude cut:
  trim c4 0.18%, c3 0.86% vs cut 1e-4 1.35%, 1e-3 6.7% in exact T.
- QuaTrEx [0,0,0] blocks are Γ-point supercells: H, S couple across the y and z faces (MoS2/HfO2: 10% / 3% of
  the pairs, minimum-image range 11.0 / 10.0 Å, periods 32.918 / 31.189 Å). meta["periodic"] is a list of
  {axis, length} (single dict still accepted); `msbasis.Geometry` / tile embeddings use minimum image on all.
  Without it the interior tiles were built with a hard cut at the faces: level-1 MS T error 99% / 73% / 63% /
  52% / 35% at Ec + 0 / 0.05 / 0.1 / 0.15 / 0.2 eV (results_ms/mos2_hfo2_nocore_nonperiodic).
- Tile embedding boundary (`tile_modes(boundary=)`, run_ms `--tile-boundary`): "cap" (default) or "schur"
  (explicit Sigma_P = A_PE A_EE^-1 A_EP of the closed environment = embedding without CAP). trim_c3 tile 4, exact
  12-pt contour, occupation-weighted weight of direct-L2 top-200 modes outside the tile basis at occ > 1e-4:
  CAP 12/3 6.7e-5, Schur 12 2.4e-4, Schur 18 4.1e-5, CAP 18/6 2.8e-5 -> keep CAP; occupied modes are captured.
- Iterated tile self-energies (`msbasis.TileBP`, cavity = block Gaussian BP) are exact only when the tile graph is a
  tree (one tile per slab = RGF). With 2 tiles per slab (loops) it converges to tile G blocks 33% / 53% off
  (trim_c3, Im z 0.37 / 0.04) at ~1750 s per point vs 75 s for exact slab RGF; "full" (no cavity) 49% / 78%.
  Known BP limitation (variances wrong on loopy graphs); exact alternative = selected inversion / nested dissection.
- trim_c3 level 1 (exact 12-pt CAP tiles) + level 2 (results_ms/mos2_hfo2_trim_c3_nocore_tc12_t*.log), T error at
  Ec + 0.02…0.5 eV: tile cut 1e-3 (N1 1038 → N2 1003) 0.07–8%; 1e-4 (1405 → 1280) 0.1–7%; 1e-5 (1939 → 1529)
  0.05–2.9%. MS2 ≈ MS1 (level 1 sets the floor); direct level 2 n200 ≤ 3e-5. On mos2_hfo2 the H1-Ritz enrichment
  (enrich_round) halved the median (2.7 → 1.2%) but its residuals do not shrink between rounds (closed clusters,
  cannot see states absent from U1).
- Residual enrichment from the original open system (`msbasis.residual_enrich`, run_ms `--res-enrich n --res-corr
  raw|slab|cluster|exact --res-rhs inject|random --res-probes k --res-contour n`): e ~ A^-1 (b - A psi_MS) per slab,
  orthogonalized, SVD, appended to U1_s. "exact" = dense block-tridiagonal LU over the fine slabs. trim_c3 t1e-3 + 3
  rounds exact: random probes per arc point (20, tol 1e-2) → N1 1998, T err 0.4–14% (worse than tile cut 1e-5 at
  N1 1939: flat singular values, (I − π)A⁻¹ω is dominated by states outside the window); injection (tol 1e-2) kept
  2 vectors (sv ~0.009) yet T err went 1.5% → 9.5% at Ec + 0.02 (MS T very sensitive to single vectors).
  Retry (run_trim_c3_resenr2.sh, --res-tol 1e-3): injection at 10 energies between the test energies → N1 1316,
  T err median 0.25% max 1.3% (best so far: tile cut 1e-4 N1 1405 median 0.6% max 7%, 1e-5 N1 1939 0.19% / 2.9%);
  converged in 2 rounds (|e|/|ψ| 0.58 → 0.17). `--res-integrate` (arc-integrated (ρ − ρ_MS)ω, 40 global probes) →
  N1 1248, median 1.2% max 9.4%: |(ρ − ρ_MS)ω|/|ρω| stays 0.62, 40 probes ≪ rank of the missing part (all slabs).
- Tile solves use a dense LU when the embedding is > 2% filled (MoS2/HfO2: ~1300 nonzeros per row,
  11 000–16 000 orbital embeddings): 2–5 min per tile instead of ~1.7 h sparse.
- Exact T cached per input set in results_ms/exact/<name>.json keyed by dE with a fixed reference "Ec"
  (--Ec 0.50435 for mos2_hfo2); runs refuse a json with another Ec.
- Goals: T and psi / charge accuracy, both (2026-11-03; corrects "psi, not T" of 2026-11-02). MoS2/HfO2 n400 (N 1636),
  10 meV grid: MS T error 4e-7–4e-5, but |psi − x_MS|/|psi| 2e-3–1.5e-2 (η-channels included; real E ~2x lower),
  charge per atom 99% 1e-3–1.6e-2, max up to 7e-2 (band edge), per slab 1e-5–2e-4, total 1e-7–9e-5.
  Why T is so good: Galerkin (U^H r = 0 to 1e-11) => dt = −φ^H r exactly (φ = adjoint, right-injected state), and the
  MS system is itself a flux-conserving open system, so dt is a near-unitary rotation of t (Ec+0.05: |dt|/|t|
  2.7e-5, Re part 3.5e-9, eigen-transmissions change 2e-6) => T error second order (6e-9).
  Residual is a bad accuracy metric: P psi (best approximation, error 3e-4) has a residual as large as Galerkin's;
  minimal-residual coarse solve cuts |r| 13–50x but makes psi 4–40x and T 1e-2 worse. Keep the Galerkin coarse
  level; the residual (outside span(U), at slab interfaces, high-energy / oxide) must be removed by a fine smoother.
  span(U) holds psi to 3e-4, Galerkin finds it to 1e-3–4e-3 (non-optimality of Galerkin for indefinite A).
- Tile-basis quality metric: occupation-weighted missing tile modes does not track T (trim_c3) and is not what a
  coarse level needs. Use MS error in T / psi / charge, the complement gap min|λ − E| of the local problem (smoother
  conditioning) and the two-level convergence factor.
- Injection: QuaTrEx injects flux-normalized Bloch modes (B B^H = Γ), qtx_ms Γ_L eigenchannels: same span, psi equal up
  to a unitary U (real E: span residual 6e-8, density 2e-8). The 17 extra qtx_ms channels on MoS2 (19 vs 2) come from
  Σ(E + 1e-6 i): at real E Γ has exactly 2 eigenvalues (549, 58; next 1e-10).
- Lead OBC: qttools spectral OBC (QuaTrEx) inside qtx_ms (`msbasis.QttLead`, Device(obc=auto|qttools|folded), drivers
  --obc; needs the qtx extra and QTTOOLS_SRC=<quatrex>/src). Real axis only (|Im z| <= 1e-5); auto falls back to the
  folded lead. The lead cell must be reordered into contiguous primitive cells (`lead_cell_perm`; input order breaks
  the NEVP: recursion error 0.72), numbered inwards (reversed for the right lead) and periodized from the outermost
  primitive cell (the MoS2 lead cell is not exactly periodic: 8.4e-3 eV between cells). Σ vs QuaTrEx 7e-4 (L) /
  5e-3 (R, block-0 cell reused); T 1.24239 / 1.87235 vs QuaTrEx 1.24320 / 1.87182 at Ec+0.05 / 0.15. Not faster
  (≈240 s per side for the 5-cell lead). Exact / scan results with it go to *_qtt.json.
- Smoother for MoS2/HfO2 (smoother_diag, one local step on the MS error at Ec+0.05): closed local solves amplify the
  error (exact LU: slab 7.6x, 6 Å interface windows 5.3x; deflation 2.2x) — in-band E puts local eigenvalues at E
  (Helmholtz-type). Absorbing layer (−i s P S P, 2 eV over 3 Å) on the artificial cuts of windows centred on the slab
  interfaces works: 16 Å windows remove 81% of the zone error (0.55 overall); slab blocks (cuts at the interfaces)
  do not. Device S is not positive definite (interface windows: λ_min −1e-2), so (H, S) deflation fails there.
  Two-level FGMRES (Galerkin MS coarse, cap-Schwarz 16 Å, 7 windows, x0 = MS): Ec+0.05 err 1.2e-3 → 4.7e-7 (16 its,
  49 matvecs) → 6.6e-13 (30 its); Ec+0.2 3.6e-3 → 1.5e-4 (16) → 1.9e-8 (30). Without smoother FGMRES stalls
  (1e-3 / 1e-2). Stationary cycle diverges slowly (×~1.8 per cycle): Krylov needed. Early iterations make T worse
  (6e-9 → 1e-4) before converging. Cost here: 55 s window LUs (dense, up to 8762 orbitals) + ~100 s Python
  iterations vs ~250 s sparse direct.
  Cost reduction (schwarz.py --cap-solver/--cap-spacing/--cap-centres/--cap-dE0, results_ms/qtbm_cmp/mgc_*.log,
  Ec+0.05 / 0.2): complex64 window LU = identical convergence, factor 50 → 29 s; PARDISO 150 s (fill). Windows must be
  ≥ ~16 Å (≈ slab length): 14 Å 40 its, ≤ 12 Å stall even with a clean core/ramp buffer; 20 Å: 23 its. Windows on
  the interfaces only stall (even 24 Å, which leaves only the lead-end strips uncovered); interfaces + the two
  lead-face windows (Σ, no CAP) converge when they overlap (20 Å: 38–40 its, factor 25 s; 16 Å just touching: stall).
  Factor at E + dE0 and reuse: ±0.05 eV costs 1–3 extra its, 0.15 eV 8. All converging setups ≈ 95–105 s/energy
  (factor + Python FGMRES incl. per-iteration diagnostics) vs 250 s exact; with reuse factor ≈ 3–5 s/energy.
  Block FGMRES without diagnostics (`--accel block`), windows in parallel (`--workers 4`, MKL 1 thread): Ec+0.05
  27 its, 26 s factor + 29 s iterations (smoother 38 → 11.5 s vs MKL 4 threads serial windows); post-smoothing only
  doubles the iterations. Outliers (direct n400): Ec+0.31 62–71 its, Ec+0.39 80 its (err 0.28 → 1e-6), 600–780 s
  for 30–34 columns (η channels) vs 280–300 s LU; σ_min(A_c) at Ec+0.31 2.4e-4 (coarse level at E, Im −2e-7).
- Tile-built levels only (no direct basis; `run_ms --save-basis --basis-only`, `qtx_ms/vcycle.py`, schwarz `--basis
  --l1`): tile_t1e-5 N1 1686 / N2 933, t1e-6 N1 2440 / N2 1018 (level 2 from the level-1 contour ρ, 24 s). Fine
  smoother cap 16 Å, its at Ec+0.05 / 0.2: coarse U1V 53 / 70 (t1e-6 55 / 78), coarse U1 exact 37 / 51 (both cuts),
  V-cycle with level-1 Schwarz (slab-pair windows, projected CAP) 58 / 76; direct n400 27 / ~40. Level-1 Schwarz
  amplifies the level-1 error 20–300x per cycle (closed or CAP 0.5–5 eV): tile vectors span whole slabs in x, so
  level-1 windows can only be cut at slab boundaries (the fine slab-block failure); a real level-1 smoother needs
  tiles cut along x as well.
  Lead-Σ tiles + end-slab lead RMS (results_ms/run_tobc_basis_mos2_hfo2.sh, qtbm_cmp/run_vcycle3.sh, mgv_*_tobc_*):
  t1e-6 N1 2301 / N2 995, its at Ec+0.05 / 0.2: coarse U1V 49 / 67, U1 exact 36 / 49, V-cycle 62 / 77; t1e-7 N1 3399 /
  N2 1074: 57 / 78, 39 / 49, 70 / 84. Same as the old CAP tiles (better MS1 T does not mean faster convergence); the
  level-1 Schwarz is still the weak part (V-cycle worse than skipping it).
  Level-1 windows cut transversely along the tile cuts (schwarz `--l1-split tile --l1-tov --l1-wmin`, t1e-6, Ec+0.05 /
  0.2): all converge to 1e-10; overlap 2.5 Å keeps the windows full size (tile modes fill the 11–14 Å tiles) 55 / 76 its;
  no overlap (windows 229–639 vs 1125–1679) 74 / 96. Tiles cut in x too (run_ms `--tile-xsplit 2`, 16 lead-Σ tiles,
  results_ms/run_tobc_tx2_mos2_hfo2.sh; t1e-6 N1 3705 / N2 1123): coarse U1V 69 / 92, U1 exact 46 / 58, V-cycle slab-pair
  level-1 windows 53 / 68 (first case where the level-1 smoother beats skipping it), 16 Å level-1 windows 95 / 123.
  Fine smoother on tile windows (schwarz `--cap-centres tile --cap-xsplit --cap-tov`: tile + tov Å shell, CAP on the
  outer d of the shell), 8-tile basis, Ec+0.05: stalls (150 its, residual 1e-4…1) except 8 tiles + 4 Å (145 its); even
  with exact level 1. Culprit = the cut in y (in the MoS2 plane, tiles 11 Å wide): 16 Å x-windows + --cap-split 2,1 (y)
  150 its 2e-8, + 1,2 (z, out of plane, windows ≤ 6046 vs 8762) 82 its; half-slab x-windows 104. In-plane cuts need ≥ 16 Å
  windows like x; tiles for fine windows must be cut out of plane (z) only.
- Tile oracle (`qtx_ms/tile_oracle.py`, results_ms/tile_oracle_c3.*; trim_c3, tile basis = eigvecs of the exact
  open-device ρ_tt from the direct-L2 top-800 slab ρ): MS1 T err median / max at cut 1e-5 N1 1525 3.1e-4 / 7.8e-3,
  1e-6 N1 1818 3.1e-5 / 1.5e-4, 1e-7 N1 2094 1.7e-6 / 2.9e-5 (≈ untiled slab ρ at 25–30% more N1); CAP tiles 1e-6
  N1 2818 6.3e-4 / 4.0e-3. Tiling costs nothing; the CAP embedding is the level-1 floor. CAP tiles miss exact
  occupation 3e-3 in tile 8 (slab 5) and 2e-4 in the end tiles at every cut (lead faces); interior tiles 1–4x cut.
- Lead Σ in the tile embedding (`tile_modes(obc=dev)`: lead-block atoms dropped, −Σ_L/R on the embedded face
  orbitals, CAP on the other cuts) for the 7 tiles whose 12 Å embedding reaches a lead face: missed exact occupation
  ≤ 1e-5 everywhere (tile 8 3e-3 → 1e-5), cut 1e-6 N1 2754, MS1 median 3.9e-4 max 2.2e-3 (CAP 6.3e-4 / 4.0e-3).
- 5-cell MoS2/HfO2 level 1 from lead-Σ tiles (run_ms `--tile-obc 24 --tile-obc-dir`, `--tile-worker k,n` for parallel
  tile builds; results_ms/run_tobc_mos2_hfo2.sh, mos2_hfo2_nocore_tobc24_t*.log): all 8 tiles reach a lead face within
  12 Å, so every tile gets the exact 24-pt ρ with Σ (Σ shared with the level-2 cache), 5–60 min per tile, 1 h on
  4×6 threads. MS1 T err on the 51-pt grid Ec+0…0.5 (median / max, # > 1e-2): cut 1e-4 N1 1150 2.7e-3 / 0.11; 1e-5
  N1 1563 9.9e-4 / 4.0e-2; 1e-6 N1 2264 1.9e-4 / 2.9e-2 (3 of 51 > 1e-2: Ec+0, +0.01, +0.23). Old CAP tiles (tc6,
  sketch 600): 1e-4 N1 1212 8.5e-3 / 0.37, 1e-5 N1 1686 6.3e-3 / 0.22 (20 of 51 > 1e-2). MS2 (N2 954) ≈ MS1.
- Level 1 with the slab partition (one block per slab, `--tile-orbs 100000`, CAP+OBC, 24-pt exact local ρ) on
  mos2_hfo2 (2808 atoms; results_ms/run_slabl1_mos2_hfo2.sh, mos2_hfo2_slabl1_tobc24_t*.log; slabs 0 / 3 = the single
  tiles of the tile partition; slab embeddings 9744 / 16998 / 16120 / 7194 orbitals, 28–33 min per slab at 12 threads).
  51-pt grid, raw level 1, MS1 median / max (# > 1e-2), MS2: cut 1e-5 N1 933 1.1e-3 / 0.20 (3), MS2 1.3e-3 / 0.14;
  1e-6 N1 1305 1.5e-4 / 2.5e-2 (1), MS2 4.4e-4 / 4.8e-2; 1e-7 N1 1831 3.3e-5 / 5.5e-3 (0), MS2 3.2e-4 / 2.0e-2.
  Tile partition (8 tiles): 1e-5 N1 1563 9.9e-4 / 4.0e-2, 1e-6 N1 2264 1.9e-4 / 2.9e-2: slabs reach the same MS1
  accuracy with ~40% fewer columns (no transverse tile cuts), N2 ≈ the same (823–904).
  Lower cuts (slabs rebuilt with the fast path, 4 × 6 threads: 116 / 708 / 612 / 67 s, i.e. slab 1 708 s vs 1961 s
  at 12 threads serial; run_slabl1_low_mos2_hfo2.sh): MS1 median / max 1e-8 N1 2373 5.2e-6 / 3.4e-4; 1e-9 N1 2982
  6.9e-7 / 1.9e-5; 1e-10 N1 3658 2.7e-7 / 8.1e-6 (below the 1.2e-4 quadrature error of ρ, still converging).
  MS2 (l2-thr 1e-5, N2 889–913) stays at median 1.6–3.7e-4, max 1.7–2.6e-2: level 2 sets the floor now.
- Accelerated local ρ (`msbasis.tile_rho_contour`, `tile_modes(fast=)`, run_ms `--tile-fast schur --tile-env-prec
  --tile-refine --tile-emb-far imz:d,... --tile-pts-workers --tile-pts-threads`; bench `qtx_ms/tile_fast_bench.py` →
  results_ms/tile_fast/): G_PP = (A_PP − A_PE A_EE⁻¹ A_EP)⁻¹ with A_EE LU in complex64 + refinement (sparse residual),
  smaller embeddings for points with large Im z, contour points over forked processes (threadpoolctl). Slab 0, 4 × 6
  threads, vs exact local ρ (637 s serial, 6 threads): whole-embedding LU 161 s, Schur double 115 s (both exact to the
  1e-8 reference truncation) -> use `--tile-fast schur --tile-env-prec double --tile-pts-workers 4 --tile-pts-threads 6`.
  Single LU + 1 refinement 132 s (dense residual; 356 s with the single-threaded sparse one), ρ 1.3e-8, same counts:
  no gain for slabs (m ≈ n_E, solves + residual GEMM ≈ LU); no refinement 88 s but ρ 8e-4 (counts change). Smaller
  embeddings off the axis fail: 6–8 Å at Im z ≥ 0.1 changes ρ by O(1), 2x the kept modes (arc radius 0.375 eV, G not
  yet local).
- Spurious-state removal on that level 1 (results_ms/run_tobc_lc_mos2_hfo2.sh, run_tobc_enr_mos2_hfo2.sh; figures
  results_ms/figures/lead_bands_tobc24_t*_slab*.png via results_ms/plot_lead_bands_tobc24.py). End slabs, lead-band RMS
  (`--lead-clean`): raw level 1 has ~8 dispersive spurious bands in the gap (−0.2 eV … Ec) and more above Ec; RMS
  converges in 16–22 steps, 252/293 → 0 (cut 1e-6, +17/+20 columns), 363/441 → 2 (1e-5). MS1 T err cut 1e-6:
  median 1.9e-4 → 2.5e-4, max 2.9e-2 → 7.4e-3 (0 of 51 > 1e-2; Ec+0 2.9e-2 → 7.4e-3, Ec+0.01 1.3e-2 → 1.7e-4); 1e-5
  max 4.0e-2 → 2.5e-2. Level 2 brings 300–500 spurious lead bands back (RMS columns forced in, but the cut drops
  what they pair with): MS2 2–3x worse than MS1. Interior slabs: dropped (user decision 2026-11-03). Lead-band RMS
  on the matched MoS2 atoms (`--lead-clean-slabs all`) sees ~750 flat in-gap levels (oxide-localized, not lead
  bands), stalls at 150 steps (771 → 310, matched-band error 31 meV); device residual (`--enrich 3 --enrich-slabs
  interior`, closed s±1 clusters) flags 85–90% of the Ritz pairs every round (|r| 1–3 eV, histogram unchanged) =
  plain basis growth (+300–500 per round): 1e-6 N1 3474 median 1.0e-4 max 4.9e-3, 1e-5 N1 2951 5.0e-4 / 8.4e-3.
  End-slab RMS only at cut 1e-7 (N1 3399, RMS 201/209 → 0): MS1 median 4.4e-5, max 2.4e-2 (single outlier Ec+0.05,
  next 3.1e-3); MS2 (N2 1074) median 3.3e-4 max 1.6e-2.
  Level-2 lead RMS (`--lead-clean-l2`, results_ms/run_tobc_lcl2_mos2_hfo2.sh): RMS on U1 V of the end slabs, candidates =
  level-1 complement U1(1 − VV^H) on the cell (keeps U = U1 V nested). 1e-7: 492/496 → 0/2 (+134/+72 columns, N2 1280),
  MS2 median 3.3e-4 → 1.8e-4 but Ec+0 2.7e-1 (was < 1.6e-2; 3 of 51 > 1e-2). 1e-6: candidates run out (81/39), 434/525
  → 221/336, MS2 median 4.2e-4 → 2.7e-4, max 1.8e-2 → 8.5e-3 (46 of 51 energies at the time of writing).
  RMS.run now re-orthogonalizes Cs and stops when a new direction lies in their span (the 1e-6 run crashed with
  an indefinite reduced S after 82 stalled steps).
- Tile ρ by eigen-decomposition instead of LU per contour point (validated on a 382-orbital slab-0 tile, 6 Å
  embedding, 2286 orbitals, 1083 lead-face orbitals): K = H − iW (+ Σ₀ on face F), QZ K v = λ S v, vᵀSv = 1,
  G₀ = V(z−Λ)⁻¹Vᵀ, Σ(z) − Σ₀ by Woodbury on F, ∫G₀ analytic (Log(μ₂−λ) − Log(μ₁−λ)). Same quadrature as LU: 4e-11;
  analytic G₀ with Σ₀ = 0 or Σ(z near axis): 1.2e-4 = the 24-pt quadrature error of the LU tile ρ (CAP only: same
  1.2e-4). Σ₀ at the arc top: 14 eigenvalues with Im λ > 0 break the Log formula (5.5e-3). Cost: QZ 450–500 s vs
  24 LUs 32 s (2 threads), both O(n³): not viable for 15k-orbital tiles unless reused for ≥ ~15 windows / bias points.
  Iterative solves for random probes: GMRES (none / Jacobi) > 3000 its to 1e-6 at Im z 0.003–0.37 on tile 1 (15398 orb).
- Self-consistent tile update (`qtx_ms/tile_scf.py`; level-1 RGF `msbasis.rgf_blocks`, tile cavity
  Δ_t = A_tE U G1^(-t) U^H A_Et with tile t removed from G1 by Schur, new U1_t from ρ_tt; also level 2 from G1):
  does not converge (7 iterations, 3.6 h). The cavity sees the environment only through span(U1): the missing
  complement lets tile boundary states fall into the window (ρ_tt trace 1.1–2x exact, 50–100 extra eigenvalues
  > 1e-6, λ_min −4e-4), the update moves 0.3–1 outside the basis every iteration, N1 grows 2754 → 3236, interior
  missed occupation 1e-5 → 3e-5–1e-4 (only the lead-end tiles improve, 1e-6), MS1 median 1e-4 (max erratic up to
  4.8e-2), MS2 ≥ 6.6e-4. Not pursued: Δ_t = Δ_t^loc (CAP / lead-Σ embedding) + level-1 correction in span(U1).
- Tile basis from the local nonlinear OBC eigenproblem (Beyn on the lead-Σ tile embedding, `qtx_ms/tile_beyn.py`,
  results_ms/tile_beyn/; contours gap / c1 / c2 / c3 between lead edges, 400 probes): dropped (user decision
  2026-11-04), keep the contour density matrix. End tiles 7 / 0: c1 0 eigenvalues inside; c2 on tile 7 29 eigenvalues
  with residual 1–12 eV, several Im > 0 (dense CAP / lead-broadened continuum poles at Im −0.005…−0.03 cut by the
  contour, trapezoid filter fails); no eigenpair below 1e-3 eV residual. A pole basis misses the band continuum that
  ρ_tt (integrated above the poles) captures smoothly.

- OFF state with the flat-band bases reused (`prep/gate_barrier.py`: 0.4 eV tanh barrier over slabs 1-2, w 2 Å,
  inputs/mos2_hfo2_gate0.4; results_ms/run_gate_off.sh, results_ms/gate_off/, report fig12): every basis fails below the
  barrier top (Ec+0.4). T err median / max on the 20 meV exact grid: slab level 1 1e-6 0.24 / 430, 1e-8 0.08 / 41, 1e-10
  0.011 / 22; tile level 1 1e-6 0.40 / 69; slab level 2 cut 5e-14 0.05 / 100 (Ec+0: T_MS 8e-7 vs 7.7e-9). Narrow spurious
  resonance spikes (3–1000x the neighbours on the 10 meV grid) = mode-space states shifted into the barrier gap opening
  tunnelling paths (user's concern confirmed). Linear-response G at EF = Ec−0.1…+0.1 (kT 25.9 meV) off by +220…+860%
  (best bases) up to 50–300x; above the barrier top errors return to 1e-4–1e-2. Stand-alone MS needs a bias-aware
  basis (or the two-level FGMRES correction) for OFF-state / subthreshold work.
  Spurious-state fixes on the slab level 2 n400 basis (results_ms/gate_off/{spurious,deflate,enrich}): the OFF-state
  spikes sit on narrow MS resonances (Ec+0.0598 −5.5e-6i, 0.1694, 0.3599) with fine residual 11–13 eV spread over all
  slabs; physical resonances 0.2–3 eV (closed MS eigenpairs do not separate: 2–12 eV). Found per 10 meV point with
  Coarse.ritz (k 12, thr 5 eV): 24–28 (n400), ~50 (slab L1 1e-8). Deflation (`ms_deflate.py`, S_c-complement): gated
  T 100x too large, flat err 1e-6 → 1e-3. Absorbing iηS_cYY^HS_c: flat err 1e-4…9e-2 (η 1e-3…1e-2), gated worse for
  η ≥ 1e-2 (up to 1e4x: the nonlocal absorber bridges the barrier). Residual enrichment (`spurious_enrich.py`, slab
  blocks of the resonance residual appended, 3 rounds): flat band unchanged (2e-6 / 4e-5), gated median 5e-2 → 5e-3–2e-2
  (raw) but max stays 1.4–230 (block-Jacobi corrected: up to 4e3): every round leaves 7–40 new spurious resonances.
  Basis built AT the gate potential (results_ms/run_gate_basis.sh, results_ms/gate_off/gate_basis/, report fig13) works:
  T err median / max on the 20 meV grid: slab level 1 1e-6 (N 1140) 1.1e-3 / 0.11; 1e-8 (N 2250) 1.4e-4 / 1.1e-2;
  1e-10 (N 3554) 4.3e-6 / 2.1e-4; slab level 2 n400 + lead RMS (N 1639) 1.3e-6 / 7.8e-2 (max at Ec+0, T 7.7e-9; next
  1e-3). G at EF = Ec−0.1…+0.2 exact to < 0.1% (1e-8, 1e-10, L2), 1.5% (1e-6). Build: slab level 1 1426 s (4 × 6 threads),
  level 2 rho eigendecomposition 122 s with the contour Sigma cache shared (same leads). => the basis must follow the
  potential (bias / Poisson loop: rebuild or update per iteration); spurious states are harmless when they sit where
  the basis was built.
  Two-level block FGMRES in the OFF state (results_ms/gate_off/fgmres/run_mg_gate.sh; CAP-Schwarz 16,2,2, 7 windows,
  dense64, 4 workers; fig14 windows but ramp 2 Å instead of 3; coarse = Galerkin MS in slab level 1; tol 1e-10; its min..max over columns at
  Ec+0.05 / 0.20 / 0.45 (19 / 24 / 36 columns)): flat-band 1e-6 (N 1305, x0 T err +7.1 / +17 / +0.55) 23-27 / 30-35 /
  44-49; flat-band 1e-8 (N 2373) 22-26 / 26-31 / 40-44; gate 1e-8 (N 2250, x0 T err 3e-4 / 6e-3 / 7e-6) 24-28 /
  37..150 (stall 1e-7…1e-10: spurious MS resonance at Ec+0.20, MS residual 12.6, cf. 0.04 at Ec+0.21) / 41-47. All
  others reach err 3-5e-11, T 1e-9. => the FGMRES correction removes the OFF-state MS failure at flat-band iteration
  counts (no bias-aware basis needed); a coarse level sitting on a narrow spurious resonance is what hurts.
  Cost: factor 26 s + iterations 115-420 s (Python, 1 thread per window) vs exact sparse LU 390-470 s.
  Follow-up (run_mg_gate2.sh, report fig15): one-level FGMRES (cap smoother only, x0 = 0, qtx_ms.cap_fgmres) converges
  too: 30-40 / 41-54 / 60-75 its, 112 / 172 / 451 s, i.e. ~1.5x the two-level its but the same or less wall time (a
  two-level iteration = pre + post smoothing + coarse + 3 matvecs, a one-level one = 1 smoothing + 1 matvec): in this
  short device the mode-space coarse level buys little. Coarse shift (schwarz --coarse-eta, A_c at E + i eta_c, fine A at
  E): gate 1e-8 at Ec+0.20 stall -> 32-37 (1e-3) / 34-37 (1e-2), elsewhere unchanged (±1-4 its) although x0 gets worse
  (T err 3e-4 -> 6e-2 / 0.46): cheap fix for spurious coarse resonances. Window LU reused off-energy (--cap-dE0, flat
  1e-8): −0.05 eV +4 / +2 / −4 its, +0.05 eV +4 / +3 / +14, +0.10 eV +10 / +23 / +36 (factor below E is better: the
  barrier region is less resonant).
  Energy sweep Ec + 0:0.5:0.05 (run_mg_gate3.sh, report fig16; its max over columns, wall time 6 jobs x 4 workers):
  one-level 40-61 below the barrier top, 69 / 75 / 97 at 0.40 / 0.45 / 0.50, 110-330 s; two-level (flat 1e-8, pre +
  post) 26-38, 41 / 44 / 55, 140-370 s; two-level pre-smoothing only 50-60, 61 / 63 / 69 (≈ one-level its, more time).
  The coarse level saves ~40% of the iterations at every E but costs more per iteration, so one-level is faster below
  the barrier top and two-level only at Ec+0.45 / 0.50 (more channels, longer Krylov). Post-smoothing is essential.
- FEAST probes for the level-1 local ρ (`qtx_ms/feast_bench.py`, results_ms/feast_bench/, report fig17; slab 0, 3483
  orbitals, CAP+OBC embedding 9744, 24-pt arc, one dense LU per point): Q = Σ w_e G(z_e) R, S-orthonormalized
  (Cholesky S_PP + SVD; the Gram eigh squares the condition number and lost half the columns), R = S_e Y (feast),
  S_e Y_P (feastP), S_e^1/2 Y (feastL; S_e is indefinite, λ_min −1.9e-2, 10 eigenvalues clipped) vs density matrix
  (identity RHS on P, eigh ρ_PP; k 319 / 550 at cut 1e-6 / 1e-8) vs range finder (Ω on P, Nyström). Missed exact
  occupation at cut 1e-8: range finder M 550 6e-7, M 900 1e-11 (k 550/550); all FEAST variants saturate in M
  (M 3000: own cut keeps 175–264 of 550, miss 6e-4–3e-3; top-550 by count 3e-7–6e-6). Weighting S Y vs S^1/2 Y
  makes no difference: any probe spread over the embedding ranks P directions by their environment coupling, not by
  ρ_PP. FEAST equals the range finder only for R = Y on P without S, which is the range finder itself. MS T (slab 0
  replaced, rest slab level 1 1e-8): all 550-column sets median 3–7e-6, max 0.5–4e-4 (dm 5.9e-6 / 3.4e-4); at cut
  1e-6 feast M400 2.2e-4 / 2.5e-2 vs rf M400 7.5e-5 / 3.3e-4. Cost: 24 LUs 5.9e13 flops (332 s summed), solves
  1.8e10 flops per column (dm 3483 columns 6.3e13, 296 s; M 3000 257–286 s); the range finder at M 900 saves ~75%
  of the solves (≈ 40% of the pass). Keep the range finder (Ω on P); FEAST probes bring nothing.
## Code state (package `qtx_ms/`; Python 3, numpy, scipy only)
All modules live in `qtx_ms/`; drivers are run from the repo root as `python -m qtx_ms.<name>`
(outputs written to the current directory). Package `mgms` renamed `qtx_ms` (distribution `qtx-ms`) on 2026-11-02;
the old single-file `mgms.py` is `qtx_ms/em.py`. The folded-lead OBC value `--obc mgms` is now `--obc folded` (Device still accepts
"mgms", e.g. from old psi files); json keys like `T_mgms` keep the old name.
Effective-mass track
- `device.py`  — `Device`: 3D finite-difference rough sheet (default 20×5×3 nm, a = 0.25 nm, m* = 0.26,
  3 eV oxide, Gaussian-correlated top/bottom roughness rms 0.3 nm Λ 1.5 nm, ideal ends), analytic uniform-lead
  Σ via transverse modes, `gamma_inj` (QTBM columns, padded to full length), `rgf_column` (exact T).
- `em.py`      — bases `basis_cms` (lowest M slab eigvecs), `basis_ldm` (local density matrix from Neumann-ended
  (2w+1)-slab clusters, states below E_b), `TwoLevel` (Chebyshev smoother, balancing preconditioner, GMRES).
- `run.py [L]` — sweep of bases × Chebyshev degree vs RGF → `results_L{L}.json`, `log_L20.txt`.
- `scaling.py` — device-length scaling (10/20/40 nm), no coarse vs LDM E_b = 1 eV.

Ab initio TB track
- `abtb.py`    — readers (`read_xyz`, `parse_norb` int | 'Si:10,H:1' | per-atom file, `read_sparse` .npz/.mtx/
  text triplets, 0/1-based auto, upper-triangle completion), `TBDevice` (sort/partition, block-tridiagonal
  assert, lead periodicity check, non-orthogonal Sancho–Rubio Σ, lead bands), `injection`,
  `rgf_transmission` (variable slab sizes), bases `basis_slab_window` (count-based), `basis_ldm` (generalized
  eigh of clusters in window).
- `twolevel.py`— generic `TwoLevel(A, U, smoother='gmres'|'cheb', k)` with flexible GMRES, `gmres_fixed`,
  `sigma_min_coarse` (spurious-band monitor), `build_A` (sparse A with dense Σ blocks).
- `run_tb.py`  — CLI driver: read → partition → lead bands (Ev, Ec) → bases → per-energy solve vs RGF →
  `results_tb.json`. See README for flags.
- `make_test_tb.py` — synthetic non-orthogonal test device (rough H-passivated wire, Si 2 orb, H 1 orb,
  1st+2nd-neighbour H, NN overlap, gap ~3 eV, shuffled atoms, upper-triangle 1-based text files).
- `README.md`  — usage and result tables.

Two-level mode space from the generic inputs (the qtmode method ported; any input set)
- `msbasis.py` — `Device` (slabs = device blocks, `FoldedLead` exact Σ(z) for complex H / any S, Σ cache),
  `CSolver` (PARDISO real-equivalent + refinement + GMRES fallback), level 1: `partition` (cores / tiles by
  transverse bisection), `BulkFromDevice` + `core_basis` (bulk Bloch cores, H(k) read off the device H),
  `tile_modes` (CAP-embedded local spectral function, randomized range finder, SVD merge over energies),
  `slab_U`, `slab_blocks`; level-1 enrichment: `enrich_round` (see Decisions); level 2: `rho_contour`
  (contour slab ρ of the open level-1 system), `level2_basis`; coarse solvers `BlockTridiag`, `MRCoarse`.
- `run_ms.py`  — level 1 (± cores, ± `--enrich n --enrich-width w`) → level 2 → MS1 / MS2 / MG1 / MG2 vs exact;
  `--direct-l2` (U⁽¹⁾ = I), `--l2-metric S`, `--mg` levels, `--exact-from` (skip exact solves). Caches tiles,
  cores, enrichment rounds, ρ in results_ms/<name>[_nocore|_l2direct]/.
- `diag_mg.py` (two-level solver diagnostics), `check_rho_lead.py` (contour ρ vs k-integral on the lead),
  `plot_hs.py` (H, S vs mode-space H, S), `cms_union.py` + `run_cms.py` (conventional vs union-cross-section
  CMS on the Si sheet), `prep/mos2.py` (GPAW LCAO MoS2 inputs).
- `tests/test_msbasis.py` — unit tests (synthetic strips, real / complex gauge, orthogonal / non-orthogonal).

QuaTrEx / MoS2-HfO2 track (see README section "QuaTrEx devices")
- `prep/quatrex.py` — deviceBuilder folder → generic inputs (`--cells`, `--trim`, `--cut`, `--cut-S`, `--potential`);
  `detect_periodic` writes the transverse periods into meta["periodic"]. `prep/set_periodic.py` patches old meta.json.
- `exact_T.py` — exact QTBM T by sparse direct solve; `--json` merges into results_ms/exact/<name>.json, `--Ec`.
- `run_l2_clean.py` — direct level-2 basis (`--nmodes` sweep), end-slab RMS cleanup, optional block residual
  enrichment, MS T vs exact, `--mg` two-level GMRES; `--exact-from`, `--Ec`; saves modes/<tag>_slab<s>.npy.
- `run_lead_cms.py` — lead-CMS basis on lead-like atoms of every slab (+ `--mg`). Negative result. Revisited 2026-11-04 with lower cuts
  (results_ms/run_leadcms_sweep.sh, window Ec −0.15…+0.6, 64 k): the lead-cell ρ (trace 56) decays fast, N per cell
  only 204 / 245 / 291 at 1e-7 / 1e-9 / 1e-11 (+8–10 RMS); T err median 0.21–0.27 on the 51-pt grid, flat in the cut
  (old N 84: 0.33). Representability-limited (no coefficients on ~880 unmatched oxide atoms, no slab-face vectors).
- `lead_rms.py` — lead bands in a slab basis, spurious-band removal (Huang et al. 2018, non-orthogonal S).
- `ms_scan.py` — saved basis on a fine energy grid vs exact (coarse-first order, exact json reuse), MS residual.
- `prep/compare_qtbm.py` — QuaTrEx QTBM (generate_inputs style) vs qtx_ms: injection, psi fit, T (two stages / envs).
- `plot_psi.py` (QuaTrEx vs qtx_ms, or exact vs MS with --modes), `plot_residual.py` (r, e maps; `ms_solution`).
- `density_scan.py` — exact vs MS on an energy grid, L + R injection: T, residual, psi error, |(I−UU^H)psi|,
  Mulliken charge per atom / region; workers share the grid; `summary` → summary.png, span.png.
- `tile_oracle.py` (tile bases from the exact slab ρ vs CAP tiles), `tile_scf.py` (lead-Σ tile embedding +
  self-consistent tile update via level-1 RGF; negative result, see Decisions).
- `feast_bench.py` — level-1 local ρ of one block: density matrix vs range finder vs FEAST probes (accuracy, flops,
  time, optional mode files for ms_scan).
- `schwarz.py` — overlapping Schwarz smoother along x (local closed eigensolves, states within delta of E deflated
  to the coarse level, open LU windows at the leads) + multiplicative two-level cycle vs exact psi.
- `release/qtx_modespace/obc_eig.py` (0.2.0.dev0) — full-size OBC eigenmodes (continued Σ, Beyn, span vs --modes, --plot);
  tests/test_obc_eig.py (1D chain: closed-form Σ, winding number); MoS2 c1 reproduces qtx_ms (λ, vector identical).
- `release/qtx_modespace/` 0.2.0 (2026-11-05, in preparation): + `level1.py` (slab / tile partition with --tile-orbs,
  --tile-xsplit; local ρ per block, CAP or CAP + lead Σ embedding, Schur G_PP, forked contour points, block cache;
  `--basis l1`), `schwarz.py` (CAP-Schwarz x-windows ± transverse split, block FGMRES, two-level cycle; `--fgmres`,
  `--fgmres-coarse ms|none`), `read_quatrex` detects meta["periodic"]; tests/test_level1.py (slab OBC with a
  whole-device embedding = global ρ_ss to 1e-10). No FEAST / range finder (user, 2026-11-05). Real-axis OBC =
  qttools by default at real E (`--eta 0`, 2 channels at Ec+0.05), folded lead at E + 1e-6 i only as fallback
  (user, 2026-11-05). `--system closed` (A0 = zS − H, no Σ on fine and coarse level, unit RHS on the contact-slab
  orbitals, chunks). `schwarz.FastOp` (real split + row threads: 64-col matvec 2.8 → 0.5 s). CapSchwarz workers limit
  BLAS to 1 thread (threaded MKL + concurrent LU segfaulted). Example: release/qtx_modespace/examples/
  mos2_hfo2_workflow.sh (A slab CAP+OBC 1e-8, B slab CAP 1e-8, C tile CAP+OBC 1e-6, D FGMRES 2/1-level, E closed);
  run results_ms/release020/wf: T err A ≤ 2.5e-4, B ≤ 4e-5, C ≤ 2.4e-4 (Ec+0.05/0.15/0.2); FGMRES two-level 25/35/41 its,
  one-level 55/61/84, T 1e-12–2e-11; closed 128 cols two-level 30–52 its, one 46–87, err ≤ 6e-11.
- `release/qtx_modespace/` — self-contained package (numpy/scipy/h5py, optional pypardiso): QuaTrEx folder →
  contour ρ → basis npy (+ QuaTrEx orbital indices) → optional T_MS / T_exact. Checked bit-equivalent to qtx_ms
  (results_ms/release_check/). `obc_eig.py` (0.2.0.dev0): full-size OBC eigenmodes (continued Σ, Beyn, span vs
  --modes, --plot), tests/test_obc_eig.py (1D chain: closed-form Σ, winding number); MoS2 c1 reproduces qtx_ms exactly
  (results_ms/obc_eig/release_c1).

What works (verified)
- Effective mass, 20 nm (25.6k points): LDM E_b = 1 eV (N = 799) → 7 GMRES its / ~76 matvecs per RHS,
  final T error ~1e-11 (MS alone 1.2%); no coarse: 42 its / 459 matvecs.
- Length independence: 7.0 its at 10/20/40 nm with LDM; no coarse 33 → 46 → 76 its.
- TB synthetic (5928 orbitals): LDM (N = 221) 8–9 its, final error ~5e-10; slab M=4 10–12 its; no coarse 56–90.
- Reader: .npz / .mtx / text, per-species / per-atom orbitals, shuffled atoms all give identical T.
- MoS2/HfO2 (29 088 orbitals total = lead block 3120 + device 22 848 in slabs 3483/8383/7862/3120 + lead block 3120): level 2 n200 (N 800) T error ≤ 1e-4; n400 + end-slab
  cleanup (N 1636) ≤ 4.3e-5 on a 10 meV grid 0–0.5 eV; n100 15–40%. 3-cell trim n200 ≤ 3e-5.
  MS solve ~1 s vs exact 250–650 s per energy; lead Σ 170–300 s per energy; basis ~1.5–2 h (contour Σ-dominated).

Input preparation (`qtx_ms/prep/`, see README)
- `inputs.py`  — `write_inputs` / `read_inputs` of the generic input set.
- `si_sheet.py`— CLI `python -m qtx_ms.prep.si_sheet --model sp3s|sp3d5s --xsec 3x5|3x15|5x10 --out DIR`;
  also writes `structure.npz` + TB params; `load_structure(dir, meta)` → (Structure, P).
- `structure.py`, `hamiltonian.py` (sp3s*), `hamiltonian5.py` + `sp3d5s.py` (sp3d5s*), `model.py` — moved
  unchanged from qtmode (only `model.device_cfg(xsec=None)` gained an argument). `model.select()` still mutates
  `structure.NORB` globally. Verified: H, lead H00/H01 bit-identical to ../qtmode_cms (3×5, both models).

User's two-level mode-space code (`qtmode/`, from ../qtmode_cms; pipeline scripts live there, not here)
- `cms.py` level 1 (partition into cores/tiles, tile spectral modes, union core basis), `enrich.py` residual
  enrichment, `lsq.py` slab blocks / transmission, `lead_ms.py` reduced leads, `transport.py`, `negf.py`,
  `reduced.py`, `clean.py`. Imports structure/TB from `qtx_ms.prep`. Not yet wired to the generic inputs.

Untested / known limitations
- No real ab initio files yet (CP2K/Wannier/OMEN; sp3d5s*, d orbitals, spin, complex H).
- The user's two-level basis is ported to the generic inputs (`msbasis.py`, `run_ms.py`), but only as a nested
  basis U = U⁽¹⁾U⁽²⁾ for a two-grid solve; the `qtmode/` package itself is still not wired to read_inputs.
- Only two grids; no three-level V-cycle.
- LDM build uses dense (generalized) eigh of (2w+1)-slab clusters: fine here (1.5 s TB, 76 s EM) but will not
  scale to sp3d5s* slabs with thousands of orbitals.
- Python wall times are not representative; compare matvec counts / flops. In the TB test RGF is still faster
  (≈100 orbitals per slab); the advantage is expected for large n_slab.
- A_c rebuilt per energy (U†HU, U†SU not cached); Σ added as dense blocks (dominates nnz for small devices).
- Lead check requires 3 identical end slabs (warns if only 2).
- `run_tb.py` Chebyshev β is a crude estimate; use the GMRES smoother for TB.
- No bias / Poisson / SCBA loop, no energy-adaptive basis, no carrier density (needs diag(G) or many RHS).
- Level-1 enrichment (`msbasis.enrich_round`, run_ms `--enrich`) uses local closed slab clusters (default
  s±1) instead of the global H1 eigenproblem; tested on synthetic strips only so far (MoS2/HfO2 run pending).
- Spurious coarse bands only monitored (σ_min(A_c)); no mitigation implemented.
- Two-level GMRES on MoS2/HfO2 stalls (residual 3e-3 → 2e-4 in 60 its, independent of smoother k = 1…60): the
  GMRES(k) smoother is ineffective there; the release ships stand-alone MS T only.
- Lead Σ (folding + Sancho–Rubio in Python) dominates every MoS2 run.

## Open problems
1. Implement the nested two-level basis U = U⁽¹⁾U⁽²⁾ and a three-level V-cycle (smoother on level 1 with A₁)
   to cut fine-grid matvecs — the user's level-1 construction is in `qtmode/` (cms.py tiles/cores, enrich.py).
2. Cheap basis construction for TB (sparse/shift-invert or randomized range finder for ρ in the window;
   reuse across energies and bias points).
3. Smoother for indefinite TB complement: compare GMRES(k) vs two-interval Chebyshev / least-squares
   polynomial on the gap-separated spectrum; cost per iteration. On MoS2/HfO2 GMRES(k) fails outright:
   try slab block-Jacobi / additive Schwarz (exact slab solves) as smoother.
4. Spurious bands: detect via σ_min(A_c(E)) and mitigate (complex shift E + iη_c in the coarse solve only,
   or basis filtering).
5. Adaptive enrichment: append smoothed residuals to U (bootstrap AMG) energy by energy.
6. Cost model vs RGF as a function of n_slab, S, number of channels, energies.

## Next steps
1. Done in qtx_ms (`msbasis.py`: tiles only for ab initio, cores from the device H for the Si sheet; enrichment
   with local clusters). Evaluate enrichment on MoS2/HfO2 (results_ms/run_enrich_mos2_hfo2.sh: baseline,
   w = 1, global) and on the Si sheet vs the qtmode numbers.
2. Wrap qtmode's U⁽¹⁾, U⁽²⁾ in a 3-level version of `twolevel.TwoLevel`;
   test on the prepared Si sheet inputs vs the current 2-grid numbers.
3. Run `run_tb.py` on a real ab initio device (start small); set `--slab-width` to the lead cell length.
4. Cache U†HU and U†SU; rebuild only A_c(E) = E·U†SU − U†HU − U†ΣU per energy.
5. Replace dense cluster eigh in `basis_ldm` with a sparse/randomized construction.
6. Write up the formulation + results.
7. Faster lead Σ (cache across runs exists; need a faster decimation / eigenvalue-based Σ for 3120-orbital cells).

## How to run
Environment: `uv sync` (Python 3.12, numpy, scipy, pypardiso/MKL on x86-64; `qtx_ms` and `qtmode` installed
editable). Prefix commands with `uv run`, or activate `.venv`.
```
python -m qtx_ms.run 20     # EM sweep (~5 min)
python -m qtx_ms.scaling    # EM length scaling (~5 min)
python -m qtx_ms.make_test_tb # writes test.xyz, test_H.txt, test_S.txt
python -m qtx_ms.run_tb --xyz test.xyz --H test_H.txt --S test_S.txt --norb "Si:2,H:1" --dE 0.02,0.08,0.15,0.25 --Eb 0.6
python -m qtx_ms.prep.si_sheet --model sp3d5s --xsec 3x5 --out inputs/si3x5_sp3d5s   # ~2 min, writes generic inputs
```
