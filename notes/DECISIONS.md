# Decisions and why

Moved verbatim from CLAUDE.md on 2026-11-15; CLAUDE.md keeps a digest. Add new results here.

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
- GPU backend (cupy; `qtx_modespace.gpu` / `qtx_ms.gpu`, `--gpu` or QTX_MODESPACE_GPU=1; plan sessions/2026-11-06-gpu-cupy-plan.txt):
  folded-lead Σ + lead edges, level-1 local ρ (blocks assembled on the device, Schur), level-2 recursion, exact solve
  (block-tridiagonal LU), Schwarz window LUs + block FGMRES. Numpy results out of every module; no fork in GPU mode.
  Hosts: mont-fort15 x86 P100 16 GB (`.venv`, cupy-cuda12x) and attelas2 aarch64 GH200 96 GB, 72 cores (`.venv-aarch64`
  built with /usr/local/anaconda3 python, cupy-cuda13x, CUDA_PATH=/usr/local/cuda-13.3, OpenBLAS; no pypardiso / qttools
  / uv / git there -> folded lead at E + 1e-6 i, 19-24 channels). Release tests on attelas2: 40 pass (CPU + GPU).
  2808a workflow A + D on the GH200 (results_ms/gpu/run_wf_gpu.sh, wf_gh200/): slab level 1 1e-8 154 s total (local ρ
  5 / 96 incl. 24 Σ / 15 / 4 s vs CPU 116 / 708 / 612 / 67 s), same N1 2373; Σ real axis 4 s, exact solve 2 s (CPU
  150-180 s), window factor 2 s (24 s), FGMRES two-level 4 / 5 / 7 s (24-47 its), one-level 4 / 5 / 8 s (46-86 its), T err
  ≤ 1.5e-10. P100: slab 0 local ρ 92 s incl. Σ (2.4-4.1 s per point), = CPU to 7e-14. "max channel" ψ errors of 3-99
  with the folded lead are the η channels (norm 1e-4 of the 2 physical ones).
  9072a (72 672 device orbitals, 10 slabs, wf_gh200_9072a/, before the projection port): slab level 1 1e-8 in 366 s
  (interior slabs embedding ~20 400, 23-25 s each), N1 8053 (925-968 per interior slab); Σ 4 s, exact 7 s per energy;
  MS T err 1.2e-4 / 1.7e-5 at Ec+0.05 / 0.15; FGMRES factor 6-7 s (19 windows), two-level 40-46 / 49-57 / 68-75 its,
  24 / 24 / 34 s, T err 1e-12; one-level 82-93 / 85-100 / 106-128 its, 22 / 22 / 32 s, T err ≤ 1.1e-10 (run totals
  321 vs 188 s: the two-level run paid ~150 s of CPU projection).
  CPU leftovers in GPU mode (cProfile, results_ms/gpu/prof/, 2808a; step A 128 s with Σ cached, D 88 s): Coarse
  projection U^H H U, U^H S U (scipy csr x dense, 1 thread) 32 s per run (9072a, N1 8053: ~150 s), injection eigh of
  Γ_L 2.6 s per energy, psi_errors 1.9 s per energy, mode / cache file writes ~8 s, level-1 embedding slicing + cache
  ~9 s. Projection (`core._slab_blocks_gpu`, cupyx SpMM) and injection (core.eigh) moved to the GPU 2026-11-10: D
  85 → 55 s, same its / T. cProfile's 110 s in matplotlib text metrics is an artifact (wall time unchanged).
  Rerun 2026-11-10 with prep + GPU projection / injection + plots moved out of the run (`python -m qtx_modespace.plot
  OUT...`, cli writes numbers only; first-run logs in results_ms/gpu/run1_logs/), everything recomputed, same N1 / its /
  T: 2808a prep 24 s, A 219 → 162 s, D two-level 85 → 40 s, one-level 83 → 40 s, whole script 4.5 min; 9072a prep 54 s
  (1.32 GB), A 581 → 400 s, D two-level 321 → 144 s, one-level 188 → 139 s, whole script 12.4 min.
- Prepared inputs (release `qtx_modespace.prep`, `quatrex.save_inputs / load_inputs`, console script qtx-modespace-prep;
  user request 2026-11-10): `python -m qtx_modespace.prep DEVICE --out PREP [--cells --trim --potential --gpu
  --sigma-cache]` writes prep.json (options, source file size / mtime, meta), arrays.npz, H.npz / S.npz (upper
  triangle, uncompressed), lead.npz; `read_quatrex(PREP)` (so cli, obc_eig) loads it instead of the QuaTrEx folder.
  2808a: 0.53 GB, load 0.9 s vs 8 s read + lead edges (255 s on the aarch64 OpenBLAS CPU, 15-30 s GPU), H, S, lead,
  index arrays bit-identical (edges differ 1e-13 GPU vs CPU: cache keys round to 8 digits). Conflicting --cells /
  --trim / --potential stop the run. to_quatrex still needs the QuaTrEx folder. Workflow scripts prepare into $OUT/prep.
- 6 nm MoS2/HfO2 (43 200 atoms, 344 640 device orbitals, 22 slabs ≤ 16 858, lead cell 6240; deviceBuilder folder with
  separate [0,±1,0] blocks -> release 0.2.2 fix), GH200, folded lead at E + 1e-6 i (/scratch/jiacao/mg_modespace/6nm/):
  prep 177 s (6.86 GB); slab level 1 CAP+OBC 1e-8 N1 38 054 (1843–1897 per interior slab, 110–130 s per slab,
  embeddings ~40 900), 2670 s; tile level 1 CAP+OBC 1e-6 (4 y strips x 10 Å x pieces, 176 tiles) N1 54 568, 2586 s.
  Device Ec above the lead Ec: T_exact 1.17e-8 / 0.0994 / 0.907 / 4.40 at Ec+0.02 / 0.05 / 0.15 / 0.3 (37–62 channels).
  Slab L1 1e-8 MS T err +7.3e-3 / −6.6e-4 / −1.4e-4 / −8.7e-5; best-in-U psi 2.0–2.6e-4. Per energy Σ 9–11 s, MS 17–19 s,
  exact 137–163 s (residual ≤ 6e-14). The dense GPU block LU needs ~417 GB -> `core.StreamBlockTridiag` (release 0.3.0,
  automatic when 5 Σn_s² x 16 B > 0.7 x free GPU memory): blocks densified one at a time, couplings sparse on the
  device, LU factors on the host (~97 GB), matches the dense LU to 2e-12 on 2808a. Memory estimate
  (results_ms/gpu/6nm/mem_estimate.log): FGMRES 16 Å windows unsplit fail ("window cores must partition the device"),
  y split 2 / 4: 86 / 172 windows, 59 / 37 GB complex64 LUs; Krylov 10.3 MB per column and iteration.
- Release 0.3.0 (2026-11-11, built on attelas2; committed f77cd83 and tagged qtx_modespace-v0.3.0 on 2026-11-12 from mont-fort15): GPU backend, streaming exact solve,
  prep, plot, `scripts/setup_env.sh` (venv + cupy matching the driver's CUDA major + VENV/bin/qtx_env.sh; 46 tests pass
  on the GH200), `examples/workflow.sh` (generic CPU / GPU steps A-E + plots; mos2_hfo2_workflow.sh calls it; 2808a
  GPU A + D reproduces N1 2373, FGMRES 24–47 / 46–86 its). dist/ wheel + sdist + SHA256SUMS. Rebuilt 2026-11-12 (still
  0.3.0) with level 2 on level 1 (`--l1-modes`), `--cap-lu-host`, `--fgmres-log`, Alps scripts, atomic
  caches, clean.py multi-axis minimum image; README notes the 6 nm two-level stall. 48 tests pass (source tree and the
  installed wheel). Build on attelas2 with /usr/sepp/bin removed from PATH: hatchling calls git, the broken SEPP git
  wrapper drops a 0-byte garbage-named file into the release folder that ends up in the sdist (it was in the first
  0.3.0 sdist).
- 6 nm two-level FGMRES stall (2026-11-11/12, results_ms/gpu/6nm/, scripts run_*_t015*.sh, coarse_*.py, figure
  figures/coarse_mode_6nm_t015.png): CAP-Schwarz 16 Å + y split 4 (tov 2.5 or 10.7) two-level stalls at Ec+0.02 / 0.15
  with slab level 2 on tile level 1 (E_L2_c1e-8) and slab level 1 1e-8 (A_slab_obc); one-level converges slowly (43
  x-windows, T = 0 after 40 its). 2808a with the same splits converges (y split 4 tov 2.5 slow, 150 its). Cause: Galerkin
  A_c has a dense set of almost real spurious eigenvalues (|mu| 5e-5, Im 1e-7, spacing ~5e-4 eV, fine residual 6.7 eV)
  = cut states on the slab interfaces (mostly HfO2; tile y cut for E_L2); the MS error is this state. 1/sigma_min(A_c)
  1.6e4 vs ~200 on 2808a; |(I − A P_c) r|/|r| 100–360 vs 10–17. Coarse shift: 1e-3 still stalls, 1e-2 brings the
  amplification to 2808a level (12–13) but FGMRES still stalls (median 0.86 → 0.13 in 40 its). Higher occupation cut
  is non-monotonic (|A_c^-1| 1.6e4 / 1.5e4 / 2.8e3 / 3.4e4 / 634 at 1e-8 … 1e-4): not a low-occupation effect; cut 1e-4
  (amplification 8–21) also stalls. Next idea (user, 2026-11-12): overlapping / partition-of-unity smoothed basis
  vectors across the slab faces (GenEO-type), prototype on 2808a first.
- SA-style smoothing of the coarse basis (2026-11-12, results_ms/gpu/pou/sa_smooth.py, run_sa_2808a.sh, sa_2808a*.log;
  literature notes sessions/2026-11-12-pou-basis-literature.txt): U = (I − M A)^k U0, U0 = slab level 1 1e-8 (N 2373),
  QR, dense Galerkin A_c, 2808a, two-level FGMRES its (max over columns) at Ec+0.05 / 0.15 / 0.2: U0 29 / 38 / 47;
  M = damped Jacobi 33 / 36 / 41 (diag(A) ~ 0, ω 0.002–0.04: useless); M = CAP-Schwarz 16 Å at the solve energy, k = 1
  20 / 21 / 22, k = 2 9 / 9 / 10. The smoothed span is no mode space (MS psi err 2–4, T err up to −50%, exact psi in
  span only to 0.2) and σ_min(A_c) is not better (4e-4…1.3e-3 vs 1.4e-4…4.8e-3; near-zero eigenvalues get Im
  1e-4…2e-3 instead of 1e-7): the gain is in the cycle, not in the spurious eigenvalues. Smoothed once at Ec+0.10 and
  reused: k = 1 25 / 26 / 42, k = 2 18 / 22 / 36 (gain mostly lost 0.1 eV away). Cost N smoother applications per
  smoothing step: does not pay per energy. Not yet tried on 6 nm (needs block-local storage of U instead of dense).
- Partition-of-unity level-1 basis (2026-11-12, results_ms/gpu/pou/pou_basis.py, run_pou_2808a.sh, pou_2808a_d*.log;
  rho cache /scratch/jiacao/mg_modespace/pou): slab extended by delta into its neighbours, local rho of the extended
  block (CAP + lead Sigma, 12 A), smoothstep ramp over 2 delta across each slab face (sum chi = 1), vectors chi v
  ("chiv") or eigenvectors of chi rho chi ("chirho"), QR, dense Galerkin. 2808a, delta 4 A, cut 1e-8, Ec+0.05 / 0.15 /
  0.2, two-level FGMRES its (max): U0 (N 2373) 29 / 38 / 47; chiv (N 2805) 49 / 54 / 63; chirho (N 1827) 45 / 53 / 60.
  Negative: the near-zero eigenvalues of A_c stay (sigma_min 2e-4…3e-3, same count, almost real, fine residual 6–7 eV,
  70–80% of |w|^2 within 3 A of a slab face = 32% of the device). MS T err 5e-8…5e-5 (U0 1e-6…3e-4), best psi in span
  the same or 3–6x better (chiv). Control "hard" (rho of the extended block cut back to the slab, i.e. only the
  embedding is larger: 16 A on the interior sides): N 1666 instead of 2373 at the same cut with the same its (30 / 37 /
  48), T err 1e-6…4e-5 and best-in-span: a larger embedding gives a more compact level 1.
  delta 7.5 A (ramp = almost the whole slab): chiv (N 2448) 43 / 48 / 53, chirho (N 1459) 38 / 44 / 51, still above U0;
  smallest eigenvalues still almost real with residual 6.5–7 eV at Ec+0.05 / 0.15 (face weight 0.42–0.46 instead of
  0.7): the ramp width is not the cause. MS T err chiv 5e-8…1e-7, chirho 2e-7…4e-6. "hard" N 1249 with 30 / 38 / 48
  its and T err 2e-7…2e-6, but here the embeddings (20 000 of 22 848 orbitals, both lead faces) are nearly the whole
  device, i.e. this is close to direct level 2: 2808a is too short to separate embedding size from global rho.
- Coarse-level tests on 2808a, 2026-11-12/13 (GH200; two-level FGMRES, CAP-Schwarz 16,2,2, tol 1e-10, its = max over
  columns at Ec + 0.02 / 0.05 / 0.1 / 0.15 / 0.2 / 0.3 / 0.4; one-level 57 / 56 / 57 / 62 / 86 / 126 / 150 (3e-6)):
  * Contour points of the slab level-1 local rho 24 -> 48 (results_ms/gpu/npts/): N1 the same to one column at cuts
    1e-8 / 1e-6 / 1e-4 (2373 / 1305 / 705), best-in-span the same, its the same: the 24-point filter is sharp enough,
    the cut removes weakly occupied in-window directions, not out-of-window leakage. 96 points not run (the contour
    Sigma files, 149 MB per point, filled /usr/scratch/mont-fort15; deleted). Keep big caches on /scratch/jiacao.
  * Its barely depend on the cut: slab level 1 1e-8 (N 2373) 30 / 29 / 31 / 38 / 46 / 150 (stall, 9e-9) / 94; 1e-6
    (1305) 32 / 31 / 31 / 36 / 44 / 65 / 97; 1e-4 (705) 37 / 38 / 39 / 44 / 50 / 76 / 113; 1e-2 (454, MS T err 4%)
    32 / 30 / 35 / 38 / 49 / 79 / 119. The 1e-8 stall at Ec+0.3: sigma_min(A_c) 4e-5, amplification 680-1570.
  * PoU at cuts 1e-2 / 1e-4 / 1e-6, ramp 4 and 7.5 A (run_pou_cuts_2808a.sh, pou_2808a_d*_c*.log): chiv / chirho
    never below the block-diagonal basis (+0…+16 its), sigma_min(A_c) unchanged; "hard" = same its with fewer
    columns (1e-6: 985 / 899 vs 1305).
  * Residual-minimising coarse correction P_c = U (C^H C)^-1 C^H, C = A U (results_ms/gpu/pou/mr_coarse.py,
    mr_2808a.log): amplification 0.94-0.99 by construction, but ~2x the its of Galerkin at every cut (1e-8: 62 / 62 /
    70 / 86 / 109), no convergence at Ec+0.3, true residual O(1) after 150 its at Ec+0.4 (FGMRES breakdown, cause
    not found). The MR start has a 30-400x smaller residual and T err 0.2-0.8% (Galerkin 1e-6…3e-4). Reading (not
    tested): |A e| weights error by distance from E, so MR spends U on far-from-E components the smoother removes
    anyway and leaves the near-E states. Keep Galerkin; large amplification alone is not what stalls 2808a.
  * End-slab lead-band cleanup of slab level 1 (--clean-slabs 0,-1, results_ms/gpu/clean/): 1e-8 184 / 221 spurious
    -> 0 with 9 / 14 vectors (N 2396), MS T err max 3.4e-4 -> 1.5e-5, its 32 / 31 / 33 / 38 / 44 / 65 / 96 (Ec+0.3
    stall gone); 1e-6 (N 1334) 34 / 35 / 35 / 38 / 46 / 70 / 100, T err unchanged (max 1.6e-3).
  * Lead-CMS basis as the Galerkin coarse level (user, 2026-11-13; bases of results_ms/mos2_hfo2_leadcms_sweep linked
    in results_ms/gpu/leadcms/, same H / atom order as the release prep): cut 1e-9 (N 980) 34 / 33 / 32 / 38 / 50 /
    80 / 119; + RMS (1012) 33 / 31 / 29 / 32 / 44 / 75 / 121; 1e-11 + RMS (1204) 36 / 34 / 33 / 37 / 49 / 71 / 117,
    with stand-alone MS T err 1-79%. Same its as slab level 1 up to Ec+0.2 at 40% of the columns and no device
    contour; worse at Ec+0.3 / 0.4. Not yet tried on 6 nm (the case that matters).
  GenEO family (papers/GenEO/, 11 PDFs; read from abstracts / arXiv HTML summaries only): GenEO selects what the
  one-level method fails on, chi v is part of the definition; H_k-GenEO needs all negative local eigenvalues (half
  the orbitals in TB) and subdomains < wavelength; MS-GFEM = local solutions at the solve energy on an oversampled
  patch (our level 1 is its window-integrated, hard-cut analogue); Nataf-Parolin extended GenEO (non-symmetric, RAS):
  dominant singular vectors of the local error operator, coarse-projector norm sigma left uncontrolled (= our
  amplification). Open: Nataf-Parolin vectors from the CAP windows (per energy), MS-GFEM space at E.
- Coarse level from the smoother's slow error modes (user, 2026-11-13; results_ms/gpu/slow/slow_modes.py,
  run_slow_2808a.sh, slow_2808a.log; hypothesis note sessions/2026-11-13-coarse-level-hypothesis.txt; 2808a, CAP-Schwarz
  16,2,2, modes built at the solve energy, Ec+0.05 / 0.2 / 0.3). Smoother alone T = I − M A (64 Ritz values, block
  subspace iteration q 8): |θ| > 1: 4 / 12 / 42, > 0.5: 35 / 53 / 57 (the smoother diverges on more modes as channels
  open). Part of the slow modes (|θ| > 0.7) outside span(U), median: slab level 1 1e-2 (N 454) 0.48 / 0.54 / 0.41, 1e-8
  (N 2373) 0.36 / 0.36 / 0.30, lead CMS (N 980) 0.52 / 0.51 / 0.42: every mode-space basis holds only half to two
  thirds of each slow mode, almost independent of N (hypothesis right in direction; predicted "close to 1").
  Enrichment U_r = [U_{r−1}, 64 dominant vectors of the cycle error operator T(I − P_c A)T], global vectors, FGMRES its
  (min..max over columns) at round 0 / 1 / 2 / 3:
    none (round 0 = 2 smoothing steps): Ec+0.05 34-41 / 11-15 / 11-14 / 10-13; Ec+0.2 60-69 / 18-23 / 13-17 / 11-15;
      Ec+0.3 98-107 / 29-37 / 18-23 / 14-18 (N 64 / 128 / 192; cycle |θ|max 0.24-0.75 after round 1).
    slab level 1 1e-2: 24-30 → 13-16; 43-49 → 17-21; 72-79 → 26-31.   slab level 1 1e-8: 24-29 → 17-20; 40-47 → 20-25;
      63-150 (stall) → 32-38.   lead CMS: 27-33 → 19-23; 44-50 → 27-32; 73-80 → 42-46.
  => 64-192 slow modes alone beat every mode-space basis (N 454-2373) by 2-4x, and a mode-space base makes the
  enriched level worse (isolated cycle factors 1-30 remain = its spurious Galerkin eigenvalues). Modes cut into slab
  blocks (block-diagonal U, 4x the columns): none 16-20 / 24-30 / 34-41 after round 1, 20-24 / 23-27 / 25-30 after
  round 3 (worse than global at Ec+0.05 / 0.2, not monotonic). Build cost per round: m (q + 1) = 576 cycle applications
  ≈ one two-level solve of 19 columns x 30 its, so per energy it does not pay unless the modes are reused in energy or
  built with fewer steps.
  Follow-up (run_slow_reuse_2808a.sh, slow_2808a_reuse0.10.log, _m32q3.log, _m64q3.log; its = max over columns, rounds
  0 / 1 / 2 / 3). Cheaper build at the solve energy, base none: m 64 q 3 (256 cycle applications per round) Ec+0.05
  41 / 16 / 16 / 14, Ec+0.2 69 / 35 / 23 / 19, Ec+0.3 107 / 44 / 29 / 22 (q 8: 15 / 14 / 13, 23 / 17 / 15, 37 / 23 /
  18); m 32 q 3 (128 per round) 41 / 21 / 16 / 16, 69 / 40 / 32 / 25, 107 / 69 / 44 / 34. Build in its-equivalents
  (cycle applications / columns): 64 x 4 = 13.5 / 10.7 / 8.5 per round, so solve + build ≈ 16 + 13.5, 23 + 21 (2
  rounds), 29 + 17 (2 rounds) vs slab level 1 1e-2 30 / 49 / 79 and none 41 / 69 / 107: break-even at Ec+0.05, ahead
  where the smoother is weak, and no basis build. Reuse of modes built at Ec+0.10 (m 64 q 8): base none Ec+0.05
  41 / 27 / 24 / 22, Ec+0.10 41 / 17 / 15 / 14, Ec+0.15 46 / 25 / 27 / 28, Ec+0.2 69 / 43 / 42 / 45 (0.05 eV away half
  the gain is lost, 0.1 eV away 2/3, extra rounds do not help); base slab level 1 1e-2 Ec+0.05 30 / 21 / 19 / 17 (= built
  at E), Ec+0.10 35 / 21 / 17 / 18, Ec+0.15 38 / 25 / 21 / 21, Ec+0.2 49 / 34 / 30 / 27 (at E: 28 / 22 / 21). => slow
  modes alone are energy specific; mode-space base + reused slow modes is robust over ±0.05–0.1 eV (the mode space
  supplies the energy-independent part).
  6 nm, Ec+0.15, slow modes only (run_slow_6nm_t015.sh, slow_6nm_t015.log; windows 16 A, y split 4, tov 10.7: 172
  windows of 3792..10083 orbitals, factor 78 s, 117 GB complex64 LUs on the host; m 64, q 3, 4 rounds; FGMRES on the 12
  strongest of 40 channels, maxit 60): smoother T: 18 of 64 |θ| > 1 (max 1.58), cycle without coarse level 22 of 24
  > 1 (max 2.79). Largest cycle |θ| / FGMRES residual estimate (max over columns) after 60 its, N 64 / 128 / 192 /
  256: 1.94 / 1.34 / 1.07 / 5.21 (σ_min(A_c) 4.6e-2 → 5.3e-4, 9 of 24 > 1); 0.60 / 0.22 / 0.12 / 6.0e-4 (psi err 0.73 /
  0.56 / 0.20 / 1.3e-3; N 256 falls ×0.77 per it over the last 20 its). First coarse level that makes FGMRES converge
  with these windows (mode-space levels stall at 14 / 1.5), but 256 modes are not yet enough (2808a: 64): the count
  grows with the device (172 vs 7 windows). Wall time: a smoother pass streams all LUs and barely depends on the
  number of columns, so a round of 64 vectors costs 17–19 s vs 7.1 s per FGMRES iteration (427 s per 60 its): the
  build is cheap here, use more / larger rounds.
  m 128 per round, 8 rounds, maxit 100 (slow_6nm_t015_m128.log; smoother T: 40 of 128 |θ| > 1): N 256 100 its,
  residual 2.5e-6 (not converged); N 512 24–30 its to 1e-10 (psi err 7.8e-10, cycle |θ|max 1.00, 6 of 24 > 0.5);
  N 768 23–29; N 1024 26–31 (saturated at 512; σ_min(A_c) 5e-4…9e-4 from N 384 on, isolated factors 1–2 remain).
  Wall time for N 512: factor 79 s + build 4 x 21–25 s + FGMRES 203 s (12 columns, 6.8 s per it) ≈ 380 s vs exact
  streamed block LU 137–163 s per energy for all channels: converges, not yet faster (pass time = streaming 117 GB of
  host LUs).
  Transverse window shell 10.7 -> 6 A (--cap-tov 6, slow_6nm_t015_tov6.log; windows 2952..7690 orbitals, LUs 68 GB,
  factor 55 s; smoother T: 21 of 128 |θ| > 1): N 512 28–36 its (182 s, 5.1 s per it), N 768 27–33, N 1024 27–33, all
  to 1e-10. +4–6 its for 42% less LU memory; ≈ 310 s in total, still above the exact solve.
  Energy sweep with a recycled coarse space (previous solutions, Ritz vectors of the FGMRES Hessenberg with the largest
  |1 − θ| as Z g; sweep_recycle.py, run_sweep_recycle_2808a.sh, sweep_recycle_2808a.log; 2808a Ec+0…0.3 in 10 meV steps, 31 energies,
  19–30 channels, maxit 100, 128 slow modes (2 x 64, q 3), recycled part capped at 256 columns). Its (max over
  columns) at Ec+0 / 0.05 / 0.1 / 0.15 / 0.2 / 0.25 / 0.3, total over the sweep, solve time:
    no coarse level                      43 / 41 / 41 / 46 / 69 / 100* / 100*   1935   211 s   (* = not converged)
    slow modes fixed at Ec+0             14 / 21 / 25 / 31 / 45 / 82 / 88       1330   154 s
    fixed + previous solutions           14 / 14 / 19 / 24 / 35 / 62 / 69        988   118 s
    fixed + recycled Ritz vectors        14 / 36 / 39 / 47 / 69 / 100* / 100*   1801   211 s
    fixed + both                         14 / 32 / 34 / 45 / 66 / 100* / 100*   1770   211 s
    Ritz only / both, no slow modes      43 / 57 / 56 / 67 / 88 / 100* (worse than no coarse level)   2271 / 2211
    slow modes rebuilt at every energy   14 / 16 / 17 / 19 / 22 / 23 / 29        618    99 s incl. ~1 s build per energy
  => rebuild per energy wins everywhere and its build is cheap (512 cycle applications ≈ 1 s on 2808a, 17–27 s on
  6 nm): no reason to carry the space. Previous solutions help while the fixed modes are fresh (≤ fixed − 7 its up to
  Ec+0.13; median start residual 1e-3…1e-5, max still O(1)) but do not replace energy-specific slow modes. The Ritz
  vectors as selected (largest |1 − θ| of non-converged per-column Krylov spaces) damage the coarse level (start
  residual 10–20); σ_min(A_c) not logged, cause not verified.
  Report with method write-up (how the slow modes / coarse solver are built) and figures: reports/2026-11-13-coarse-level-
  findings/ (summary.md, fig1–6, make_figures.py). Reading (user, 2026-11-13): mode space is a separate single-shot
  product; the solver's coarse level = slow modes rebuilt per energy. 6 nm open item: cost (factor 79 + build ~90 +
  FGMRES 203 s vs exact 150 s; tov 6 Å ~310 s).
  Spectrum check (slow_modes.py --spec, run_slow_spec_2808a.sh, slow_spec_2808a.log): the slow-mode A_c has as many
  small singular values as mode space (σ < 3e-2: 23–29 at N 192) but no spurious states: eigenvalues next to 0 have
  Im ≈ Re (1e-4…1e-2, damped) and fine residual 0.07–3 eV; the 0.02–0.17 eV state at Ec+0.2 (4.3e-4 + 1.4e-4i) is a
  physical resonance present in every basis. Mode space keeps its almost real (Im 1e-7) 6.5–7.3 eV states after 3
  rounds of slow-mode enrichment. Related methods: adaptive / bootstrap AMG, Lüscher inexact deflation, DD-αAMG
  (papers/AMG/, note in the report): same setup, but they aggregate the test vectors into a sparse block-local P
  (local coherence); our "slab" cut of the slow modes was worse on 2808a.
  Implemented 2026-11-13 (`SlowCoarse` in release schwarz.py and qtx_ms/schwarz.py, identical): release cli
  `--fgmres-coarse slow --slow-m 64 --slow-q 3 --slow-rounds 3` (open and closed system), `--basis none` (no mode
  space; only with --fgmres coarse slow|none); qtx_ms.cap_fgmres `--coarse slow`. Tests: release
  tests/test_slow_coarse.py (2D strip + RAS; a 1D chain is degenerate: RAS error rank 2 per cut), test_level1
  (wire, closed cli), test_gpu (GPU = CPU); repo tests/test_slow_coarse.py. End-to-end check on attelas2 pending:
  results_ms/gpu/slow/run_release_slow.sh (2808a; SIXNM=1 adds 6 nm Ec+0.15 m 128 x 4, tov 6).
  2808a done (release_slow_2808a.log): 11-15 / 14-17 / 16-20 its at Ec+0.05 / 0.15 / 0.2 (MS coarse 24-29 / 32-38 /
  40-48), T err ≤ 1.2e-11, factor 2 + build 3 + iterations 3-5 s. 6 nm Ec+0.15 (release_slow_6nm_t015.log, all 40
  channels in 4 chunks of 12, m 128 x 4, tov 6, LUs on the host 68 GB): 28-36 its, T err +1.3e-11, psi err 3.5e-11;
  factor 60 + build 96 + iterations 1025 s (~8.5 s per it per chunk; GPU shared with another job) vs exact ~150 s:
  converges on every channel, cost per channel chunk is the problem. Release 0.4.0 (2026-11-14, commit a659d0d, tag
  qtx_modespace-v0.4.0, built with uv on zijin x86; dist/ wheel + sdist + SHA256SUMS; 53 tests pass from source incl.
  GPU (P100), wheel 39 pass / 14 GPU-skipped).
- Multilevel method (user, 2026-11-13; next task: O(N), less FGMRES / window LU / solve time) = additive Schwarz with
  CAP (RAS smoother on every level) + GenEO for indefinite problems (coarse vectors from the smoother's own local
  window operators A_j = R_j A R_j^T − iW_j: local contour ρ over the energy window, or the eigenvectors of A_j next to
  0 by inverse iteration with the already factored window LUs; partition of unity) + bootstrap AMG (algebraic recursion
  A_k = P_k^H A_{k−1} P_k with the CAP projected by the composite prolongation; slow modes of the current cycle as
  correction). Consistency (user): the prolongation must come from the same embedding / CAP as the smoother. Level k+1
  only has to hold the slow modes of the level-k smoother, so the "missing complement" failure of tile_iter (target =
  the true fine ρ) does not apply. Risks: Galerkin spurious states on every level; indefinite GenEO needs many local
  vectors. Plan: (1) two-level test on 2808a with the 16 Å smoother windows (fraction of global slow modes in span,
  its, A_c spectrum), (2) three-level V-cycle, (3) 6 nm. Session: sessions/2026-11-13-slow-coarse-release040-multilevel.txt.
  Step 1 done 2026-11-14 (results_ms/gpu/geneo/geneo_2808a.py, run_geneo_2808a.sh, geneo_2808a.log; OOM at N 15k in the
  first run = full svd of A_c, now inverse iteration above N 6000). 2808a, CAP 16,2,2 (7 windows), its max over columns
  at Ec+0.05 / 0.2 / 0.3; one-level 56 / 86 / 126; slow 3 x 64 (N 192) 15 / 20 / 22. Window local rho (24-pt contour of
  A_jj + CAP, Ec −0.15…+0.6) cut to the window cores ("core"): 1e-2 (N 4040) 16 / 20 / 21, 1e-4 (10 688) 9 / 9 / 10,
  1e-6 (15 362) 7 / 8 / 7; captures 96–100% of the smoother slow modes (mode space 50–70%), but A_c keeps the almost
  real 5.4–9 eV spurious states and N is 14–53% of n. chi-weighted ("pou", 4 Å ramp) fails: 47–105 its, often worse
  than one-level, cycle |theta| up to 79 (6–27% of the slow modes outside span): the prolongation must be cut like the
  RAS core. "eig" (window inverse iteration at E, k 16/32/64, core) 45–110 its: the most amplified window states are
  not the slow modes (out 0.2–0.6). One bootstrap round (64 cycle modes): global on DM core −0…−5 its, split over the
  window cores (SVD, tol 1e-2, ~60 of 64 kept per window) −2…−9; on the slow base global −0…−2, split −1…+4.
  => consistency fixes capture, not the spurious states or the size; slow modes stay the best per column.
  Note: sessions/2026-11-14-geneo-step1-window-bases.txt.
  Local coherence of the slow modes (results_ms/gpu/geneo/coherence.py, coherence_2808a*.log, coherence_6nm_t015.log;
  global slow modes cut into blocks per aggregate of 1/2/4 window cores, SVD per block, singular values > tol x max):
  2808a (7 windows) Ec+0.05 / 0.2 / 0.3, N 256 global 12-15 / 14-17 / 16-19 its; per window at tol 1e-2 (N 1034-1188)
  12-16, only 0-7% of Q outside the span, no spurious states (eigenvalues next to 0 damped, residual 1-3.6 eV); rank per
  window core (interior median, of 256) 118 / 163 / 196 at tol 0.1 / 0.03 / 0.01: weakly local. Ec+0.15, 8 x 64: global
  N 256 13-16, N 512 14-16 (saturated); per window of N 512 at 3e-2 / 1e-2 (N 1526 / 1880) 9-11 / 9-10.
  6 nm Ec+0.15 (172 windows, tov 6, 12 strongest channels, chunk 6, maxit 100): rank per window core of N 256 only
  18 / 39 / 63 / 122 at 0.1 / 0.03 / 0.01 / 1e-3 (15-25% of the global count vs 60-75% on 2808a: much more local),
  but global N 256 does not converge (100 its, 3e-6) and the per-window cut at 0.1 (N 3197) loses 17% of Q (2808a 6%),
  residual 3e-2 after 100 its, A_c 48 singular values < 1e-2 (global 6), next-to-0 states damped, residual 3.5 eV.
  (Cuts 0.03 / 0.01 and N 512 were running at the time of writing.) Bugs fixed on the way: `--ncol k` took the k
  WEAKEST channels (core.injection returns ascending Gamma eigenvalues; slow_6nm_t015*.log ran on T 3e-6 channels,
  release_slow_6nm_t015 used all 40), now b[:, -k:]; GPU OOM from Q.conj().T copies (hdot) and strided Qall[:, :N].
  Step 2 done 2026-11-14: three-level V-cycle on 2808a (results_ms/gpu/mlevel/vcycle3_2808a.py, run_vcycle3_2808a.sh,
  vcycle3_2808a.log; attelas1, 1054 s). Level 1 P1 = blockdiag over the 7 S0 window cores (RAS partition), A1 = P1^H A
  P1 (dense, prototype); level-1 smoother = RAS, window = cores within graph distance `halo` in the A1 block graph
  (cores couple to c±2: halo 1 = 5 of 7 cores, halo 2 = all), CAP i s1 U_w^H P S P U_w projected from the fine level
  (x cuts only); level 2 = 3 x 64 slow modes of the level-1 smoother / cycle in level-1 coordinates. FGMRES x0 = 0, its
  (max over columns) at Ec+0.05 / 0.2 / 0.3, halo 1: one-level 56 / 86 / 126; fine slow N 192 14 / 19 / 22;
  dm 1e-2 core (N1 4040, n/N1 5.7) exact level 1 16 / 19 / 21, V 17 / 23 / 33, W 16 / 19 / 21, K4 = exact;
  dm 1e-4 core (N1 10 688, n/N1 2.1) exact 9 / 10 / 10, V 10 / 11 / 15, W = K4 = exact;
  slow:512:3e-2 (512 fine slow modes cut per core, N1 ~1450, n/N1 15, level-1 windows 556-1278) exact 11 / 13 / 14,
  V 18 / 26 / 45, W 12 / 16 / 19, K4 = exact; level-1 smoother skipped (level 2 only) 35 / 47-66 / 74-104.
  Halo 2 = exact everywhere (windows = whole level 1). The level-1 smoother with the projected CAP is nearly stable
  (|theta|max 0.7-1.45, <= 6 of 32 > 1; the tile V-cycle amplified 20-300x) and needed (skip 3-7x the its); one V-cycle
  is not enough at high E, K4 (8 S1 passes per fine it) recovers exact level 1. The 32-Ritz estimate of the level-1
  cycle factor (0.00-0.22) misses the V-cycle loss: judge by fine its. Not yet measured: cost (2808a too small; dm 1e-4
  does not coarsen); no global 512 two-level baseline at these E. Next: 6 nm (43 x columns x 4 y strips, halo-1 windows
  really local) with block-sparse A1, transverse projected CAP, and a level-1 cut that keeps enough (0.1 does not).
  Step 3 on 6 nm, 2026-11-15 (results_ms/gpu/mlevel/vcycle3.py: block-sparse A1, restricted assembly O(nnz), CAP ramp
  = distance to the nearest atom outside the level-1 window (covers y cuts; on 2808a V 26-30 -> 23-27, W 21-25 -> 19-23
  vs the x-plane ramp), level-1 LUs complex64; run_vcycle3_6nm.sh, vcycle3_6nm.log, vcycle3_6nm_c1e-2.log; Ec+0.15,
  12 strongest channels, chunks of 6): two-level with the 512 global fine slow modes 30-36 its (419 s, smoother 404 s);
  level 1 = the same 512 modes cut per core, EXACT level-1 solve: 3e-2 (N1 10 155) 58-65 its (780 s), 1e-2 (N1 15 706)
  36-40 (482 s). Core graph 10-15 cores per core, halo-1 level-1 windows 572-1483 (really local), level 2 512 modes
  settles (|theta| 9.5-10 -> 1.0-1.1). => per-core cuts of the global slow modes never beat keeping them global; a
  V / W / K-cycle can only add to the exact-level-1 count, so the three-level route needs a level 1 that holds MORE
  than the global slow modes (e.g. + local slow modes of the level-1 windows), not a localized copy. The smoother is
  95% of every solve: the practical gain is distributing it (sessions/2026-11-15-alps-multigpu-multilevel-plan.txt).
  Stage 3.1 of that plan done: restricted A1 assembly (31.5 s for N1 15 706 vs 79 s full build for N1 10 155; blocks
  equal to 1.4e-14 on 2808a), U^H S U / U^H H U cached for dm bases (A1 = E S1 - H1 - Sigma1), --fgmres-log, and the
  release CapSchwarz solver inv / inv64 (window inverses in zero-padded size batches, one gather + batched matmul +
  scatter per batch; resident only, not with --cap-lu-host; tests test_level1 / test_gpu inverse_batched pass; 6 nm
  needs ~75-90 GB of inverses, i.e. > 1 GPU).
  Smoothed-aggregation level 1 (2026-11-15, vcycle3.py --l1-smooth nu --l1-drop; P1 = (I − S0 A)^nu P_tent, P_tent =
  blockdiag of the per-core cut slow modes, supports from the core graph + S0 read sets, cores coloured so nu S0 passes
  per colour, per-core QR, A1 = P1^H A P1 by coloured products). 2808a Ec+0.05 slow:128:3e-2: exact level 1 12-14 its
  (unsmoothed 14-17), but level-1 V-cycle 91-97 (unsmoothed 26-30). 6 nm Ec+0.15 nu 1 (vcycle3_6nm_sa1.log, CYC=exact,
  1321 s): supports 16..28 cores per block, 28 colours; 3e-2 (N1 10 155, P1 8.3 GB, smoothing 76 s + A1 46 + 122 s
  assembly) EXACT level 1 25-29 its (346 s; unsmoothed 58-65, global two-level N 512 30-36 / 419 s); 1e-2 (N1 15 707,
  P1 12.8 GB) 24-28 (337 s; unsmoothed 36-40). T err ≤ 2e-12. First level 1 that beats the global slow modes; the
  cut-state loss in A1 is gone (3e-2 = 1e-2). But level 2 on it with the halo-1 level-1 RAS does not settle: cycle
  |theta|max per round 206 / 151 / 44 / 44 (3e-2), 750 / 251 / 239 / 141 (1e-2) vs 9.5 -> 0.97 unsmoothed: the
  smoothed basis functions (28 cores) reach beyond the halo-1 windows (572-1483 columns). Next: halo 2.
  Halo 2 (vcycle3_6nm_sa1_h2.log, 6346 s): level-1 windows 1222-2238, level 2 |theta|max 65 / 40 / 39 / 22; V and W
  stall (residual 0.93-1.0 after 100 its), K4 6e-3 / 0.12 after 100 its. SA given up (user, 2026-11-15).
  Density-matrix hierarchy (user, 2026-11-15; results_ms/gpu/mlevel/merge.py, run_merge_9072a.sh, merge_9072a.log,
  merge_9072a_dm.log, smoke merge_2808a_smoke.log; run scripts are listed in job.sh): level 1 = contour rho (Ec
  -0.15...+0.6, 24 pts) of the S0 window operators cut to the cores, selection "dmc" = eigenvectors of the core block
  rho_cc (mode-space style) or "dm" = eigenvectors of the whole window rho restricted to the core (5-9x more columns);
  level 2 = clusters of 2 cores along x, rho of cluster + coupled cores IN THE LEVEL-1 BASIS (z S1 - H1 - Sigma1 +
  projected CAP), same selection, cut 1e-2, wide contour or the slice Ec+0.05...+0.25; RAS smoothers S1 (core + coupled
  cores on A1) and S2 (cluster + coupled clusters on A2), projected CAP; top = slow modes of the cycle below in rounds
  of 32 (SubCoarse = first N). 9072a (n 72 672, 19 unsplit windows 16,2,2, 10 clusters), Ec+0.15, 20 columns, tol
  1e-10, V-cycle 1 + 1 smoothing per level, its min-max. Two-level, slow modes on the fine level, N 32 / 64 / 128 /
  256: 38-49 / 26-35 / 21-28 / 19-24 (2808a Ec+0.05: 64 -> 12-17, saturated).
    level 1          N1      exact L1   three-level N 32 … 256     level 2 wide: N2, exact, four-level 32 … 256      slice: N2, exact, four-level
    dmc 1e-2         2407    41-48      68-77 … 57-66              1313, 53-62, 64-74 … 61-69                       349, 72-80, 89-99 … 73-82
    dmc 1e-3         6796    32-38      61-70 … 55-64              2026, 60-68, 79-88 … 75-83                       613, 63-71, 95-100 … 86-95
    dm 1e-1          4485    40-46      75-83 … 63-71              3993, 48-55, 64-72 … 56-63                       1793, 72-80, 84-92 … 80-87
    dm 1e-2          13 674  16-20      39-45 … 24-30              7275, 34-40, 38-45 … 38-45                       3726, 31-37, 38-45 … 39-45
  => everything converges, nothing beats two-level: only dm 1e-2 (n / N1 5.3, level-1 windows 1442-4057 columns ~ the
  fine windows) gives a good exact level 1 (16-20); every cycle step below an exact solve costs 15-30 its; the top
  count hardly matters once levels sit in between (its set by the levels below). Level-1 rho of the middle cluster vs
  the same window on the fine level (same orbitals / ramp, unprojected CAP; trace, modes > 1e-2): wide 96.5 / 65.7
  (x1.47), 126 / 97 (dmc 1e-2); 147 / 66.4 (x2.2), 217 / 102 (dmc 1e-3); x1.9-2.1 (dm); slice: fine trace 2.04, 14
  modes, level 1 x5.8-12, 35-70 modes (dmc), 51-52 in rho_CC (dm). No negative eigenvalues; reference modes missed
  2e-4...3e-2. The fine slice is small (narrowing argument right on the fine level), but the level-1 operator puts
  2.5-5x more states into it, more with a larger level-1 basis (which holds the fine rho to 1e-3): read as the
  spurious Galerkin cut states of the block-cut basis (not verified). 6 nm step 1 (run_dmcore_6nm.sh) stopped during the
  window density-matrix build (154 of 172 windows cached in /scratch/jiacao/mg_modespace/6nm/dmw).
  Level-2 smoother formed in the level-1 basis (user's design; merge.py --sweep c1,cexact2,c2, Level2.smoother1 = RAS
  on A1 with the CLUSTERS as blocks, window = cluster + coupled cores = the window of its rho, 1442-4827 columns for
  dm 1e-2) and W-cycle (--gamma 2), merge_9072a_cl.log / merge_9072a_w.log, same case, its for top N 32 / 128 / 256:
    dm 1e-2 (exact L1 16-20)              V-cycle                       W-cycle
    single-core S1, slow modes on L1      39-45 / 31-38 / 24-30         28-34 / 18-23 / 16-21
    cluster S1c, slow modes on L1         31-36 / 25-30 / 18-24         21-27 / 17-22 / 16-20
    single-core S1, exact L2 wide/slice   34-40 / 31-37                 31-37 / 28-33
    cluster S1c, exact L2 wide/slice      16-20 / 16-21                 16-20 / 16-20
    four-level (S1, S2, top on L2)        38-45                         33-41 (slice 31-42)
    dmc 1e-3 (exact L1 32-38): cluster S1c + exact L2 38-46 (wide, N2 2026) / 37-44 (slice, N2 613 = n / 119);
      cluster S1c + slow modes on L1 51-59 / 50-57 / 45-52 (V), 47-54 / 38-44 / 34-41 (W).
  => with the cluster-block smoother the merged level 2 loses nothing against an exact level-1 solve (the earlier
  34-40 was the separate single-core S1), also for the energy slice (N2 3726 instead of 7275; dmc 1e-3: 613); the
  W-cycle recovers the exact-level-1 count with 128-256 slow modes on level 1. What is missing is the solver for A2:
  slow modes of A2 seen through the projected level-1 smoother P2^H S1c P2 fail (|theta| 400-980 -> 10-40; V 37-56,
  slice / W / dmc stall): the next level needs its own cluster smoother (level 3 = merge again), not built yet.
  The floor is the level-1 quality (16-20 only at n / N1 5.3). Time per solve ~ the two-level one (S0 dominates).
  Repeated merging (mlevel.py, run_mlevel_9072a.sh, mlevel_9072a*.log; --up AXIS:CUT[:mu1:mu2] per level; level
  k+1 = pairs of level-k blocks, rho of pair + coupled blocks in the level-k basis, CAP / Sigma projected through all
  bases; the pairs are the blocks of the smoother of A_k; bottom = exact level m, or own-block smoother + first N slow
  modes on the deepest level). 9072a, dm 1e-2, x, x merges: wide N 13 674 / 7275 / 4283, exact level 1 / 2 / 3 16-20 /
  16-20 / 16-20, 3 levels + top N 32 / 128 / 256 24-28 / 21-25 / 18-22; slices (level 2 Ec+0.05..0.25, level 3
  Ec+0.1..0.2) N 13 674 / 3726 / 1105 (3.7x, 3.4x per merge), exact 16-20 / 16-21 / 18-23, top 30-35 / 27-32 / 24-29.
  => merges are lossless and slices give geometric coarsening (n / N3 66); solve times = two-level (S0 dominates).
  Cores also cut in y (--cap-split 2,1, tov 6, 38 cores): level 1 gets LARGER (N1 31 707, n / N1 2.3; pair windows
  3364-11 207 columns > the fine windows); the three runs died with GPU OOM building level 3 (no its). Open: smaller
  level 1 (slice on level 1, selection between dmc and dm, face-neighbour embedding), memory of mlevel.py, other
  energies, length series, 6 nm. Docs: reports/2026-11-15-density-matrix-hierarchy/summary.md (method for others),
  notes/2026-11-15-density-matrix-hierarchy.md (findings, next steps), sessions/2026-11-15-density-matrix-hierarchy.txt.
  Memory fix + row-restricted inverses (2026-11-15 afternoon; mlevel.py --l1-solver rows (default), schwarz.py
  --cap-solver rows64 | rows; run_mlevel_9072a_y2_rows.sh, run_mlevel_9072a_s0rows.sh, mlevel_9072a_*_rows.log,
  mlevel_9072a*_s0_{dense64,rows64}.log). The y-cut OOM was not the cluster LUs (9.7 GB) but dense projected CAPs
  kept per window (~22 GB), the CAP of a level-3 window formed in level-1 coordinates, S / H / A blocks of finished
  levels and a fixed 24 GB batch. Now: CAP projected as a (ramp orbitals x columns) factor and not stored, blocks of
  a finished level freed, batches from the free GPU memory: 33 GB in use instead of > 99 GB.
  Rows trick: RAS keeps only the rows of the window inverse on the written blocks -> store (A_w^-1)[write, :]
  (n_write x n_window, one GEMM) instead of the LU; the same correction. Cluster smoothers: 2.0 instead of 9.7 GB
  (y merge of level 1), 2.4 instead of 13.8 GB (x merge), levels >= 1 2-4 s -> < 1 s per solve; unsplit x, x
  regression 16-20 / 16-20 / 16-20 unchanged. Fine smoother S0 (rows64 vs dense64, identical its): unsplit 4.4
  instead of 9.1 GB, S0 5-13 s -> < 1 s per solve, solve 7-16 s -> 2-3 s (two-level N 32 / 128 / 256 38-49 / 21-28 /
  19-24; exact levels 16-20); y-cut 3.8 instead of 13.7 GB, S0 7-32 s -> <= 1 s, solve 8-35 s -> 2-4 s. Factor
  time unchanged (6-7 s, 10 s). The A matvec is now the largest part of a solve. Not yet the release default.
  y-cut results (dm 1e-2, 38 cores, N1 31 707; exact level 1 skipped, N1 > 30 000): y, x, x wide N 19 680 / 10 665 /
  6271, exact level 2 / 3 / 4 13-16 / 13-16 / 13-16, top N 32 / 128 / 256 19-22 / 16-20 / 15-18; y, x, x slices N
  19 680 / 5569 / 1866, exact 13-16 / 13-16 / 16-19, top 23-28 / 25-29 / 24-27 (erratic slow modes); x, x, y wide N
  19 931 / 12 188 / 7347, exact 13-16 / 13-16 / 14-17, top 15-19 / 14-17 / 14-17. => the y cut gives fewer its
  than unsplit (13-16 vs 16-20) and needs fewer top modes, merges lossless in both orders, but every level is
  larger (5 final blocks: 6271-7347 vs 4283) and the build is slow (pair-window rho 164-272 s, run 350-540 s vs 111 s).
  Intermediate devices in examples/: 9072a (x 188 A), 21600a (x 378 A, y 33 A: 6 nm length at the 2808a cross
  section; prepared, inputs/mos2_hfo2_21600a), 6 nm (y 66 A), 12 nm (y 132 A): length series 2808a / 9072a / 21600a, width series
  21600a / 6 nm / 12 nm.
  Window halo (user, 2026-11-15 evening; mlevel.py / qtx_ms.mlevel --halo H,... per --up entry, --halo-smooth
  same | full). Why the level-2 rho build dominates the y-cut runs: one full dense inverse per window and contour
  point, windows = pair + every coupled level-1 block; H reaches the second core along x (cores 7.9 A), so with the y
  cut a pair window is the whole cross section at x-2 .. x+2 = 10 blocks x ~926 columns (9287; the pair itself 1831).
  Idea: level 2 covers only part of the interaction range (halo 1 block = 7.9 A per side, as level 1 does with its
  ~4 A), level 3 (blocks 15.8 A) covers the full range with 1 block anyway; the Galerkin operators keep all couplings.
  The log now lists the coupling blocks of A_k, S_k by distance (2-norm, 12 power steps) and the window size / rho
  cost per halo. 2808a, Ec+0.05, y-cut dm 1e-2, y + x merges (run_mlevel_2808a_halo.sh, mlevel_2808a_halo_{all,h1,
  h1full}.log, attelas1): level-1 couplings |A1| median dx 0 other y 60.7 eV, dx 1 same / other y 63.6 / 29.4, dx 2
  same / other y 0.34 / 0.010 (rel 0.01 / 0.00; |S1| 2.8 / 1.3 / 0.014 / 0.002) -> the second neighbours are 1% of
  the first. Halo 1 at level 2: windows 3364..5933..8000 -> 1672..4101..5417 columns (predicted cost 0.29), rho 89 ->
  49 s, smoother rows 0.45 -> 0.30 GB, N2 4864 -> 5171 (+6%), N3 2450 -> 2581; exact level 1 / 2 / 3 12-15 / 13-15 /
  13-15 for all, halo 1 and halo 1 with the full smoother windows (12-15 on levels 2, 3): no iteration lost.
  9072a, Ec+0.15, y-cut dm 1e-2, y, x, x merges, halo 1 at level 2 only (qtx_ms.mlevel, run_qtxms_mlevel_9072a_h1.sh,
  qtxms_9072a_y2_yxx_h1.log / .json; attelas1, GPU shared with another user's job at 99%: times not comparable) vs
  mlevel_9072a_y2_yxx_rows.log (all coupled blocks): level-2 windows 3364..9287..9385 -> 1672..5557..5650 columns
  (cost 0.229), smoother rows 2.01 -> 1.23 GB, GPU after level 2 30.9 -> 18.7 GB; N 18 719 / 10 789 / 6388 (reference
  19 680 / 10 665 / 6271); couplings |A1| dx 1 same / other y 64.8 / 29.7 eV, dx 2 0.35 / 0.011 (1%), |A2| dx 1 57.1,
  dx 2 0.23. Iterations: exact level 2 / 3 / 4 14-17 / 14-17 / 16-19 (reference 13-16 / 13-16 / 13-16), 4 levels + top
  N 32 / 128 / 256 25-29 / 23-26 / 20-24 (reference 19-22 / 16-20 / 15-18). => on 9072a the halo costs 1 iteration
  at level 2 and 3-6 below it, unlike 2808a (none lost); the 1% coupling weight did not predict that. Level-2 rho 175 s
  on the shared GPU (reference 164 s idle): the time gain is not demonstrated. Not separated yet: basis vs smoother
  (--halo-smooth full), and no same-code run without halo on 9072a (the reference is the script; equal on 2808a).
  Both done (qtxms_9072a_y2_yxx_{h1full,all}.log, same shared GPU, same code): no halo N 19 680 / 10 665 / 6271, exact
  13-16 / 13-16 / 13-16, top 19-22 / 16-19 / 15-18 (= the script), level-2 rho 485 s in 8 batches, run 841 s; halo 1
  for rho + full smoother windows: N 18 719 / 10 789 / 6388, exact 14-16 / 14-16 / 14-17, top 20-24 / 17-20 / 16-19,
  level-2 rho 176 s in 2 batches, run 534 s. => the level-2 basis from the short windows costs ~1 iteration; the
  3-6 lost with halo 1 everywhere come from the weaker smoother of A1 (short windows). Use --halo 1 --halo-smooth full
  on the y cut: rho 2.8x faster under equal conditions (n^3 count 4.4x; fewer batches), smoother memory as without halo.
  FEAST for the level-2 rho: no (same LU per point; probes only save back-solves, at most ~2x with LU + columns on the
  pair, and that gives the dmc selection only).
  qtx_ms.mlevel (2026-11-15): the hierarchy as a library on the qtx_ms stack (see notes/code-state.md), with the GPU
  windows / row inverses / block FGMRES ported into qtx_ms/schwarz.py. 2808a check (run_qtxms_mlevel_2808a.sh,
  qtxms_2808a_{h1,all}.log): halo 1 N 9297 / 5171 / 2581, exact level 1 / 2 / 3 12-15 / 13-15 / 13-15, T err <= 1.2e-12
  = the script; all N 9297 / 4864 / 2450, 12-15 / 13-15 / 13-15 = the script; coupling tables identical. Level-2 rho
  89 s (all) as the script; 77 s (halo 1) vs 49 s in the script, on a GPU shared with another job (not investigated).
  21600a (2026-11-16; length series, 172 320 device orbitals, 22 slabs 3120..8485, x 378 A, own lead: Ec 0.5062, 20
  channels at Ec+0.15; Sigma cache run_sigma_21600a.sh, 25 points 1-4 s each; qtx_ms.mlevel, run_qtxms_mlevel_21600a.sh,
  qtxms_21600a_{hall,h1full}.log / .json, attelas1, GPU shared with another user's 39.7 GB). No exact psi for this
  device: iterations and residuals only (all runs res <= 1e-10). Unsplit level 1 (43 x windows of 16 A, dm 1e-2), x, x
  merges, Ec+0.15. No halo: N 32 960 / 17 788 / 10 522 (n / N 5.2 / 9.7 / 16.4), exact level 2 / 3 18-23 / 19-24, 3
  levels + top N 32 / 128 / 256 34-40 / 35-41 / 32-37, level-2 rho 131 s in 2 batches, run 1307 s (852 s = the 43
  window density matrices, now cached in /scratch/jiacao/mg_modespace/21600a/dmw). Halo 1 for rho + full smoother
  windows: N 32 960 / 16 114 / 10 275, exact 19-25 / 19-25, top 32-39 / 30-36 / 28-33, level-2 rho 73 s in 1 batch
  (windows 1586..4762..4953 -> 772..3179..3382 columns, predicted cost 0.304), run 390 s. Couplings |A1| dx 1 51.0 eV,
  dx 2 0.21 (0.4%). GPU peak 27.6 GB; fine solve 20-33 s per 20 columns. => halo 1 for rho on the unsplit device too:
  ~1 iteration on the exact levels, N2 -9% (the opposite sign of the y-cut 2808a, +6%), rho 1.8x faster, top level 2-5
  iterations better (slow-mode rounds settle: |theta|max 0.13 vs 0.99 in the last round). Length scaling against the
  unsplit 9072a reference (mlevel_9072a_xx_rows.log: exact 16-20, top 23-28 / 21-26 / 19-23, N 13 674 / 7275 / 4283):
  2.4x the length -> +2..5 iterations on the exact levels, +9..15 with the top level (the top N 32-256 does not scale
  with the length; 11 final blocks instead of 5); n / N per level the same (5.3 / 10.0 / 17.0 on 9072a). The y-cut
  case (SPLIT=2,1, halo 1 + full smoother, needs 75 GB free) had not started when this was written.
  Exact psi of 21600a (run_exact_21600a.sh, mb.exact_solve on the GPU of attelas2: 24 s, residual 8.6e-14, T_exact(20)
  0.937877 at Ec+0.15; results_ms/gpu/wf_gh200_21600a/psi/psi_+0.65619413.npz, use --psi-dir). No-halo case again
  with it (attelas1, 476 s): exact level 2 / 3 T err -3.3e-12 / -3.8e-12, psi err 4.8e-12 / 4.6e-12; top N 32 / 128 /
  256 T err +2.2e-12 / -6.2e-13 / +3.7e-12, psi err 7.2e-12 / 3.6e-12 / 5.1e-12, its 32-38 / 35-41 / 33-38 (top
  counts vary by 1-2 between runs). Bug: CapSchwarz transverse split on 21600a failed ("window cores must partition
  the device"): 3 atoms at y = 0 exactly, the tie rule (du == L / 2n -> box 0) broke by rounding (du1 - L/4 =
  -1.8e-15) -> covered twice. Fixed: owner box = floor(u mod L / (L / n)) as in the release (6 nm has 5 such atoms and
  ran only through the release code). y-cut case restarted on attelas2 2026-11-16 10:32 (86 cores of 1092..2176).
  Halo 1 + full smoother again with the exact psi (attelas1, 411 s): exact level 2 / 3 T err +9.3e-12 / +8.7e-12, psi
  err 4.1e-12 / 4.0e-12; top N 32 / 128 / 256 T err +2.0e-12 / +2.7e-12 / +6.8e-13, psi err 5.4e-12 / 5.9e-12 /
  3.3e-12, its 33-39 / 31-37 / 28-34. y-cut on 21600a, first full attempt (qtxms_21600a_y2_h1full_oom.log): level 1 86
  blocks, N1 79 285 (n / N1 2.2; unsplit 5.2), per block 273..956..1038, A1 10.99 GB, GPU 63 GB after level 1, window
  rho 650 s in 5 batches (cached on attelas2); couplings |A1| dx 0 other y 64.3 eV, dx 1 same / other y 66.0 / 31.4,
  dx 2 0.39 / 0.011; level-2 windows halo 1 5759 columns (cost 0.221), all 9549. Out of memory (94 GB) in the
  smoother of A1: L.Sb, L.Hb, L.Ab (3 x A1 = 33 GB) were freed only after it. Fixed in qtx_ms/mlevel.py: BlockOp.sub
  reads the block rows R, the three block lists are freed before ras(); restarted 10:56.
  Result (qtxms_21600a_y2_h1full.log, attelas2 alone, 609 s with the window rho cached): N 79 285 / 46 185 / 27 096 /
  16 132 (n / N 2.2 / 3.7 / 6.4 / 10.7; 86 / 43 / 22 / 11 blocks), GPU 52.2 GB after level 2 (was 94 GB + OOM), ~80 GB
  peak during the level-3 rho; rho 181 s (6 batches, halo 1) / 131 s / 53 s. Exact level 3 / 4 12-15 / 12-15 its, T
  err -3.7e-13 / +5.2e-13, psi err 2.6e-12 / 3.4e-12 (exact level 2 skipped: N > --exact-max 30000); 4 levels + top N
  32 / 128 / 256 27-32 / 26-30 / 23-27, T err <= 1.4e-12, psi err <= 4.0e-12; fine solve 7-13 s per 20 columns
  (unsplit 21-32 s, shared GPU). => with the y cut the exact-level count does not grow with the length (9072a same
  settings 14-17, 21600a 12-15; unsplit 16-20 -> 19-25); with the slow-mode top it grows (9072a 16-24 -> 23-32).
  Price: level 1 n / 2.2, last level 16 132 (unsplit 10 275), 3x the GPU memory: one GH200 is nearly full, the width
  series (6 nm) does not fit this way -> "smaller level 1" and "memory of mlevel.py" stay the open points.
- PoU slab basis on 6 nm, stand-alone MS (results_ms/gpu/pou/pou_ms_6nm_d4.log, delta 4 Å, 6690 s), T err at Ec + 0.02 /
  0.05 / 0.15 / 0.3: block-diagonal slab level 1 1e-8 (N 38 054) +7.3e-3 / −6.6e-4 / −1.4e-4 / −8.7e-5; chiv 1e-8 (N
  41 225) −6.6e-2 / −3.9e-5 / +9.8e-5 / +1.1e-5; "hard" 1e-8 (N 22 648) −1.1e-1 / +1.2e-3 / +2.2e-3 / −4.0e-6; chiv 1e-6
  (N 18 633) +5.5 / +4.2e-4 / +2.6e-4 / +7.9e-5; "hard" 1e-6 (N 10 592) +176 / +0.49 / −1.2e-2 / +4.8e-6. chiv 10–20x
  better at Ec+0.05 / 0.3, equal at 0.15, worse at 0.02 (T 1e-8); best-in-span 3–4x better but the Galerkin psi error of
  the strong channels stays 1e-2…1.8. "hard" is not more compact at equal accuracy on 6 nm (unlike 2808a). Mixed gain.
- Iterated tile level 1 (user, 2026-11-13; results_ms/gpu/pou/tile_iter.py, run_tile_iter_2808a.sh, tile_iter_2808a_*.log;
  2808a, cut 1e-6, iteration 0 = 6 Å CAP shell (N 6400), then each tile re-solved with its neighbour tiles as the
  environment, all tiles updated together; reference = 12 Å CAP shell, 8 tiles N 2264, T err ≤ 2.5e-4, missed exact
  occupation 6.3e-4). All negative:
  * env = neighbours in their basis: diverges (local / exact trace up to 4.4, missed occupation 1.1e-2 → 7e-2, T err
    1.7e-2 → 7.9e-2), same mechanism as tile_scf.py (environment seen only through span(U) lets cut states into the window).
  * "hybrid" (fine 6 Å shell + neighbour bases beyond): trace right, N 1692–1803, missed occupation 3e-3–1.2e-2, no
    settling (30–50% of new vectors outside the previous span; iteration 3 −21% at Ec+0.05, σ_min 2.8e-5).
  * neighbours' CAP projected into their basis (`--env-cap outer`): N 2262 → 2686 → 3273, trace up to 3.9, T err 2.8e-4 →
    3.7e-3 → 3.2e-3. With tiles y split 2, x = the 4 slabs (`--tsplit 2,1`; reference N 2055, T err ≤ 1.8e-3):
    iteration 2 N 2382 T err ≤ 1.1e-4 (best tile variant), but N keeps growing 250–400 per iteration.
  * cluster ρ (centre tile + neighbours in their basis) restricted by a PoU ramp (8 / 15 Å): chiv N 7874 / 8300 (no
    better T, one energy 5e-3), chirho N 2279 / 2640 (≈ reference at 8 Å, −5% at Ec+0.05 for 15 Å); cluster ρ
    over-occupies the centre tile by 1–11% in trace.
  * spurious coarse states never reduced: σ(A_c) < 3e-2 count 15–21 (reference) → 23–38, all almost real, fine residual
    6–7 eV; a wider ramp is worse. "spurious vectors" metric (exact occupation < cut/10) = 0 for every basis: useless.
  => stop iterating tile bases with neighbours represented by their basis vectors; the fine extended embedding (slab
  PoU) is better (inference across different cuts / partitions, not a controlled test).
- CSCS Alps (plan sessions/2026-11-11-alps-env-plan.txt; not yet run there): install into the QuaTrEx uenv venv
  (prgenv-gnu/26.3, cupy-cuda13x, mpi4py on cray-mpich) with scripts/setup_env.sh (reuses the venv's cupy / mpi4py,
  CUDA from $CUDA_HOME, kernel caches on $SCRATCH); job templates scripts/alps/{gpu_bind.sh,basis,sweep,exact}.sbatch
  (1 process per GPU, --l1-only split of the basis build, energy split; exact = 1 task per node, --mem=0). Shared cache
  writes (Sigma, level-1 blocks, lead edges) are atomic (temp file + os.replace).
