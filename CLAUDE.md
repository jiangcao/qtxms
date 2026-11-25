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

## Decisions and findings (digest)
The full log (every experiment with numbers, scripts, log files, negative results) is in `notes/DECISIONS.md`:
read it only when touching architecture (method, levels, bases, smoother, coarse level), and add new results
there, not here. The module-by-module code state, verified results and known limitations are in
`notes/code-state.md`. Change this digest only when a standing decision changes.

Standing decisions
- Mode space is a coarse level / deflation space, not a stand-alone approximation, inside flexible GMRES;
  stand-alone MS is a separate single-shot product (T second-order accurate, psi / charge 1e-3–1e-2).
- Goals: T and psi / charge accuracy, both. The residual is a bad accuracy metric.
- Galerkin coarse operator A_c = U†AU; exact lead Σ on the fine level only. Minimal-residual coarse solves are worse.
- Smoother = CAP-Schwarz (RAS) windows ≥ ~16 Å along x with an absorbing layer on the artificial cuts; closed local
  solves amplify the error; in-plane (y) cuts need ≥ 16 Å windows too; post-smoothing is essential; Krylov needed.
- Solver (user, 2026-11-25) = density-matrix hierarchy `qtx_ms.hier` stopped at a level that is solved exactly by a
  block-tridiagonal LU (`--bottom bt`, default): level 3 (level 2 is faster at 6 nm but too large for wider devices),
  V-cycle in FGMRES. Deeper merged levels cost iterations.
  Slow modes of the smoother / cycle as the only coarse level (`--slow`, `--fgmres-coarse slow`): enough on 2808a and at
  low energy, not at 6 nm above Ec + 0.15 (energy specific, their number grows steeply with energy and device size).
- Mode-space bases keep almost-real spurious Galerkin states (cut states at block faces).
- Bases are density matrices over a complex contour (level 1 local ρ with CAP + lead Σ embedding, level 2 global ρ);
  label a basis by partition, level and cut. Lead CMS, FEAST probes, Beyn pole bases, tile SCF / iterated tiles, PoU
  and SA-smoothed bases, per-core cuts of slow modes: tried, negative or no gain (see the log).
- The basis must follow the potential (OFF state / bias): flat-band bases fail below a barrier; the FGMRES correction
  removes that failure.
- Oxide devices (MoS2/HfO2): direct level 2 or slab level 1, not lead CMS. End-slab lead-band RMS cleanup only.
- H/S truncation only as block-tridiagonal trim, never a magnitude cut.
- QuaTrEx blocks are Γ-point supercells: use meta["periodic"] minimum image on all periodic axes.
- Exact T is cached per input set in results_ms/exact/<name>.json with a fixed reference Ec; runs refuse another Ec.
- Injection channels: Γ_L eigenvalues > 1e-4 eV; `core.injection` returns them ascending (strongest = last columns).
- Density-matrix hierarchy (2026-11-15): merges with cluster-block smoothers are lossless, energy slices coarsen
  geometrically; open: smaller level 1, memory of mlevel.py, 6 nm.

Operational
- Hosts: mont-fort15 (x86, P100, `.venv`), attelas1 / attelas2 (aarch64 GH200, `.venv-aarch64`, no pypardiso /
  qttools / uv / git -> folded lead at E + 1e-6 i). GPU backend via `--gpu` or QTX_MODESPACE_GPU=1.
- Big caches (contour Σ, ρ, window LUs) go to /scratch/jiacao (per host), never the repo NFS.
- Build releases on attelas2 with /usr/sepp/bin removed from PATH (the SEPP git wrapper pollutes the sdist).
- Python wall times on the CPU are not representative; compare iteration / matvec counts.
- Reports go to Mathieu Luisier; run scripts are listed in job.sh.

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
Environment setup and run commands are in `docs/HOWTORUN.md`. Read and update it only when adding new modules or
making a new release.
