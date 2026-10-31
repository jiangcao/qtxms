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
- All models go through one generic input set (mgms/prep/inputs.py: xyz, norb, blocks, H, S, lead cell,
  meta.json), so solvers and bases never call a TB builder directly; model-specific extras (Si structure.npz)
  are optional side files.
- Per-slab "slab" (CMS-like) basis in TB is selected by count (M lowest above mid-gap), not by window,
  because isolated-slab energies sit above the device band edge.
- QuaTrEx devices with oxide (MoS2/HfO2): basis = direct level 2 (top-K eigenvectors of the contour slab ρ of
  the open device, U⁽¹⁾ = I), not lead CMS — the lead-CMS basis cannot represent oxide / displaced atoms (−17…−39%).
  Count K per slab (not an occupation threshold); end-slab spurious lead bands removed with RMS (generalized S).
- H/S truncation only as block-tridiagonal trim (drop couplings beyond the next block), never a magnitude cut:
  trim c4 0.18%, c3 0.86% vs cut 1e-4 1.35%, 1e-3 6.7% in exact T.
- Exact T cached per input set in results_ms/exact/<name>.json keyed by dE with a fixed reference "Ec"
  (--Ec 0.50435 for mos2_hfo2); runs refuse a json with another Ec.

## Code state (package `mgms/`; Python 3, numpy, scipy only)
All modules live in `mgms/`; drivers are run from the repo root as `python -m mgms.<name>`
(outputs written to the current directory). `mgms.py` was renamed `mgms/em.py`.
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

QuaTrEx / MoS2-HfO2 track (see README section "QuaTrEx devices")
- `prep/quatrex.py` — deviceBuilder folder → generic inputs (`--cells`, `--trim`, `--cut`, `--cut-S`, `--potential`).
- `exact_T.py` — exact QTBM T by sparse direct solve; `--json` merges into results_ms/exact/<name>.json, `--Ec`.
- `run_l2_clean.py` — direct level-2 basis (`--nmodes` sweep), end-slab RMS cleanup, optional block residual
  enrichment, MS T vs exact, `--mg` two-level GMRES; `--exact-from`, `--Ec`; saves modes/<tag>_slab<s>.npy.
- `run_lead_cms.py` — lead-CMS basis on lead-like atoms of every slab (+ `--mg`). Negative result.
- `lead_rms.py` — lead bands in a slab basis, spurious-band removal (Huang et al. 2018, non-orthogonal S).
- `ms_scan.py` — saved basis on a fine energy grid vs exact (coarse-first order, exact json reuse).
- `release/qtx_modespace/` — self-contained package (numpy/scipy/h5py, optional pypardiso): QuaTrEx folder →
  contour ρ → basis npy (+ QuaTrEx orbital indices) → optional T_MS / T_exact. Checked bit-equivalent to mgms
  (results_ms/release_check/).

What works (verified)
- Effective mass, 20 nm (25.6k points): LDM E_b = 1 eV (N = 799) → 7 GMRES its / ~76 matvecs per RHS,
  final T error ~1e-11 (MS alone 1.2%); no coarse: 42 its / 459 matvecs.
- Length independence: 7.0 its at 10/20/40 nm with LDM; no coarse 33 → 46 → 76 its.
- TB synthetic (5928 orbitals): LDM (N = 221) 8–9 its, final error ~5e-10; slab M=4 10–12 its; no coarse 56–90.
- Reader: .npz / .mtx / text, per-species / per-atom orbitals, shuffled atoms all give identical T.
- MoS2/HfO2 (29 088 orbitals, slabs 3483/8383/7862/3120): level 2 n200 (N 800) T error ≤ 1e-4; n400 + end-slab
  cleanup (N 1636) ≤ 4.3e-5 on a 10 meV grid 0–0.5 eV; n100 15–40%. 3-cell trim n200 ≤ 3e-5.
  MS solve ~1 s vs exact 250–650 s per energy; lead Σ 170–300 s per energy; basis ~1.5–2 h (contour Σ-dominated).

Input preparation (`mgms/prep/`, see README)
- `inputs.py`  — `write_inputs` / `read_inputs` of the generic input set.
- `si_sheet.py`— CLI `python -m mgms.prep.si_sheet --model sp3s|sp3d5s --xsec 3x5|3x15|5x10 --out DIR`;
  also writes `structure.npz` + TB params; `load_structure(dir, meta)` → (Structure, P).
- `structure.py`, `hamiltonian.py` (sp3s*), `hamiltonian5.py` + `sp3d5s.py` (sp3d5s*), `model.py` — moved
  unchanged from qtmode (only `model.device_cfg(xsec=None)` gained an argument). `model.select()` still mutates
  `structure.NORB` globally. Verified: H, lead H00/H01 bit-identical to ../qtmode_cms (3×5, both models).

User's two-level mode-space code (`qtmode/`, from ../qtmode_cms; pipeline scripts live there, not here)
- `cms.py` level 1 (partition into cores/tiles, tile spectral modes, union core basis), `enrich.py` residual
  enrichment, `lsq.py` slab blocks / transmission, `lead_ms.py` reduced leads, `transport.py`, `negf.py`,
  `reduced.py`, `clean.py`. Imports structure/TB from `mgms.prep`. Not yet wired to the generic inputs.

Untested / known limitations
- No real ab initio files yet (CP2K/Wannier/OMEN; sp3d5s*, d orbitals, spin, complex H).
- The user's two-level basis (qtmode) is not yet coupled to mgms: the mgms LDM is a single-step per-slab basis.
- Only two grids; no three-level V-cycle.
- LDM build uses dense (generalized) eigh of (2w+1)-slab clusters: fine here (1.5 s TB, 76 s EM) but will not
  scale to sp3d5s* slabs with thousands of orbitals.
- Python wall times are not representative; compare matvec counts / flops. In the TB test RGF is still faster
  (≈100 orbitals per slab); the advantage is expected for large n_slab.
- A_c rebuilt per energy (U†HU, U†SU not cached); Σ added as dense blocks (dominates nnz for small devices).
- Lead check requires 3 identical end slabs (warns if only 2).
- `run_tb.py` Chebyshev β is a crude estimate; use the GMRES smoother for TB.
- No bias / Poisson / SCBA loop, no energy-adaptive basis, no carrier density (needs diag(G) or many RHS).
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
1. Make qtmode level 1 run from the generic inputs (read_inputs) with tiles only (no cores) for ab initio;
   keep cores as a Si-sheet option via structure.npz.
2. Wrap qtmode's U⁽¹⁾, U⁽²⁾ in a 3-level version of `twolevel.TwoLevel`;
   test on the prepared Si sheet inputs vs the current 2-grid numbers.
3. Run `run_tb.py` on a real ab initio device (start small); set `--slab-width` to the lead cell length.
4. Cache U†HU and U†SU; rebuild only A_c(E) = E·U†SU − U†HU − U†ΣU per energy.
5. Replace dense cluster eigh in `basis_ldm` with a sparse/randomized construction.
6. Write up the formulation + results for Mathieu.
7. Faster lead Σ (cache across runs exists; need a faster decimation / eigenvalue-based Σ for 3120-orbital cells).

## How to run
Environment: `uv sync` (Python 3.12, numpy, scipy, pypardiso/MKL on x86-64; `mgms` and `qtmode` installed
editable). Prefix commands with `uv run`, or activate `.venv`.
```
python -m mgms.run 20     # EM sweep (~5 min)
python -m mgms.scaling    # EM length scaling (~5 min)
python -m mgms.make_test_tb # writes test.xyz, test_H.txt, test_S.txt
python -m mgms.run_tb --xyz test.xyz --H test_H.txt --S test_S.txt --norb "Si:2,H:1" --dE 0.02,0.08,0.15,0.25 --Eb 0.6
python -m mgms.prep.si_sheet --model sp3d5s --xsec 3x5 --out inputs/si3x5_sp3d5s   # ~2 min, writes generic inputs
```
