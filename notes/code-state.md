# Code state

Moved verbatim from CLAUDE.md on 2026-11-15.

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
- `mlevel.py` (2026-11-15) — density-matrix hierarchy with repeated merging as a library + driver
  (`python -m qtx_ms.mlevel`), the qtx_ms form of results_ms/gpu/mlevel/mlevel.py (which runs on the release stack
  and stays as the experiment script). `Hierarchy(dev, inp, sw)` on a `schwarz.CapSchwarz` fine smoother: `window_dm`
  (cached contour rho of the S0 windows), `basis1` (dm | dmc), `level1(U, E, SL, SR)`, `merge(axis, cut, kind, mu1,
  mu2, npts, halo)` per further level (pair windows in the level-k basis, projected CAP / Sigma, Galerkin S / H / A,
  smoother of A_k as RAS over the pair windows; `halo` = x halo of the windows in blocks, `halo_smooth`; logs the
  coupling blocks of A_k, S_k by distance and the window size / rho cost per halo, also in `Lev.info`), `cycle(m,
  bottom, A0, S0, gamma)`, `exact(m)`, `own_smoother()`. Driver: sweeps fine / exact / top as the script, `--halo`,
  `--out` json; reads the release caches (Sigma files, window density matrices, psi_<E>_<eta>.npy) and the qtx_ms
  psi_<E>.npz. tests/test_mlevel.py (synthetic strip, CPU: Galerkin level operators to 1e-9, FGMRES with exact level
  1 / 2 / 3 reaches the direct solution, halo 0 with both smoother choices, dmc + own-block smoother, rows = LU
  smoother). GPU check on 2808a (results_ms/gpu/mlevel/run_qtxms_mlevel_2808a.sh): same N and iterations as the script.
- `schwarz.py` additions for it (2026-11-15, twins of the release schwarz.py): `CapSchwarz` solvers rows / rows64 (core
  rows of the window inverses), GPU windows when gpu.on (every solver but pardiso; r and result cupy), `ramp(k)`,
  `free()`, `set_energy(..., log=)`; `GpuOp`, `FastOp`, `operator`; `fgmres_block` takes numpy or cupy blocks (the
  release algorithm, `clock` kept). Not ported: inv / inv64 batches, host-resident streamed LUs.
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

