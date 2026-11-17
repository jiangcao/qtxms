# Mode space as the coarse level of a multigrid transport solver

Mode-space (basis-reduced) quantum transport for rough nanosheet / nanowire FETs, made exact by using the mode-space
basis U as the coarse level of a multigrid / deflated-Krylov solver for the open-boundary Schrödinger (QTBM) equation
A ψ = b, A(E) = E S − H − Σ_L − Σ_R, instead of as a stand-alone approximation. Stand-alone mode space remains a
separate single-shot product (T second-order accurate, ψ / charge 1e-3…1e-2).

Status (2026-11-17)
- Effective mass and synthetic TB: two-level GMRES with a density-matrix basis converges in a length-independent
  7–9 iterations.
- MoS2/HfO2 (QuaTrEx, 29k–172k orbitals): stand-alone mode space reaches T errors of 1e-5…1e-4. Flexible GMRES with a
  CAP-Schwarz smoother converges to 1e-10 (T error ~1e-11). The coarse level is either the mode-space basis, slow modes
  of the smoother rebuilt per energy, or a multilevel density-matrix hierarchy (`qtx_ms.mlevel`).
- The self-contained release `release/qtx_modespace` is at 0.4.0.

## Naming

- **Slab**: one block of the block-tridiagonal partition along transport (x); covers the whole cross section.
  **Tile**: a piece of a slab (transverse bisection, optionally also cut along x).
- **Level 1 = local density matrix**: per block (tile or slab), ρ of the block embedding closed by a CAP, or by a
  CAP + the exact lead Σ on the lead faces ("CAP+OBC"). U⁽¹⁾ = its dominant eigenvectors per block.
- **Level 2 = global density matrix**: ρ of the whole open device (RGF). The basis is the dominant eigenvectors of its
  diagonal blocks. It is either built on level 1 (U = U⁽¹⁾V) or directly on the fine device ("direct level 2").
- Both ρ are contour integrals over the energy window. Label a basis by partition, level and cut, e.g.
  "slab level 1, cut 1e-8".

## Repository layout

| path | content |
|---|---|
| `qtx_ms/` | research package (all drivers: `python -m qtx_ms.<name>` from the repo root) |
| `qtx_ms/prep/` | input preparation: Si sheet (sp3s*, sp3d5s*), QuaTrEx / deviceBuilder, GPAW MoS2 |
| `release/qtx_modespace/` | self-contained, pip-installable release (own README) |
| `workflows/` | run / rebuild / summary scripts (Σ caches, exact ψ, bases, cut sweeps); invoked from `job.sh` |
| `tests/` | `test_msbasis.py`, `test_slow_coarse.py`, `test_mlevel.py` (pytest) |
| `job.sh` | the list of run commands behind every reported result |
| `inputs/`, `results_ms/`, `examples/`, `reports/`, `notes/`, `docs/` | local only, not in git |

## Setup

```
uv sync                                    # .venv: Python 3.12, numpy, scipy, pypardiso (x86-64)
uv run python -m qtx_ms.run_tb --help        # or: source .venv/bin/activate
uv run pytest tests
```
GPU (cupy) backend: `--gpu` or `QTX_MODESPACE_GPU=1`. On the aarch64 GH200 hosts use `.venv-aarch64` (no pypardiso or
qttools; the lead is folded at E + 1e-6 i). Put large caches (contour Σ, ρ, window LUs, ψ) on the local `/scratch`,
not on the repo file system.

---

# 1. Effective-mass prototype (`device.py`, `em.py`, `run.py`, `scaling.py`)

Rough Si nanosheet (5 × 3 nm, a = 0.25 nm, m* = 0.26, 3 eV oxide barrier, top/bottom roughness rms 0.3 nm, Λ = 1.5 nm,
uniform ideal leads). Reference: exact RGF.

Method
- Prolongation P = U (block-diagonal per-slab basis), restriction U†, Galerkin A_c = U† A U.
- Coarse-grid projector π = U A_c⁻¹ U† A (idempotent to ~1e-14). The mode-space solution is π ψ_exact.
- Right-preconditioned GMRES on A ψ = b (b = Γ_L^{1/2} injection columns), started from the mode-space solution.
- Balancing two-level preconditioner M⁻¹ = P_c + (I − P_c A) S (I − A P_c), P_c = U A_c⁻¹ U†.
- Smoother S: Chebyshev polynomial for (−A)⁻¹ on [E_b − E, λ_max]. It is valid in effective mass because the complement
  is (nearly) definite.
- The fine level uses the exact lead Σ; only the coarse solve sees U†ΣU.

Bases: CMS M (lowest M eigenvectors of each slab H_ii), LDM E_b (local density matrix of 2w+1-slab clusters with
zero-flux ends, states below e1 + E_b, eigenvectors with weight > 1e-3 · max).

Results (`python -m qtx_ms.run 20`: 80 slabs × 320 = 25 600 points, 6 energies 10–160 meV above the first subband,
tol 1e-8, Chebyshev degree 8):

| basis | N/slab | mode-space max T error | after GMRES | GMRES its (mean) | fine matvecs / RHS |
|---|---|---|---|---|---|
| none (smoother + GMRES only) | 0 | — | 7e-8 | 41.7 | 459 |
| CMS M=2 | 2 | 85% | 4e-8 | 15.8 | 175 |
| CMS M=16 | 16 | 32% | 1e-10 | 14.8 | 164 |
| LDM E_b=0.6 eV | 5.3 | 3.3% | 1e-10 | 8.9 | 99 |
| LDM E_b=1.0 eV | 10.0 | 1.2% | 2e-11 | 6.8 | 76 |

Device-length scaling (`python -m qtx_ms.scaling`, 3 energies):

| L | n | no coarse: its / matvecs | LDM E_b=1: N, its / matvecs |
|---|---|---|---|
| 10 nm | 12 800 | 32.5 / 342 | 383, 6.8 / 74 |
| 20 nm | 25 600 | 46.0 / 485 | 799, 7.0 / 78 |
| 40 nm | 51 200 | 76.2 / 789 | 1648, 7.0 / 78 |

Takeaways: the coarse space makes the iteration count independent of device length. The basis no longer needs to be
accurate on its own (a 1.2%-error basis converges to 1e-11 in 7 iterations). What matters is how high in energy the
basis reaches (E_b), because E_b sets the smoother's condition number.

---

# 2. Tight-binding input (`abtb.py`, `twolevel.py`, `run_tb.py`)

```
python -m qtx_ms.run_tb --xyz device.xyz --H H.mtx --S S.mtx --norb "Si:10,H:1" --units eV \
                 --dE 0.02,0.08,0.15 --Eb 0.6 --basis ldm,slab
```

Input
- `--xyz`: standard XYZ. The atom order must match the matrix orbital order.
- `--H`, `--S`: `.npz`, `.mtx`, or text triplets `i j Re [Im]` (0/1-based auto-detected). Upper-triangle-only files are
  completed to Hermitian. If `--S` is omitted, the TB is treated as orthogonal. `--units eV|Ry|Ha` scales H.
- `--norb`: one integer, a per-species list (`Si:10,H:1`), or a file with one integer per atom.
- Assumes a device-only matrix, the orbitals of an atom stored consecutively, and device ends that repeat the lead cell.

What it does: sorts and bins atoms into slabs (coupling range, or `--slab-width`), asserts block tridiagonality, builds
Sancho–Rubio leads and injection, finds Ev/Ec from the lead bands (basis window [mid-gap, Ec + Eb]), builds the `ldm` /
`slab` bases, and solves with flexible GMRES. The default smoother is 10 GMRES steps (indefinite complement). It reports
σ_min(A_c) to flag spurious bands, with RGF as the reference (`--no-ref` to skip).

Synthetic non-orthogonal test (`python -m qtx_ms.make_test_tb`, rough H-passivated wire, 5928 orbitals):

| E − Ec | channels | none: its / matvecs | ldm (N=221): MS err → final, its / matvecs | slab M=4 (N=240): MS err → final, its |
|---|---|---|---|---|
| 0.15 eV | 2 | 56 / 616 | 3.4e-3 → 5e-10, 8 / 105 | 2.5e-1 → 6e-10, 10 |
| 0.25 eV | 2 | 90 / 984 | 2.4e-4 → 6e-10, 9 / 118 | 2.2e-2 → 6e-10, 12 |

---

# 3. Generic input sets (`qtx_ms/prep/`)

Every TB model, empirical or ab initio, is converted into one generic input set (`prep/inputs.py`,
`write_inputs` / `read_inputs`):

| file | content |
|---|---|
| `device.xyz` | atoms in orbital order, `label x y z` [Å] |
| `norb.txt` | orbitals per atom |
| `blocks.txt` | transport block (slab) index per atom |
| `H.npz`, `S.npz` | device H [eV], overlap (`S = I` for orthogonal TB) |
| `lead_{H,S}{00,01}.npz` | periodic lead cell and its coupling to the next cell (+x) |
| `meta.json` | units, axis, block length, device / lead blocks, band edges, transverse periods (`periodic`) |

```
python -m qtx_ms.prep.si_sheet --model sp3d5s --xsec 3x5 --out inputs/si3x5_sp3d5s    # rough Si sheet + oxide
python -m qtx_ms.prep.quatrex examples/device_MoS2HfO2_2808a_29088o --out inputs/mos2_hfo2          # 5 cells/block
python -m qtx_ms.prep.quatrex ... --out inputs/mos2_hfo2_trim_c3 --cells 3 --trim                     # tridiagonal trim
```
QuaTrEx blocks are Γ-point supercells: the minimum-image convention is applied on the periodic axes recorded in
`meta["periodic"]`. Truncate H/S only with the block-tridiagonal trim (`--trim`): the magnitude cut (`--cut`) costs
1–40% in T.

---

# 4. Mode-space bases on QuaTrEx MoS2/HfO2 devices

Devices (deviceBuilder, non-orthogonal; Mo 26, S 13, Hf 10, O 4 orbitals per atom; lead cell 3120 orbitals):
2808a (29 088 orbitals), 9072a, 21600a (172 320 device orbitals, 2808a cross section at 6 nm length), 6 nm, 12 nm
(width series).

Drivers
- `msbasis.py` / `run_ms.py`: level 1 (tiles / cores, optional enrichment) → level 2 → MS1 / MS2 vs exact (`--direct-l2`).
- `run_l2_clean.py`: direct level-2 basis with end-slab lead-band cleanup (`lead_rms.py`).
- `exact_T.py`: exact QTBM T, cached in `results_ms/exact/<name>.json` with a fixed `--Ec` (a cache built with a
  different Ec is refused).
- `ms_scan.py` (a saved basis on a fine energy grid), `density_scan.py` (T, ψ, Mulliken charge vs exact).
```
python -m qtx_ms.exact_T --inputs inputs/mos2_hfo2 --dE 0.02,0.05 --Ec 0.50435
python -m qtx_ms.run_l2_clean --inputs inputs/mos2_hfo2 --nmodes 200,400 --no-res --dE 0.05,0.15,0.3,0.45 \
       --exact-from results_ms/exact/mos2_hfo2.json --Ec 0.50435
python -m qtx_ms.ms_scan --inputs inputs/mos2_hfo2 --modes results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead \
       --dE 0:0.5:0.01
```

Stand-alone mode-space T error on 2808a (window [Ec − 0.15, Ec + 0.6] eV, 24 contour points):

| basis | N | max T error |
|---|---|---|
| direct level 2, 100 per slab | 400 | 15–40% |
| direct level 2, 200 per slab | 800 | ≤ 1e-4 |
| direct level 2, 400 per slab + end-slab cleanup | 1636 | 4.3e-5 (10 meV grid, 0–0.5 eV) |
| slab level 1 CAP+OBC, cut 1e-8 | 2373 | ≤ 2.5e-4 (Ec + 0.05 / 0.15 / 0.2) |
| trim 3 cells, direct level 2, 200 per slab | ~800 | ≤ 3e-5 |

Negative results (details in the local notes): lead CMS (−17…−39%), FEAST probes, Beyn pole bases, tile SCF /
iterated tiles, partition-of-unity and SA-smoothed bases, minimal-residual coarse solves.
The basis must follow the potential: bases built at flat band fail below a gate barrier. The FGMRES correction removes
that failure.

---

# 5. Iterative solver: CAP-Schwarz smoother + flexible GMRES (`schwarz.py`, `cap_fgmres.py`)

GMRES(k) as a smoother stalls on MoS2/HfO2 (residual ~2e-4 for any k). The working smoother is restricted additive
Schwarz over overlapping windows along x (≥ ~16 Å) with an absorbing CAP ramp on the artificial cuts and the lead Σ on
the lead faces (`CapSchwarz`; optional transverse split, ≥ 16 Å windows in y). Closed local solves amplify the error,
and post-smoothing is essential. Outer solver: block flexible GMRES over all channels (`fgmres_block`).

Coarse levels
- Mode-space basis (Galerkin A_c = U†AU): 2808a, slab level 1 1e-8: 30–46 its up to Ec + 0.2 eV vs one-level 56–86.
  Mode-space bases keep almost-real spurious Galerkin states (cut states at block faces), so this can stall higher up.
- **Slow modes** of the smoother / cycle, rebuilt at every energy (`SlowCoarse`, `--coarse slow`; m vectors per round,
  q subspace steps, a few rounds): 2808a 11–20 its at Ec + 0.05…0.2 (T error ≤ 1.2e-11). 6 nm (40 channels): 28–36 its.
  This is the standing default.
```
python -m qtx_ms.cap_fgmres --inputs inputs/mos2_hfo2_9072a --dE 0.05,0.2 --workers 4 --exact            # one-level
python -m qtx_ms.cap_fgmres --inputs inputs/mos2_hfo2 --dE 0.05 --coarse slow --slow-m 64 --slow-q 3 --slow-rounds 3
```

---

# 6. Density-matrix hierarchy (`mlevel.py`)

Multilevel preconditioner built from density matrices with repeated merging:
- Level 0: the fine A with the CAP-Schwarz smoother.
- Level 1: contour ρ of each Schwarz window restricted to its core (`dm` or `dmc` selection, cut).
- Level k + 1: pairs of level-k blocks merged along x or y. ρ of each pair window is computed in the level-k basis
  (projected CAP and Σ, optional halo, optional narrower energy slice). Galerkin operators are block-sparse, and each
  level gets its own RAS smoother.
- Top level: an exact solve or slow modes.

Each level of the V-cycle is applied inside FGMRES.

```
python -m qtx_ms.mlevel --inputs inputs/mos2_hfo2 --dE 0.05 --gpu --sigma-cache /scratch/jiacao/mg_modespace/sigma_2808a \
    --psi-dir results_ms/gpu/wf_gh200/psi --dm-cache /scratch/jiacao/mg_modespace/geneo/dmw --cap 16,2,2 --cap-split 2,1 \
    --cap-tov 6 --cap-solver rows64 --l1 dm:1e-2 --up y:1e-2,x:1e-2 --halo 1 --sweep exact   # 2808a, ~3 min on a GH200
```

Findings on 21600a (Ec + 0.15 eV, 20 channels, dm cut 1e-2, halo 1):
- y-cut level 1, then merges to 6 levels (N 79 285 → 9189): 10–15 FGMRES its at every level. T error ≤ 1.3e-11,
  ψ error ≤ 1e-11.
- Unsplit (no y cut): 18–25 its.
- Merges below level 2 are lossless, both in the plain-MS error and in the iteration count.
- Energy slices cost FGMRES little (12–15 → 16–22 its), but they break stand-alone MS, so slice bases serve only as
  coarse levels.
- The basis is ~70% of a one-energy run (y cut, 634 s).
- Open: a smaller level 1, memory use, and the 6 nm device.

---

# Release (`release/qtx_modespace/`, 0.4.0)

A self-contained package (numpy, scipy, h5py; optional pypardiso, qttools, cupy). It reads a QuaTrEx device folder,
builds the level-1 / level-2 bases, and writes `.npy` bases with QuaTrEx orbital indices. Optional outputs: T_MS,
T_exact, FGMRES with CAP-Schwarz and a mode-space or slow-mode coarse level (`--fgmres-coarse ms|slow|none`,
`--basis none`), and the closed system. GPU backend and streaming exact solve. Setup, workflow scripts and tested
numbers are in `release/qtx_modespace/README.md`. `qtx_ms` is the development superset; release-only features are
ported back into it.
