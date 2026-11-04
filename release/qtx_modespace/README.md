# qtx_modespace

Mode-space basis for QuaTrEx / deviceBuilder devices. It reads a QuaTrEx device folder, builds a per-slab mode-space
basis from the slab density matrices of the open device, and saves the basis as `.npy` files. Optionally it computes
the mode-space transmission and checks it against the exact transmission.

Dependencies: numpy, scipy, h5py. Optional: pypardiso (fast sparse LU for the lead interior and the exact solve,
x86-64), matplotlib (transmission plot).

```
pip install .                 # or: pip install ".[fast,plot]"
python -m qtx_modespace DEVICE_FOLDER --out ms_out                                   # basis only
python -m qtx_modespace DEVICE_FOLDER --out ms_out --transmission --exact --dE 0:0.5:0.05
python -m qtx_modespace DEVICE_FOLDER --out ms_out --cells 3 --trim --nmodes 200     # smaller slabs, see below
python -m qtx_modespace DEVICE_FOLDER --out t_out --cells 3 --trim \
       --modes data/mos2_hfo2_trim_c3_n200/modes --transmission --dE 0:0.5:0.05      # T from a saved basis
```
`qtx-modespace` is installed as a console script with the same arguments. Set `OMP_NUM_THREADS` / `MKL_NUM_THREADS`.

Options can also come from a TOML file, `--config run.toml`, keyed by the long option names (dashes or underscores;
`folder` is the device folder); the command line overrides the file. Lists are accepted for list options
(`dE = [0.05, 0.15]`, windows as `mu = [[-0.15, 0.6], [0.5, 0.9]]`). Every run writes its resolved options to
`<out>/config.toml`; `python -m qtx_modespace --config <out>/config.toml` repeats the run (relative paths are taken
from the launch directory, which the file's first line records).
```
folder = "DEVICE_FOLDER"
out = "t_out"
modes = "data/mos2_hfo2_n400/modes"
transmission = true
exact = true
dE = [0.02, 0.05, 0.15, 0.3, 0.45]
sigma-cache = "sigma_mos2_hfo2"
psi-dir = "psi_exact"
```

## Input

A deviceBuilder folder:
```
quatrex_config.toml       [device] transport_direction, num_orbitals_per_atom, contacts (origin, lattice vectors,
                          mid_gap_energy)
inputs/structure.xyz      extended xyz, atoms in orbital order
inputs/hamiltonian.h5     group "[0, 0, 0]": coo row / col / data, attrs shape   (eV)
inputs/overlap.h5         same for S
inputs/potential.npy      optional (--potential): added as H += (V S + S V) / 2
```
Only the R = [0, 0, 0] block is used (no transverse k). Triangle-only storage is completed to Hermitian.

## Method

1. **Slabs.** The contact lattice vector defines primitive cells along transport. A slab (block) is `--cells` cells,
   by default the smallest number covering the coupling range of H and S, so that H and S are block tridiagonal.
   `--trim` drops the H, S entries that couple non-neighbouring blocks, which allows fewer cells per block (smaller
   slabs, cheaper basis); the largest dropped |H|, |S| are printed. The first and last blocks are the leads (the
   QuaTrEx contacts are the device's own end cells); the device slabs are the blocks in between.
2. **Leads.** Exact self-energies Sigma_L, Sigma_R(z): the lead-cell interior is folded onto the orbitals coupled to
   the neighbouring cells, then Lopez-Sancho decimation (complex Hermitian H, non-orthogonal S).
3. **Slab modes (level 2).** Slab density matrices of the open device,
   rho_ss = -(1/pi) Im ∫ G_ss(z) dz, G = (zS - H - Sigma(z))^-1, over the upper half circle from Ec + mu1 to Ec + mu2
   (`--mu`, default -0.15,0.6 eV; `--npts` Gauss-Legendre points), by a left-connected block recursion.
   Per slab the `--nmodes` eigenvectors of rho_ss with the largest occupation are kept, out of an eigensolve of the
   `--neig` largest (default 2 x nmodes). `--floor f` keeps only occupations > f x the largest.
4. **Optional cleanup (off by default).** The level-2 bases of the slabs in `--clean-slabs` (e.g. `--clean-slabs=0,-1`,
   the slabs next to the leads) are put on the lead cell (slab atoms matched to lead-cell atoms; rows of atoms without a lead image, e.g. oxide, dropped). Reduced lead
   bands eig(V^H H(k) V, V^H S(k) V) that are more than `--tol` from every exact lead band in
   [Ec + win1, Ec + win2] (`--win`, default -0.6,0.45 eV) are spurious; they are removed by appending optimized
   vectors one at a time (RMS method: Huang et al., JAP 123, 044303 (2018), generalized to non-orthogonal S).
   T does not need it (see the table), and its convergence varies between runs (below).
5. **Transmission (optional).** At E = Ec + dE: QTBM injection b from Gamma_L, mode-space solution
   x = U A_c^-1 U^H b with the Galerkin A_c = U^H (E S - H - Sigma) U (block tridiagonal, exact Sigma projected),
   T = x_N^H Gamma_R x_N. With `--exact` also the exact T from a sparse direct solve of the full device.

## Output (`--out`)

| file | content |
|---|---|
| `modes/slab<s>.npy` | basis of slab s: complex array (slab orbitals x modes), orthonormal columns |
| `modes/slab<s>_orbitals.npy` | for each row: the orbital index in the QuaTrEx `hamiltonian.h5` |
| `modes/slab<s>_occupations.npy` | density-matrix occupations of the level-2 modes (largest first) |
| `basis.json` | slab sizes, N per slab, window, occupations, cleanup summary per end slab, device info, arguments |
| `eig_mu…_top<neig>.npz` | cached slab eigenpairs; a rerun with the same window and `--neig` skips the contour |
| `rho_mu….npz` | slab density matrices, only with `--save-rho` (2.2 GB for the device below) |
| `config.toml` | resolved options of the run (input for `--config`) |
| `sigma/` | cached lead self-energies and lead band edges `lead_edges.json` (`--sigma-cache none` to disable) |
| `transmission.json`, `.png` | per energy: channels, T_MS, T_exact, relative error, timings; with `--exact` also the errors of the QTBM solution vectors (`err_psi`: Frobenius over channels, `err_psi_max`: worst channel, `err_psi_S`: S norm, `err_psi_last` / `err_psi_slab`: right end slab / each slab, `err_psi_best`: best approximation in range(U), `err_res`: residual), all relative to the exact solution |
| `--psi-dir` | exact QTBM solutions `psi_<E>_<eta>.npy`, reused by any later run (checked by their residual) |

The full basis is U = diag(U_0, …, U_{S-1}) over the slabs; in QuaTrEx orbital indices,
`U[orbitals_s, cols_s] = slab<s>.npy`. Orbitals of the two lead blocks are not in any slab. Inside the code the
right lead block is reordered into the left-lead atom order; device slabs keep the QuaTrEx order (contiguous index
ranges for the device below), so `slab<s>_orbitals.npy` is simply a range there.

`--modes DIR` reads `slab<s>.npy` back, checks `slab<s>_orbitals.npy` against the device (same `--cells`, `--trim`),
and skips steps 3-4.

## Included bases (`data/`)

Not in git; shipped as `qtx_modespace-0.1.0-data-<name>.tar.gz` (unpack into `data/`). Each `data/<name>/` is the
output of one basis build of the device below, in the `--out` layout above (only `modes/` and `basis.json`):
```
data/<name>/basis.json                   how it was built and checked (see below)
data/<name>/modes/slab<s>.npy            U_s: complex128, (n_s slab orbitals) x (N_s modes)
data/<name>/modes/slab<s>_orbitals.npy   int64, length n_s: row i of U_s is orbital orbitals[i] of hamiltonian.h5
data/<name>/modes/slab<s>_occupations.npy  float64, length N_s: weight of each column (largest first)
```
s = 0 … S-1 runs over the device slabs from the left lead to the right lead; the lead blocks (first and last
block) are not in any slab and stay in full orbitals.

How to read the files:
- **Orbital indices.** `slab<s>_orbitals.npy` holds 0-based row / column indices of the QuaTrEx `hamiltonian.h5` /
  `overlap.h5` (orbital order = atom order of `structure.xyz`, `num_orbitals_per_atom` orbitals per atom). They
  depend only on the device options (`--cells`, `--trim`), not on the basis; for the included devices each slab is
  a contiguous range (e.g. `mos2_hfo2_trim_c3_*` slab 0 = orbitals 1872 … 3743, after the 1872-orbital lead block).
- **Basis vectors.** Columns of `slab<s>.npy` are orthonormal in the plain inner product (U_s^H U_s = I, not
  S-orthonormal); the mode-space overlap U^H S U is formed with the projection. The full basis is block diagonal,
  U = diag(U_0, …, U_{S-1}), i.e. U[orbitals_s, columns of slab s] = slab<s>.npy, zero elsewhere. The reduced
  device matrices are
      H_red[s, t] = U_s^H H[orbitals_s][:, orbitals_t] U_t,   S_red likewise,
  nonzero for |s - t| <= 1 (block tridiagonal; with `--trim` the dropped couplings are dropped here too). The
  lead blocks couple to slab 0 / S-1 through H[lead, orbitals_0] U_0. Any potential is added to H (and S-weighted)
  before the projection: a basis is a fixed set of vectors, reused for every potential and energy.
- **Occupations.** For a single window: eigenvalues of the contour slab density matrix rho_ss of each kept column
  (electrons per mode over the window; columns sorted by it). For several windows (`*_2win`): the
  squared singular values s^2 of the merged, weighted stack (relative weights; the smallest kept s^2 / s0^2 is the
  merge threshold 1e-12). Only the ratio to the first value is meaningful: it shows how far down the basis reaches
  (about 1e-7 or below is converged for T here).
- **basis.json.** `device` (blocks, slab sizes `slabs`, lead edges `edges`, trim info), `mu` (window(s) in eV,
  absolute), `npts`, `nmodes`, `neig`, `merge_thr`, `N` (columns per slab), `clean` (end-slab cleanup, if any),
  `args` (command line), and the transmission checks (`transmission`: own solver; `quatrex_check_*`: QuaTrEx QTBM
  through `to_quatrex`, mode space vs full orbitals).

Use a basis with `--modes data/<name>/modes` and the device options of the table (the orbital check refuses a
mismatch), or apply it to QuaTrEx with `python -m qtx_modespace.to_quatrex FOLDER --modes data/<name>/modes ...`.
In Python:
```python
import numpy as np, h5py, scipy.sparse as sp
d = "data/mos2_hfo2_n400/modes"
with h5py.File("DEVICE/inputs/hamiltonian.h5") as f:          # one triangle stored: complete it
    g = f["[0, 0, 0]"]; H = sp.coo_matrix((g["data"][()], (g["row"][()], g["col"][()])), g.attrs["shape"]).tocsr()
H = H + sp.triu(H, 1).conj().T if (sp.tril(H, -1).nnz == 0) else H
U = [np.load(f"{d}/slab{s}.npy") for s in range(4)]
o = [np.load(f"{d}/slab{s}_orbitals.npy") for s in range(4)]
H01 = U[0].conj().T @ (H[o[0]][:, o[1]] @ U[1])                # reduced coupling slab 0 - slab 1
```

All included bases are direct level-2 bases of the device below, built at zero potential; single-window ones over
Ec - 0.15 … Ec + 0.6 eV with 24 points from an 800-vector eigensolve. Every check reads the
saved files back through `--modes` (own solver) or `to_quatrex --modes` (QuaTrEx).

| `data/<name>` | device options | slabs | N | T_MS vs exact (same device) |
|---|---|---|---|---|
| `mos2_hfo2_n400` | default (5 cells per block) | 4 | 400 per slab, no cleanup = 1600 | 1.4e-5 at Ec + 0.02 / 0.05 / 0.15 / 0.30 eV |
| `mos2_hfo2_trim_c3_n200` | `--cells 3 --trim` | 8 | 213 / 200 x 6 / 214 (end-slab cleanup) = 1627 | 6e-6 at Ec + 0.05 / 0.15 / 0.30 / 0.45 eV |
| `mos2_hfo2_2win` | default (5 cells per block) | 4 | 512-542 per slab = 2127 (two windows, below) | QuaTrEx: 4e-6 without, 2.2e-4 with `--potential` |
| `mos2_hfo2_trim_c3_2win` | `--cells 3 --trim` | 8 | 438-479 per slab = 3661 (two windows, below) | QuaTrEx: 5e-5 without, 4e-5 with `--potential` |
| `mos2_hfo2_2win` | default (5 cells per block) | 4 | 537 / 542 / 536 / 512 = 2127 (two windows, below) | QuaTrEx: 4e-6 without, 2.2e-4 with `--potential` |

The per-energy values are in each `basis.json`.

`mos2_hfo2_trim_c3_2win` is for biased devices. It is still built at zero potential, from two windows
(`--mu=-0.15,0.6;0.5,0.9 --npts 24,16 --nmodes 200,400 --merge-thr 1e-12`): the second window covers the states that a
potential lowering the bands (here inputs/potential.npy, 0 -> -0.2 V across the device) brings into the transport
window. Each window's modes are weighted by sqrt(occupation / largest) and the two sets are merged by one SVD per
slab (as in qtmode). QuaTrEx QTBM, mode space vs full orbitals at Ec + 0.05 / 0.15 / 0.30 / 0.45 eV with the
potential: single-window basis (N 1627) up to 0.45% (at Ec + 0.45), this basis <= 0.0042%. The second window needs
its occupations down to rounding level (400 modes); with 200 (occupation 1e-5) the error moves between energies
(up to 0.05%).

`mos2_hfo2_2win` is the same construction on the default 5-cell device (`--nmodes 400,800 --neig 800`, other
options as above; second-window contour 2.2 h). QuaTrEx, mode space vs full orbitals at Ec + 0.05 / 0.15 / 0.30 /
0.45 eV with the potential: +0.0026 / +0.022 / +0.0010 / -0.014% (the one-window `mos2_hfo2_n400`: +0.008 / +0.045 /
+0.024 / +0.009%); without the potential <= 0.0004%. On this device the one-window basis (occupations down to
rounding level at 400 modes) already holds under this bias; the second window halves the largest error.

`mos2_hfo2_2win` is the same construction on the default 5-cell device (`--mu=-0.15,0.6;0.5,0.9 --npts 24,16
--nmodes 400,800 --neig 800 --merge-thr 1e-12`). The two windows overlap strongly here, so the merge keeps only
about 530 columns per slab. QuaTrEx QTBM, mode space vs full orbitals at Ec + 0.05 / 0.15 / 0.30 / 0.45 eV:

| basis | N | no potential | with `--potential` (0 -> -0.2 V) |
|---|---|---|---|
| `mos2_hfo2_n400` (one window) | 1600 | 0.0002% | +0.008 / +0.045 / +0.024 / +0.009% |
| `mos2_hfo2_2win` | 2127 | 0.0004% | +0.003 / +0.022 / +0.001 / -0.014% |

On this device the single-window basis already holds up under the potential (its 400 modes per slab reach
occupation ~1e-14, unlike the 200 of the 3-cell basis); the second window halves the largest error for 33% more
columns. Neither reaches 0.01% at every energy with the potential.

## Tested: MoS2 on HfO2 (2808 atoms, 29 088 orbitals)

Device `device_MoS2HfO2_2808a_29088o` (Mo 26, S 13, Hf 10, O 4 orbitals per atom; non-orthogonal), lead cell 3120
orbitals, Ec = 0.504 eV, Ev = -1.203 eV. Default: 5 cells per block, device slabs [3483, 8383, 7862, 3120] orbitals.

Transmission error of the mode space against the exact T of the same device:

| device | basis | N total | max error, Ec + 0 … 0.5 eV |
|---|---|---|---|
| full (5 cells) | 400 per slab + end-slab cleanup | 1636 | 0.0043% (fine scan, 10 meV steps) |
| full (5 cells) | 400 per slab, this code (defaults, `--neig 800`) | 1600 | 0.0002% (Ec + 0.05, 0.15, 0.30, 0.45) |
| full (5 cells) | 400 per slab | 1600 | 0.001% (Ec + 0.02, 0.05, 0.15, 0.30) |
| full (5 cells) | 200 per slab | 800 | 0.010% (Ec + 0.02, 0.05, 0.15, 0.30) |
| full (5 cells) | 100 per slab | 400 | 40% |
| `--cells 3 --trim` | 200 per slab, with or without cleanup | 1600 / 1627 | 0.003% (4 energies) |

`--cells 3 --trim` changes the device itself by at most 0.86% in T (vs the untruncated H, S; 0.15% except near
Ec + 0.2 eV); `--cells 4 --trim` by at most 0.18%. Global magnitude cuts of H, S are much worse (1e-4: 1.4%,
1e-3: 7%) and are not offered.

Cost on this device (4 threads, shared machine): basis 1.5–2 h (contour 24 points ~80 min, of which most is the lead
self-energies; eigenvectors and cleanup ~10 min); per energy: lead self-energy ~3 min, mode-space solve ~1 s, exact
solve 4–9 min.

## Open-boundary eigenmodes (`qtx_modespace.obc_eig`)

The resonances and bound states of the open device, i.e. the solutions of the nonlinear eigenproblem
T(z) v = 0, T(z) = zS − H − Σ_L(z) − Σ_R(z), inside an ellipse, by Beyn's contour-integral method. Their span
against a saved basis (`--modes`) tells whether the basis holds the physical resonant states; `--plot` draws each mode.
```
python -m qtx_modespace.obc_eig DEVICE --out eig_c1 --center 0.017 --a 0.015 --b 0.01 --npts 24 --probes 200 \
       --modes data/mos2_hfo2_n400/modes --plot
for w in 0 1 2; do                                   # parallel: points per worker, then one run that combines them
  python -m qtx_modespace.obc_eig DEVICE --out eig_c1 ... --points $(seq -s, $w 3 23) &
done; wait
```
`qtx-modespace-obc-eig` is the same as a console script.

- **Second sheet.** The resonances lie below the real axis on the continuation of the retarded Σ^R(E + i0). The lead
  Bloch modes of one decimation at z are classified by their flux, and below the axis the n propagating modes that
  have the wrong direction are swapped. Here n is the channel count at the contour centre on the real axis. This
  continuation is analytic only between consecutive lead subband edges, which are its branch points. **A contour must
  lie inside one such interval and stay a few meV away from its edges.** The lead band edges are printed by the main
  driver (`sigma/lead_edges.json`); on MoS2/HfO2 the edges near Ec are at Ec + 0, 0.034, 0.136 and 0.232 eV, and
  above that they are only meV apart.
- **Probes.** The number of eigenvalues inside the contour must stay below `--probes` × `--kmom`. The rank of the
  moment matrix is printed, with a warning when it comes close to that limit. The singular values are in `eig.json`.
- **Cost.** One dense block-tridiagonal LU of T(z) per contour point, with `--probes` right-hand sides, plus one
  continued Σ per point. Below the axis, each continued Σ also needs two lead mode eigendecompositions. The residual
  check costs one more Σ per eigenvalue (`--resid n` limits it to the n eigenvalues closest to the centre). On the
  device below: about 2 min per solve and 2.5–4 min per Σ (4 threads). The solves and Σ are cached, so a run
  can be resumed or extended.
- **Output** (`--out`): `eigpairs.npz` (`lam`; `vec`, device orbitals × modes; `orbitals`, the QuaTrEx index of each
  row; with `--modes` also `missing` = |(1 − P)v|/|v| and `missing_slab`), `eig.json` (eigenvalues, E − Ec, residuals
  |T(λ)v|/(max|T| |v|), span, contour, channel counts), `eigmodes.png` (Mulliken weight per atom in two projections,
  and the profile along transport per species; dashed lines = slab boundaries), and `X/` (the solves per point).

MoS2/HfO2 (the device below, level-1 basis N1 2301 of the qtx_ms study; three contours between the lead edges, all
eigenvalues found):

| contour (E − Ec, eV) | eigenvalues | Im λ (eV) | missing in span(U1) | residual |
|---|---|---|---|---|
| 0.002–0.032, depth 0.01 (also 0.05: the same one) | 1 | −1.5e-4 | 8.2e-4 | 4.5e-10 |
| 0.040–0.130, depth 0.03 | 3 | −1.3e-7 … −2.0e-4 | 3.2e-4 – 6.0e-4 | ≈ 2e-14 |
| 0.144–0.224, depth 0.03 | 5 | −2.0e-4 … −2.5e-3 | 5.2e-4 – 9.6e-4 | ≈ 2e-11 |

All of them are narrow states (Γ ≤ 5 meV) at the two ends of the HfO2 layer. Below Ec + 0.17 eV they are almost
pure Hf/O. Above that they move into the MoS2 channel near the right oxide end (57–98 % on Mo/S).

## Notes and limits

- Use enough modes: with too few (100 per slab here) the error is tens of percent and erratic. The smallest kept
  occupation relative to the largest (printed, and in `basis.json`) should be ~1e-7 or below (200 per slab here).
- Occupations decay smoothly down to rounding level (~1e-14 relative at 400 per slab). Modes below ~1e-12 are not
  reproducible: they depend on the eigensolve size and on rounding-level changes of rho. They are harmless for T.
- Cleanup: on the same 400-mode basis the number of vectors needed varied between 14 and > 150 (not converged)
  between runs that differ only at rounding level, so it is off by default.
- Only Gamma (no transverse k); no spin handling beyond what is in H, S.
- The lead self-energy is a dense Python decimation on the lead faces; for large leads it dominates the run time.
- The coarse system is meant as a stand-alone mode-space solver. Used as the coarse level of a two-level GMRES with a
  GMRES smoother it converges slowly on this device (residual 3e-3 -> 2e-4 in 60 iterations, independent of the
  number of smoother steps), so no iterative refinement is included.

## Release 0.1.0

Tag `qtx_modespace-v0.1.0`. Release files (archived with a `SHA256SUMS` file, which is not in git so that the
package can carry this README): `qtx_modespace-0.1.0-py3-none-any.whl`, `qtx_modespace-0.1.0.tar.gz` (`uv build`),
and one data tarball per included basis, `qtx_modespace-0.1.0-data-<name>.tar.gz` with `<name>/modes/` and
`<name>/basis.json`. Unpack one and pass `--modes <name>/modes` with the device options of the table above.
The QuaTrEx production runs (`to_quatrex`, `compare_quatrex`) used github.com/quatrex/quatrex at commit 95abf986.
