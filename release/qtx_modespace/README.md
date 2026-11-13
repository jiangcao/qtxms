# qtx_modespace

Mode-space basis for QuaTrEx / deviceBuilder devices. It reads a QuaTrEx device folder, builds a per-slab mode-space
basis and saves it as `.npy` files. There are two kinds of basis, both density-matrix mode space:
- **level 2** (global density matrix, the default; `--basis l2`): the slab density matrices of the whole open device;
- **level 1** (local density matrix, `--basis l1`, new in 0.2.0): one density matrix per block, where a block is a
  slab or a tile of a slab, embedded in its surroundings and closed by a CAP, or by a CAP plus the exact lead
  self-energy on the lead faces.

Optionally it computes the mode-space transmission and checks it against the exact transmission. It can also solve
the full QTBM system iteratively (`--fgmres`, new in 0.2.0), using flexible GMRES with a CAP-Schwarz smoother and the
mode space as the coarse level.

Dependencies: numpy, scipy, h5py. Optional: qttools from QuaTrEx (real-axis lead OBC, the default when it imports:
on sys.path or `QTTOOLS_SRC=<quatrex>/src`; needs Python >= 3.12, mpi4py and numba; without it the folded lead is used), pypardiso (fast sparse LU for the lead interior
and the exact solve, x86-64), threadpoolctl (`--l1-workers` and `--workers`: BLAS threads per process / window),
matplotlib (only for the post-processing plots, `python -m qtx_modespace.plot`), cupy (`--gpu`, see "GPU" below).

Environment (0.3.0): `scripts/setup_env.sh [VENV]` creates a venv with the package and its optional dependencies,
detects the GPU driver's CUDA version (nvidia-smi) and installs the matching cupy (cupy-cuda12x / cupy-cuda13x), finds
the CUDA toolkit (`/usr/local/cuda-<major>.*`, or `CUDA_PATH`, or `CTK=1` for the pip CUDA runtime) and writes
`VENV/bin/qtx_env.sh` (activate + `CUDA_PATH` + `QTTOOLS_SRC`); `TEST=1` runs the tests. Options in the script header.
```
PYTHON=python3.12 TEST=1 scripts/setup_env.sh ~/venvs/qtx      # GPU auto-detected; CUDA=none for CPU only
source ~/venvs/qtx/bin/qtx_env.sh
```
Workflow (0.3.0): `examples/workflow.sh DEVICE_FOLDER [OUT]` runs prep, the level-1 bases (A slab CAP+OBC, B slab
CAP, C tile CAP+OBC) with T vs exact, FGMRES (D, open) and the closed system (E), then the plots; `GPU=1` for the GPU,
`STEPS="A D"` to pick steps, `EXACT=0`, `CAP_SPLIT`, cuts and energies by environment variables (script header).
```
GPU=1 STEPS="A D" DE=0.02,0.05,0.15,0.3 examples/workflow.sh DEVICE_FOLDER out_gpu
```
`examples/mos2_hfo2_workflow.sh DEVICE_FOLDER [OUT]` is the same workflow with the defaults used for the MoS2/HfO2
device below (results in "Tested").

```
pip install .                 # or: pip install ".[fast,plot]"
python -m qtx_modespace DEVICE_FOLDER --out ms_out                                   # basis only
python -m qtx_modespace DEVICE_FOLDER --out ms_out --transmission --exact --dE 0:0.5:0.05
python -m qtx_modespace DEVICE_FOLDER --out ms_out --cells 3 --trim --nmodes 200     # smaller slabs, see below
python -m qtx_modespace DEVICE_FOLDER --out t_out --cells 3 --trim \
       --modes data/mos2_hfo2_trim_c3_n200/modes --transmission --dE 0:0.5:0.05      # T from a saved basis
python -m qtx_modespace DEVICE_FOLDER --out l1_out --basis l1 --l1-cut 1e-8 \
       --l1-workers 4 --l1-threads 6 --transmission --dE 0:0.5:0.01                 # slab level 1, CAP + OBC
python -m qtx_modespace DEVICE_FOLDER --out l1t_out --basis l1 --l1-partition tile --tile-orbs 2500 \
       --l1-boundary cap --l1-cut 1e-6                                              # tile level 1, CAP only
python -m qtx_modespace DEVICE_FOLDER --out mg_out --modes l1_out/modes --fgmres --dE 0.05,0.2 \
       --exact --psi-dir psi                                                       # FGMRES + CAP-Schwarz to 1e-10
```
Read the device folder once (`qtx_modespace.prep`) and give later steps the prepared folder instead of DEVICE_FOLDER:
```
python -m qtx_modespace.prep DEVICE_FOLDER --out PREP [--cells m] [--trim] [--potential] [--gpu]
python -m qtx_modespace PREP --out ms_out --basis l1 --l1-cut 1e-8 ...               # loads PREP in seconds
```
See "Prepared inputs" below.

The run writes numbers only; plots are a separate post-processing step: `python -m qtx_modespace.plot OUT [OUT ...]`
(or `qtx-modespace-plot`) draws `transmission.png` from each run's `transmission.json`.

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

## Prepared inputs (`qtx_modespace.prep`)

Reading a QuaTrEx folder (h5 files, sum of the R blocks, Hermitian completion, block partition, reordering, lead cell,
lead band edges) is repeated by every run; on the 2808-atom MoS2/HfO2 device it takes 8-35 s plus the lead band
edges (41 dense generalized eigensolves of the lead cell: 20 s - 4 min on the CPU, cached in `--sigma-cache`), and it
grows with the device. `python -m qtx_modespace.prep DEVICE_FOLDER --out PREP` does it once and writes

    PREP/prep.json     format, options (cells, trim, potential), source folder with size / mtime of its files, meta
    PREP/arrays.npz    labels, xyz, norb, blocks, perm (orbital -> QuaTrEx orbital)
    PREP/H.npz, S.npz  device + lead blocks in the internal order, upper triangle (Hermitian), uncompressed
    PREP/lead.npz      lead cell H00, H01, S00, S01 (and the shifted right lead with a bias)

`PREP` is then accepted wherever DEVICE_FOLDER is (`python -m qtx_modespace PREP ...`, `qtx_modespace.obc_eig`,
`read_quatrex(PREP)`); it loads in about 1 s here (0.53 GB, about 12 bytes per stored nonzero) and gives bit-identical
H, S, lead and index arrays. `--cells`, `--trim`, `--potential` are fixed at the prep step: later steps need not
repeat them and stop if they ask for something else. A warning is logged when the source files changed after the
prep. `--gpu` computes the lead band edges on the GPU, `--sigma-cache DIR` reuses / stores them in
`DIR/lead_edges.json`. `qtx_modespace.to_quatrex` still needs the QuaTrEx folder itself. From Python:
`save_inputs(read_quatrex(folder), "PREP")`, `read_quatrex("PREP")`. Console script: `qtx-modespace-prep`.

## Method

1. **Slabs.** The contact lattice vector defines primitive cells along transport. A slab (block) is `--cells` cells,
   by default the smallest number covering the coupling range of H and S, so that H and S are block tridiagonal.
   `--trim` drops the H, S entries that couple non-neighbouring blocks, which allows fewer cells per block (smaller
   slabs, cheaper basis); the largest dropped |H|, |S| are printed. The first and last blocks are the leads (the
   QuaTrEx contacts are the device's own end cells); the device slabs are the blocks in between.
2. **Leads.** Exact self-energies Sigma_L, Sigma_R(z).
   - **Real energies.** Transmission, FGMRES and the exact solves run at real E (`--eta 0`, the default). There the
     self-energies and the injected modes come from qttools' spectral OBC, the QuaTrEx one (`--obc auto`, the
     default; `--obc qttools` fails instead of falling back). Sigma is evaluated at E itself, with no imaginary
     part. The injection uses the flux-normalized incoming Bloch modes, one column per propagating channel, so
     there are 2 channels at Ec + 0.05 eV on MoS2/HfO2. qttools reads the lead cell as `cells_per_block` primitive
     cells, periodized from the outermost one.
   - **Fallback.** Without qttools, or with `--obc folded`, the folded lead is used at E + 1e-6 i, with the Gamma_L
     eigenchannels above 1e-4 eV as injection. That gives 19 channels at Ec + 0.05: 17 of them are
     eta-broadened evanescent modes that carry no current, so T is the same.
   - **Complex z.** The contour points of the level-1 and level-2 density matrices always use the folded lead: the
     lead-cell interior is folded onto the orbitals coupled to the neighbouring cells, then Lopez-Sancho decimation
     (complex Hermitian H, non-orthogonal S).
   - **Cache.** Results are cached in `--sigma-cache`: `qsig_<E>_*.npz` (qttools, with the injected modes) and
     `sig_<Re z>_<Im z>.npz` (folded lead).
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
3'. **Level-1 basis (`--basis l1`, instead of 3).** See "Level-1 basis" below.
5. **Transmission (optional).** At E = Ec + dE: QTBM injection b from Gamma_L, mode-space solution
   x = U A_c^-1 U^H b with the Galerkin A_c = U^H (E S - H - Sigma) U (block tridiagonal, exact Sigma projected),
   T = x_N^H Gamma_R x_N. With `--exact` also the exact T from a sparse direct solve of the full device.

6. **Iterative solve (optional, `--fgmres`).** See "Iterative solve" below.

### Level-1 basis (`--basis l1`)

Partition and boundary are chosen independently:
- **Partition** (`--l1-partition`). `slab` makes each device slab one block; the slab size is set by `--cells`
  (and `--trim`). `tile` cuts each slab transversely by recursive bisection into round(n_slab / `--tile-orbs`)
  tiles. `--tile-xsplit k` then cuts every block, slab or tile, into k pieces of equal orbital weight along
  transport. The basis is block diagonal over the blocks: U_s = diag(blocks of slab s).
- **Embedding** (`--l1-emb`, default 12 A). The block plus every atom within that distance. Distances use the
  minimum image across the periodic transverse faces, which `read_quatrex` detects from the couplings of the
  Gamma-point supercell. The embedding is closed by a CAP eta ((d - (d_emb - d_cap)) / d_cap)^2 on its outer
  d_cap shell (`--l1-cap d_cap,eta`, default 3 A, 0.3 eV).
- **Boundary** (`--l1-boundary`). `obc` (CAP + OBC, the default) drops the lead-block atoms from the embedding and
  adds the exact lead self-energy -Sigma_L/R(z) on the lead-face orbitals inside it. Only blocks whose embedding
  reaches a lead face are affected. `cap` keeps the lead atoms and closes the embedding with the CAP only.
- **Local rho.** rho_PP = -(1/pi) Im ∫ G_PP(z) dz over the upper half circle of `--l1-mu` (default: the `--mu`
  window) with `--l1-npts` points. These are the same points as the level-2 contour, so the Sigma cache is shared.
  G_PP is the inverse of the Schur complement of the environment, (A_PP - A_PE A_EE^-1 A_EP)^-1, with a dense LU
  when the embedding is more than 2% filled. `--l1-gpp full` uses one LU of the whole embedding instead, and
  `--l1-env-prec single` a complex64 environment LU plus one refinement step (not faster for slabs). The contour
  points of a block can run in parallel: `--l1-workers k --l1-threads t` uses k forked processes with t BLAS
  threads each.
- **Basis.** Per block, the eigenvectors of rho_PP with occupation > `--l1-cut`, largest first. Blocks are cached
  in `<out>/level1/<partition>_<boundary>_<hash>/block<i>.npz` (or `--l1-cache`) with vectors down to
  `--l1-store-cut` (1e-10), so a rerun with any cut >= that reuses them. `--l1-only K,N` builds only the blocks
  i % N == K and exits, for building blocks in parallel jobs. The end-slab cleanup (`--clean-slabs 0,-1`) works on
  level-1 bases too. It is recommended there, because the raw end slabs carry spurious lead bands; it lowered the
  largest T error, see below.
- **Level 2 on level 1** (`--basis l2 --l1-modes DIR`, 0.3.0). Slab level 2 computed in a saved level-1 basis U1
  (the `modes/` folder of a `--basis l1` run): the global rho of the open level-1 system (Galerkin blocks
  U1^H (H, S) U1, lead Sigma projected on the end-slab blocks, contour `--mu`, `--npts`), per slab its eigenvectors
  V_s with occupation > `--l2-cut` (or the top `--nmodes`), and the nested basis U = U1 V. Much cheaper than direct
  level 2 (the recursion runs on N1 instead of the device orbitals); it cannot be more accurate than its level 1.
  The eigenpairs are cached in `<out>/eig_l2on1_*.npz`.

### Iterative solve (`--fgmres`)

The QTBM system A x = b (all injected channels) is solved to `--fgmres-tol` (relative residual per channel,
default 1e-10) by right-preconditioned flexible GMRES. A block of channels is iterated together: each channel has
its own Krylov space, but the preconditioner acts on all active channels at once.
- **CAP-Schwarz smoother.** Restricted additive Schwarz along transport. The windows are `--cap w,s,d` (default
  16 A wide, absorbing strength 2 eV, ramp 2 A), centred on the slab interfaces and between them; `--cap-spacing`
  sets the largest distance between centres, and `--cap-centres ends` keeps the interfaces plus the two lead-face
  windows. Each window is solved with an LU of A_jj + i s P S_jj P, where P^2 ramps from 0 to 1 over the outer d A
  towards each artificial cut. The lead faces keep Sigma. `--cap-split ny,nz` additionally cuts every window into
  boxes along the periodic transverse axes, with `--cap-tov` A of overlap. Cut only out of the 2D plane: in-plane
  cuts need boxes of 16 A or more. `--cap-solver dense64` (default) gives the same convergence as complex128 with
  half the factor time. `--workers` sets how many windows are factored and solved concurrently (threads).
  `--cap-dE0` factors at E + dE0, so one factorization can be reused over an energy interval (about 0.05 eV costs a
  few iterations). `--cap-lu-host` (GPU, 0.3.0) keeps the window LUs in pinned host memory and streams them to the
  card during each sweep (two windows on the device at a time, the next one copied while the current one is solved):
  the 6 nm device with 172 windows holds 117 GB of complex64 LUs; the sweep is as fast as with device-resident LUs
  on the 2808-atom device (identical iterations).
- **Channels and logging.** `--fgmres-chunk k` iterates the channels k at a time (independent Krylov spaces, so the
  result is the same; GPU memory ~ k x (2 maxit + 1) x n x 16 B). `--fgmres-log n` (0.3.0) logs the residual
  estimates (largest / median over the chunk, active channels, time) every n iterations.
- **Coarse level** (`--fgmres-coarse ms`, the default). The Galerkin mode space of the basis (`--modes`, or the
  one just built). The preconditioner is one multiplicative cycle: `--pre` smoothing steps, coarse correction,
  `--post` smoothing steps; keep post-smoothing, without it the iterations double. The start vector is the
  mode-space solution. `--coarse-eta` moves only the coarse solve to E + i eta_c, which removes the stall of a coarse
  level that sits on a narrow spurious mode-space resonance. `--fgmres-coarse none` is one-level: the smoother alone
  with x0 = 0. It needs about 1.5x the iterations of two-level, but each iteration is cheaper.
- **Slow-mode coarse level** (`--fgmres-coarse slow`, 0.4.0). Instead of mode space, the coarse space is built at
  each energy from the error components the smoother and the cycle damp worst (`schwarz.SlowCoarse`): block subspace
  iteration (`--slow-q` steps from `--slow-m` random vectors) on the error operator of two smoothing steps, then of
  the two-level cycle with the coarse space built so far, `--slow-rounds` times; Galerkin A_c = Q^H A Q, x0 = P_c b.
  No basis is needed: `--basis none --fgmres --fgmres-coarse slow` skips the basis build and the mode-space solve.
  Prototype numbers (two-level its, max over channels): 2808 atoms, Ec + 0.05 / 0.2 / 0.3, N 192: 13 / 15 / 18 vs
  29 / 47 / 150 with slab level 1 1e-8 (N 2373); 6 nm Ec + 0.15 (y split 4), m 128 x 4 rounds (N 512): 24-36 its
  on the 12 strongest channels,
  where every mode-space coarse level stalls. Its coarse matrix has no spurious mode-space states (almost real
  eigenvalues with ~7 eV fine residual). The modes are energy specific: build them at every energy (2808 atoms ~1 s,
  6 nm ~20-25 s per round of 128 on a GH200). Q is dense (n x N).
- **Output.** `transmission.json` → `E.<dE>.fgmres`: T, iterations and true residual per channel, residual history,
  factor and iteration time, the settings, and with `--exact` the T and psi errors (`err_T`, `err_psi`).

### Closed system with unit right-hand sides (`--system closed --fgmres`)

This mode builds the system matrix without the lead self-energy, A0 = z S - H with z = E + i `--closed-eta`, on both
the fine and the coarse level (the Galerkin A_c then carries no projected Sigma). The right-hand sides are the
identity on the contact slab: one unit vector per orbital of the slabs in `--rhs-slabs` (default 0, the slab next
to the left lead; `0,-1` for both), or with `--rhs-face` only on their lead-face orbitals. These columns are solved
by the same FGMRES / CAP-Schwarz / coarse-level cycle, `--rhs-chunk` columns at a time. Krylov memory is about
maxit x chunk device vectors, so on MoS2/HfO2 64 columns need ~3.5 GB.
- The result X = A0^-1 I_C holds the columns of the closed-device Green's function on C. The lead self-energies
  can be added afterwards on the face orbitals (Dyson / Woodbury), so these solves do not depend on Sigma.
- The closed device has real eigenvalues. At `--closed-eta 0`, A0 (and the coarse A_c) is singular at those
  energies and the iteration stalls near them; use a small eta, or `--coarse-eta` for the coarse level only.
- `--exact` adds a sparse direct solve of the same columns and the error of X. `--save-x DIR` writes
  `closed_<E>_<eta>.npz`, holding X, the device columns and their QuaTrEx orbital indices. `--rhs-max n` keeps only
  the first n columns, for tests.
- Output: `<out>/closed.json`, per energy the iterations per column, the largest residual, timings and the errors.

## Output (`--out`)

| file | content |
|---|---|
| `modes/slab<s>.npy` | basis of slab s: complex array (slab orbitals x modes), orthonormal columns |
| `modes/slab<s>_orbitals.npy` | for each row: the orbital index in the QuaTrEx `hamiltonian.h5` |
| `modes/slab<s>_occupations.npy` | density-matrix occupations of the level-2 modes (largest first) |
| `basis.json` | slab sizes, N per slab, window, occupations, cleanup summary per end slab, device info, arguments; `basis` = l2 / l1, `level1` = partition, boundary, embedding, cut, columns per block |
| `level1/` | `--basis l1`: cached local-rho eigenvectors per block (`block<i>.npz`: U, occupations s, atoms, stored cut) |
| `eig_mu…_top<neig>.npz` | cached slab eigenpairs; a rerun with the same window and `--neig` skips the contour |
| `rho_mu….npz` | slab density matrices, only with `--save-rho` (2.2 GB for the device below) |
| `config.toml` | resolved options of the run (input for `--config`) |
| `sigma/` | cached lead self-energies and lead band edges `lead_edges.json` (`--sigma-cache none` to disable) |
| `transmission.json` (`.png` from `python -m qtx_modespace.plot OUT`) | per energy: channels, T_MS, T_exact, relative error, timings; with `--exact` also the errors of the QTBM solution vectors (`err_psi`: Frobenius over channels, `err_psi_max`: worst channel, `err_psi_S`: S norm, `err_psi_last` / `err_psi_slab`: right end slab / each slab, `err_psi_best`: best approximation in range(U), `err_res`: residual), all relative to the exact solution |
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

### Release 0.2.0 on this device (`examples/mos2_hfo2_workflow.sh`)

Real-axis OBC = qttools (2 channels at every energy below), default 5-cell slabs [3483, 8383, 7862, 3120], local rho
window Ec - 0.15 ... Ec + 0.6 eV with 24 points, embedding 12 A, CAP 0.3 eV over 3 A; 4 processes x 6 threads.

Level-1 bases (mode-space T error vs exact; 51-point grid Ec + 0 ... 0.5 eV against the folded-lead exact T, and
Ec + 0.05 / 0.15 / 0.2 with the qttools OBC):

| basis | N1 per slab | N | 51-pt grid: median / max | Ec + 0.05 / 0.15 / 0.2 (qttools) | build |
|---|---|---|---|---|---|
| slab level 1, CAP + OBC, cut 1e-8 | 550 / 702 / 726 / 395 | 2373 | 5.2e-6 / 3.4e-4 | -7e-7 / +1.1e-5 / -2.5e-4 | 40 min |
| slab level 1, CAP only, cut 1e-8 | 649 / 704 / 728 / 492 | 2573 | 3.3e-6 / 3.5e-4 | -3.1e-5 / -1.0e-5 / -4.0e-5 | 42 min |
| tile level 1 (8 tiles, 2500 orb), CAP + OBC, cut 1e-6 | 319 / 840 / 839 / 266 | 2264 | 1.9e-4 / 2.9e-2 | -2.4e-4 / -8.6e-5 / +1.9e-4 | 70 min |

CAP only changes only the end slabs (their embedding keeps the lead atoms): more columns there, the same accuracy.
The slab partition reaches a given accuracy with fewer columns than tiles (no transverse cuts). The mode-space psi
error is much larger than the T error (3e-3 / 6e-2 / 6e-2 for the slab CAP + OBC basis, while range(U) holds psi to
2-4e-4): T is second order in the error, psi is not (Galerkin is not optimal for this indefinite A).

FGMRES + CAP-Schwarz (16 A windows, 7 of them, 3186-8762 orbitals, dense64, 4 workers), coarse level = the slab
CAP + OBC basis, tol 1e-10; times are window factorization + iterations; the exact sparse solve takes 150-180 s:

| E - Ec | two-level its | T / psi error | time | one-level its | T / psi error | time |
|---|---|---|---|---|---|---|
| 0.05 | 25 | 1e-12 / 1.1e-11 | 24 + 26 s | 55 | 8e-11 / 7.5e-11 | 24 + 23 s |
| 0.15 | 35 | 6e-13 / 7.2e-12 | 24 + 37 s | 61 | 1e-11 / 6.2e-11 | 24 + 25 s |
| 0.20 | 41 | 4e-12 / 5.9e-12 | 23 + 44 s | 84 | 2e-11 / 2.3e-10 | 23 + 37 s |

The coarse level halves the iterations; a two-level iteration (pre + post smoothing, coarse solve, three matvecs)
costs about twice a one-level one, so on this short device both take the same time.
These times are with scipy's complex sparse product. The fine matvec now uses `schwarz.FastOp`: real products on
[Re X, Im X], with the rows split over `--workers` threads. That cut the 64-column product from 2.8 s to 0.5 s, and
the two-level solve at Ec + 0.2 from 44 to 30 s of iterations (same 41 iterations, T error 5e-12).

Closed system (`--system closed`, no Sigma, unit right-hand sides on the first 128 lead-face orbitals of slab 0,
z = E real), vs a sparse direct solve of the same columns (~300 s): Ec + 0.05 two-level 30-36 its, one-level 46-59;
Ec + 0.2 two-level 47-52, one-level 70-87; error of X <= 6e-11 (worst column 3e-9). At real E the closed device is
close to its own eigenvalues, so the iteration counts are higher than for the open system. One 64-column chunk (two-level, Ec + 0.05)
took 403 s with scipy's product and 117 s with `FastOp`, against about 300 s for the direct solve of 128 columns.
The iterative path pays off only when the window factors are reused over energies (`--cap-dE0`), or for more
columns than one direct factorization serves well.

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

## GPU (`--gpu`)

`--gpu` (or `QTX_MODESPACE_GPU=1`) moves the dense algebra to an NVIDIA GPU with cupy; `--gpu-device n` picks the
card. Without it nothing changes (numpy / scipy, cupy is never imported). On the GPU:
- lead Sigma of the folded lead (interior fold and decimation) and the lead band edges;
- level-1 local density matrices (`--basis l1`, dense embeddings): blocks are built on the device from the sparse
  H, S, one LU + Schur complement per contour point;
- level-2 slab recursion, the eigensolves, and the exact QTBM solve (`--exact`, block-tridiagonal LU; when the dense
  blocks do not fit on the card, a streaming variant keeps the couplings sparse on the device and every LU factor
  in host memory, one dense slab block each: 6 nm device ~100 GB host, ~2.5 min per energy on the GH200);
- `--fgmres`: Schwarz window LUs (complex64), sparse matvec and the whole Krylov loop.

Install cupy for the CUDA version of the driver: `pip install ".[gpu]"` (cupy-cuda12x) or `pip install cupy-cuda13x`;
set `CUDA_PATH` if cupy does not find the toolkit. Results leave every module as numpy arrays; caches and mode files
are the same as on the CPU. In GPU mode the contour points and Schwarz windows run one after another in one process
(`--l1-workers`, `--workers` are ignored: no fork after CUDA is initialized). Memory: the largest dense block is the
embedding of a block (complex128, n = 17 000 here: 4.6 GB, a few copies of it at the peak) or a Schwarz window.
The qttools OBC is not routed through `--gpu`; sparse (non-dense) embeddings and PARDISO windows stay on the CPU.

Measured on the 2808-atom MoS2/HfO2 device (workflow steps A and D, `examples/mos2_hfo2_workflow.sh`), GH200 96 GB
(cupy-cuda13x 14.2, folded lead at E + 1e-6 i, 19-24 columns) against the CPU run of "Tested" below (28 cores,
4 x 6 threads, qttools OBC, 2 columns):

| step | GPU | CPU |
|---|---|---|
| slab level 1, cut 1e-8: 4 local rho, 24 points (slabs 0 / 1 / 2 / 3) | 5 / 96 (with the 24 Sigma) / 15 / 4 s | 116 / 708 / 612 / 67 s (Sigma cached) |
| basis total (Sigma + local rho + eigensolves) | 154 s | ~40 min |
| real-axis lead Sigma, per energy (folded) | 4 s | 170-300 s |
| exact solve, per energy | 2 s | 150-180 s |
| Schwarz window factorization (7 windows <= 8762) | 2 s | 23-24 s |
| FGMRES two-level, iterations at Ec + 0.05 / 0.15 / 0.2 | 4 / 5 / 7 s (24-47 its) | 26 / 37 / 44 s (25-41 its) |
| FGMRES one-level | 4 / 5 / 8 s (46-86 its) | - |

Same basis as on the CPU (N1 550 / 702 / 726 / 395), T_MS error 1e-6 / 7e-6 / 3.4e-4, FGMRES T error <= 1.5e-10.
On a Tesla P100 (16 GB, cupy-cuda12x) the slab-0 local rho takes 92 s including the 24 Sigma (2.4-4.1 s each) and
agrees with the CPU block to 7e-14 in the occupations. With the folded lead the injection includes the eta-broadened
channels (norm 1e-4 of the physical ones); "max channel" psi errors refer to those.

## Notes and limits

- Use enough modes: with too few (100 per slab here) the error is tens of percent and erratic. The smallest kept
  occupation relative to the largest (printed, and in `basis.json`) should be ~1e-7 or below (200 per slab here).
- Occupations decay smoothly down to rounding level (~1e-14 relative at 400 per slab). Modes below ~1e-12 are not
  reproducible: they depend on the eigensolve size and on rounding-level changes of rho. They are harmless for T.
- Cleanup: on the same 400-mode basis the number of vectors needed varied between 14 and > 150 (not converged)
  between runs that differ only at rounding level, so it is off by default.
- Only Gamma (no transverse k; all R blocks of the h5 files are summed); no spin handling beyond what is in H, S.
- The lead self-energy is a dense Python decimation on the lead faces; for large leads it dominates the run time.
- The mode-space system is also the coarse level of the iterative solve (`--fgmres`). A GMRES(k) smoother does not
  work on this device (residual 3e-3 -> 2e-4 in 60 iterations, whatever k); the CAP-Schwarz smoother does. Windows
  narrower than about 16 A stall.
- `--workers` > 1 factors the Schwarz windows in concurrent threads, with BLAS limited to 1 thread inside them
  (threadpoolctl). Concurrent LAPACK calls on a multi-threaded MKL crash (segfault). Without threadpoolctl the
  windows run one after another.
- Two-level FGMRES on long devices: on the 6 nm MoS2/HfO2 device (22 slabs) the two-level iteration stalls at
  Ec + 0.15 eV with every basis tried (slab level 1 cuts 1e-8 to 1e-4, slab level 2 on tile level 1), also with
  `--coarse-eta` 1e-3 / 1e-2, while one-level converges (slowly: about one window per iteration along x). Cause:
  the Galerkin coarse operator has a dense set of almost undamped spurious eigenvalues (Ritz values of the
  indefinite operator) from basis vectors cut at the slab faces; their fine residual is ~7 eV, they sit on the slab
  interfaces, mostly in the oxide, and one of them is within ~1e-4 eV of almost any E once N reaches tens of
  thousands. On the 4- and 10-slab devices (N 2 373 / 8 053) two-level converges. Use `--fgmres-coarse slow` (see
  above; converges on 6 nm) or `--fgmres-coarse none` on long devices, and check the iteration log (`--fgmres-log`).

## Release 0.4.0

Tag `qtx_modespace-v0.4.0`. New:
- Slow-mode coarse level for the iterative solve (`--fgmres-coarse slow`, `schwarz.SlowCoarse`; see "Iterative solve"):
  the coarse space is built at each energy from the error components the CAP-Schwarz smoother and the two-level cycle
  damp worst (block subspace iteration, `--slow-m` vectors per round, `--slow-q` steps, `--slow-rounds` rounds),
  Galerkin coarse matrix, x0 = P_c b. It has no spurious mode-space states (almost real eigenvalues with ~7 eV fine
  residual), which is what made the mode-space coarse level stall on long devices. Open and closed system, CPU and GPU.
- `--basis none`: no mode-space basis build or mode-space solve (only with `--fgmres --fgmres-coarse slow|none`).
- 2808 atoms (GH200, `--basis none --fgmres-coarse slow`, m 64 x 3 rounds, N 192): Ec + 0.05 / 0.15 / 0.2 11-15 /
  14-17 / 16-20 its (slab level 1 1e-8 coarse level: 24-29 / 32-38 / 40-48), T error <= 1.2e-11, per energy factor
  2 s + build 3 s + iterations 3-5 s.
- 6 nm MoS2/HfO2 (344 640 device orbitals, 22 slabs; GH200, Ec + 0.15, CAP windows 16 A with y split 4 and 6 A
  transverse shell, LUs on the host, m 128 x 4 rounds, N 512, all 40 channels in chunks of 12): 28-36 its, T error
  1.3e-11, psi error 3.5e-11, where every mode-space coarse level stalls. Time: factor 60 s + build 96 s + iterations
  1025 s, vs ~150 s for the streamed exact solve: it converges, but is not yet faster (each chunk of channels pays
  ~8.5 s per iteration, mostly streaming 68 GB of window LUs).
- Tests: tests/test_slow_coarse.py (square-lattice strip, RAS smoother), slow coarse level on the wire (open,
  closed `cli.closed_solves`, GPU vs CPU).

## Release 0.3.0

Tag `qtx_modespace-v0.3.0`. New:
- GPU backend (`--gpu`, `QTX_MODESPACE_GPU=1`, module `gpu`): folded-lead Sigma and lead edges, level-1 local rho,
  level-2 recursion, eigensolves, exact solve, Galerkin projection U^H H U / U^H S U, injection, Schwarz window LUs
  and FGMRES on an NVIDIA GPU with cupy (see "GPU"). Extras `gpu` (cupy-cuda12x) and `gpu13` (cupy-cuda13x).
- Streaming exact solve for devices whose dense slab blocks exceed the GPU memory (`core.StreamBlockTridiag`, chosen
  automatically): the 6 nm MoS2/HfO2 device (344 640 device orbitals, 22 slabs of up to 16 858) needs ~417 GB for the
  dense block LU; the streaming one runs on a 95 GB GH200 in ~2.5 min per energy (residual 3e-14).
- Prepared inputs (`qtx_modespace.prep`, `qtx-modespace-prep`): read the QuaTrEx folder once, every later run loads it
  in seconds.
- Plots out of the run (`qtx_modespace.plot`, `qtx-modespace-plot`): the run writes numbers only.
- `scripts/setup_env.sh` (venv + cupy matching the driver + `qtx_env.sh`) and `examples/workflow.sh` (generic CPU / GPU
  workflow, steps A-E by environment variables); `examples/mos2_hfo2_workflow.sh` now calls it.
- Level 2 on a saved level 1 (`--basis l2 --l1-modes DIR`, `--l2-cut`): see "Level-1 basis".
- FGMRES: window LUs in pinned host memory (`--cap-lu-host`) for devices whose windows do not fit on the card;
  per-iteration log (`--fgmres-log n`).
- CSCS Alps: `scripts/setup_env.sh` reuses an existing venv (e.g. the QuaTrEx one in the prgenv-gnu uenv) with its cupy
  and mpi4py, takes CUDA from `$CUDA_HOME`, puts the kernel caches on `$SCRATCH`; Slurm templates
  `scripts/alps/{basis,sweep,exact}.sbatch` (one process per GPU; level-1 blocks split with `--l1-only`, energies
  split over the GPUs; exact solve one task per node) with `gpu_bind.sh` / `common.sh` (usage in the headers).
  Shared cache files (Sigma, level-1 blocks, lead edges) are written atomically (temporary file + rename), so
  concurrent ranks never read a partial file.
- Fix: the lead-band cleanup (`--clean-slabs`) took a single periodic axis; with both transverse axes periodic
  (`meta["periodic"]` a list) it now uses the minimum image on every one.
- Extra `test` (pytest). Tested on attelas2 (aarch64, GH200, cupy-cuda13x 14.2, Python 3.12): CPU and GPU tests.

## Release 0.2.2

Tag `qtx_modespace-v0.2.2`. Fix: `read_quatrex` used only the `[0, 0, 0]` block of `hamiltonian.h5` / `overlap.h5`.
Newer deviceBuilder folders (e.g. the 6 nm MoS2/HfO2 device, 357 120 orbitals) store the transverse periodic images as
separate `[0, -1, 0]` / `[0, 1, 0]` blocks instead of folding them into `[0, 0, 0]`; dropping them gave wrong lead
bands and a wrong device H, S. All R blocks are now summed (Gamma point, M = sum_R M_R), as in qtx_ms; each extra block
is logged with its nnz and overlap with `[0, 0, 0]`. Folders with a single `[0, 0, 0]` block (2808 / 9072 / 21600-atom
MoS2/HfO2) are unchanged. Caches (`<out>/sigma`, `<out>/level1`, `rho_*.npz`, `eig_*.npz`) are keyed by settings, not by
H: after an 0.2.1 run on a multi-block folder, use a new `--out` (or delete them).

## Release 0.2.1

Tag `qtx_modespace-v0.2.1`. Fix: on Python 3.11 the run stopped with `SyntaxError: invalid syntax` at
`type NDArray[...]` in `qttools/__init__.py` (qttools uses Python 3.12 syntax, PEP 695), because only `ImportError`
was caught. `--obc auto` now falls back to the folded lead and logs the reason; `--obc qttools` stops with
"qttools needs Python >= 3.12". The qttools OBC needs Python >= 3.12; the rest of the package still runs on 3.11.

## Release 0.2.0

Tag `qtx_modespace-v0.2.0`. New:
- Level-1 basis (`--basis l1`): local density matrices per block (slab or tile), embedding closed by CAP + lead Sigma or
  CAP only; `--l1-partition`, `--tile-orbs`, `--tile-xsplit`, `--l1-emb`, `--l1-cut`, block cache, parallel contour points.
- Iterative solve (`--fgmres`): flexible GMRES with the CAP-Schwarz smoother, two-level (mode-space coarse level) or
  one-level; closed system without Sigma with unit right-hand sides on the contact slab (`--system closed`).
- Real-axis lead OBC from qttools (QuaTrEx spectral OBC) by default, at real E (`--eta 0`), with the folded lead
  as fallback (`--obc`); injection = the flux-normalized incoming modes.
- `read_quatrex` detects the periodic transverse faces of Gamma-point supercells (meta["periodic"], minimum image).
- `examples/mos2_hfo2_workflow.sh`: the whole workflow on the MoS2/HfO2 device; `tests/test_level1.py`.
Not included: FEAST / range-finder probes for the local rho.

## Release 0.1.0

Tag `qtx_modespace-v0.1.0`. Release files (archived with a `SHA256SUMS` file, which is not in git so that the
package can carry this README): `qtx_modespace-0.1.0-py3-none-any.whl`, `qtx_modespace-0.1.0.tar.gz` (`uv build`),
and one data tarball per included basis, `qtx_modespace-0.1.0-data-<name>.tar.gz` with `<name>/modes/` and
`<name>/basis.json`. Unpack one and pass `--modes <name>/modes` with the device options of the table above.
The QuaTrEx production runs (`to_quatrex`, `compare_quatrex`) used github.com/quatrex/quatrex at commit 95abf986.
