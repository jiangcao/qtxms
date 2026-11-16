# Density-matrix hierarchy: merges, bases, smoothers (2026-11-15)

A multilevel preconditioner for the open-boundary transport equation in which every coarse level is a local
density-matrix basis, and each level is obtained from the one below by merging two neighbouring blocks. On the
9072-atom MoS2/HfO2 device (72 672 orbitals) three such levels shrink the problem to 1105 unknowns and the fine
iteration count stays at 18-23, against 16-20 when the first level is solved exactly. It is not faster than the
two-level method on this device, and the first level is still large.

Code: `results_ms/gpu/mlevel/mlevel.py` (any number of levels), `merge.py` (two levels, reference checks).
Logs: `results_ms/gpu/mlevel/mlevel_9072a*.log`, `merge_9072a*.log`. All numbers below: 9072a, E = Ec + 0.15 eV,
all 20 injection columns, flexible block GMRES to 1e-10, iterations given as min-max over the columns.

## 1. The problem and the fine level (level 0)

Per energy E we solve A psi = b with A = E S - H - Sigma_L - Sigma_R (72 672 orbitals) and b = the injected lead
channels. The solver is flexible block GMRES; everything described here is the preconditioner.

**Fine smoother S0 (CAP-Schwarz).** The device is covered by 19 overlapping windows along the transport direction x,
each 16 A wide. A window is closed on its artificial cuts by an absorbing potential (CAP, 2 eV over 2 A); windows
that touch a lead keep the exact lead self-energy there. Each window matrix is LU-factorized once per energy.
Every orbital belongs to exactly one window **core** (the part of the window nearest to its centre); a smoothing step
solves every window and keeps the solution on the cores only. The 19 cores partition the device.

    x ->   |  core 0  |  core 1  |  core 2  |  core 3  | ... |  core 18 |
                 [------ window 1 ------]                       (core 1 + embedding on both sides + CAP)

## 2. Blocks: how the levels are merged

    level 1:  19 blocks   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | ... | 18 |     one block = one window core
    level 2:  10 blocks   |  0+1  |  2+3  |  4+5  |  6+7  | ... | 18 |     pairs of level-1 blocks
    level 3:   5 blocks   |    0..3       |     4..7      | ... |16..18|   pairs of level-2 blocks
    ...                                                                    until one block, or until small enough

A merge joins two neighbouring blocks along one axis (x here; y when the cores are also cut transversally). Nothing
else changes: the blocks of every level are unions of window cores.

## 3. Bases: what the unknowns of each level are

Every level has one basis per block; the prolongation P_k is block-diagonal over the blocks of level k.

**Level 1.** For each window, take exactly the operator the smoother uses (core + embedding + CAP, plus the lead
self-energy), and integrate its Green's function over a complex contour that covers the energy window
[Ec - 0.15, Ec + 0.6] eV (24 points). This gives the local density matrix rho of the window. Keep the eigenvectors
of rho with occupation above a cut and restrict them to the core. Two selection rules were tested:

- *window selection* (`dm`): eigenvectors of the whole window rho, restricted to the core (more columns);
- *core selection* (`dmc`): eigenvectors of the core block of rho, as in mode space (fewer columns).

A1 = P1^H A P1 is stored block-sparse (one dense block per pair of coupled cores). S1 = P1^H S P1 and H1 = P1^H H P1
do not depend on the energy.

**Level k + 1.** For each pair C of level-k blocks:

1. window w = C plus every level-k block coupled to it (the embedding);
2. local operator **in the level-k basis**: z S_k[w,w] - H_k[w,w] - Sigma_k(z) + CAP_k, where the CAP on the cuts of w
   and the lead self-energy are projected through all the bases below;
3. contour density matrix of that operator; keep the modes above the cut on C. These are the columns of U_{k+1}[C],
   expressed in level-k coordinates.

A_{k+1} = P_{k+1}^H A_k P_{k+1}, again block-sparse. No fine-level work is needed above level 1.

**Energy slices.** The contour of level k + 1 need not cover the whole energy window. A larger block resolves a
finer energy range, so higher levels can use a narrower window around the solve energy (a "slice"): here
[Ec+0.05, Ec+0.25] on level 2 and [Ec+0.1, Ec+0.2] on level 3 for E = Ec + 0.15. Level 1 keeps the wide window and
can be reused for all energies; sliced levels are specific to an energy range.

## 4. Smoothers: one per level, formed with the blocks of the level above

The smoother of A_k is a block solve whose blocks are the level-(k+1) blocks:

| operator | smoother | local problems |
|---|---|---|
| A (fine) | S0: CAP-Schwarz windows, written on the cores (= level-1 blocks) | 19 windows of 3186-8768 orbitals |
| A1 | pairs of cores (= level-2 blocks) + coupled cores, projected CAP | 10 windows of 1442-4827 columns |
| A2 | pairs of level-2 blocks (= level-3 blocks) + coupled blocks | 5 windows (wide: 1556-3654, sliced: 832-1900) |
| A_k | level-(k+1) blocks + coupled level-k blocks, projected CAP | |

The windows of the smoother of A_k are the same windows used for the density matrix of level k + 1, so basis and
smoother are consistent: the basis of level k + 1 is built from exactly the local operators its smoother inverts.
Each local matrix A_k[w,w] + CAP_k is dense and LU-factorized (complex64).

An earlier variant used a separate smoother with single blocks on each level (core + coupled cores on A1). It lost
15-30 iterations per level; forming the smoother with the blocks of the level above removed that loss.

## 5. The cycle

One preconditioner application, from the fine residual r:

    smooth on A  (S0)           ->  restrict to level 1
      smooth on A1 (pair blocks)  ->  restrict to level 2
        smooth on A2 (pair blocks)  ->  restrict to level 3
          bottom solve of A3                       (see section 6)
        correct, smooth on A2
      correct, smooth on A1
    correct, smooth on A (S0)

One smoothing step before and one after the coarse correction on every level (V-cycle). `--gamma 2` does the coarse
correction twice per level (W-cycle).

## 6. What "exact level m" and "slow-mode top" mean

The deepest level needs a solver. Two were used:

- **Exact level m.** The hierarchy is cut at level m and A_m is solved exactly by a dense LU. All levels above use
  their smoothers. Running this for m = 1, 2, 3 shows what each merge costs: if "exact level 2" needs the same
  iterations as "exact level 1", the level-1 smoother plus the level-2 basis is as good as solving level 1 exactly.
  It is a diagnostic, and also the practical bottom when the last level is small.
- **Slow-mode top.** The deepest level gets a smoother over its own blocks (block + coupled blocks, projected CAP),
  and on top of it a small global space of *slow modes*: the vectors that the cycle below damps worst, found by a
  few steps of block subspace iteration on the cycle's error operator, in rounds of 32 vectors. "top N" means the
  first N of them are used, with a dense Galerkin solve on those N vectors. This is the same construction as the
  two-level method of 2026-11-13, applied on the last coarse level and not on the fine level.

The **two-level baseline** is that earlier method: the fine smoother S0 plus slow modes directly on the fine level.

## 7. Results (9072a, Ec + 0.15)

Level 1 = window selection, cut 1e-2 (13 674 columns), two x merges, V-cycle.

| | wide window on every level | slices on levels 2 and 3 |
|---|---|---|
| level 1: size, its with exact level 1 | 13 674, 16-20 | 13 674, 16-20 |
| level 2: size, its with exact level 2 | 7275, 16-20 | 3726, 16-21 |
| level 3: size, its with exact level 3 | 4283, 16-20 | 1105, 18-23 |
| coarsening per merge | 1.9, 1.7 | 3.7, 3.4 |
| 3 levels + slow-mode top, N 32 / 128 / 256 | 24-28 / 21-25 / 18-22 | 30-35 / 27-32 / 24-29 |

Two-level baseline, N 32 / 128 / 256 slow modes on the fine level: 38-49 / 21-28 / 19-24.

- Every merge is lossless when the level below is smoothed with the pair blocks.
- With energy slices each merge shrinks the level by 3.4-3.7; three levels reduce 72 672 orbitals to 1105 unknowns
  (factor 66) at a cost of 2-3 iterations.
- With a slow-mode top on the wide level 3, 32 global vectors give 24-28 iterations where two-level needs 128.
- Solve times are 8-15 s against 8-16 s for two-level: the fine smoother is applied twice per iteration in both and
  dominates. There is no time gain on this device.

Other level-1 choices (exact level 1): core selection 1e-2 (2407 columns) 41-48; core selection 1e-3 (6796) 32-38;
window selection 1e-1 (4485) 40-46. W-cycle with the single-block smoother and 128-256 slow modes on level 1: 16-23.

## 8. Limits and open points

- **Level 1 is large.** Only the window selection at cut 1e-2 gives 16-20 iterations, and it keeps 1 column in 5.3.
  Stronger compression costs iterations (32-48). The smoother of A1 has windows of up to 4827 columns, about half
  the size of the fine windows.
- **Cutting the cores in y made level 1 worse, not better** (38 cores, 6 A transverse shell): 31 707 columns (1 in
  2.3), and the level-2 pair windows have 3364-11 207 columns, more than the fine windows. These runs ran out of GPU
  memory while building level 3, so there are no iteration counts.
- **The density matrix in a coarse basis over-occupies.** Compared with the same window on the fine level, the
  level-1 density matrix of a pair has 1.5-2.2 times the trace with the wide window and 6-12 times with the slice
  (35-70 modes above 1e-2 where the fine level has 14). The excess modes are kept in the basis. This has not hurt
  convergence, but it limits the coarsening. The cause is not verified (suspect: the cut states of the block-cut
  Galerkin operator).
- **One device, one energy.** No length or width scaling yet, no other energies, no 6 nm.
- **A slow-mode top built through a projected smoother fails** (the smoother of A1 projected into level 2 is not a
  smoother of A2): the deepest level needs its own block smoother or an exact solve.

## 9. How to run

    # from the repo root on attelas2; prepared inputs + psi from results_ms/gpu/wf_gh200_9072a
    results_ms/gpu/mlevel/run_mlevel_9072a.sh                                   # x, x merges, wide window
    UP=x:1e-2:0.05:0.25,x:1e-2:0.1:0.2 LOG=mlevel_9072a_xx_slice results_ms/gpu/mlevel/run_mlevel_9072a.sh
    SPLIT=2,1 UP=y:1e-2,x:1e-2,x:1e-2 LOG=mlevel_9072a_y2_yxx results_ms/gpu/mlevel/run_mlevel_9072a.sh

`--l1 dm:CUT | dmc:CUT` selects level 1; `--up AXIS:CUT[:mu1:mu2[:npts]],...` adds one level per entry (merge axis,
occupation cut, optional energy slice relative to Ec); `--sweep fine,exact,top` chooses the two-level baseline, the
"exact level m" ladder and the slow-mode top; `--gamma 2` is the W-cycle. All launch lines are in `job.sh`.
