# 21600a: window halo and length scaling of the density-matrix hierarchy (2026-11-16)

The density-matrix hierarchy (report 2026-11-15) run on the next device of the length series: 21 600 atoms,
172 320 device orbitals, x 378 A, the 2808a / 9072a cross section (2.4x the length of 9072a). The solver converges
to 1e-10 in 18-25 fine iterations when the last level is solved exactly, 2-5 more than on 9072a. Shortening the
density-matrix windows of level 2 to one neighbour block per side ("halo 1") costs about one iteration and nearly
halves the level-2 build.

**Status.** The exact psi of this device was computed on 2026-11-16 (T_exact = 0.937877 over the 20 columns). The
two cases of section 1 have been rerun against it: T error <= 1e-11 (absolute), psi error <= 7e-12 in every sweep,
with and without halo. The y-cut level 1 (section 3) needs fewer iterations, 12-15, the same as on 9072a, at the price
of much larger levels and 3x the GPU memory.

Setup: MoS2/HfO2 21600a (`inputs/mos2_hfo2_21600a`), 22 slabs of 3120..8485 orbitals, own lead (Ec 0.5062 eV),
E = Ec + 0.15 eV, all 20 injection columns, flexible block GMRES to 1e-10 in chunks of 10 columns, iterations given as
min-max over the columns. Fine smoother: 43 CAP-Schwarz windows of 16 A along x (CAP 2 eV over 2 A), unsplit in y.
Level 1: window density matrices, cut 1e-2. Levels 2 and 3: x merges of block pairs, cut 1e-2. GH200 (attelas1), GPU
shared with another user's jobs, so wall times are indicative only.

Code: `python -m qtx_ms.mlevel`, script `results_ms/gpu/mlevel/run_qtxms_mlevel_21600a.sh`.
Logs: `results_ms/gpu/mlevel/qtxms_21600a_{hall,h1full}.log` / `.json`.

## 1. Window halo on 21600a

"No halo" = the level-2 density-matrix window of a block pair contains every level-1 block coupled to it (2 blocks
per side). "Halo 1" = one block (7.9 A) per side for the density matrix; the smoother of A1 keeps the full windows
(`--halo 1 --halo-smooth full`).

|                                   | no halo                  | halo 1 for rho, full smoother |
|-----------------------------------|--------------------------|-------------------------------|
| N1 / N2 / N3                      | 32 960 / 17 788 / 10 522 | 32 960 / 16 114 / 10 275      |
| n / N per level                   | 5.2 / 9.7 / 16.4         | 5.2 / 10.7 / 16.8             |
| level-2 rho windows (columns)     | 1586..4762..4953         | 772..3179..3382               |
| level-2 rho build                 | 131 s (2 batches)        | 73 s (1 batch)                |
| exact level 2, iterations         | 18-23                    | 19-25                         |
| exact level 3, iterations         | 19-24                    | 19-25                         |
| 3 levels + top N 32, iterations   | 34-40                    | 32-39                         |
| 3 levels + top N 128, iterations  | 35-41                    | 30-36                         |
| 3 levels + top N 256, iterations  | 32-37                    | 28-33                         |
| residual                          | <= 1e-10                 | <= 1e-10                      |
| T err, exact level 2 / 3          | -3.3e-12 / -3.8e-12      | +9.3e-12 / +8.7e-12           |
| psi err, exact level 2 / 3        | 4.8e-12 / 4.6e-12        | 4.1e-12 / 4.0e-12             |
| T err, top N 32 / 128 / 256       | +2.2e-12 / -6.2e-13 / +3.7e-12 | +2.0e-12 / +2.7e-12 / +6.8e-13 |
| psi err, top N 32 / 128 / 256     | 7.2e-12 / 3.6e-12 / 5.1e-12 | 5.4e-12 / 5.9e-12 / 3.3e-12   |
| fine solve, 20 columns            | 20-33 s                  | 22-32 s                       |
| GPU memory, peak                  | 27.6 GB                  | 27.6 GB                       |
| run                               | 1307 s (*)               | 390 s                         |

(*) includes 852 s for the 43 window density matrices of level 1, built once and cached; the halo-1 run read them.

T err = T - T_exact summed over the 20 columns (T_exact 0.937877); psi err as printed by `qtx_ms.mlevel`, against
the direct solution. The iteration counts with the slow-mode top level vary by 1-2 between runs of the same case
(the reruns with errors gave 32-38 / 35-41 / 33-38 without halo and 33-39 / 31-37 / 28-34 with halo 1).

"Exact level m" = levels 1..m with a dense LU of A_m at the bottom. "Top N" = slow modes of the level-3 cycle as a
last level of N unknowns instead of the LU.

- Halo 1 costs about one iteration on the exact levels, gives a 9% smaller level 2 and a 1.8x faster level-2 build.
- With the slow-mode top level, halo 1 is 2-5 iterations better. The slow-mode rounds settle with halo 1 (largest
  |theta| of the last round 0.13) and not without it (0.99).
- The level-1 couplings beyond the first neighbour block along x are 0.4% of the first (0.21 eV against 51 eV).

## 2. Length scaling: 9072a -> 21600a (unsplit, no halo)

|                                   | 9072a (x 188 A)        | 21600a (x 378 A)         |
|-----------------------------------|------------------------|--------------------------|
| device orbitals n                 | 72 672                 | 172 320                  |
| fine windows = level-1 blocks     | 19                     | 43                       |
| N1 / N2 / N3                      | 13 674 / 7275 / 4283   | 32 960 / 17 788 / 10 522 |
| n / N per level                   | 5.3 / 10.0 / 17.0      | 5.2 / 9.7 / 16.4         |
| final blocks                      | 5                      | 11                       |
| exact level 2, iterations         | 16-20                  | 18-23                    |
| exact level 3, iterations         | 16-20                  | 19-24                    |
| 3 levels + top N 32, iterations   | 23-28                  | 34-40                    |
| 3 levels + top N 128, iterations  | 21-26                  | 35-41                    |
| 3 levels + top N 256, iterations  | 19-23                  | 32-37                    |
| T err / psi err (exact levels)    | <= 1e-12 / <= 6e-12    | <= 4e-12 / <= 5e-12      |

9072a reference: `results_ms/gpu/mlevel/mlevel_9072a_xx_rows.log` (the script version; same level sizes and counts as
`qtx_ms.mlevel` on 2808a and on the y-cut 9072a).

- 2.4x the length costs 2-5 iterations when the last level is solved exactly.
- With the slow-mode top level it costs 9-15: a top level of 32-256 modes does not scale with the length.
- The compression per level is the same on both devices; the level sizes grow with the length.

## 3. y-cut level 1 on 21600a

Every x window is cut into two boxes along y (+6 A of embedding on each side), so level 1 has 86 blocks; then a y
merge (halo 1 for rho, full smoother windows) and two x merges. attelas2, GPU not shared.

|                                   | unsplit, halo 1 (x, x)   | y-cut, halo 1 (y, x, x)            |
|-----------------------------------|--------------------------|------------------------------------|
| level-1 blocks                    | 43                       | 86                                 |
| level sizes N1 / N2 / ...         | 32 960 / 16 114 / 10 275 | 79 285 / 46 185 / 27 096 / 16 132  |
| n / N per level                   | 5.2 / 10.7 / 16.8        | 2.2 / 3.7 / 6.4 / 10.7             |
| final blocks                      | 11                       | 11                                 |
| exact last-but-one level, its     | 19-25 (level 2)          | 12-15 (level 3)                    |
| exact last level, its             | 19-25 (level 3)          | 12-15 (level 4)                    |
| T err / psi err, exact levels     | <= 1e-11 / <= 5e-12      | <= 6e-13 / <= 4e-12                |
| all levels + top N 32, its        | 33-39                    | 27-32                              |
| all levels + top N 128, its       | 31-37                    | 26-30                              |
| all levels + top N 256, its       | 28-34                    | 23-27                              |
| T err / psi err, top              | <= 3e-12 / <= 6e-12      | <= 2e-12 / <= 4e-12                |
| fine solve, 20 columns            | 21-32 s                  | 7-13 s                             |
| level-1 window rho (once, cached) | 852 s                    | 650 s                              |
| rho of the further levels         | 73 + 36 s                | 181 + 131 + 53 s                   |
| GPU memory                        | 27.6 GB peak             | 63 GB after level 1, ~80 GB peak   |
| run (window rho cached)           | 411 s                    | 609 s                              |

Exact level 2 of the y-cut case (N 46 185) is above the size limit of the dense solve (30 000) and was skipped.

- The y cut takes 12-15 iterations on 21600a against 14-17 on 9072a (same settings): no growth with the length
  when the last level is solved exactly. The unsplit level 1 grows from 16-20 to 19-25.
- With the slow-mode top level the y cut also grows with the length: 23-32 against 16-24 on 9072a.
- The price is the size: level 1 keeps n / 2.2 unknowns and the last level 16 132 (unsplit 10 275), and the GPU
  memory is 3x. The fine solve is 2-3x faster per energy (fewer iterations, smaller fine windows); the times of the
  two columns are from different hosts (the unsplit one on a shared GPU).

## 4. Limits

- One energy (Ec + 0.15), one device.
- The y-cut case nearly fills one GH200 (about 80 GB); a wider device will not fit this way.
- Shared GPU: times are not comparable between runs; compare iteration counts.
- Level 1 is large (N1 = n / 5.2) and the last exact level still has 10 000 unknowns.

## 5. Run log

- Exact psi of 21600a at Ec + 0.15 (block-tridiagonal LU on the GPU): `results_ms/gpu/mlevel/run_exact_21600a.sh`
  -> `results_ms/gpu/wf_gh200_21600a/psi`, on attelas2. Done: 24 s, residual 8.6e-14, T_exact(20 columns) = 0.937877.
- y-cut level 1 (y, x, x merges, halo 1 for rho, full smoother windows) with T / psi errors: attelas2, log
  `qtxms_21600a_y2_h1full.log`. The first attempt stopped at once ("window cores must partition the device"): three
  atoms of 21600a lie exactly on the y = 0 face of the boxes and rounding put them into both cores. Fixed in
  `qtx_ms/schwarz.py` (owner box by floor(u / box width), as the release already does); restarted 10:32: 86 window
  cores of 1092..2176 orbitals, windows 2990..7724. That run built level 1 (86 blocks, N1 = 79 285 = n / 2.2, A1
  11 GB, GPU 63 GB; 650 s for the 86 window density matrices, now cached) and then ran out of GPU memory (94 GB)
  while building the smoother of A1: the S, H and A block lists of level 1 (3 x 11 GB) were still held next to the
  block rows of A1. `qtx_ms/mlevel.py` now frees them before that smoother is built; the restart of 10:56 ran
  through (609 s, 52 GB after level 2). Log of the failed run: `qtxms_21600a_y2_h1full_oom.log`.
- The two cases of section 1 again with `--psi-dir` (T / psi errors), attelas1, the same log names: no halo done
  (476 s), halo 1 done (411 s).

## 6. How to run

    results_ms/gpu/mlevel/run_sigma_21600a.sh                 # lead Sigma cache of the host (25 points, 1-4 s each)
    WAIT_GB=40 HALO=all results_ms/gpu/mlevel/run_qtxms_mlevel_21600a.sh
    WAIT_GB=40 HALO=1 LOG=qtxms_21600a_h1full results_ms/gpu/mlevel/run_qtxms_mlevel_21600a.sh --halo-smooth full
    SPLIT=2,1 HALO=1 WAIT_GB=85 LOG=qtxms_21600a_y2_h1full results_ms/gpu/mlevel/run_qtxms_mlevel_21600a.sh --halo-smooth full
    # T / psi errors: results_ms/gpu/mlevel/run_exact_21600a.sh once, then add --psi-dir results_ms/gpu/wf_gh200_21600a/psi
