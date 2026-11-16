# Density-matrix hierarchy: findings, what we learnt, next steps (2026-11-15)

Explanation of the method for others: reports/2026-11-15-density-matrix-hierarchy/summary.md.
Session log: sessions/2026-11-15-density-matrix-hierarchy.txt. Code: results_ms/gpu/mlevel/{merge.py, mlevel.py}.
All numbers: MoS2/HfO2 9072a (72 672 orbitals, 19 unsplit CAP-Schwarz windows 16,2,2), Ec + 0.15, 20 columns,
FGMRES tol 1e-10, its min-max over columns, V-cycle unless stated.

## Findings

1. Smoothed aggregation is dead. 6 nm, halo-2 level-1 windows: V / W stall (residual 0.93-1.0 after 100 its), K4
   6e-3 / 0.12. The smoothed basis functions (16-28 cores) do not fit in local level-1 windows.

2. Two selections for a density-matrix level 1 (from the S0 window rho, restricted to the core):
   "dm" = eigenvectors of the WINDOW rho above the cut; "dmc" = eigenvectors of the CORE block (mode-space style).
   Exact level 1: dmc 1e-2 (N1 2407, n/N1 30) 41-48; dmc 1e-3 (6796) 32-38; dm 1e-1 (4485) 40-46; dm 1e-2
   (13 674, n/N1 5.3) 16-20. Two-level with 32 / 64 / 128 / 256 slow modes on the fine level: 38-49 / 26-35 / 21-28
   / 19-24. Only dm 1e-2 beats the slow modes, and it barely coarsens.

3. The level-2 smoother must be formed in the level-1 basis with the level-2 blocks (user's design). With RAS on A1
   over the pairs (pair + coupled cores, projected CAP), exact level 2 = exact level 1 (16-20). My first version
   (separate single-core smoother on A1) gave 34-40 and lost 15-30 its per level.

4. Repeated merging is lossless: exact level 1 / 2 / 3 = 16-20 / 16-20 / 16-20 (N 13 674 / 7275 / 4283), wide
   window, x, x merges.

5. Energy slices give geometric coarsening: level 2 on Ec+0.05..0.25, level 3 on Ec+0.1..0.2: N 13 674 / 3726 /
   1105 (3.7x, 3.4x per merge; wide: 1.9x, 1.7x), exact its 16-20 / 16-21 / 18-23. Smoother windows of A2 832-1900
   columns (wide 1556-3654).

6. Slow-mode top on the deepest level (own-block smoother + first N slow modes), N 32 / 128 / 256: wide level 3
   24-28 / 21-25 / 18-22; sliced level 3 30-35 / 27-32 / 24-29. The hierarchy needs far fewer global vectors than
   two-level (32 instead of ~128 for ~25 its).

7. W-cycle (gamma 2) recovers the exact-level-1 count with slow modes on level 1: single-core smoother 28-34 /
   18-23 / 16-21, pair smoother 21-27 / 17-22 / 16-20 (V: 39-45 / 31-38 / 24-30 and 31-36 / 25-30 / 18-24).

8. Over-occupation of the density matrix computed in a coarse basis (middle pair, vs the same window on the fine
   level, same orbitals and ramp, CAP not projected): trace x1.47 (dmc 1e-2), x2.2 (dmc 1e-3), x1.9-2.1 (dm), wide
   window; slice: fine trace 2.04 and 14 modes > 1e-2, level 1 x5.8-12 and 35-70 modes. No negative eigenvalues,
   reference modes missed 2e-4...3e-2. Worse with a larger level-1 basis although that basis holds the fine rho to
   1e-3.

9. Cutting the cores in y does not shrink level 1 (--cap-split 2,1, tov 6, 38 cores of 1092-2130 orbitals): N1
   31 707 (n/N1 2.3; 926 columns per half core vs 789 per full core), pair windows of A1 3364-9385 (y merge) /
   3364-11 207 (x merge) columns, LUs 9.7 / 13.8 GB, larger than the fine windows (2786-7672). All three runs died
   with GPU out of memory while building level 3 (mlevel_9072a_y2_*.log): no iteration counts.

10. Not working: slow modes of A2 seen through the projected level-1 smoother P2^H S1c P2 (|theta| 400-980 -> 10-40;
    stalls for slice / W / dmc). The deepest level needs its own block smoother or an exact solve.

11. Solve times equal two-level (8-15 s vs 8-16 s): two S0 passes per iteration dominate either way.

## What we learnt

- Consistency is what makes a level lossless: the basis of level k+1 comes from the same local operators (window,
  embedding, CAP) that the smoother of level k inverts, and the smoother blocks are the blocks of the level above.
- The narrowing argument holds: on the fine level the slice of a pair really holds few states (14 above 1e-2 for
  +-0.1 eV), and narrowing per level gives a constant coarsening factor per merge along x.
- The top level stops growing in importance once local levels sit below it: iteration counts depend weakly on the
  number of global slow modes (the two-level method needs more of them as the device grows: 64 on 2808a, ~256 on
  9072a, 512 on 6 nm).
- The window selection keeps the modes of the whole window on each core. That is why it captures the smoother's slow
  modes, and also why level 1 is large and why smaller cores make it larger, not smaller.
- The projected CAP does its job as a smoother ingredient (stable cycles on every level) but a density matrix built
  with it counts extra states; so far these only cost columns.
- Not learnt yet: whether any of this pays in time. On 9072a it does not.

## Next steps

1. Make level 1 smaller without losing the 16-20 its.
   a. Level 1 itself on an energy slice (window rho over E +- dE; needs the window factorizations at a few contour
      points per energy range, the smoother already factors every window at E).
   b. Selection between dmc and dm: core selection plus the smoother's slow modes cut to the cores, or the window
      selection with the embedding weight discarded (weight of each window mode on the core as a second cut).
   c. Smaller pair windows: embedding = face neighbours only instead of all coupled cores.
2. DONE 2026-11-15 afternoon (see notes/DECISIONS.md, "Memory fix + row-restricted inverses": y cut 13-16 its,
   larger levels; rows64 makes S0 < 1 s per solve). Was: Fix the memory use of mlevel.py (free the projected CAPs and S / H window copies per batch, complex64 where
   possible, keep the level-1 CAP projection sparse) and rerun the y-cut cases; then decide whether a y cut has any
   use.
3. More than one case: Ec + 0.05 / 0.2 / 0.3, W-cycle with slices, deeper recursion (x, x, x, x to one block) with a
   rule for the slice width per level (dE_k ~ 1 / block length).
4. Length series at fixed cross section: 2808a / 9072a / 21600a (examples/device_MoS2HfO2_21600a_178560o, x 378 A;
   needs prep + exact psi). Question: do the last-level size and the its stay bounded with length?
5. Width series: 21600a / 6 nm / 12 nm. 154 of 172 window density matrices of 6 nm are cached
   (/scratch/jiacao/mg_modespace/6nm/dmw); expect ~20 cores per pair window there.
6. Understand the over-occupation: fine residual and position of the excess modes (cut states at the block faces?),
   and whether removing them changes the its.
7. Cost: the fine smoother is the bottleneck in every variant; the hierarchy only pays if it cuts fine iterations
   well below two-level or replaces the growing global coarse level on large devices (Alps multi-GPU plan,
   sessions/2026-11-15-alps-multigpu-multilevel-plan.txt).
