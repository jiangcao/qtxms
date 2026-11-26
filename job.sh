#M=128 ROUNDS=8 FGR=2,4,6,8 MAXIT=100 LOG=slow_6nm_t015_m128 nohup results_ms/gpu/slow/run_slow_6nm_t015.sh > /dev/null 2>&1 &

#M=128 ROUNDS=8 FGR=4,6,8 MAXIT=100 LOG=slow_6nm_t015_tov6 nohup results_ms/gpu/slow/run_slow_6nm_t015.sh --cap-tov 6 > /dev/null 2>&1 &
#nohup results_ms/gpu/slow/run_sweep_recycle_2808a.sh > /dev/null 2>&1 &
#nohup sh -c 'results_ms/gpu/pou/run_pou_ms_2808a.sh && results_ms/gpu/pou/run_pou_ms_6nm.sh' > /dev/null 2>&1 &
#
# start after the 6 nm PoU run (pou_ms_6nm_d4.log ends with "done") has finished:
#nohup results_ms/gpu/pou/run_tile_iter_2808a.sh > /dev/null 2>&1 &
#
# iterated level 1 with the CAP projected into the neighbour bases (start after tile_iter_2808a_slab.log ends with "done"):
#SUF=_cap nohup results_ms/gpu/pou/run_tile_iter_2808a.sh --env basis --env-cap outer > /dev/null 2>&1 &
#
# same with 2 y strips per slab (4 slabs in x, no z cut; 8 tiles), start after the run above has finished:
#PARTS=tile SUF=_y2_cap nohup results_ms/gpu/pou/run_tile_iter_2808a.sh --tsplit 2,1 --env basis --env-cap outer > /dev/null 2>&1 &
#
# partition-of-unity basis from the whole-cluster rho (2 y strips per slab; neighbours in the iteration-0 basis, outer CAP):
#PARTS=tile SUF=_y2_pou nohup results_ms/gpu/pou/run_tile_iter_2808a.sh --tsplit 2,1 --iters 0 --no-diag0 --env-cap outer --pou 4,7.5 > /dev/null 2>&1 &
#
# spurious-state check of the slow-mode coarse level (2808a, ~5-10 min):
#nohup results_ms/gpu/slow/run_slow_spec_2808a.sh > /dev/null 2>&1 &
#
# release CLI check of the slow-mode coarse level (2808a ~2 min; SIXNM=1 adds 6 nm Ec+0.15, ~30 min):
# SIXNM=1 nohup results_ms/gpu/slow/run_release_slow.sh > results_ms/gpu/slow/run_release_slow.out 2>&1 &
# multilevel step 1 (2808a): GenEO-type coarse spaces from the smoother's own CAP windows (~20-40 min, first run
# builds the window density matrices, 24 contour points x 7 windows):

# nohup results_ms/gpu/geneo/run_geneo_2808a.sh > /dev/null 2>&1 &

# slow-mode local coherence (after geneo finishes; GPU shared)
# nohup results_ms/gpu/geneo/run_coherence_2808a.sh > /dev/null 2>&1 &

# (a) slow-mode rank per window, 2808a vs 6 nm at Ec+0.15 (6 nm: 172 windows, 68 GB host LUs)
# nohup results_ms/gpu/geneo/run_coherence_6nm.sh > results_ms/gpu/geneo/run_coherence_6nm.out 2>&1 &
#
# rerun 6 nm only after the OOM fixes (hdot, contiguous Q) and the strongest-12 column fix; check nvidia-smi first
#SKIP2808A=1 nohup results_ms/gpu/geneo/run_coherence_6nm.sh > results_ms/gpu/geneo/run_coherence_6nm.out 2>&1 &
#
# multilevel step 2: three-level V-cycle on 2808a (level 1 = window-core dm 1e-2 / 1e-4 + per-core cut of 512 slow
# modes, halo 1 / 2, level 2 = 3 x 64 level-1 slow modes; Ec+0.05/0.2/0.3; needs the geneo dm cache on /scratch)
# nohup results_ms/gpu/mlevel/run_vcycle3_2808a.sh > results_ms/gpu/mlevel/run_vcycle3_2808a.out 2>&1 &   # done 2026-11-14 (attelas1, 1054 s)
#
# multilevel step 3: three-level cycle on 6 nm, Ec+0.15 (vcycle3.py: block-sparse level 1 = 512 fine slow modes cut per
# core at 3e-2 / 1e-2, halo 1, distance-ramp CAP, level 2 128 x 4). Host with /scratch/jiacao/mg_modespace/6nm; start
# after the 6 nm coherence run (same GPU) has finished
# nohup results_ms/gpu/mlevel/run_vcycle3_6nm.sh > results_ms/gpu/mlevel/run_vcycle3_6nm.out 2>&1 &   # stopped 2026-11-15 after slow:512:3e-2 exact (58-65 its vs two-level 30-36)
#
# 6 nm: does exact level 1 at the per-core cut 1e-2 beat the two-level 30-36 its? (only the exact cycle, ~10-15 min;
# stop the run above first)
# L1=slow:512:1e-2 CYC=exact LOG=vcycle3_6nm_c1e-2 nohup results_ms/gpu/mlevel/run_vcycle3_6nm.sh --fgmres-log 10 > results_ms/gpu/mlevel/run_vcycle3_6nm_c1e-2.out 2>&1 &   # done: 36-40 its
#
# 6 nm: smoothed-aggregation level 1, P1 = (I - M A) P_tent (one S0 step), exact level 1 at the cuts 3e-2 (unsmoothed
# 58-65 its) and 1e-2 (36-40) vs two-level 30-36; ~25-35 min
# L1=slow:512:3e-2,slow:512:1e-2 CYC=exact LOG=vcycle3_6nm_sa1 nohup results_ms/gpu/mlevel/run_vcycle3_6nm.sh --l1-smooth 1 --fgmres-log 10 > results_ms/gpu/mlevel/run_vcycle3_6nm_sa1.out 2>&1 &   # done: exact 25-29 (3e-2) / 24-28 (1e-2); halo-1 level 2 does not settle (|theta| 206 -> 44)
#
# 6 nm: smoothed level 1 at 3e-2 with halo-2 level-1 windows (smoothed basis functions span 16-28 cores, halo-1 windows
# 10-15): V / W / K4 vs exact level 1 25-29 its; ~30-45 min
# L1=slow:512:3e-2 HALO=2 CYC=v,w,k4 LOG=vcycle3_6nm_sa1_h2 nohup results_ms/gpu/mlevel/run_vcycle3_6nm.sh --l1-smooth 1 --fgmres-log 10 > results_ms/gpu/mlevel/run_vcycle3_6nm_sa1_h2.out 2>&1 &   # done (6346 s): V / W stall (residual 0.93-1.0 after 100 its), K4 6e-3 / 0.12; level 2 |theta| 65 -> 22. SA given up
#
# density-matrix hierarchy (user, 2026-11-15), step 1: 6 nm Ec+0.15, level 1 = window density-matrix modes cut to the
# cores (dm:1e-1 / 3e-2 / 1e-2, window selection), exact level 1 + V / W with 4 x 128 slow modes on top; builds the
# window density matrices (172 windows x 24 points, ~25 min) into /scratch/jiacao/mg_modespace/6nm/dmw
# nohup results_ms/gpu/mlevel/run_dmcore_6nm.sh > results_ms/gpu/mlevel/run_dmcore_6nm.out 2>&1 &   # stopped 2026-11-15 08:58 during the density-matrix build (154 of 172 windows cached): 9072a first (user)
#
# smoke test of merge.py on 2808a (Ec+0.05; level 1 dmc:1e-2, level 2 = 2-core clusters, fine reference, count sweep)
# (cd . && W=results_ms/gpu/wf_gh200 && CUDA_PATH=/usr/local/cuda-13.3 PYTHONPATH=release/qtx_modespace/src nice .venv-aarch64/bin/python -u results_ms/gpu/mlevel/merge.py $W/prep /scratch/jiacao/mg_modespace/sigma_2808a $W/psi 0.05 --dm-cache /scratch/jiacao/mg_modespace/geneo/dmw --l1 dmc:1e-2 --l2 1e-2 --merge x --ref 1 --sweep fine,exact1,l1,exact2,l2 --counts 64,128 > results_ms/gpu/mlevel/merge_2808a_smoke.log 2>&1 &)   # done (239 s)
#
# steps 2 + 3: level 2 from 2-core clusters in the level-1 basis (projected CAP), fine reference for cluster 41, top
# slow-mode count 128 / 256 / 512 on the fine level vs on level 1 vs on level 2 (6 strongest channels; ~3 h). Start
# after run_dmcore_6nm.sh has finished (needs its dm cache and the GPU):
# nohup results_ms/gpu/mlevel/run_merge_6nm.sh > results_ms/gpu/mlevel/run_merge_6nm.out 2>&1 &   # not started (queue cancelled 2026-11-15)
#
# the same on the intermediate device 9072a (2x the length of 2808a, same cross section; 19 unsplit windows, 10 x
# clusters; Ec+0.15, all columns; top count 32 / 64 / 128 / 256)
# nohup results_ms/gpu/mlevel/run_merge_9072a.sh > results_ms/gpu/mlevel/run_merge_9072a.out 2>&1 &   # done (1323 s): see CLAUDE.md
#
# 9072a with the WINDOW selection on level 1 (dm: all window-rho modes above the cut, restricted to the core; captures
# the smoother's slow modes, more columns) instead of the core-block selection; same level 2 / reference / count sweep
# L1=dm:1e-1,dm:1e-2 LOG=merge_9072a_dm nohup results_ms/gpu/mlevel/run_merge_9072a.sh > results_ms/gpu/mlevel/run_merge_9072a_dm.out 2>&1 &   # done (1010 s)
#
# 9072a, options 1 + 2 (user, 2026-11-15): level-2 smoother formed in the level-1 basis (RAS on A1 with the clusters as
# blocks: c1 / cexact2 / c2) with the V-cycle, then everything again with the W-cycle (--gamma 2); level 1 dm:1e-2
# (the good one) and dmc:1e-3; ~15 min each
# (L1=dm:1e-2,dmc:1e-3 SWEEP=c1,cexact2,c2 COUNTS=32,128,256 REF= LOG=merge_9072a_cl results_ms/gpu/mlevel/run_merge_9072a.sh; L1=dm:1e-2,dmc:1e-3 SWEEP=l1,exact2,l2,c1,cexact2,c2 COUNTS=32,128,256 REF= LOG=merge_9072a_w results_ms/gpu/mlevel/run_merge_9072a.sh --gamma 2) > results_ms/gpu/mlevel/run_merge_9072a_cl.out 2>&1 &   # done (650 s + 1677 s); see CLAUDE.md
#
# repeated merging (mlevel.py: any number of density-matrix levels, each = pairs of blocks of the level below, rho in
# that level's basis, the pairs are the blocks of the smoother of the level below; bottom = exact level m for every m,
# or own-block smoother + slow modes on the deepest level). 9072a, Ec+0.15, level 1 dm:1e-2:
#   unsplit (19 cores), levels 2, 3 = x, x merges, wide window
# nohup results_ms/gpu/mlevel/run_mlevel_9072a.sh > results_ms/gpu/mlevel/run_mlevel_9072a.out 2>&1 &   # done (164 s): exact level 1 / 2 / 3 16-20
#   then: x, x with energy slices (Ec+0.05..0.25, +0.1..0.2); level-1 cores also cut in y (--cap-split 2,1, tov 6, 38
#   cores) with y, x, x merges (wide / slices) and x, x, y merges
# nohup results_ms/gpu/mlevel/run_mlevel_9072a_queue.sh > results_ms/gpu/mlevel/run_mlevel_9072a_queue.out 2>&1 &   # done: slices N3 1105, 18-23 its; the three y-cut runs died with GPU OOM at level 3
#
# y-cut level 1 with the core selection only (dmc 1e-2 / 1e-3 on every level; y, x, x merges, wide and slices) + the
# unsplit dmc 1e-3 x, x reference (user, 2026-11-15)
# nohup results_ms/gpu/mlevel/run_mlevel_9072a_dmc.sh > results_ms/gpu/mlevel/run_mlevel_9072a_dmc.out 2>&1 &   # started 2026-11-15 11:45 (attelas2)
#
# y-cut level 1 (dm 1e-2) after the memory fixes of mlevel.py (--l1-solver rows), the three cases that died with OOM
# nohup results_ms/gpu/mlevel/run_mlevel_9072a_y2_rows.sh > results_ms/gpu/mlevel/run_mlevel_9072a_y2_rows.out 2>&1 &   # done 2026-11-15: 33 GB, exact levels 13-16 its
# Fine smoother S0: core rows of the window inverses (--cap-solver rows64) vs LUs (dense64), unsplit + y-cut
# nohup results_ms/gpu/mlevel/run_mlevel_9072a_s0rows.sh > results_ms/gpu/mlevel/run_mlevel_9072a_s0rows.out 2>&1 &   # done 2026-11-15: same its, S0 5-32 s -> <= 1 s per solve
#
# window halo of the further levels (mlevel.py --halo H,... per --up entry, --halo-smooth same|full; the log lists the
# coupling blocks of A_k, S_k by distance and the window size / rho cost per halo). Smoke test on 2808a (attelas1,
# y-cut, y + x merges, Ec+0.05): all coupled blocks vs halo 1 at level 2 vs halo 1 for rho only
# nohup results_ms/gpu/mlevel/run_mlevel_2808a_halo.sh > results_ms/gpu/mlevel/run_mlevel_2808a_halo.out 2>&1 &   # done 2026-11-15 (attelas1, 349 + 124 + 124 s): same its, rho 89 -> 49 s
#
# qtx_ms.mlevel (library + driver on the qtx_ms stack; python -m qtx_ms.mlevel) against the script on 2808a: halo 1, all
# nohup results_ms/gpu/mlevel/run_qtxms_mlevel_2808a.sh > results_ms/gpu/mlevel/run_qtxms_mlevel_2808a.out 2>&1 &   # done 2026-11-15 (attelas1, 165 + 171 s): N and its = the script
#
# 9072a y-cut, y, x, x merges with halo 1 at level 2 (qtx_ms.mlevel, GPU, Ec+0.15, exact + top); waits until 35 GB of
# the shared GPU are free (at most 24 h). Reference without halo: mlevel_9072a_y2_yxx_rows.log
# WAIT_GB=35 nohup results_ms/gpu/mlevel/run_qtxms_mlevel_9072a_h1.sh > results_ms/gpu/mlevel/run_qtxms_mlevel_9072a_h1.out 2>&1 &   # done 2026-11-15 (attelas1, shared GPU, 532 s): exact 14-17 / 14-17 / 16-19 (no halo 13-16), top 25-29 / 23-26 / 20-24 (19-22 / 16-20 / 15-18)
# the same with the full smoother windows (halo only for rho: basis vs smoother), then the same-code control without halo
# (WAIT_GB=35 LOG=qtxms_9072a_y2_yxx_h1full $R --halo-smooth full; WAIT_GB=35 HALO=all LOG=qtxms_9072a_y2_yxx_all $R) > results_ms/gpu/mlevel/run_qtxms_mlevel_9072a_h1full.out 2>&1 &   # done 2026-11-15 (attelas1, 534 + 841 s): h1full exact 14-16 / 14-16 / 14-17, all 13-16; level-2 rho 176 vs 485 s; R = results_ms/gpu/mlevel/run_qtxms_mlevel_9072a_h1.sh
#
# window halo on the next device, 21600a (172 320 device orbitals; own lead: Sigma cache built first by
# run_sigma_21600a.sh; no exact psi -> iterations / residuals only), qtx_ms.mlevel, Ec+0.15, attelas1 (shared GPU):
# unsplit x, x merges without halo, then halo 1 for rho with the full smoother windows; then the y-cut case (y, x, x,
# halo 1 + full smoother windows), which waits until 75 GB of the GPU are free (at most 24 h)
# nohup results_ms/gpu/mlevel/run_sigma_21600a.sh > results_ms/gpu/mlevel/run_sigma_21600a.out 2>&1 &   # done 2026-11-15 21:43 (attelas1, 25 points, 1-4 s each)
# (WAIT_GB=40 HALO=all $R; WAIT_GB=40 HALO=1 LOG=qtxms_21600a_h1full $R --halo-smooth full; SPLIT=2,1 HALO=1 WAIT_GB=75 LOG=qtxms_21600a_y2_h1full $R --halo-smooth full) > results_ms/gpu/mlevel/run_qtxms_mlevel_21600a.out 2>&1 &   # started 2026-11-16 09:22 (attelas1; the 2026-11-15 21:42 queue never ran: its "pgrep -f run_sigma_21600a.sh" wait matched its own command line); unsplit cases done, 1307 + 390 s: no halo exact 18-23 / 19-24, top 34-40 / 35-41 / 32-37; h1full exact 19-25 / 19-25, top 32-39 / 30-36 / 28-33, level-2 rho 73 vs 131 s; the y-cut waiter on attelas1 was stopped 2026-11-16 10:21 and the case moved to attelas2, see below); R = results_ms/gpu/mlevel/run_qtxms_mlevel_21600a.sh
#
# 21600a with T / psi errors (user, 2026-11-16): exact psi at Ec+0.15 by the block-tridiagonal LU on the GPU
# (run_exact_21600a.sh -> results_ms/gpu/wf_gh200_21600a/psi, ~75 GB of GPU memory), then the y-cut case on attelas2
# (free GPU; Sigma cache of that host built first), and the two unsplit cases again on attelas1 with --psi-dir (window
# rho cached there; they wait for the psi file, at most 6 h)
# attelas2, O=results_ms/gpu/mlevel, R=$O/run_qtxms_mlevel_21600a.sh:
# ($O/run_sigma_21600a.sh > $O/run_sigma_21600a_attelas2.out 2>&1 && WAIT_GB=80 $O/run_exact_21600a.sh > $O/run_exact_21600a.out 2>&1; SPLIT=2,1 HALO=1 WAIT_GB=75 LOG=qtxms_21600a_y2_h1full $R --halo-smooth full --psi-dir results_ms/gpu/wf_gh200_21600a/psi) > $O/run_qtxms_mlevel_21600a_attelas2.out 2>&1 &   # 2026-11-16 10:20 (attelas2): Sigma + exact psi done (24 s, T_exact 0.937877); the y-cut case failed (window cores, 3 atoms on y = 0; fixed in qtx_ms/schwarz.py) and was restarted alone at 10:32, ran out of GPU memory in the smoother of A1 (log qtxms_21600a_y2_h1full_oom.log; level-1 block lists now freed earlier in qtx_ms/mlevel.py), restarted 10:56: done (609 s; exact level 3 / 4 12-15 its, T err <= 6e-13, psi err <= 4e-12; top 27-32 / 26-30 / 23-27; N 79285 / 46185 / 27096 / 16132, ~80 GB peak)
# attelas1, P=results_ms/gpu/wf_gh200_21600a/psi:
# (until ls $P/psi_+0.656*.npz; do sleep 30; done; WAIT_GB=40 HALO=all $R --psi-dir $P; WAIT_GB=40 HALO=1 LOG=qtxms_21600a_h1full $R --halo-smooth full --psi-dir $P) > $O/run_qtxms_mlevel_21600a_psi.out 2>&1 &   # started 2026-11-16 10:24 (attelas1): no halo done (476 s, T err <= 4e-12, psi err <= 7e-12), h1full done (411 s, T err <= 1e-11, psi err <= 6e-12); overwrites the two logs of 09:22 (same runs, now with errors)
#
# More levels (user, 2026-11-16): merges down to 2 (9072a) / 3 (21600a) blocks, first on 9072a, then on 21600a; halo 1
# for rho at level 2 + full smoother windows, Ec+0.15, exact levels + top, T / psi errors. Per device 4 runs: y-cut
# y,x,x,x,x and unsplit x,x,x,x, each also with energy slices on the x merges (Ec+0.05..0.25, then 3x / 2x Ec+0.1..0.2).
# run_qtxms_mlevel_9072a_h1.sh now takes SPLIT= (empty) for the unsplit level 1. O=results_ms/gpu/mlevel,
# R9=$O/run_qtxms_mlevel_9072a_h1.sh, R=$O/run_qtxms_mlevel_21600a.sh, P="--halo-smooth full --psi-dir results_ms/gpu/wf_gh200_21600a/psi"
# attelas2 (all of 9072a, then the y-cut 21600a; touches $O/run_qtxms_mlevel_9072a_deep.done in between):
# (WAIT_GB=40 HALO=1 UP=y:1e-2,x:1e-2,x:1e-2,x:1e-2,x:1e-2 LOG=qtxms_9072a_y2_h1full_yxxxx $R9 --halo-smooth full; ... _yxxxx_slice; SPLIT= ... LOG=qtxms_9072a_h1full_xxxx; ... _xxxx_slice; touch ...done; SPLIT=2,1 HALO=1 WAIT_GB=85 UP=y:1e-2,x:1e-2,x:1e-2,x:1e-2,x:1e-2 LOG=qtxms_21600a_y2_h1full_yxxxx $R $P; ... _yxxxx_slice) > $O/run_qtxms_mlevel_deep_attelas2.out 2>&1 &   # started 2026-11-16 11:44 (attelas2)
# attelas1 (unsplit 21600a, waits for the .done file of 9072a, at most 6 h):
# (HALO=1 WAIT_GB=40 UP=x:1e-2,x:1e-2,x:1e-2,x:1e-2 LOG=qtxms_21600a_h1full_xxxx $R $P; HALO=1 WAIT_GB=40 UP=x:1e-2:0.05:0.25,x:1e-2:0.1:0.2,x:1e-2:0.1:0.2,x:1e-2:0.1:0.2 LOG=qtxms_21600a_h1full_xxxx_slice $R $P) > $O/run_qtxms_mlevel_deep_attelas1.out 2>&1 &   # queued 2026-11-16 11:45 (attelas1)
#
# Lead Sigma of the rho windows as a lead-core factor (2026-11-16 afternoon): the two y-cut 21600a runs above ran out of
# GPU memory in the merge to 3 blocks (level 6): sig_of built the window Sigma dense in level-1 coordinates (44138^2,
# 31.2 GB; logs *_yxxxx_oom.log, *_yxxxx_slice_oom.log). Lead-face cores hold 686 (L, 2 cores) + 1909 (R, 4 cores) of
# N1 79285 level-1 columns ($O/run_lead_cols_21600a.sh, log lead_cols_21600a.log); qtx_ms/mlevel.py sig_factor now
# projects only those. Rerun of both, attelas2:
# (SPLIT=2,1 HALO=1 WAIT_GB=85 UP=y:1e-2,x:1e-2,x:1e-2,x:1e-2,x:1e-2 LOG=qtxms_21600a_y2_h1full_yxxxx $R $P; SPLIT=2,1 HALO=1 WAIT_GB=85 UP=y:1e-2,x:1e-2:0.05:0.25,x:1e-2:0.1:0.2,x:1e-2:0.1:0.2,x:1e-2:0.1:0.2 LOG=qtxms_21600a_y2_h1full_yxxxx_slice $R $P) > $O/run_qtxms_mlevel_deep2_attelas2.out 2>&1 &   # started 2026-11-16 15:43, stopped after 5 min and restarted 15:50 with the memory profile ("mem |" lines: live / peak / pool / free GB per step, "held" per level; also rec["mem"] in the json) (attelas2)
#   both done 16:08: yxxxx 629 s, N 79285 / 46185 / 27096 / 16132 / 12059 / 9189, exact levels 3-6 11-15 its, top 12-15,
#   T err <= 6e-13, psi err <= 3e-12; slice 553 s, N6 2134 (n / N6 81), exact level 3 12-15, levels 4-6 16-22, top 17-24,
#   T err <= 2e-12, psi err <= 4e-12. Memory: peak 79.7 GB at the level-2 Galerkin blocks (level 1 holds 4 x A1 = 44 GB:
#   S, H, A list + the BlockOp copy), 77.5 GB (pool 87) in the dense LU of A3 (N 27096); level 6 now peaks at 64.9 GB.
#
# Plain mode-space T / psi error per exact level (psi = P A_m^-1 P^H b, "mode space level m" lines, rec["mode_space"]),
# 2026-11-16 16:20: the four deep 21600a runs again (same commands and logs as above, now with the memory profile too):
# attelas2 y-cut yxxxx + yxxxx_slice > $O/run_qtxms_mlevel_deep3_attelas2.out; attelas1 unsplit xxxx + xxxx_slice
# (WAIT_GB=40) > $O/run_qtxms_mlevel_deep3_attelas1.out
#
# Memory savings in qtx_ms/mlevel.py (2026-11-16 16:17): the A block list of a level points into the operator's block
# rows (one A_k less per level), exact level solves by SlabLU (block-tridiagonal LU over x slabs) instead of the dense
# LU. The two slice runs of the 16:10 chains already use it. Non-slice runs again with every level solved exactly
# (--exact-max 100000: mode-space T / psi error at levels 1, 2 too), queued 16:27 behind the slice runs:
# attelas2: SPLIT=2,1 HALO=1 WAIT_GB=85 UP=y:1e-2,x:1e-2,x:1e-2,x:1e-2,x:1e-2 LOG=qtxms_21600a_y2_h1full_yxxxx $R $P --exact-max 100000 > $O/run_qtxms_mlevel_deep4_attelas2.out
# attelas1: HALO=1 WAIT_GB=40 UP=x:1e-2,x:1e-2,x:1e-2,x:1e-2 LOG=qtxms_21600a_h1full_xxxx $R $P --exact-max 100000 > $O/run_qtxms_mlevel_deep4_attelas1.out
#   all done 16:41 (T_exact 0.937877). Memory, y-cut: overall peak 79.7 -> 71.0 GB (level-2 rho batches 70.8, now 4
#   batches instead of 6; FGMRES with the exact level 1 71.0); level-2 Galerkin 79.7 -> 65.1; slab LU of A3 1.54 GB,
#   no peak (dense LU: 77.5, pool 87); A1 (N 79285) 13.35 GB in 22 slabs. Unsplit peak 35.4 GB.
#   Plain mode space (cut 1e-2), T err / psi err: y-cut level 1 -1.2e-2 / 0.24, levels 2-6 -2.3e-2..-2.6e-2 / 0.54;
#   unsplit level 1 -4.1e-1 / 2.5, levels 2-5 -2.3e-1..-2.5e-1 / 3.2; unsplit slice -4.2e-1..-4.7e-1 / 3.3-3.6;
#   y-cut slice level 3 -2.5e-2 / 0.57, levels 4-6 +1.0e-1 / 5.4-6.3. FGMRES unchanged (y-cut 10-15 its, unsplit 18-25).
#
# Basis reuse over the window energies (2026-11-16 21:25, attelas2): qtx_ms/mlevel.py keeps the bases of the levels >= 2
# (energy independent) in a basis store (memory + <dm-cache>/<tag>_levels/l<k>_<hash>.npz, --basis-cache); per energy
# only A_k(E) and the smoothers are rebuilt. 2808a smoke test (run 1 two energies, run 2 from disk), exact psi of 21600a
# at Ec + 0.02,0.05,0.1,0.15,0.3,0.45,0.58,0.7, then y-cut yxxxx and unsplit xxxx with every level exact:
# setsid nohup results_ms/gpu/mlevel/run_basis_reuse.sh > results_ms/gpu/mlevel/run_basis_reuse.out 2>&1 &   # (attelas2)

# Stand-alone MS T + T error on the 2808a grid (Ec+0...0.5, 10 meV) for slab level 1 cuts 1e-8 / 1e-6 / 1e-4 on 2808a,
# 9072a, 21600a (user, 2026-11-17; big files in attelas2:/scratch/jiacao/mg_modespace/ms_cuts, table + ms_cuts.png in
# reports/2026-11-17-slab-level1-cuts by workflows/ms_cuts_summary.py):
# nohup workflows/run_ms_cuts.sh > results_ms/gpu/ms_cuts/run_ms_cuts.out 2>&1 &   # started 2026-11-17 08:04 (attelas2), stopped 08:18 (user), restarted 08:24

# Rebuild check of the npy / npz files deleted from results_ms 2026-11-17 (list: results_ms/deleted_2026-11-17.txt):
# workflows/{sigma,psi,bases} rebuild them, compared with the recorded json / logs; results in results_ms/verify_rebuild/
# nohup workflows/verify_rebuild_gpu.sh > results_ms/verify_rebuild/gpu.out 2>&1 &   # attelas1, started 08:52: 2808a + 9072a Sigma / psi, 15 release-CLI GPU runs from config.toml
# nohup workflows/verify_rebuild_cpu.sh > results_ms/verify_rebuild/cpu.out 2>&1 &   # mont-fort15, started 08:54: qtx_ms qttools Sigma / psi, release020 runs
# nohup workflows/bases/rebuild_l2direct.sh > results_ms/verify_rebuild/l2direct.out 2>&1 &   # mont-fort15, started 08:55 (~2 h): mos2_hfo2_l2direct/modes
# /scratch/jiacao/mg_modespace/queue_psi_21600a.sh -> workflows/psi/rebuild_psi_21600a.sh (attelas2, waits for the ms_cuts sweep): 21600a psi, 8 energies
# 2026-11-17 09:25 (user): release020 / ship010 re-checks stopped; mos2_hfo2_l2direct/modes restored after the rebuild
# passes its compare (waiter -> results_ms/verify_rebuild/l2direct_restore.out); mlevel exact psi now in
# /scratch/jiacao/mg_modespace/<dev>/psi on each GPU host, computed on demand by workflows/psi/ensure_psi.sh (the mlevel
# run scripts call it; the --psi-dir results_ms/gpu/wf_gh200*/psi paths in the older lines above no longer exist)
# 2026-11-17 (user): same sweep with the end slabs cleaned against the lead bands (--clean-slabs 0,-1 --win -0.6,0.55),
# queued on attelas2 after the raw sweep + 21600a psi job; runs <dev>_l1_<cut>_clean in results_ms/gpu/ms_cuts, summary
# (raw + clean) in reports/2026-11-17-slab-level1-cuts
#   CLEAN=1 nohup workflows/run_ms_cuts.sh > results_ms/gpu/ms_cuts/run_ms_cuts_clean.out 2>&1 &
# 2026-11-17 11:15 (user): OFF-state sweep, 0.4 eV barrier over the middle third (workflows/gate_device.py), cuts
# 1e-8 / 1e-6 / 1e-4 x {basis rebuilt at the gate (_gate), flat-band basis reused (_flat)}, 2808a / 9072a / 21600a
# (attelas2, needs the run_ms_cuts.sh modes in its /scratch); summary -> reports/2026-11-17-slab-level1-cuts-gate
#   nohup workflows/run_ms_cuts_gate.sh > results_ms/gpu/ms_cuts_gate/run.out 2>&1 &
# 11:39 cleaned sweep moved to attelas1 (user): /scratch/jiacao/mg_modespace/ms_cuts/copy_and_clean.sh pulls the Sigma /
# psi / 21600a prep + level-1 caches from attelas2, then CLEAN=1 workflows/run_ms_cuts.sh (log run_ms_cuts_clean.out)

# Stand-alone MS with tile level 1 cut in y, 6 nm 4 y strips / 12 nm 8 y strips per slab, CAP+OBC, cut 1e-8, 2808a grid
# Ec+0..0.5 (10 meV); exact T for 6 nm only (12 nm streamed LU ~400 GB host; EXACT_12nm=1 to force) (user, 2026-11-17).
# Prep of both devices redone (6 nm caches no longer on either host). Logs results_ms/gpu/ms_ycut, big files /scratch/jiacao/mg_modespace/<dev>
#   setsid nohup workflows/run_ms_ycut.sh > results_ms/gpu/ms_ycut/run.out 2>&1 &   # started 2026-11-17 16:50 (attelas2), stopped ~18:20 (user) after Ec+0..0.04 (level 1 N1 82 557, 2742 s, cached in /scratch/jiacao/mg_modespace/6nm/level1_tile4y; MS solve on the CPU, 118 s per energy)
# 2026-11-17 18:15 (user): mode-space solve on the GPU (core.block_solve: streaming block sweep, --ms-solve auto|gpu|cpu
# in the release CLI; ported to qtx_ms.msbasis.block_solve). Validation on the 6 nm tile4y basis at Ec+0.03/0.04/0.15,
# then the y-cut sweep restarted only if GPU / CPU T_MS agree to 1e-9 (attelas2):
#   setsid nohup results_ms/gpu/ms_ycut/validate_ms_gpu.sh > results_ms/gpu/ms_ycut/validate_ms_gpu.out 2>&1 &
#   setsid nohup results_ms/gpu/ms_ycut/queue_restart.sh > results_ms/gpu/ms_ycut/queue_restart.out 2>&1 &
# 2026-11-17 19:38 (user): y-cut tile sweep stopped after 19 energies on 6 nm (T err ~4x worse than slab level 1 at 2.2x
# N; partial json in results_ms/gpu/ms_ycut/6nm_tile4y_l1_1e-8, log run_tile.out); switched to slab level 1 (PART=slab,
# default now; CAP+OBC, emb 12, 24 pts, cut 1e-8; exact psi of the 19 energies reused), 6 nm then 12 nm (MS only):
#   setsid nohup workflows/run_ms_ycut.sh > results_ms/gpu/ms_ycut/run.out 2>&1 &   # started 19:38 (attelas2)
# 2026-11-17 20:20 (user): 12 nm as tile level 1 with 2 y strips (unsplit slab local rho ~145 GB of GPU); memory-lean GPU
# local rho (level1.GpuEmbedding in the release, msbasis.GpuEmbedding in qtx_ms: environment components, chunked A_EP / X,
# S_P^-1 into the accumulator; check: results_ms/gpu/ms_ycut/check_lean_rho.sh, 2808a block 1 = cached block to 1.3e-8).
# The 6 nm wrapper was stopped (python 2336043 keeps running); queue (attelas2):
#   setsid nohup results_ms/gpu/ms_ycut/queue_12nm.sh 2336043 >> results_ms/gpu/ms_ycut/run.out 2>&1 &
# 2026-11-18 12 nm with the own spectral OBC (obc_beyn, Woodbury, lean) + Sigma pre-pass (attelas2, 13:07):
#   DEVS=12nm PART=tile NY_12nm=2 XARGS="--obc beyn --sigma-pass" setsid nohup workflows/run_ms_ycut.sh > results_ms/gpu/ms_ycut/run_12nm_beyn.out 2>&1 &
#   setsid nohup results_ms/gpu/ms_ycut/fallback_12nm_4y.sh <pid> >> results_ms/gpu/ms_ycut/run_12nm_beyn.out 2>&1 &   # tile4y if tile2y OOMs
#   setsid nohup results_ms/gpu/ms_ycut/queue_12nm_exactT.sh >> results_ms/gpu/ms_ycut/run_12nm_exactT.out 2>&1 &      # exact T (--exact-T) after the MS run
#   setsid nohup results_ms/gpu/ms_ycut/run_12nm_exactT_odd.sh >> results_ms/gpu/ms_ycut/run_12nm_exactT.out 2>&1 &   # 2026-11-18 19:40: exact T at the 25 odd 10 meV energies (0.13, 0.17 first)
#   .venv/bin/python results_ms/gpu/ms_ycut/plot_T_6nm_12nm.py reports/2026-11-18-ms-ycut-6nm-12nm   # T / T error 6 nm + 12 nm; rerun when the odd 12 nm exact T is done
#   setsid nohup workflows/run_ms_l2onl1.sh > results_ms/gpu/ms_ycut/run_l2onl1.out 2>&1 &   # 2026-11-18 20:40 (user): slab level 2 on y-cut tile level 1, cuts 1e-6 / 1e-6: 6 nm tile2y, 12 nm tile4y (waits for the exact-T run)
# 2026-11-18 22:15 (user): OOM reruns (GPU shared with another user's ~60 GB job): 12 nm exact T Ec+0.41..0.49 (chunk 256)
# then 6 nm tile2y l1 + l2-on-l1; waits for the 12 nm l2onl1 job (pid 2383965) and >= 45 GB free GPU memory (attelas2):
#   setsid nohup results_ms/gpu/ms_ycut/rerun_oom.sh 2383965 >> results_ms/gpu/ms_ycut/rerun_oom.out 2>&1 &
#   setsid nohup results_ms/gpu/ms_ycut/rerun_12nm_l2onl1.sh >> results_ms/gpu/ms_ycut/run_l2onl1.out 2>&1 &   # 22:20: 12 nm l2-on-l1 MS solves again (first try: transient cuSOLVER getrs error at Ec+0)
# 22:45 (user): rerun_oom / rerun_12nm_l2onl1 stopped; the three pending runs in sequence: 12 nm l2-on-l1 with Ec+0 last;
# 6 nm tile2y l1 with the local rho in complex64 (new --l1-env-prec c64, half the GPU memory; L1ARGS in run_ms_l2onl1.sh)
# + l2-on-l1, after check_c64_rho.sh (12 nm block 1 c64 vs c128) passes, >= 20 GB free; 12 nm exact T 0.41..0.49, >= 35 GB:
#   setsid nohup results_ms/gpu/ms_ycut/rerun_seq.sh >> results_ms/gpu/ms_ycut/rerun_seq.out 2>&1 &   (attelas2)
# 23:10 c64 check FAILED (results_ms/gpu/ms_ycut/check_c64_rho.out): complex64 local rho has a ~1e-4 occupation noise floor
# (cut 1e-6: 686 vs 402 vectors, subspaces unrelated); 6 nm skipped; rerun_seq.sh now only waits for the 12 nm exact T.
# 23:15 (user): 12 nm slab level 2 on tile4y level 1, cuts 1e-8 / 1e-8 (level-2 recursion streamed from pinned host memory):
#   setsid nohup results_ms/gpu/ms_ycut/run_12nm_l2onl1_1e-8.sh >> results_ms/gpu/ms_ycut/run_12nm_l2onl1_1e-8.out 2>&1 &   (attelas2)
#   .venv/bin/python results_ms/gpu/ms_ycut/plot_T_12nm.py   # 2026-11-19: 12 nm T / T error, all 51 exact energies, tile4y l1 1e-8 + l2-on-l1 1e-6, 1e-8 -> figures/T_err_12nm.png
# 2026-11-19 (user): 6 nm slab level 2 on tile2y level 1, cuts 1e-8 / 1e-8 (folded lead, exact T from cached psi; Ec+0 last) (attelas2):
#   DEVS=6nm C1=1e-8 C2=1e-8 DE=0.01,...,0.50,0 setsid nohup workflows/run_ms_l2onl1.sh > results_ms/gpu/ms_ycut/run_6nm_l2onl1_1e-8.out 2>&1 &
# 2026-11-19 (user): FGMRES with the multi-level density-matrix hierarchy on the 6 nm, 21600a y-cut basis-reuse settings
# (y split 4,1 = same strip width, UP y,y,x,x,x,x at 1e-2, halo 1 full, exact levels <= 1e5, new qtx_ms.mlevel
# --host-blocks; --inputs now also takes a release prep); exact psi Ec+0.58/0.7, build at Ec+0.15, then the 8-energy
# sweep from the stored bases; queued behind the 6 nm l2-on-l1 run (attelas2):
#   setsid nohup results_ms/gpu/mlevel/run_mlevel_6nm_reuse.sh > results_ms/gpu/mlevel/run_mlevel_6nm_reuse.out 2>&1 &
# 2026-11-19 (user): before the 6 nm, the multi-level DM FGMRES over the exact-T grid (Ec+0.01..0.50) on 2808a, 9072a,
# 21600a, 21600a y-cut reuse settings (y 2,1; 2808a y,x; 9072a / 21600a y,x,x,x,x), exact psi from ms_cuts/<dev>/psi;
# 9072a / 21600a wait for the 6 nm l2-on-l1 run. qtx_ms/mlevel.py: Lev reference cycles freed per energy (gc), the
# 21600a sweep held ~35 GB of the previous energy. 6 nm queue (run_mlevel_6nm_reuse.sh) stopped until the comparison:
#   setsid nohup results_ms/gpu/mlevel/run_mlevel_grid.sh > results_ms/gpu/mlevel/run_mlevel_grid.out 2>&1 &   (attelas2)
# 2026-11-19 (user): streaming of the large operators (qtx_ms gpu.Streamer: pinned host memory, copy of item k + 1
# overlapped with the compute of item k; --stream-gb G in qtx_ms.mlevel: fine A (schwarz.StreamOp), fine smoother rows
# (CapSchwarz), A_k block rows (BlockOp host), level smoothers (ras) above G GB). 2808a, everything streamed: same its /
# errors (results_ms/gpu/mlevel/stream_2808a_{stream0,resident}.log); benchmark 360 GB/s, copy-bound. 6 nm queue
# relaunched with --stream-gb 4 --chunk 20, after the grid runs:
#   setsid nohup results_ms/gpu/mlevel/run_mlevel_6nm_reuse.sh > results_ms/gpu/mlevel/run_mlevel_6nm_reuse.out 2>&1 &   (attelas2)
# 2026-11-19 (user): matrix-free level 1 (qtx_ms.mlevel --l1-free, GalerkinOp: A1 @ y = P^H A P y through the fine
# operator, dense windows / Galerkin pairs from the pinned S1, H1 blocks; 2808a identical: results_ms/gpu/mlevel/
# l1free_2808a.log). 6 nm queue relaunched with --l1-free --stream-gb 8 --chunk 20 (after the grid runs), same command.
#   .venv/bin/python results_ms/gpu/mlevel/plot_mlevel_grid.py reports/2026-11-19-mlevel-grid   # 2026-11-19: its + plain-MS T err per level vs E, 2808a / 9072a / 21600a
# 2026-11-19 (user): 6 nm mlevel contour window widened to Ec-0.15..Ec+0.8 (MU in run_mlevel_6nm_reuse.sh, --mu at every
# level); wrapper relaunched (waits for the running psi rebuild), same command.
# 2026-11-19 (user): --free-levels K in qtx_ms.mlevel (A_1 .. A_K without stored blocks, recursive Galerkin products;
# 2808a K = 2, 3 identical: results_ms/gpu/mlevel/free{2,3}_2808a.log); 6 nm relaunched with --free-levels 2 (same command).
# 2026-11-19 (user): merge bases without the dense window eigh (the 6 nm build ran out of GPU memory in the level-3
# eigh: cusolver syevd asked 29 GB for a 13k window). qtx_ms.mlevel --merge-eig full | rand (randomized range finder on
# the formed rho) | sketch (rho Omega of a probe block straight from the contour: one LU + 2 solves per point, Nystrom,
# probes grown until they resolve the cut; as the msbasis tile sketch). 2808a comparison, Ec+0.05, 0.15 (attelas2):
#   setsid nohup results_ms/gpu/mlevel/run_mergeeig_2808a.sh > results_ms/gpu/mlevel/run_mergeeig_2808a.out 2>&1 &
#   MODES=sketch:c64 setsid nohup results_ms/gpu/mlevel/run_mergeeig_2808a.sh >> results_ms/gpu/mlevel/run_mergeeig_2808a.out 2>&1 &   # LU, probes, solves in complex64 (--sketch-prec c64)
#   MODES=sketch:lu64 (same command)   # LU in complex128, factors down-cast to complex64 for the solves (--sketch-prec lu64)
# 2026-11-19 (user): 6 nm relaunched with --merge-eig sketch (logs reuse_6nm_y4_sketch_*), gated by a 2808a check of the
# host Nystrom path (mergeeig_2808a_sketch_host.log) (attelas2):
#   setsid nohup results_ms/gpu/mlevel/run_sketch_host_check.sh > results_ms/gpu/mlevel/run_sketch_host_check.out 2>&1 &
# 2026-11-19 (user): 6 nm sketch build stopped (level 2, batch 10/82). dmc selection (--l1 dmc:1e-2: eigenvectors of the
# core / pair block of the local rho) on 2808a, 9072a, grid settings, Ec+0.05, 0.15, 0.3, 0.45 vs dm (grid_*_y2.log):
#   setsid nohup results_ms/gpu/mlevel/run_dmc_grid.sh > results_ms/gpu/mlevel/run_dmc_grid.out 2>&1 &   (attelas2)
#   dmc at 1e-2 keeps ~78 vectors per core and fails (maxit); 9072a stopped. Cut 1e-4 at every level (same script):
#   CUT=1e-4 setsid nohup results_ms/gpu/mlevel/run_dmc_grid.sh > results_ms/gpu/mlevel/run_dmc1e-4_grid.out 2>&1 &
#   dmc 1e-4: works, ~1.5x the fine its of dm at ~0.5x the coarse sizes; level-1 counts 483..500 per core, maybe capped by
#   the 1e-4 window-DM cache. Window DM files now record their keep (rebuilt when larger, filtered when smaller):
#   CUT=1e-5 DMKEEP=1e-6 setsid nohup results_ms/gpu/mlevel/run_dmc_grid.sh > results_ms/gpu/mlevel/run_dmc1e-5_grid.out 2>&1 &
#   dmc 1e-5 / keep 1e-6 on 2808a: N1 7335, N3 1776, its 11..23 (= dm at N3 2581): the 1e-4 cache capped level 1.
# 2026-11-19 (user): Schur folding of the window into the core / pair block for dmc (qtx_ms.mlevel --schur, SchurRho:
# complex64 environment LU + refinement, optional probes + Nystrom; the single-shot msbasis tile-contour techniques), same
# cut / keep, queued after the reference run (attelas2):
#   CUT=1e-5 DMKEEP=1e-6 SCHUR=single:1 setsid nohup results_ms/gpu/mlevel/run_dmc_grid.sh > results_ms/gpu/mlevel/run_dmc1e-5_schur_grid.out 2>&1 &
# 2026-11-19 (user): 21600a with dmc 1e-5 / keep 1e-6, Schur (single, 1 step); exact levels <= 50000 (shared GPU) (attelas2):
#   DEVS=21600a CUT=1e-5 DMKEEP=1e-6 SCHUR=single:1 EXMAX=50000 setsid nohup results_ms/gpu/mlevel/run_dmc_grid.sh > results_ms/gpu/mlevel/run_dmc1e-5_21600a.out 2>&1 &
#   Single precision + refinement is slower than double on GH200 above n_E ~ 4k (bench_lu_prec.py: 0.63-0.67x; FP64
#   tensor cores ~ FP32 rate); --rho-eig rand on 2808a: same N / its, slower (k/n ~ 0.5, dense fallback in most blocks).
#   21600a relaunched in double, dense eigh:
#   DEVS=21600a CUT=1e-5 DMKEEP=1e-6 SCHUR=double:0 EXMAX=50000 setsid nohup results_ms/gpu/mlevel/run_dmc_grid.sh > results_ms/gpu/mlevel/run_dmc1e-5_21600a.out 2>&1 &
#   21600a dmc 1e-5 (double Schur): N 58649/38665/14184/10766/7212/5507 (dm 79285 .. 9189); its flat at depth (Ec+0.45
#   deep levels 48-55, dm 59-60 = maxit).
# 2026-11-19 (user): 6 nm with dmc 1e-5 (keep 1e-6, Schur double), merges y,x,y,x,x,x (attelas2):
#   setsid nohup results_ms/gpu/mlevel/run_mlevel_6nm_dmc.sh > results_ms/gpu/mlevel/run_mlevel_6nm_dmc.out 2>&1 &
# 2026-11-24 (user): daint.alps dev env (envs/daint/: setup.sh, activate.sh, test.sbatch; account lp82, normal queue) and
# 2808a check run, settings of run_mergeeig_2808a.sh MODES=full, fresh sigma / dm caches in cache_daint/:
#   sbatch envs/daint/mlevel_2808a.sbatch   (daint, repo root) -> .tools/mlevel_2808a-5445893.out, results_ms/daint/
# 2026-11-24 (user): the larger devices on daint after the 2808a check (reference 9072a / 21600a: grid_*_y2.log; 6 nm dmc):
#   sbatch --time=0:30:00 --job-name=mlevel-9072a  envs/daint/mlevel_dev.sbatch 9072a
#   sbatch --time=1:00:00 --job-name=mlevel-21600a envs/daint/mlevel_dev.sbatch 21600a
#   sbatch --time=4:00:00 --job-name=mlevel-6nm    envs/daint/mlevel_dev.sbatch 6nm
# 2026-11-24 (user): energy sweep Ec + 0.01..0.50 (10 meV) on daint with the stored bases, 2808a (GPU 0) and 9072a (GPUs 1-3):
#   sbatch envs/daint/mlevel_grid.sbatch   (daint, repo root) -> job 5446160, results_ms/daint/grid_{2808a,9072a_1..3}_5446160.{log,json}
# 2026-11-24 (user): 12 nm on daint: prep (debug queue), then the hierarchy at Ec + 0.15 (7 merges, no exact psi; mlevel-12nm, 12 h):
#   P=$(sbatch --parsable envs/daint/prep_12nm.sbatch); sbatch --dependency=afterok:$P envs/daint/mlevel_12nm.sbatch   -> 5448056, 5448057
# 2026-11-24 (user): 12 nm memory test in the debug queue first (2 contour points; GPU 0 free-levels 2, GPU 1 all streamed); 5448057 waits for it:
#   sbatch --dependency=afterok:<prep> envs/daint/memtest_12nm.sbatch   -> 5448063, results_ms/daint/memtest_12nm_5448063*
# 2026-11-24 (user): domain decomposition (mlevel --dist nccl) test on 4 GPUs, debug: sbatch envs/daint/dist_test.sbatch -> 5448143; code copy dist_dev/ on daint
#   12 nm job 5448057 held (scontrol release 5448057); memory test 5448063 out of memory at level 8.
# 2026-11-24 (user): 5448057 cancelled; 4-GPU version (--dist nccl): 6 nm in debug (5448244), 12 nm in normal after it (5448245):
#   A=$(sbatch --parsable -p debug -t 00:30:00 envs/daint/mlevel_wide_dist.sbatch 6nm); sbatch --dependency=afterany:$A envs/daint/mlevel_wide_dist.sbatch 12nm
# 2026-11-24 (user): two-node test of --dist (contour points over ranks, DistSlabLU), code copy dist_dev2/ on daint, debug:
#   sbatch envs/daint/mn_test.sbatch   -> 5448357 (failed: libmpi.so.12 not found, cray library has another name), resubmitted below
#   sbatch envs/daint/mn_test.sbatch   -> 5448425 (cancelled: the wheel took the venv MPICH, 8 one-rank runs), 5448430 (MPI ok on 2 nodes, NCCL fails), 5448488 (NCCL probes), results_ms/daint/mn_*
# 2026-11-24 (user): dmc 1e-5 + Schur double on 2808a / 9072a / 21600a, 4 GPUs (read "2808 - 12k" as this trio), after 5448488:
#   sbatch --dependency=afterany:5448488 envs/daint/dmc_dist.sbatch   -> 5448557, results_ms/daint/dmc1e-5_schur_*
# 2026-11-24 (user): 6 nm, shared contour points + distributed bottom solve on / off, 4 GPUs, after 5448557:
#   sbatch --dependency=afterany:5448557 envs/daint/onoff_6nm.sbatch   -> 5448594, results_ms/daint/onoff_6nm_*
# 2026-11-25 (user): 12 nm 4-GPU job 5448245 ran out of host memory at setup (4 private copies of H, S). Device store: H, S sliced
# by slab on disk, every rank holds only its own rows (qtx_ms.devstore, mlevel --dev-store; code copy dist_dev3/ on daint);
# 2808a check, full matrices vs store, 4 GPUs, debug:
#   sbatch envs/daint/devstore_test.sbatch   -> 5452393 (N, its, T equal), results_ms/daint/devstore_2808a_{ref,store}_5452393.*
# 2026-11-25 (user): 6 nm on 4 GPUs with the device store (fresh window rho and bases, --no-share-points), debug:
#   sbatch -p debug -t 00:30:00 --job-name=mlevel-6nm-store envs/daint/mlevel_wide_dist.sbatch 6nm   -> 5452416 (T 0.9074057, 675 s, host peak ~196 GiB), results_ms/daint/mlevel_6nm_4g_5452416.*
# 2026-11-25 (user): S_k, H_k formed from sparse fine core-pair blocks on the GPU (mlevel --fine-blocks K, code copy dist_dev4/):
# strip tests on 4 GPUs (nccl) + 2808a K = 0, 1, 2, debug:
#   sbatch envs/daint/fine_blocks_test.sbatch   -> 5453053 (PASS, its and T equal), results_ms/daint/fine_blocks_2808a_k{0,1,2}_5453053.*
# 2026-11-25 (user): A_ij = E S_ij - H_ij directly, dense S_k, H_k pairs kept on the GPU during the basis build, smoothers (rows and LU)
# streamed (--stream-smoother-gb 0): strip tests (nccl) + 2808a, 5 variants, code copy dist_dev5/, debug:
#   sbatch envs/daint/fine_blocks_test2.sbatch   -> 5453937, results_ms/daint/fine_blocks2_2808a_*_5453937.*
#   the strip test alone with A_k spectra (entries differ by a block gauge): sbatch envs/daint/strip_test.sbatch   -> 5454029 (PASS)
# 2026-11-25 (user): 6 nm, --fine-blocks 1 (level 1 S_k, H_k never stored), smoothers by the --stream-gb rule, fresh dm cache, code dist_dev5/, debug:
#   FB=1 DMW=$PWD/cache_daint/6nm_store/dmw_fb1 sbatch -p debug -t 00:30:00 --job-name=mlevel-6nm-fb1 envs/daint/mlevel_wide_dist.sbatch 6nm   -> 5454090 (T 0.9074057, 727 s, GPU peak 49.8 GB), results_ms/daint/mlevel_6nm_4g_5454090.*
# 2026-11-25 (user): single-purpose drivers mlbasis / mlsolve / mlmode / mlexact on 2808a, 4 GPUs, exact-T grid Ec + 0.01 .. 0.50 (against
# job 5446160 and results_ms/exact), code copy dist_dev6/, two debug jobs (5454430, normal queue, cancelled before its start):
#   A=$(PART=a sbatch --parsable envs/daint/drivers_2808a.sbatch); PART=b sbatch --dependency=afterany:$A envs/daint/drivers_2808a.sbatch   -> 5454441 (exact, dm), 5454442 (dmc), results_ms/daint/drivers_2808a_<job>*
# 2026-11-25 (user): 6 nm with mlbasis (fresh cache dmw_drv), then mlsolve K = 2, streamed smoothers, 8 energies with exact T, two debug jobs, dist_dev6/:
#   A=$(MODE=basis sbatch --parsable envs/daint/drivers_6nm.sbatch); MODE=solve sbatch --dependency=afterok:$A envs/daint/drivers_6nm.sbatch   -> 5454677, 5454678, results_ms/daint/drivers_6nm_<mode>_<job>*
# 2026-11-25 (user): 6 nm, Ec + 0.40, 0.50 (not converged in 5454678) with maxit 150, then the study driver at 0.50 on the same bases, debug:
#   MODE=rerun sbatch envs/daint/drivers_6nm.sbatch   -> 5454926, results_ms/daint/drivers_6nm_rerun_5454926*
# 2026-11-25 (user: ways to reduce the smoother time): profile of mlsolve at 6 nm (phase times, window microbenchmark, smoothers kept over energies,
# A_k windows with halo 1), code copy dist_dev7/ (coupling norms off in mlsolve), debug:
#   sbatch envs/daint/prof_smoother.sbatch   -> 5455040, results_ms/daint/prof_smoother_5455040_*

# 2026-11-25 fresh minimal hierarchy qtx_ms.hier (daint, 4 GPUs; code copies dist_dev8 / 9 / 10 = commit 91fcd56 + qtx_ms/hier)
#   MODE=2808a MU=-0.15,0.8 sbatch envs/daint/hier.sbatch       # done (99 s of solver, job 5455491, contour to 0.8): N 7837/4707/2119, its max 22, T err 1.3e-6
#   MODE=2808a sbatch envs/daint/hier.sbatch                    # done (job 5455514, contour to 0.6 as the old driver): N 7335/4283/1776 = mlsolve, 1.0 s per energy, T err 1.3e-6
#   MODE=6nm sbatch envs/daint/hier.sbatch                      # done (578 s, job 5455515): T = reference to 10 digits, its 20-115, GPU peak 66.1 GiB
#   sbatch envs/daint/hier_slices.sbatch                        # done (8 min, job 5455615): rho from several weighted contours, 2808a; weights 1,3,0.3 best
#   sbatch envs/daint/hier_schwarz.sbatch                       # done (41 min, job 5455652, lp82): FGMRES with the CAP-Schwarz smoother alone
#   sbatch -A c33 envs/daint/hier_control.sbatch                # done (8 min, job 5455684): equal-N control of the weighted contours, 2808a (account c33, user)
#   sbatch envs/daint/hier_l3.sbatch                            # done (7:42, job 5456734, c33, debug): 6 nm, cuts of 5455515, stop at level 3, exact block-tridiagonal LU there (hier.run --bottom bt; dist_dev11)
#   LV=2 sbatch envs/daint/hier_l3.sbatch                       # done (about 6 min, job 5456840, c33, debug): the same stopped at level 2 (N 96281), exact block-tridiagonal LU there
#   CUT3=1e-6 DE=0.40,0.50 sbatch envs/daint/hier_l3.sbatch     # done (5:34, job 5456861, c33, debug): exact level 3 with its cut at 1e-6 (is the level 3 cut the cause of the its at high E?)
# 2026-11-25 (user): slow modes of the fine smoother as coarse level, dense exact solve (hier.run --slow M,Q,ROUNDS, qtx_ms/hier/slow.py; dist_dev12, c33, debug)
#   A=$(MODE=2808a sbatch --parsable envs/daint/hier_slow.sbatch); MODE=6nm sbatch --dependency=afterany:$A envs/daint/hier_slow.sbatch   # done (2:14 and 10:12, jobs 5456903, 5456904)
# 2026-11-25 (user): 12 nm (714 240 orbitals), hierarchy with exact level 3 (--bottom bt, now the default; dist_dev13), 2 nodes, c33, debug: device store first
#   A=$(sbatch --parsable envs/daint/store_12nm.sbatch); sbatch --dependency=afterok:$A envs/daint/hier_12nm.sbatch   # store done (1:36, 5456968); run failed (13:18, 5456969): GPU out of memory in the level 3 LU, bases and Sigma cached
#   CODE=$PWD/dist_dev14 sbatch envs/daint/hier_12nm.sbatch   # done (job 5457039): projected CAP blocks on the host, memory logged; +0.50 not converged in 100 its
#   OBC=beyn-real MAXIT=200 CODE=$PWD/dist_dev15 sbatch envs/daint/hier_12nm.sbatch   # done (job 5457068, 15:03): Beyn Sigma at the real energies, cached bases; all 3 energies converge (maxit 200)
# closed device (--obc closed: no Sigma, RHS = 80 of the 4128 / 8256 unit vectors of the left contact face), submitted 21:04:
#   OBC=closed CODE=$PWD/dist_dev16 sbatch envs/daint/hier_l3.sbatch     # 6 nm, done (job 5457149, 8:19, debug)
#   OBC=closed MAXIT=200 CODE=$PWD/dist_dev16 sbatch -A lp82 envs/daint/hier_12nm.sbatch   # 12 nm, done (job 5457150, 13:52; +0.50 not converged in 200 its: max residual 1.1e-9) (submitted normal / c33, moved pending to debug / lp82, 30 min, by the user at 21:10)
# Beyn incoming modes as injection (no Gamma_L eigenproblem), timings of the Beyn steps:
#   OBC=beyn-real MAXIT=200 CODE=$PWD/dist_dev17 sbatch envs/daint/hier_12nm.sbatch   # done (job 5457257, 14:58): T = reference to 1e-10
# profile of the Beyn lead setup (leads 131 s, coupling bases 153 + 149 s of 464 s):
#   CODE=$PWD/dist_dev17 sbatch envs/daint/prof_beyn.sbatch   # done (job 5457498, 6:12) (5457489 failed at import: wrong code path)
#   CODE=$PWD/dist_dev17 sbatch envs/daint/prof_beyn_cpl.sbatch   # done (job 5457524): 'leads 131 s' = page-wise mmap reads of the 8 dense coupling files (~40 MB/s)
# coupling-basis products on the GPU (qtx_ms/obc_beyn.py), time and Sigma / B against the cached files of 5457257:
#   CODE=$PWD/dist_dev18 sbatch envs/daint/prof_beyn_new.sbatch   # done (job 5457529, 5:02): basis 153 + 149 s -> 22.8 + 18.5 s
# sparse lead-device couplings (both stores converted in place, dense files kept) and nonzero lead blocks, checked on the 12 nm lead:
#   CODE=$PWD/dist_dev19 sbatch envs/daint/prof_beyn_new.sbatch   # done (job 5457611, 2:46)
#   CODE=$PWD/dist_dev19 sbatch envs/daint/prof_folded_cpl.sbatch   # done (job 5457623): folded Sigma, sparse against dense couplings: 1e-15
# 12 nm T with the sparse couplings and nonzero lead blocks (fresh Sigma folder: links to the folded contour files only):
#   SIG=$PWD/cache_daint/12nm/hier_sigma_new OBC=beyn-real MAXIT=200 CODE=$PWD/dist_dev19 sbatch envs/daint/hier_12nm.sbatch   # done (job 5457667, 8:22): T = reference to 1e-10
# phases of the Beyn Sigma (12 nm, left) and Sigma error after 1, 2, ... fixed-point steps:
#   CODE=$PWD/dist_dev20 sbatch envs/daint/prof_fp.sbatch   # done (job 5457714, 12:36): stalls after 4-6 steps, 8-16 used; chunks = fixed point in time
# 12 nm T with the earlier stop of the fixed point (dist_dev21, fresh Sigma folder):
#   SIG=$PWD/cache_daint/12nm/hier_sigma_stop OBC=beyn-real MAXIT=200 CODE=$PWD/dist_dev21 sbatch envs/daint/hier_12nm.sbatch   # done (job 5457786, 8:03): T moved 5e-10 vs old stop, Sigma 53.6 s (55.2)
# 12 nm with Beyn on the contour too (bases rebuilt; dist_dev22: contour Sigma only when a density matrix is built):
#   SIG=$PWD/cache_daint/12nm/hier_sigma_stop OBC=beyn MAXIT=200 CODE=$PWD/dist_dev22 sbatch envs/daint/hier_12nm.sbatch   # done (job 5457798, 16:05): T = 5457786 to 6e-13, its -1..-3, contour 99 s
# 12 nm full sweep Ec+0.00..0.50 step 0.01 (51 energies), Beyn everywhere, per-side Beyn tasks, cached bases (dist_dev23); table: envs/daint/sweep_table.py:
#   SIG=$PWD/cache_daint/12nm/hier_sigma_stop OBC=beyn MAXIT=300 DE=0.00,0.01,...,0.50 CODE=$PWD/dist_dev23 sbatch --partition=normal --time=02:00:00 --job-name=hier-12nm-sweep envs/daint/hier_12nm.sbatch   # job 5457909, cancelled before it started (user, 2026-11-26)
# 12 nm on 4 nodes (16 GPUs), debug, same 3 energies, fresh Sigma folder (per-side Beyn tasks on 16 ranks), cached bases:
#   SIG=$PWD/cache_daint/12nm/hier_sigma_4n OBC=beyn MAXIT=200 CODE=$PWD/dist_dev23 sbatch --nodes=4 --job-name=hier-12nm-4n envs/daint/hier_12nm.sbatch   # done (job 5457910, 5:41): solve 1.5x faster at +0.50, bottom LU slower
# 12 nm on 6 nodes (24 GPUs), debug, same 3 energies, fresh Sigma folder (coupling bases loaded):
#   SIG=$PWD/cache_daint/12nm/hier_sigma_6n OBC=beyn MAXIT=200 CODE=$PWD/dist_dev23 sbatch --nodes=6 --job-name=hier-12nm-6n envs/daint/hier_12nm.sbatch   # done (job 5457977, 4:34): +0.50 105.7 s (4 nodes 131.1, 2 nodes 195.8)
# 12 nm 6 nodes, timing breakdown of the solve (--profile, GPU synchronised at every timer; dist_dev24):
#   SIG=$PWD/cache_daint/12nm/hier_sigma_6n OBC=beyn MAXIT=200 EXTRA=--profile CODE=$PWD/dist_dev24 sbatch --nodes=6 --job-name=hier-12nm-prof envs/daint/hier_12nm.sbatch   # done (job 5458061, 5:08): least squares 23 s, bottom solve 21 s, fine products 23 s of 111 s at +0.50
# 12 nm 6 nodes, lower cuts of levels 2 and 3 (user: fewer its at +0.50); bases of all levels rebuilt (new cache folder):
#   CUT2=1e-6 CUT3=1e-6 SIG=$PWD/cache_daint/12nm/hier_sigma_stop OBC=beyn MAXIT=200 CODE=$PWD/dist_dev24 sbatch --nodes=6 --job-name=hier-12nm-c1e-6 envs/daint/hier_12nm.sbatch   # done (job 5458088, 8:17): +0.50 its 100..103, 71.0 s
#   CUT2=1e-7 CUT3=1e-7 (same)   # done (job 5458089, 7:46): +0.50 its 49..51, 43.3 s
# 12 nm full sweep Ec+0.00..0.50 step 0.01 (51 energies), cuts 1e-7 at levels 2 / 3, 2 nodes, normal 2 h (cancelled pending, user: debug, fewer points), cached bases of 5458089 (user, 2026-11-26):
#   CUT2=1e-7 CUT3=1e-7 SIG=$PWD/cache_daint/12nm/hier_sigma_stop OBC=beyn MAXIT=300 DE=0.00,0.01,...,0.50 CODE=$PWD/dist_dev24 sbatch --partition=normal --time=02:00:00 --job-name=hier-12nm-sweep-c1e-7 envs/daint/hier_12nm.sbatch   # job 5461019, cancelled
#   same on debug (30 min), 11 energies Ec+0.00..0.50 step 0.05 (user):
#   CUT2=1e-7 CUT3=1e-7 SIG=$PWD/cache_daint/12nm/hier_sigma_stop OBC=beyn MAXIT=300 DE=0.00,0.05,...,0.50 CODE=$PWD/dist_dev24 sbatch --job-name=hier-12nm-sweep-c1e-7 envs/daint/hier_12nm.sbatch   # failed (job 5461022, 3:27): GPU out of memory in the bottom block-tridiagonal LU at Ec+0.05 (2 nodes too small for cuts 1e-7)
# distributed bottom LU (hier --bottom bt now = domains of slabs + separator system on one rank; old replicated LU: --bottom btrep), dist_dev25:
#   CUT2=1e-7 CUT3=1e-7 SIG=$PWD/cache_daint/12nm/hier_sigma_stop OBC=beyn MAXIT=300 DE=0.00,0.05,...,0.50 CODE=$PWD/dist_dev25 sbatch --job-name=hier-12nm-sweep-c1e-7 envs/daint/hier_12nm.sbatch   # done (job 5461473, 14:18, 2 nodes): all 11 converge, T = 6-node runs to 1e-10, +0.50 its 49..52, 84.6 s; factors 11.5 GB on the largest rank (8 domains, 7 separators), node peak 77 GB
# OFF state 12 nm (user, 2026-11-26): barrier 0.2 eV above the source on the middle third, drain -0.4 eV (qtx_ms.prep.offstate, dist_dev26):
#   A=$(sbatch --parsable envs/daint/prep_off_12nm.sbatch); A=$(DEV=12nm_off CODE=$PWD/dist_dev26 sbatch --parsable --dependency=afterok:$A envs/daint/store_12nm.sbatch)   # prep done (job 5461633, 11:32; right_lead_shift -0.4, V [-0.4, 0.2]), store 5461634
#   DEV=12nm_off CUT2=1e-7 CUT3=1e-7 OBC=beyn MAXIT=300 DE=0.00,0.05,...,0.50 CODE=$PWD/dist_dev26 sbatch --nodes=6 --dependency=afterok:<store> --job-name=hier-12nm-off envs/daint/hier_12nm.sbatch   # done (job 5461684, 11 min: setup 383 s (bases 273 s), its 28..43, T(+0.25) 0.844, T(+0.50) 8.263)
# Gate sweep on the Vb 0.2 bases (user, 2026-11-26: reusability of the bases for varying gate voltages): dV = V(VB) - V(0.2) as
# (dV S + S dV) / 2 on the fine H (hier --dV), Sigma and bases of 12nm_off reused; reference with new bases of H + dV (MODE=build)
#   VBS="0.1 0.3 0.0" CODE=$PWD/dist_dev27 sbatch envs/daint/hier_gate_12nm.sbatch              # done (job 5462593, 13:54: Vb 0.1 / 0.3 / 0.0 in 291 / 230 / 243 s, setup 10-61 s, its 25..53)
#   MODE=build VBS="0.1 0.3" CODE=$PWD/dist_dev27 sbatch envs/daint/hier_gate_12nm.sbatch       # done (job 5462594, 18:31: 555 / 494 s, bases 264 / 250 s; T = reuse to 1e-10)
# Source-drain bias on the same bases (user, 2026-11-26): dV = V(0.2, VDS) - V(0.2, 0.4) = -(VDS - 0.4) on the drain and the right lead face
# -> Sigma_R of the 0.4 lead at E + VDS - 0.4 (OBC shift_R, Beyn), Sigma_L / injection / bases reused
#   VBS=0.2 VDSS="0.1 0.2 0.3 0.5" CODE=$PWD/dist_dev28 sbatch --job-name=hier-bias envs/daint/hier_gate_12nm.sbatch   # submitted (job 5463970, 6 nodes, debug)
