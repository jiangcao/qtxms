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
