#!/bin/bash
# CPU / qttools part of the 2026-11-17 rebuild check (see workflows/verify_rebuild_gpu.sh), mont-fort15 (x86, .venv,
# qttools + pypardiso): the runs that were made on the CPU with the qttools lead OBC.
#   * qtx_ms stack, qttools OBC (sigma_qtx*, qtbm_cmp, sigma_qtt, psi_qtt caches): Sigma + psi of inputs/mos2_hfo2 vs
#     results_ms/exact/mos2_hfo2_qtt.json, of inputs/mos2_hfo2_9072a vs results_ms/cap_fgmres/exact_qtt_mos2_hfo2_9072a.json
#     (qttools gives the 2 physical channels; the folded lead at E + 1e-6 i gives 19-24 incl. eta channels and a T that
#     differs by ~1e-3: compare each OBC only with a reference made with the same OBC)
#   * release 0.2.0 runs (results_ms/release020, wf/ and the 51-point ones on 3 of their energies) from config.toml
# 4 threads, nice (shared host). Output: results_ms/verify_rebuild/cpu (logs, json, rebuilt files; ~2 GB).
#   nohup workflows/verify_rebuild_cpu.sh > results_ms/verify_rebuild/cpu.out 2>&1 &     (repo root, mont-fort15)
cd "$(dirname "$0")/.."
L=results_ms/verify_rebuild/cpu
S=$L/data
mkdir -p $S
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 THREADS=4
export QTTOOLS_SRC=${QTTOOLS_SRC:-/home/jiacao/quatrex_project/quatrex/src}   # qttools is not installed in .venv
PY=.venv/bin/python
step() { echo "[$(date +%T)] $*"; }
res() { echo "$1: $(tail -1 $2)" >> $L/summary.txt; }
: > $L/summary.txt
REX=results_ms/exact/mos2_hfo2_qtt.json
R9Q=results_ms/cap_fgmres/exact_qtt_mos2_hfo2_9072a.json
step "sigma qtx_ms 2808a (qttools)"
nice $PY -u workflows/sigma/rebuild_sigma.py --stack qtx_ms --obc qttools --eta 0 --inputs inputs/mos2_hfo2 \
  --cache $S/qtx_sigma_2808a --dE 0.05,0.15,0.2 --ref $REX --out $L/sigma_qtx_2808a.json > $L/sigma_qtx_2808a.log 2>&1
res sigma_qtx_2808a $L/sigma_qtx_2808a.log
step "psi qtx_ms 2808a (qttools)"
nice $PY -u workflows/psi/rebuild_psi.py --stack qtx_ms --obc qttools --eta 0 --inputs inputs/mos2_hfo2 \
  --sigma-cache $S/qtx_sigma_2808a --psi-dir $S/qtx_psi_2808a --dE 0.05,0.15,0.2 --ref $REX \
  --out $L/psi_qtx_2808a.json > $L/psi_qtx_2808a.log 2>&1
res psi_qtx_2808a $L/psi_qtx_2808a.log
step "sigma qtx_ms 9072a (qttools)"
nice $PY -u workflows/sigma/rebuild_sigma.py --stack qtx_ms --obc qttools --eta 0 --inputs inputs/mos2_hfo2_9072a \
  --cache $S/qtx_sigma_9072a --dE 0.05,0.2 --ref $R9Q \
  --out $L/sigma_qtx_9072a.json > $L/sigma_qtx_9072a.log 2>&1
res sigma_qtx_9072a $L/sigma_qtx_9072a.log
step "psi qtx_ms 9072a (qttools, one energy: ~25 min CPU)"
nice $PY -u workflows/psi/rebuild_psi.py --stack qtx_ms --obc qttools --eta 0 --inputs inputs/mos2_hfo2_9072a \
  --sigma-cache $S/qtx_sigma_9072a --psi-dir $S/qtx_psi_9072a --dE 0.05 --ref $R9Q \
  --out $L/psi_qtx_9072a.json > $L/psi_qtx_9072a.log 2>&1
res psi_qtx_9072a $L/psi_qtx_9072a.log
# release 0.2.0 runs: wf/ (3 energies) first, then the 51-point A / B / C on 3 of their energies (same caches)
for r in release020/wf/A_slab_obc release020/wf/B_slab_cap release020/wf/C_tile_obc \
         release020/A_slab_obc release020/B_slab_cap release020/C_tile_obc; do
  n=$(echo $r | tr / _)
  step "bases $r"
  CACHE=$S/rel_2808a workflows/bases/rebuild_run.sh results_ms/$r $S/runs/$n --dE 0.05,0.15,0.2 --exact \
    > $L/bases_$n.log 2>&1
  cp $S/runs/$n/rebuild.log $L/bases_${n}_run.log 2>/dev/null
  res bases_$n $L/bases_$n.log
done
step done; cat $L/summary.txt
