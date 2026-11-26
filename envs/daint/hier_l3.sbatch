#!/bin/bash
# qtx_ms.hier, 6 nm, 4 GPUs: the hierarchy stopped at level 3 (merges y:1e-5,x:1e-5, the cuts of job 5455515), level 3 solved exactly by a
# block-tridiagonal LU over its x slabs (--bottom bt; user, 2026-11-25); 8 energies against T and its of job 5455515 (7 levels).
#   sbatch envs/daint/hier_l3.sbatch        (repo root; code copy dist_dev11)
#  LV=2: stopped at level 2 (merge y:1e-5 only; the x positions are joined in pairs)
#   LV=2 sbatch envs/daint/hier_l3.sbatch
#  CUT3: cut of level 3 (LV=3), DE: energies; is the level 3 cut or the level 2 smoother the cause of the its growth?
#   CUT3=1e-6 DE=0.40,0.50 sbatch envs/daint/hier_l3.sbatch
# OBC=closed: no lead, RHS evenly spaced columns of the identity on the left contact face (4128 in all; RHS=0: all); bases of the closed device
#   OBC=closed CODE=$PWD/dist_dev16 sbatch envs/daint/hier_l3.sbatch
#SBATCH --job-name=hier-l3
#SBATCH --account=c33
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
R=$PWD; O=$R/results_ms/daint; J=$SLURM_JOB_ID; CODE=${CODE:-$R/dist_dev11}; C=$R/cache_daint
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
LV=${LV:-3}; CUT3=${CUT3:-1e-5}; UP=y:1e-5; N=l$LV
[ $LV = 3 ] && UP=y:1e-5,x:$CUT3
[ $LV = 3 ] && [ $CUT3 != 1e-5 ] && N=l3c$CUT3
OBC=${OBC:-folded}; [ $OBC != folded ] && N=${N}_$OBC
L=$O/hier_${N}_6nm_$J; cd $CODE
echo "$(hostname) job $J, level $LV, merges $UP, obc $OBC, code $CODE"
( while sleep 5; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' '; done ) > $L.mon 2>&1 & MON=$!
QTX_RANK_LOG=$L.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.hier.run \
  --inputs $R/inputs/mos2_hfo2_6nm --dev-store $C/6nm_store/store --sigma-cache $C/6nm_dist/sigma --cache $C/6nm_store/hier \
  --cap 16,2,2 --cap-split 4,1 --cut 1e-5 --up $UP --bottom bt --obc $OBC --rhs-cols ${RHS:-80} --chunk 20 --maxit ${MAXIT:-150} --fgmres-log 25 --gpu --dist nccl \
  --dE ${DE:-0.15,0.05,0.30,0.20,0.10,0.25,0.40,0.50} --out $L.json > $L.log 2>&1
rc=$?
kill $MON
echo "exit $rc after $SECONDS s; GPU MiB peak $(tr ' ' '\n' < $L.mon | sort -n | tail -1)"
grep -E "^ *level|^setup|^obc|memory of|^Ec|Error|done" $L.log | cut -c1-380
[ $OBC = closed ] && exit $rc                               # no T to compare
$R/.venv-daint/bin/python - $L.json $O/hier_6nm_5455515.json $O/hier_l3_6nm_5456734.json $O/hier_l2_6nm_5456840.json <<'PY'
import json, os, sys
a, b, c, d = (json.load(open(f)) if os.path.exists(f) else None for f in sys.argv[1:])
for n, r in (("this", a), ("7 levels", b), ("level 3", c), ("level 2", d)):
    print(n, "N", r and r["N"])
for k in sorted(a["E"]):
    v = a["E"][k]
    print(k, f"T {v['T']:.10f}, its {min(v['its'])}..{max(v['its'])}, {v['t_energy']:.1f} s, solve {v['t']['solve']:.1f}, bottom LU {v['t']['bottom LU']:.1f}, conv {v['converged']}; "
          + "; ".join(f"{n}: rel dT {abs(v['T'] / r['E'][k]['T'] - 1):.1e}, its {min(r['E'][k]['its'])}..{max(r['E'][k]['its'])}, {r['E'][k]['t_energy']:.1f} s"
                      for n, r in (("7 levels", b), ("level 3", c), ("level 2", d)) if r))
PY
exit $rc
