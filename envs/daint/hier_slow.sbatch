#!/bin/bash
# qtx_ms.hier --slow: two-level, coarse level = slow modes of the fine smoother S_0, dense exact solve, built at every energy (user, 2026-11-25).
#  --slow M,Q,ROUNDS: ROUNDS 1 = slow modes of S_0 alone; more rounds: slow modes of the cycle of the rounds before (same total N).
#  MODE=2808a  N 192 as 192x1, 64x3; N 384 as 128x3; Ec + 0.05 .. 0.50, against Schwarz only (5455652) and the hierarchy (5455514)
#  MODE=6nm    N 512 as 512x1, 128x4; N 1024 as 256x4; Ec + 0.15, 0.30, 0.50 against the hierarchy with exact level 2 (5456840)
#   MODE=2808a sbatch envs/daint/hier_slow.sbatch       (repo root; code copy dist_dev12)
#SBATCH --job-name=hier-slow
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
MODE=${MODE:?2808a | 6nm}
R=$PWD; O=$R/results_ms/daint; J=$SLURM_JOB_ID; CODE=${CODE:-$R/dist_dev12}; C=$R/cache_daint
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
P=$O/hier_slow_${MODE}_$J; cd $CODE
echo "$(hostname) job $J, $MODE, code $CODE"
COM="--gpu --dist nccl --tol 1e-10 --cap 16,2,2"
if [ $MODE = 2808a ]; then
  X="--inputs $R/inputs/mos2_hfo2 --dev-store $C/devstore_2808a/store --sigma-cache $C/hier_sigma_2808a_5455514 --cap-split 2,1 --maxit 100 --dE 0.05,0.10,0.20,0.30,0.40,0.50"
  RUNS="192,3,1 64,3,3 128,3,3"
else
  X="--inputs $R/inputs/mos2_hfo2_6nm --dev-store $C/6nm_store/store --sigma-cache $C/6nm_dist/sigma --cap-split 4,1 --maxit 150 --chunk 20 --dE 0.15,0.30,0.50"
  RUNS="512,3,1 128,3,4 256,3,4"
fi
( while sleep 5; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' '; done ) > $P.mon 2>/dev/null & MON=$!
for s in ${SLOWS:-$RUNS}; do
  n=${s//,/x}; t0=$SECONDS
  QTX_RANK_LOG=${P}_$n.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.hier.run $X $COM --slow $s --out ${P}_$n.json > ${P}_$n.log 2>&1
  echo "=== slow $s exit $? after $(( SECONDS - t0 )) s"; grep -E "^Ec|slow modes round|Error" ${P}_$n.log | cut -c1-260
done
kill $MON
echo "GPU MiB peak $(tr ' ' '\n' < $P.mon | sort -n | tail -1)"
$R/.venv-daint/bin/python - $MODE $O ${P}_*.json <<'PY'
import json, os, sys
mode, O = sys.argv[1:3]
f = dict(S=f"{O}/hier_schwarz_5455652_2808a_w16_y2.json", H=f"{O}/hier_2808a_5455514.json") if mode == "2808a" else \
    dict(L3=f"{O}/hier_l3_6nm_5456734.json", L2=f"{O}/hier_l2_6nm_5456840.json")
ref = {n: json.load(open(p)) for n, p in f.items() if os.path.exists(p)}
T0 = list(ref.values())[-1]
for p in sys.argv[3:]:
    r = json.load(open(p))
    for k in sorted(r["E"]):
        v = r["E"][k]
        print(os.path.basename(p)[:-5], k, f"N {v['slow']['N']}, its {min(v['its'])}..{max(v['its'])}, conv {v['converged']}, rel dT {abs(v['T'] / T0['E'][k]['T'] - 1):.1e}, "
              f"build {v['slow']['t_build']:.1f}, solve {v['t']['solve']:.1f}, energy {v['t_energy']:.1f} s; "
              + ", ".join(f"{n} its {max(q['E'][k]['its'])}, {q['E'][k]['t_energy']:.1f} s" for n, q in ref.items() if k in q["E"]))
PY
