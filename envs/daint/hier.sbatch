#!/bin/bash
# qtx_ms.hier (fresh minimal hierarchy; user, 2026-11-25) on 4 GPUs.
#  MODE=2808a  fresh lead Sigma (OBC on the GPUs, parallel over the points) and fresh bases, 50 energies, against results_ms/exact/mos2_hfo2.json
#  MODE=6nm    cached Sigma, fresh bases, 8 energies, against the T and its of mlsolve (jobs 5454678, 5454926)
#   MODE=2808a sbatch envs/daint/hier.sbatch        (repo root; code copy dist_dev8 = commit 91fcd56 + qtx_ms/hier)
#SBATCH --job-name=hier
#SBATCH --account=lp82
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
R=$PWD; O=$R/results_ms/daint; J=$SLURM_JOB_ID; CODE=${CODE:-$R/dist_dev8}; C=$R/cache_daint
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
L=$O/hier_${MODE}_$J; mkdir -p $O; cd $CODE
echo "$(hostname) job $J, $MODE, code $CODE"
if [ $MODE = 2808a ]; then
  X="--inputs $R/inputs/mos2_hfo2 --dev-store $C/devstore_2808a/store --sigma-cache $C/hier_sigma_2808a_$J --cache $C/hier_$J --cap-split 2,1 \
     --up y:1e-5,x:1e-5 --mu=${MU:--0.15,0.6} --maxit 60 --dE ${DE:-$(LC_ALL=C seq -f %.2f 0.01 0.01 0.50 | paste -sd,)}"
else
  X="--inputs $R/inputs/mos2_hfo2_6nm --dev-store $C/6nm_store/store --sigma-cache $C/6nm_dist/sigma --cache $C/6nm_store/hier --cap-split 4,1 \
     --up y:1e-5,x:1e-5,y:1e-5,x:1e-5,x:1e-5,x:1e-5 --chunk 20 --maxit 150 --dE ${DE:-0.15,0.05,0.30,0.20,0.10,0.25,0.40,0.50}"
fi
( while sleep 5; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' '; done ) > $L.mon 2>&1 & MON=$!
QTX_RANK_LOG=$L.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.hier.run $X \
  --cap 16,2,2 --gpu --dist nccl --out $L.json ${EXTRA} > $L.log 2>&1
rc=$?
kill $MON
echo "exit $rc after $SECONDS s; GPU MiB peak $(tr ' ' '\n' < $L.mon | sort -n | tail -1)"
grep -vE "^ *$" $L.log | cut -c1-260 | tail -80
$R/.venv-daint/bin/python - $L.json $R $MODE <<'PY'
import json, sys
import numpy as np
r, R, mode = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
if mode == "2808a":
    ex = {k: v["T_exact"] for k, v in json.load(open(f"{R}/results_ms/exact/mos2_hfo2.json"))["E"].items()}
else:
    ex = {"0.0500": 0.0994021662, "0.1000": 0.0442350528, "0.1500": 0.9074056942, "0.2000": 2.5087172780, "0.2500": 3.0366838765,
          "0.3000": 4.4012290977, "0.4000": 8.3426670089, "0.5000": 7.7957237868}
v = [(k, abs(w["T"] - ex[k]) / abs(ex[k]), min(w["its"]), max(w["its"]), w["t_energy"], w["t"]) for k, w in r["E"].items() if k in ex]
print(f"{len(v)} energies, N {r['N']}, converged {sum(w['converged'] for w in r['E'].values())}, max rel dT {max(x[1] for x in v):.1e}, its max {max(x[3] for x in v)}, "
      f"mean time per energy {np.mean([x[4] for x in v]):.1f} s")
for k in sorted(v[0][5]):
    print(f"   {k}: mean {np.mean([x[5].get(k, 0) for x in v]):.2f} s")
if mode == "6nm":
    for x in v:
        print(x[0], f"rel dT {x[1]:.1e}, its {x[2]}..{x[3]}, {x[4]:.1f} s")
PY
exit $rc
