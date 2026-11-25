#!/bin/bash
# 2808a, qtx_ms.hier: control of the weighted contours (job 5455615): is a weight on a slice better than a lower cut everywhere,
# at about equal N?  (weights 1,3,0.3: N 8748/5403/2662, mean its 14.28)
#   sbatch envs/daint/hier_control.sbatch       (repo root; code copy dist_dev9)
#SBATCH --job-name=hier-control
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
R=$PWD; O=$R/results_ms/daint; J=$SLURM_JOB_ID; CODE=${CODE:-$R/dist_dev9}; C=$R/cache_daint
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
P=$O/hier_control_$J; cd $CODE
X="--inputs $R/inputs/mos2_hfo2 --dev-store $C/devstore_2808a/store --sigma-cache $C/hier_sigma_2808a_5455514 --cap-split 2,1 --cap 16,2,2 \
   --maxit 60 --gpu --dist nccl --dE ${DE:-$(LC_ALL=C seq -f %.2f 0.01 0.01 0.50 | paste -sd,)}"
run() {  # name, cut, contours
  local t0=$SECONDS
  QTX_RANK_LOG=${P}_$1.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.hier.run $X --cut $2 --up y:$2,x:$2 "--mu=$3" --out ${P}_$1.json > ${P}_$1.log 2>&1
  echo "=== $1 (cut $2, $3) exit $? after $(( SECONDS - t0 )) s"; grep -E "Error|NOT CONV" ${P}_$1.log | head -3 | cut -c1-200
}
run c08_5e6   5e-6 -0.15,0.8,48
run c08_3e6   3e-6 -0.15,0.8,48
run c12_7e6   7e-6 -0.15,1.2,48
run w2_13     1e-5 -0.15,0.4,24:0.4,0.8,24,3
run w3_131    1e-5 -0.15,0.4,16:0.4,0.8,16,3:0.8,1.2,16
echo "##### comparison"
$R/.venv-daint/bin/python - $P $O $R <<'PY'
import json, sys, os
import numpy as np
P, O, R = sys.argv[1:]
ex = {k: v["T_exact"] for k, v in json.load(open(f"{R}/results_ms/exact/mos2_hfo2.json"))["E"].items()}
S = f"{O}/hier_slices_5455615_"
f = {n: f"{S}{n}.json" for n in ("n48", "to12", "s1303")}
f.update({n: f"{P}_{n}.json" for n in ("c08_5e6", "c08_3e6", "c12_7e6", "w2_13", "w3_131")})
ks = ["0.1000", "0.2000", "0.3000", "0.4000", "0.4500", "0.5000"]
print(f"{'run':8s} {'N':>20s} {'conv':>4s} {'dT max':>8s} {'its mean':>8s} {'mean>0.3':>8s} " + " ".join(f"{k:>6s}" for k in ks) + "  s/E")
for n, p in f.items():
    if not os.path.exists(p):
        print(n, "missing"); continue
    r = json.load(open(p)); E = r["E"]
    m = {k: max(v["its"]) for k, v in E.items()}
    print(f"{n:8s} {'/'.join(map(str, r['N'])):>20s} {sum(v['converged'] for v in E.values()):4d} {max(abs(E[k]['T'] - ex[k]) / abs(ex[k]) for k in E if k in ex):8.1e} "
          f"{np.mean(list(m.values())):8.2f} {np.mean([v for k, v in m.items() if float(k) > 0.3]):8.2f} " + " ".join(f"{m[k]:6d}" for k in ks)
          + f" {np.mean([v['t_energy'] for v in E.values()]):5.2f}")
PY
