#!/bin/bash
# 2808a, qtx_ms.hier: rho as a weighted sum of several contours (user, 2026-11-25: several smaller contours instead of one big one).
# Does it lower the iterations at the top energies (Ec + 0.30 .. 0.50) and what does it cost in N? Fresh bases per variant, cached
# real-axis Sigma (job 5455514). References: one contour with 24 points to 0.6 (job 5455514) and to 0.8 (job 5455491).
#   sbatch envs/daint/hier_slices.sbatch       (repo root; code copy dist_dev9)
#SBATCH --job-name=hier-slices
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
R=$PWD; O=$R/results_ms/daint; J=$SLURM_JOB_ID; CODE=${CODE:-$R/dist_dev9}; C=$R/cache_daint
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
P=$O/hier_slices_$J; cd $CODE
X="--inputs $R/inputs/mos2_hfo2 --dev-store $C/devstore_2808a/store --sigma-cache $C/hier_sigma_2808a_5455514 --cap-split 2,1 --cap 16,2,2 \
   --up y:1e-5,x:1e-5 --maxit 60 --gpu --dist nccl --dE ${DE:-$(LC_ALL=C seq -f %.2f 0.01 0.01 0.50 | paste -sd,)}"
run() {  # name, contours
  local t0=$SECONDS
  QTX_RANK_LOG=${P}_$1.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.hier.run $X "--mu=$2" --out ${P}_$1.json > ${P}_$1.log 2>&1
  echo "=== $1 ($2) exit $? after $(( SECONDS - t0 )) s"; grep -E "^contours|^setup|Error|NOT CONV" ${P}_$1.log | head -3 | cut -c1-200
}
run n48       -0.15,0.8,48
run to12      -0.15,1.2,48
run s111      -0.15,0.4,16:0.4,0.8,16:0.8,1.2,16
run s1101     -0.15,0.4,16:0.4,0.8,16:0.8,1.2,16,0.1
run s1303     -0.15,0.4,16:0.4,0.8,16,3:0.8,1.2,16,0.3
echo "##### comparison"
$R/.venv-daint/bin/python - $P $O $R <<'PY'
import json, sys, os
import numpy as np
P, O, R = sys.argv[1:]
ex = {k: v["T_exact"] for k, v in json.load(open(f"{R}/results_ms/exact/mos2_hfo2.json"))["E"].items()}
f = dict(to06_24=f"{O}/hier_2808a_5455514.json", to08_24=f"{O}/hier_2808a_5455491.json")
f.update({n: f"{P}_{n}.json" for n in ("n48", "to12", "s111", "s1101", "s1303")})
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
