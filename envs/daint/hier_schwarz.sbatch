#!/bin/bash
# FGMRES with the CAP-Schwarz smoother S0 alone, no coarse level (qtx_ms.hier.run --schwarz; user, 2026-11-25).
#  2808a: windows 16 A with y strips (the S0 of the hierarchy runs), 16 / 24 / 32 A over the full width; Ec + 0.05 .. 0.50
#  6 nm: 16 A windows, y strips 4 and full width; Ec + 0.05, 0.15, 0.30, 0.50
#   sbatch envs/daint/hier_schwarz.sbatch       (repo root; code copy dist_dev10)
#SBATCH --job-name=hier-schwarz
#SBATCH --account=lp82
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
R=$PWD; O=$R/results_ms/daint; J=$SLURM_JOB_ID; CODE=${CODE:-$R/dist_dev10}; C=$R/cache_daint
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
P=$O/hier_schwarz_$J; cd $CODE
COM="--schwarz --gpu --dist nccl --tol 1e-10"
A="--inputs $R/inputs/mos2_hfo2 --dev-store $C/devstore_2808a/store --sigma-cache $C/hier_sigma_2808a_5455514 --maxit 400 --dE 0.05,0.10,0.20,0.30,0.40,0.50"
B="--inputs $R/inputs/mos2_hfo2_6nm --dev-store $C/6nm_store/store --sigma-cache $C/6nm_dist/sigma --maxit 400 --chunk 20 --fgmres-log 50 --dE 0.15,0.05,0.30,0.50"
run() {  # name, options
  local n=$1 t0=$SECONDS; shift
  QTX_RANK_LOG=${P}_$n.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.hier.run $COM "$@" --out ${P}_$n.json > ${P}_$n.log 2>&1
  echo "=== $n exit $? after $(( SECONDS - t0 )) s"; grep -E "^Ec|Error" ${P}_$n.log | cut -c1-230
}
run 2808a_w16_y2 $A --cap 16,2,2 --cap-split 2,1
run 2808a_w16 $A --cap 16,2,2
run 2808a_w24 $A --cap 24,2,2 --cap-spacing 12
run 2808a_w32 $A --cap 32,2,2 --cap-spacing 16
run 6nm_w16_y4 $B --cap 16,2,2 --cap-split 4,1
run 6nm_w16 $B --cap 16,2,2
