#!/bin/bash
# Energy sweep on daint with the stored bases: FGMRES with the density-matrix hierarchy over the exact-T grid
# Ec_lead + 0.01 .. 0.50 (10 meV, as results_ms/gpu/mlevel/run_mlevel_grid.sh; reference logs grid_{2808a,9072a}_y2.log).
# 2808a and 9072a on the four GPUs of one node: GPU 0 = 2808a (all energies), GPUs 1-3 = 9072a (every third energy).
# Window rho from the check runs (cache_daint); 9072a levels 2-6 stored by mlevel_dev.sbatch 9072a. The 2808a check
# ran with --basis-cache none: its levels 2-3 are rebuilt at the first energy from the stored window rho, then reused.
# Exact reference: direct solve at every energy (--exact), no psi files. Options as in the check runs.
# repo root:  sbatch envs/daint/mlevel_grid.sbatch
#SBATCH --job-name=mlevel-grid
#SBATCH --account=lp82
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
C=$PWD/cache_daint; O=$PWD/results_ms/daint; J=$SLURM_JOB_ID; mkdir -p $O
grid() { LC_ALL=C seq -f %.2f $1 $2 0.50 | paste -sd,; }

run() {  # gpu label inputs sigma dmw up energies
  local t0=$SECONDS L=$O/grid_$2_$J
  CUDA_VISIBLE_DEVICES=$1 taskset -c $(( $1 * 72 ))-$(( $1 * 72 + 71 )) \
    python -u -m qtx_ms.mlevel --inputs $3 --dE $7 --gpu --sigma-cache $4 --exact --dm-cache $5 \
    --cap 16,2,2 --cap-split 2,1 --cap-tov 6 --cap-solver rows64 --l1 dm:1e-2 --up $6 --halo 1 --halo-smooth full \
    --sweep exact --exact-max 100000 --chunk 10 --maxit 60 --host-blocks --out $L.json > $L.log 2>&1
  rc=$?
  echo "[$(date +%T)] GPU $1 $2 exit $rc after $(( SECONDS - t0 )) s"
}

echo "$(hostname) job $J"
run 0 2808a inputs/mos2_hfo2 $C/sigma_2808a $C/dmw y:1e-2,x:1e-2 $(grid 0.01 0.01) &
for k in 1 2 3; do
  run $k 9072a_$k inputs/mos2_hfo2_9072a $C/9072a/sigma $C/9072a/dmw y:1e-2,x:1e-2,x:1e-2,x:1e-2,x:1e-2 \
    $(grid 0.0$k 0.03) &
done
wait
