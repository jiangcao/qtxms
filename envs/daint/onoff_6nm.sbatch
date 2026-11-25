#!/bin/bash
# 6 nm (344 640 orbitals, dmc 1e-5, Schur double, 24 points) on 4 GPUs of one node: shared contour points + distributed
# bottom solve on / off (user, 2026-11-24). Same code, node, window rho (stored by 5448244, cache_daint/6nm_dist) and
# options as mlevel_wide_dist.sbatch 6nm; levels rebuilt in both runs (--basis-cache none), Ec + 0.15, no exact psi.
# off first, then on. Compare: levels time and merge rho per level, solve per bottom level, peak memory per rank.
#   sbatch [--dependency=afterany:<job>] envs/daint/onoff_6nm.sbatch
#SBATCH --job-name=onoff-6nm
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
R=$PWD; CODE=${CODE:-$R/dist_dev2}; O=$R/results_ms/daint; C=$R/cache_daint/6nm_dist; J=$SLURM_JOB_ID
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
test -d $C/dmw || { echo "no window rho in $C"; exit 1; }
mkdir -p $O; cd $CODE
echo "$(hostname) job $J, code $CODE"; $R/.venv-daint/bin/python -c "import qtx_ms; print(qtx_ms.__file__)"
run() {  # label options
  local L=$O/onoff_6nm_$1_$J t0=$SECONDS rc; shift
  QTX_RANK_LOG=$L.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.mlevel \
    --dist nccl --inputs $R/inputs/mos2_hfo2_6nm --dE 0.15 --gpu --sigma-cache $C/sigma --dm-cache $C/dmw --cap 16,2,2 \
    --cap-split 4,1 --cap-tov 6 --cap-solver rows64 --l1 dmc:1e-5 --dm-keep 1e-6 --schur --schur-prec double \
    --up y:1e-5,x:1e-5,y:1e-5,x:1e-5,x:1e-5,x:1e-5 --halo 1 --halo-smooth full --sweep exact --exact-max 100000 \
    --chunk 20 --maxit 60 --free-levels 2 --stream-gb 40 --mu=-0.15,0.8 --basis-cache none "$@" --out $L.json > $L.log 2>&1
  rc=$?
  echo "=== $L exit $rc after $(( SECONDS - t0 )) s ($*)"
  grep -E "domain decomp|window density|contour points shared|^  level [2-9] \(|levels at|slab LU|exact level|peak per rank|Error|done," $L.log | cut -c1-230
  grep -oE "level [0-9] \(. merge|rho [0-9]+ s in [0-9]+ batches|smoother [0-9]+ s;" $L.log | paste -sd" "
}
run off --bottom repl --no-share-points
run on
