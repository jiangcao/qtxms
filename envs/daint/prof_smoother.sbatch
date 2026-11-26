#!/bin/bash
# Where the smoother time of mlsolve goes at 6 nm (user, 2026-11-25): microbenchmark of a level window, phase times, smoothers of
# Ec + 0.15 reused at 0.20, 0.10, 0.25, 0.30 (its of fresh smoothers: 28..35, 24..28, 25..31, 51..57), A_k windows with x halo 1
#   sbatch [--dependency=afterany:JOB] envs/daint/prof_smoother.sbatch
#SBATCH --job-name=prof-smoother
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
R=$PWD; O=$R/results_ms/daint; CODE=${CODE:-$R/dist_dev7}; L=$O/prof_smoother_$SLURM_JOB_ID
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
C=$R/cache_daint/6nm_store
cd $CODE
$R/.venv-daint/bin/python -u $R/envs/daint/prof_smoother.py bench
B="--inputs $R/inputs/mos2_hfo2_6nm --gpu --sigma-cache $R/cache_daint/6nm_dist/sigma --dm-cache $C/dmw_drv --dev-store $C/store --dist nccl --no-share-points \
  --cap-split 4,1 --up y:1e-5,x:1e-5,y:1e-5,x:1e-5,x:1e-5,x:1e-5 --mu=-0.15,0.8 --fine-blocks 2 --stream-smoother-gb 0 --stream-gb 40 --chunk 20 --maxit 100"
run() { n=$1; shift
  QTX_RANK_LOG=${L}_$n.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u $R/envs/daint/prof_smoother.py $B "$@" --out ${L}_$n.json > ${L}_$n.log 2>&1
  echo "== $n exit $? after $SECONDS s"; grep -E "PROF|: T |Error" ${L}_$n.log | cut -c1-330; }
FREEZE=1 run freeze --dE 0.15,0.20,0.10,0.25,0.30
FREEZE=0 run halo1 --dE 0.15,0.30 --halo-smooth same
