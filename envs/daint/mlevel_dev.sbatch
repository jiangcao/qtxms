#!/bin/bash
# Density-matrix hierarchy on the larger MoS2/HfO2 devices on daint (after the 2808a check, mlevel_2808a.sbatch), one GPU
# of an otherwise empty node, all host memory (streaming exact psi, host blocks). Everything fresh in cache_daint/<dev>/.
#   9072a, 21600a: settings of results_ms/gpu/mlevel/run_mlevel_grid.sh (reference logs grid_{9072a,21600a}_y2.log)
#   6nm: settings of run_mlevel_6nm_dmc.sh (no finished reference: the attelas2 build ran out of GPU memory), Ec+0.15
# repo root:  sbatch --time=0:30:00 --job-name=mlevel-9072a envs/daint/mlevel_dev.sbatch 9072a
#SBATCH --job-name=mlevel-dev
#SBATCH --account=lp82
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
D=${1:?device 9072a | 21600a | 6nm}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
source envs/daint/activate.sh
C=$PWD/cache_daint/$D; O=$PWD/results_ms/daint; mkdir -p $C/sigma $C/dmw $O
case $D in
  9072a)  IN=inputs/mos2_hfo2_9072a;  DE=${DE:-0.05,0.15} ;;
  21600a) IN=inputs/mos2_hfo2_21600a; DE=${DE:-0.15} ;;     # T(Ec + 0.05) = 0 to 6 digits: no meaningful relative error
  6nm)    IN=inputs/mos2_hfo2_6nm;    DE=${DE:-0.15} ;;
  *) echo "unknown device $D" >&2; exit 2 ;;
esac
case $D in
  6nm) OPTS="--cap-split 4,1 --l1 dmc:1e-5 --dm-keep 1e-6 --schur --schur-prec double --up y:1e-5,x:1e-5,y:1e-5,x:1e-5,x:1e-5,x:1e-5 --chunk 20 --free-levels 2 --stream-gb 8 --mu=-0.15,0.8" ;;
  *)   OPTS="--cap-split 2,1 --l1 dm:1e-2 --up y:1e-2,x:1e-2,x:1e-2,x:1e-2,x:1e-2 --chunk 10 --host-blocks" ;;
esac
echo "$D on $(hostname), Ec + $DE: $OPTS"
time python -u -m qtx_ms.mlevel --inputs $IN --dE $DE --gpu --sigma-cache $C/sigma --exact --dm-cache $C/dmw \
  --cap 16,2,2 --cap-tov 6 --cap-solver rows64 --halo 1 --halo-smooth full --sweep exact --exact-max 100000 --maxit 60 \
  $OPTS --out $O/mlevel_${D}_$SLURM_JOB_ID.json
