#!/bin/bash
# Density-matrix hierarchy on the 12 nm MoS2/HfO2 device (window 16 A, 8 y strips of 16.5 A as the 4 of 6 nm), one GPU of
# an otherwise empty node: the 6 nm settings of mlevel_dev.sbatch with --cap-split 8,1 and one more y merge (y 8->4->2->1).
# No --exact: the fine A (~34 GB resident) plus the slab LU (slabs 33k, 3-5 dense blocks of 17 GB) exceeds one GPU. T
# reference: exact T of the 2026-11-18 run (--obc beyn), reports/2026-11-18-ms-ycut-6nm-12nm/12nm_exactT_beyn*.
# Out of GPU memory: one retry with fine A and every level operator streamed (bases already built are reused).
# The estimate (scaled from the 6 nm build: ~2 h to level 5) is 6-10 h; stopped by the limit: resubmit, it resumes.
# after the prep job (sbatch envs/daint/prep_12nm.sbatch -> cache_daint/12nm/prep):
#   sbatch --dependency=afterok:<prep> envs/daint/mlevel_12nm.sbatch
#SBATCH --job-name=mlevel-12nm
#SBATCH --account=lp82
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
source envs/daint/activate.sh
C=$PWD/cache_daint/12nm; O=$PWD/results_ms/daint; mkdir -p $C/dmw $O
UP=y:1e-5,x:1e-5,y:1e-5,x:1e-5,y:1e-5,x:1e-5,x:1e-5
run() {  # tag, extra options
  local L=$O/mlevel_12nm_${SLURM_JOB_ID}_$1; shift
  time python -u -m qtx_ms.mlevel --inputs $C/prep --dE ${DE:-0.15} --gpu --sigma-cache $C/sigma --dm-cache $C/dmw \
    --cap 16,2,2 --cap-split 8,1 --cap-tov 6 --cap-solver rows64 --l1 dmc:1e-5 --dm-keep 1e-6 --schur \
    --schur-prec double --up $UP --halo 1 --halo-smooth full --sweep exact --exact-max 100000 --chunk 20 --maxit 60 \
    --mu=-0.15,0.8 "$@" --out $L.json > $L.log 2>&1
  local rc=$?; echo "[$(date +%T)] $L exit $rc"; LOG=$L.log; return $rc
}
echo "12nm on $(hostname), Ec + ${DE:-0.15}"
run free2 --free-levels 2 --stream-gb 8 && exit 0
grep -q OutOfMemory $LOG || exit 1
run stream --free-levels 0 --host-blocks --stream-gb 2
