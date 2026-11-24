#!/bin/bash
# Real-device check on daint: density-matrix hierarchy on MoS2/HfO2 2808a, settings of
# results_ms/gpu/mlevel/run_mergeeig_2808a.sh (MODES=full); sigma and window density matrices computed fresh in
# cache_daint/ (no /scratch on daint). Compare with results_ms/gpu/mlevel/mergeeig_2808a_full.log (attelas2).
# sbatch envs/daint/mlevel_2808a.sbatch   (repo root)
#SBATCH --job-name=mlevel-2808a
#SBATCH --account=lp82
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
source envs/daint/activate.sh
C=$PWD/cache_daint; O=$PWD/results_ms/daint; mkdir -p $C/sigma_2808a $C/dmw $O
time python -u -m qtx_ms.mlevel --inputs inputs/mos2_hfo2 --dE 0.05,0.15 --gpu --sigma-cache $C/sigma_2808a \
  --psi-dir results_ms/gpu/wf_gh200/psi --dm-cache $C/dmw --cap 16,2,2 --cap-split 2,1 --cap-tov 6 --cap-solver rows64 \
  --l1 dm:1e-2 --up y:1e-2,x:1e-2 --halo 1 --halo-smooth full --sweep exact --exact-max 100000 --chunk 10 --maxit 60 \
  --host-blocks --basis-cache none --merge-eig full --out $O/mlevel_2808a_$SLURM_JOB_ID.json
