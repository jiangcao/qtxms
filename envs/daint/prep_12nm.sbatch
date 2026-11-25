#!/bin/bash
# Prepared inputs of the 12 nm MoS2/HfO2 device (86 400 atoms, 714 240 orbitals; as workflows/run_ms_ycut.sh on attelas2,
# 602 s there) -> cache_daint/12nm/prep. Raw QuaTrEx folder examples/12nm_experiment.
# repo root:  sbatch envs/daint/prep_12nm.sbatch
#SBATCH --job-name=prep-12nm
#SBATCH --account=lp82
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=.tools/%x-%j.out
set -e
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0
source envs/daint/activate.sh
C=$PWD/cache_daint/12nm; mkdir -p $C
time python -u -m qtx_modespace.prep examples/12nm_experiment --out $C/prep --gpu --sigma-cache $C/sigma
test -f $C/prep/prep.json
