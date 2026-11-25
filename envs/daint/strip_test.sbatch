#!/bin/bash
# tests/dist_mpi.py (the strip, every case) on 4 GPUs over NCCL: spectra of A_k, cycle, its and solutions against one rank.
#   sbatch envs/daint/strip_test.sbatch   (repo root; CODE=dist_dev5)
#SBATCH --job-name=strip-test
#SBATCH --account=lp82
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
R=$PWD; CODE=${CODE:-$R/dist_dev5}; W=$R/cache_daint/devstore_2808a
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
cd $CODE
QTX_TEST_SIGMA=$W/strip_sigma_$SLURM_JOB_ID $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python tests/dist_mpi.py nccl 2>&1 | grep -v CapSchwarz
