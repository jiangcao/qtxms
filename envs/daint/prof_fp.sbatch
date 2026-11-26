#!/bin/bash
# Phases and fixed-point steps of the Beyn Sigma; see prof_fp.py
#   CODE=$PWD/dist_dev20 sbatch envs/daint/prof_fp.sbatch
#SBATCH --job-name=prof-fp
#SBATCH --account=c33
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
R=$PWD; CODE=${CODE:-$R/dist_dev20}; C=$R/cache_daint/12nm
source envs/daint/activate.sh
cd $CODE; export PYTHONPATH=$CODE${PYTHONPATH:+:$PYTHONPATH}
for n in 8; do
  echo "== $n threads"
  OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n numactl --physcpubind=0-71 --membind=0 $R/.venv-daint/bin/python -u $R/envs/daint/prof_fp.py $C/store $C/prep
done
