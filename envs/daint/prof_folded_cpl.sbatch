#!/bin/bash
# Folded Sigma: sparse against dense couplings; see prof_folded_cpl.py
#   CODE=$PWD/dist_dev19 sbatch envs/daint/prof_beyn_new.sbatch
#SBATCH --job-name=prof-folded-cpl
#SBATCH --account=c33
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
R=$PWD; CODE=${CODE:-$R/dist_dev19}; C=$R/cache_daint/12nm
source envs/daint/activate.sh
cd $CODE; export PYTHONPATH=$CODE${PYTHONPATH:+:$PYTHONPATH}
for n in 8; do
  echo "== $n threads"
  OMP_NUM_THREADS=72 OPENBLAS_NUM_THREADS=72 numactl --physcpubind=0-71 --membind=0 $R/.venv-daint/bin/python -u $R/envs/daint/prof_folded_cpl.py $C/store $C/prep $C/hier_sigma
done
