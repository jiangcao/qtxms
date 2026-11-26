#!/bin/bash
# Beyn Sigma with the coupling-basis products on the GPU, sparse couplings (both stores converted first), nonzero lead blocks; see prof_beyn_new.py
#   CODE=$PWD/dist_dev19 sbatch envs/daint/prof_beyn_new.sbatch
#SBATCH --job-name=prof-beyn-new
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
for st in $C/store $R/cache_daint/6nm_store/store; do echo "== $st"; $R/.venv-daint/bin/python -u -m qtx_ms.devstore --sparse-couplings $st; done
for n in 8; do
  echo "== $n threads"
  OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n numactl --physcpubind=0-71 --membind=0 $R/.venv-daint/bin/python -u $R/envs/daint/prof_beyn_new.py $C/store $C/prep $C/hier_sigma
done
