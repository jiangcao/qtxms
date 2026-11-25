#!/bin/bash
# Device store (qtx_ms.devstore) of the 12 nm inputs (cache_daint/12nm/prep, 714 240 orbitals) for qtx_ms.hier --dev-store.
#   sbatch envs/daint/store_12nm.sbatch       (repo root; code copy dist_dev13)
#SBATCH --job-name=store-12nm
#SBATCH --account=c33
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
R=$PWD; CODE=${CODE:-$R/dist_dev13}; C=$R/cache_daint/12nm
export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64
source envs/daint/activate.sh
cd $CODE; test -f $C/store/store.json || time python -u -m qtx_ms.devstore --inputs $C/prep --out $C/store
python - <<PY
import json; print(json.load(open("$C/prep/prep.json"))); print({k: v for k, v in json.load(open("$C/store/store.json")).items() if k != "source"})
PY
