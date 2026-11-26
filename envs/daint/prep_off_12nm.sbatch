#!/bin/bash
# OFF-state 12 nm device: barrier VB above the source on the middle third, drain at -VDS (qtx_ms.prep.offstate; defaults 0.2 / 0.4 eV),
# device folder examples/12nm_off (links to examples/12nm_experiment + inputs/potential.npy), prepared with --potential
# (right lead = left lead - VDS S) -> cache_daint/12nm_off/prep. Then the device store and the hierarchy with DEV=12nm_off:
#   A=$(CODE=$PWD/dist_dev26 sbatch --parsable envs/daint/prep_off_12nm.sbatch)
#   A=$(DEV=12nm_off CODE=$PWD/dist_dev26 sbatch --parsable --dependency=afterok:$A envs/daint/store_12nm.sbatch)
#   DEV=12nm_off OBC=beyn ... sbatch --dependency=afterok:$A envs/daint/hier_12nm.sbatch
#SBATCH --job-name=prep-12nm-off
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
R=$PWD; CODE=${CODE:-$R/dist_dev26}; VB=${VB:-0.2}; VDS=${VDS:-0.4}; OUT=${OUT:-12nm_off}
export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0
source envs/daint/activate.sh
C=$R/cache_daint/$OUT; mkdir -p $C
(cd $CODE && python -u -m qtx_ms.prep.offstate $R/examples/12nm_experiment --out $R/examples/$OUT --Vb $VB --Vds $VDS)
test -f $C/prep/prep.json || time python -u -m qtx_modespace.prep examples/$OUT --out $C/prep --potential --gpu --sigma-cache $C/sigma
python - <<PY
import json; m = json.load(open("$C/prep/prep.json")); print(m["options"]); print(m["meta"]["edges"], m["meta"]["lead_checks"])
PY
