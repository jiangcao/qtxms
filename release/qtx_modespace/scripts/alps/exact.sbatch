#!/bin/bash
# Exact T (and optionally MS T) of a large device, one task per node with the whole node's host memory: the streaming
# block LU (core.StreamBlockTridiag) keeps its factors on the host (~97 GB for the 6 nm MoS2/HfO2 device, more than one
# Grace module's ~120 GB once the arrays are added), so no --membind. Several nodes: energies split over the nodes.
# Usage: sbatch --account=<proj> [--nodes=n] scripts/alps/exact.sbatch PREP MODES OUT DE_LIST
# Environment: QTX_ENV (required), OPTS ["--transmission --exact"], SIGMA_CACHE [OUT/sigma], THREADS [64]
#SBATCH --job-name=qtx-exact
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=qtx-exact-%j.out
set -e
PREP=${1:?}; MODES=${2:?}; OUT=${3:?}; DE=${4:?usage: exact.sbatch PREP MODES OUT DE_LIST}
THREADS=${THREADS:-64}
source "${QTX_ALPS:-$SLURM_SUBMIT_DIR/scripts/alps}/common.sh"   # submit from the package root or set QTX_ALPS
SIG=${SIGMA_CACHE:-$OUT/sigma}
NT=${SLURM_NTASKS:-1}
mkdir -p $OUT
for k in $(seq 0 $((NT - 1))); do
  mine=$(split_list "$DE" $k $NT); [ -n "$mine" ] && mkdir -p $OUT/part$k && echo "$mine" > $OUT/part$k/dE.txt
done
export BIND_ALL=1                                   # all 4 Grace modules and their memory, GPU 0
srun -n $NT --cpu-bind=none bash -c "[ -f $OUT/part\$SLURM_PROCID/dE.txt ] || exit 0; \
  $BIND python -u -m qtx_modespace $PREP --gpu --modes $MODES --out $OUT/part\$SLURM_PROCID --sigma-cache $SIG \
  ${OPTS:---transmission --exact} --dE \$(cat $OUT/part\$SLURM_PROCID/dE.txt) > $OUT/part\$SLURM_PROCID/run.log 2>&1"
grep -h -E "^E = Ec" $OUT/part*/run.log || true
