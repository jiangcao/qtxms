#!/bin/bash
# Energy sweep on a saved basis, one process per GPU, energies split round robin over all tasks (any node count).
# Each task writes OUT/part<k>/ (transmission.json, log); the Sigma cache and psi folder are shared (atomic writes).
# Usage: sbatch --account=<proj> [--nodes=n] scripts/alps/sweep.sbatch PREP MODES OUT DE_LIST
#   DE_LIST: comma list of E - Ec [eV], e.g. $(python -c "print(','.join(f'{0.01*i:.2f}' for i in range(51)))")
# Environment: QTX_ENV (required), OPTS ["--transmission --exact"], e.g. add "--fgmres --cap 16,2,2"; SIGMA_CACHE
#              [OUT/sigma], THREADS [16]
#SBATCH --job-name=qtx-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=qtx-sweep-%j.out
set -e
PREP=${1:?}; MODES=${2:?}; OUT=${3:?}; DE=${4:?usage: sweep.sbatch PREP MODES OUT DE_LIST}
source "${QTX_ALPS:-$SLURM_SUBMIT_DIR/scripts/alps}/common.sh"   # submit from the package root or set QTX_ALPS
SIG=${SIGMA_CACHE:-$OUT/sigma}
NT=${SLURM_NTASKS:-4}
mkdir -p $OUT
for k in $(seq 0 $((NT - 1))); do
  mine=$(split_list "$DE" $k $NT); [ -n "$mine" ] && mkdir -p $OUT/part$k && echo "$mine" > $OUT/part$k/dE.txt
done
srun -n $NT --cpu-bind=none bash -c "[ -f $OUT/part\$SLURM_PROCID/dE.txt ] || exit 0; \
  $BIND python -u -m qtx_modespace $PREP --gpu --modes $MODES --out $OUT/part\$SLURM_PROCID --sigma-cache $SIG \
  --psi-dir $OUT/psi ${OPTS:---transmission --exact} --dE \$(cat $OUT/part\$SLURM_PROCID/dE.txt) \
  > $OUT/part\$SLURM_PROCID/run.log 2>&1"
grep -h -E "^E = Ec" $OUT/part*/run.log | sort -t+ -k2 -g || true
