#!/bin/bash
# Level-1 basis on one Alps GH200 node: the blocks (slabs or tiles) over the 4 GPUs (--l1-only k,4 into the shared
# block cache), then one task assembles the basis from the cache (+ optional mode-space T).
# Usage: sbatch --account=<proj> [--uenv=prgenv-gnu/26.3:v1 --view=default] scripts/alps/basis.sbatch PREP OUT
# Environment: QTX_ENV (required), L1_OPTS ["--l1-boundary obc --l1-cut 1e-8"], SIGMA_CACHE [OUT/sigma],
#              EXTRA (options of the assembly run, e.g. "--transmission --dE 0.05,0.15"), THREADS [16]
# Reference: 6 nm MoS2/HfO2 slab level 1 (22 slabs, ~120 s each on one GH200) 45 min on 1 GPU -> ~12 min.
#SBATCH --job-name=qtx-basis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=qtx-basis-%j.out
set -e
PREP=${1:?usage: basis.sbatch PREP OUT}; OUT=${2:?usage: basis.sbatch PREP OUT}
source "${QTX_ALPS:-$SLURM_SUBMIT_DIR/scripts/alps}/common.sh"   # submit from the package root or set QTX_ALPS
SIG=${SIGMA_CACHE:-$OUT/sigma}
L1="--basis l1 ${L1_OPTS:---l1-boundary obc --l1-cut 1e-8} --l1-cache $OUT/level1"
mkdir -p $OUT
# part k builds blocks i % NT == k into the shared block cache (own --out; the contour Sigma may be computed by
# several parts at once, cache writes are atomic)
NT=${SLURM_NTASKS:-4}
srun -n $NT --cpu-bind=none bash -c "$BIND python -u -m qtx_modespace $PREP --gpu --out $OUT/l1_part\$SLURM_PROCID \
     --sigma-cache $SIG $L1 --l1-only \$SLURM_PROCID,$NT > $OUT/l1_part\$SLURM_PROCID.log 2>&1"
srun -n 1 --cpu-bind=none $BIND python -u -m qtx_modespace $PREP --gpu --out $OUT --sigma-cache $SIG $L1 $EXTRA \
     > $OUT/basis.log 2>&1
grep -h "^level 1: N1" $OUT/basis.log || true
