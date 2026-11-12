#!/bin/bash
# One GH200 module per Slurm task: GPU $SLURM_LOCALID, the Grace CPU next to it and (unless NOMEMBIND=1) its LPDDR.
# Usage inside a job: srun scripts/alps/gpu_bind.sh python -m qtx_modespace ...
#   NOMEMBIND=1  CPUs of module $SLURM_LOCALID, host memory from all 4 modules
#   BIND_ALL=1   no numactl at all: all 288 cores and ~480 GB (one task per node, e.g. exact.sbatch)
L=${SLURM_LOCALID:-0}
export CUDA_VISIBLE_DEVICES=$L
if [ "${BIND_ALL:-0}" != 1 ] && command -v numactl >/dev/null; then
  if [ "${NOMEMBIND:-0}" = 1 ]; then exec numactl --cpunodebind=$L "$@"; fi
  exec numactl --cpunodebind=$L --membind=$L "$@"
fi
exec "$@"
