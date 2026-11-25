#!/bin/bash
# One MPI rank = one GH200 module: its GPU, its 72 cores, host memory preferably on the CPU (LPDDR, 480 GB per node) NUMA
# nodes. Without a policy, host pages of a full module land on the HBM of the GPUs (12 nm memory test: ~40 GB of GPU
# memory lost). preferred-many spills to the HBM only when the LPDDR of all modules is full; membind (older numactl) kills.
# QTX_LAUNCH=srun: local rank from Slurm (several nodes); otherwise mpiexec -launcher fork on one node (in the batch step
# SLURM_LOCALID is 0 for every rank).
if [ "$QTX_LAUNCH" = srun ]; then l=$SLURM_LOCALID; else
  r=${PMI_RANK:-${MPI_LOCALRANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}}; l=$(( r % 4 )); fi
export CUDA_VISIBLE_DEVICES=$l
M=--membind=0-3; numactl --preferred-many=0-3 true 2>/dev/null && M=--preferred-many=0-3
[ "$l" = 0 ] && echo "rank.sh: numa policy $M" >&2
exec numactl --physcpubind=$(( l * 72 ))-$(( l * 72 + 71 )) $M "$@"
