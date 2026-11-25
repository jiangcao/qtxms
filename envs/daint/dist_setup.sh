#!/bin/bash
# MPI + NCCL for qtx_ms --dist in .venv-daint (compute node; called by dist_test.sbatch): mpi4py + mpich wheels (one node,
# mpiexec fork launcher) and the NCCL library (nvidia-nccl-cu13, bound with ctypes in qtx_ms/dist.py).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export UV_CACHE_DIR=$PWD/.tools/uv-cache UV_LINK_MODE=copy
.tools/uv-venv/bin/uv pip install --python .venv-daint/bin/python mpi4py mpich nvidia-nccl-cu13
.tools/uv-venv/bin/uv cache clean -q
.venv-daint/bin/python -c "import mpi4py; print('mpi4py', mpi4py.__version__)"
