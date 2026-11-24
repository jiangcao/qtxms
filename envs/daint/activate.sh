# source envs/daint/activate.sh
# CPU threads: a GH200 module has 72 cores; set OMP_NUM_THREADS before sourcing to override.
_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$_repo/.venv-daint/bin/activate"
export PYTHONPATH=$_repo/release/qtx_modespace/src${PYTHONPATH:+:$PYTHONPATH}
export CUPY_CACHE_DIR=${CUPY_CACHE_DIR:-$_repo/.tools/cupy_cache}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-64} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS:-64}}
unset _repo
