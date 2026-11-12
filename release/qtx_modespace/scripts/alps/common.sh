# sourced by the *.sbatch templates: environment of one qtx_modespace task on an Alps GH200 node
#   QTX_ENV  qtx_env.sh written by scripts/setup_env.sh (required)
: "${QTX_ENV:?set QTX_ENV=<venv>/bin/qtx_env.sh (scripts/setup_env.sh)}"
source "$QTX_ENV"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BIND=$HERE/gpu_bind.sh
THREADS=${THREADS:-16}                       # BLAS threads per task (CPU leftovers in GPU mode)
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export CUPY_CACHE_DIR=${CUPY_CACHE_DIR:-$SCRATCH/.cache/qtx/cupy} NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$SCRATCH/.cache/qtx/numba}
mkdir -p "$CUPY_CACHE_DIR" "$NUMBA_CACHE_DIR"
# part K of N of a comma list (round robin): split_list "a,b,c,d,e" 1 4 -> b
split_list() { echo "$1" | tr ',' '\n' | awk -v k=$2 -v n=$3 '(NR - 1) % n == k' | paste -sd, -; }
