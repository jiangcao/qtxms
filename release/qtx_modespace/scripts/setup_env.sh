#!/bin/bash
# qtx_modespace environment: a Python venv with the package, its optional dependencies and, on a host with an NVIDIA
# GPU, the cupy wheel that matches the driver's CUDA version. Writes VENV/bin/qtx_env.sh (activate + CUDA_PATH +
# QTTOOLS_SRC) to source before every run.
#
# Usage: scripts/setup_env.sh [VENV]          (default VENV = ./.venv-qtx; run from anywhere)
# Environment:
#   PYTHON       interpreter for the venv (default python3; >= 3.11, 3.12 for the qttools OBC)
#   PKG          what to install: the package folder (default: the folder above this script) or a wheel file
#   EDITABLE=1   install the package folder in editable mode (pip install -e)
#   CUDA         auto (default: from nvidia-smi) | 12 | 13 | none (CPU only)
#   CUDA_PATH    CUDA toolkit folder; default: $CUDA_HOME (uenv), else the newest /usr/local/cuda-<major>.* of the
#                chosen major version; if none is found, CTK=1 is used
#   CTK=1        install the CUDA runtime libraries as pip wheels (cupy-cudaXXx[ctk]) instead of using CUDA_PATH
#   SYSTEM_SITE=1  create the venv with --system-site-packages (on top of a uenv / module Python)
#   CACHE_DIR    CUPY_CACHE_DIR / NUMBA_CACHE_DIR recorded in qtx_env.sh (default: $SCRATCH/.cache/qtx when $SCRATCH
#                is set, e.g. on CSCS Alps; kernel caches in $HOME go incoherent across nodes)
# An existing venv (e.g. the QuaTrEx one, uenv prgenv-gnu) is reused: the package goes into it, and a cupy already
# installed there is kept (no second cupy-cudaXXx).
#   QTTOOLS_SRC  QuaTrEx src folder (real-axis qttools OBC); recorded in qtx_env.sh; mpi4py and numba are installed
#   TEST=1       run the test suite (CPU + GPU tests) at the end
# Examples:
#   scripts/setup_env.sh ~/venvs/qtx                                   # auto-detect the GPU
#   CUDA=none PYTHON=python3.12 scripts/setup_env.sh                   # CPU only
#   PYTHON=/usr/local/anaconda3/bin/python TEST=1 scripts/setup_env.sh # GH200 host (aarch64, CUDA 13 driver)
#   CSCS Alps (on a compute node, inside `uenv start --view=default prgenv-gnu/26.3:v1`), into the QuaTrEx venv:
#     QTTOOLS_SRC=$QUATREX/src TEST=1 scripts/setup_env.sh $QUATREX/.venv
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
VENV=${1:-.venv-qtx}
PYTHON=${PYTHON:-python3}
PKG=${PKG:-$HERE}
CUDA=${CUDA:-auto}

die() { echo "setup_env: $*" >&2; exit 1; }
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null || die "$PYTHON is not Python >= 3.11"
PYV=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "python $PYV ($("$PYTHON" -c 'import sys; print(sys.executable)')), $(uname -m)"

# ---- CUDA major version of the driver
if [ "$CUDA" = auto ]; then
  if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    # "CUDA Version: 12.4" (driver <= 5xx) or "CUDA UMD Version: 13.3" (driver >= 6xx)
    CUDA=$(nvidia-smi | grep -oE 'CUDA (UMD )?Version: *[0-9]+' | grep -oE '[0-9]+$' | head -1)
    [ -n "$CUDA" ] || die "nvidia-smi works but shows no CUDA version; set CUDA=12|13|none"
  else
    CUDA=none
  fi
fi
case $CUDA in
  none) CUPY= ;;
  12) CUPY=cupy-cuda12x ;;
  13) CUPY=cupy-cuda13x ;;
  *) die "CUDA=$CUDA: supported 12, 13 or none" ;;
esac
[ -n "$CLUSTER_NAME" ] && echo "cluster: $CLUSTER_NAME${UENV_VIEW:+, uenv view $UENV_VIEW}"
if [ "$CUDA" = none ] && [ -n "$SLURM_CLUSTER_NAME$CLUSTER_NAME" ] && [ -z "$SLURM_JOB_ID" ]; then
  echo "setup_env: warning: no GPU visible (login node?); run on a compute node (salloc / srun) for the GPU part"
fi
if [ -n "$CUPY" ] && [ "${CTK:-0}" != 1 ] && [ -z "$CUDA_PATH" ]; then
  [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ] && CUDA_PATH=$CUDA_HOME
  [ -n "$CUDA_PATH" ] || CUDA_PATH=$(ls -d /usr/local/cuda-$CUDA.* 2>/dev/null | sort -V | tail -1)
  [ -n "$CUDA_PATH" ] || { [ -d /usr/local/cuda ] && CUDA_PATH=/usr/local/cuda; }
  if [ -z "$CUDA_PATH" ]; then
    echo "setup_env: no CUDA $CUDA toolkit found (CUDA_HOME, /usr/local); CUDA runtime from pip (CTK=1)"
    CTK=1
  fi
fi
echo "GPU: ${CUPY:-none}${CUDA_PATH:+, CUDA_PATH=$CUDA_PATH}${CTK:+, CUDA runtime from pip}"

# ---- venv and packages
if [ -x "$VENV/bin/python" ]; then
  echo "existing venv $VENV ($("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])'))"
else
  SSP=; [ "${SYSTEM_SITE:-0}" = 1 ] && SSP=--system-site-packages
  "$PYTHON" -m venv $SSP "$VENV"
fi
VENV=$(cd "$VENV" && pwd)
PIP="$VENV/bin/python -m pip"
$PIP install -q --upgrade pip
EXTRAS="fast,plot,test"
if [ "${EDITABLE:-0}" = 1 ]; then
  $PIP install -q -e "$PKG[$EXTRAS]"
else
  $PIP install -q "$PKG[$EXTRAS]"
fi
HAVE_CUPY=$($PIP list 2>/dev/null | grep -oE '^cupy(-cuda[0-9]+x)? ' | head -1 | tr -d ' ')
if [ -n "$CUPY" ] && [ -n "$HAVE_CUPY" ]; then
  [ "$HAVE_CUPY" = "$CUPY" ] || echo "setup_env: keeping the installed $HAVE_CUPY (driver suggests $CUPY)"
elif [ -n "$CUPY" ]; then
  if [ "${CTK:-0}" = 1 ]; then $PIP install -q "$CUPY[ctk]"; else $PIP install -q "$CUPY"; fi
fi
if [ -n "$QTTOOLS_SRC" ]; then
  [ "$PYV" != 3.11 ] || echo "setup_env: warning: the qttools OBC needs Python >= 3.12 (folded lead used instead)"
  # mpi4py: keep an installed one (e.g. built against cray-mpich in the QuaTrEx venv)
  $VENV/bin/python -c 'import mpi4py' 2>/dev/null || $PIP install -q mpi4py || echo "setup_env: warning: mpi4py failed"
  $PIP install -q numba || echo "setup_env: warning: numba failed to install (qttools OBC unavailable)"
  $VENV/bin/python -c "import sys; sys.path.insert(0, '$QTTOOLS_SRC'); import qttools" 2>/dev/null \
    || echo "setup_env: warning: qttools not importable from $QTTOOLS_SRC"
fi

# ---- qtx_env.sh
[ -z "$CACHE_DIR" ] && [ -n "$SCRATCH" ] && CACHE_DIR=$SCRATCH/.cache/qtx
cat > "$VENV/bin/qtx_env.sh" <<EOF
# source $VENV/bin/qtx_env.sh   (written by qtx_modespace scripts/setup_env.sh)
source "$VENV/bin/activate"
${CUDA_PATH:+export CUDA_PATH="$CUDA_PATH"}
${QTTOOLS_SRC:+export QTTOOLS_SRC="$QTTOOLS_SRC"}
${CACHE_DIR:+export CUPY_CACHE_DIR="$CACHE_DIR/cupy" NUMBA_CACHE_DIR="$CACHE_DIR/numba"}
${UENV_VIEW:+# built inside uenv view $UENV_VIEW: start it first (uenv start / sbatch --uenv --view)}
EOF

# ---- check
source "$VENV/bin/qtx_env.sh"
python - <<'EOF'
import qtx_modespace, numpy, scipy
print(f"qtx_modespace {qtx_modespace.__version__}, numpy {numpy.__version__}, scipy {scipy.__version__}")
try:
    import pypardiso  # noqa: F401
    print("pypardiso: yes")
except ImportError:
    print("pypardiso: no (SuperLU for the sparse solves)")
from qtx_modespace import gpu
try:
    gpu.enable()
    print(gpu.describe())
    a = gpu.cp.random.standard_normal((512, 512)) + 1j
    lu = gpu.lu_factor(a.copy())
    x = gpu.lu_solve(lu, a)
    print(f"GPU LU check: {float(gpu.cp.abs(x - gpu.cp.eye(512)).max()):.1e}")
except Exception as e:  # CPU-only environment
    print(f"GPU: not available ({type(e).__name__}: {e})")
EOF
if [ "${TEST:-0}" = 1 ]; then
  [ -d "$HERE/tests" ] || die "TEST=1: no tests/ folder in $HERE"
  (cd "$HERE" && python -m pytest tests -q -p no:cacheprovider)
fi
echo "done: source $VENV/bin/qtx_env.sh"
