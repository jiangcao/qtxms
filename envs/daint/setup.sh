#!/bin/bash
# Dev environment on daint.alps (GH200, aarch64): <repo>/.venv-daint, same package versions as .venv-aarch64 (attelas).
# Run on a compute node (not a login node):  srun -A lp82 -p normal -N1 -t 20 bash envs/daint/setup.sh
# Use:  source envs/daint/activate.sh   Check:  sbatch envs/daint/test.sbatch
# uv, its python and its cache stay inside the repo folder (.tools/); no uenv / system CUDA needed (cupy[ctk] wheels).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$REPO"
TOOLS=$REPO/.tools
export UV_CACHE_DIR=$TOOLS/uv-cache UV_PYTHON_INSTALL_DIR=$TOOLS/python UV_PYTHON_BIN_DIR=$TOOLS/bin UV_LINK_MODE=copy
unset CONDA_PREFIX VIRTUAL_ENV PYTHONPATH
if [ ! -x "$TOOLS/uv-venv/bin/uv" ]; then
  /usr/bin/python3.11 -m venv "$TOOLS/uv-venv" 2>/dev/null || "$HOME/miniconda3/bin/python3" -m venv "$TOOLS/uv-venv"
  "$TOOLS/uv-venv/bin/pip" install -q uv
fi
UV=$TOOLS/uv-venv/bin/uv
[ -d .venv-daint ] || $UV venv --python 3.12 --python-preference only-managed .venv-daint
$UV pip install --python .venv-daint/bin/python -r envs/daint/requirements.txt
$UV pip install --python .venv-daint/bin/python --no-deps -e . -e release/qtx_modespace
$UV cache clean -q   # link mode copy: the cache is not needed any more
.venv-daint/bin/python -c "import cupy, numpy, scipy, qtx_ms, qtx_modespace; print(cupy.__version__, numpy.__version__, scipy.__version__)"
