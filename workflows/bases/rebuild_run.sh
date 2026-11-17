#!/bin/bash
# Rebuild a release-CLI (qtx_modespace) run from the config.toml it recorded: the saved basis (modes/), level-1 cache,
# Sigma cache and exact psi are regenerated, then the new transmission.json is compared with the recorded one
# (workflows/bases/compare_runs.py: N per slab identical, T_MS / T_exact / channels within tolerance).
#   workflows/bases/rebuild_run.sh RUN_DIR NEW_OUT [extra qtx_modespace options, e.g. --dE 0.05,0.15]
#     RUN_DIR  recorded run (RUN_DIR/config.toml, RUN_DIR/transmission.json)
#     NEW_OUT  output directory of the rebuild; pass RUN_DIR itself to restore the deleted files in place
# Caches go to $CACHE (default NEW_OUT/cache: sigma, psi, level1) unless the extra options set them. FOLDER= replaces
# the device folder of the config (e.g. a prep folder that no longer exists). GPU runs: --gpu is in the config.
# Run from any directory; host: a GH200 (attelas1/2) for gpu = true configs, mont-fort15 for CPU / qttools configs.
set -e
cd "$(dirname "$0")/../.."
RUN=$1; NEW=$2; shift 2
CACHE=${CACHE:-$NEW/cache}
case $(uname -m) in
  aarch64) PY=${PY:-.venv-aarch64/bin/python}; export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-13.3} ;;
  *) PY=${PY:-.venv/bin/python} ;;
esac
export PYTHONPATH=release/qtx_modespace/src OMP_NUM_THREADS=${THREADS:-8} OPENBLAS_NUM_THREADS=${THREADS:-8} MKL_NUM_THREADS=${THREADS:-8}
mkdir -p $NEW
opts=(--config $RUN/config.toml --out $NEW --sigma-cache $CACHE/sigma --psi-dir $CACHE/psi --l1-cache $CACHE/level1)
[ -n "$FOLDER" ] && opts=("$FOLDER" "${opts[@]}")
[ "$NEW" != "$RUN" ] && cp $RUN/transmission.json $NEW/transmission.ref.json 2>/dev/null || true
echo "[$(date +%T)] rebuild $RUN -> $NEW ($(hostname))"
nice $PY -u -m qtx_modespace "${opts[@]}" "$@" > $NEW/rebuild.log 2>&1
REF=$NEW/transmission.ref.json; [ "$NEW" = "$RUN" ] && REF=${REF_JSON:?NEW_OUT = RUN_DIR: set REF_JSON to a copy of the old transmission.json}
$PY workflows/bases/compare_runs.py $REF $NEW/transmission.json --basis-json $RUN/basis.json $NEW/basis.json
