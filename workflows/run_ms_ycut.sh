#!/bin/bash
# Stand-alone mode-space T on the 2808a energy grid (Ec_lead + 0 ... 0.5 eV, 10 meV) for the wide MoS2/HfO2 devices with
# tile level 1 cut in y (user, 2026-11-17): 6 nm = 4 y strips per slab, 12 nm = 8 y strips per slab (no x split, no z
# split), CAP+OBC embedding, 24-point local rho, cut 1e-8 (cache down to 1e-10). Release CLI (qtx_modespace), GPU.
# PART=slab (default since 2026-11-17 19:40, user): slab level 1 instead (same settings as workflows/run_ms_cuts.sh);
# PART=tile: the y-cut tiles above (6 nm stopped after 19 energies: T err 4x worse than slab level 1 at 2.2x N).
# 12 nm (user, 2026-11-17 20:20): PART=tile with 2 y strips (NY_12nm=2): the unsplit slab local rho needs ~145 GB of GPU;
# 2 strips ~35-40 GB with the memory-lean level1.GpuEmbedding (environment components, chunked A_EP / X and S_P^-1).
# The exact psi in <dev>/psi does not depend on the basis and is shared by both.
# Prep, Sigma, level-1 cache, exact psi, modes -> /scratch/jiacao/mg_modespace/<dev> (host-local); logs and the json files
# -> results_ms/gpu/ms_ycut.
# EXACT_<dev>=1 adds the exact T (T error). Default: 6 nm yes (streamed block LU, ~97 GB host), 12 nm no (~4x that,
# ~400 GB host: too much on the shared attelas2).
#   nohup workflows/run_ms_ycut.sh > results_ms/gpu/ms_ycut/run.out 2>&1 &   (attelas2, repo root)
cd "$(dirname "$0")/.."
R=${R:-results_ms/gpu/ms_ycut}
DEVS=${DEVS:-6nm 12nm}
CUT=${CUT:-1e-8}
PART=${PART:-slab}
DE=${DE:-0:0.5:0.01}
EXACT_6nm=${EXACT_6nm:-1}
EXACT_12nm=${EXACT_12nm:-0}
# largest level-1 embedding solved densely on the GPU (environment LU); above it the CPU sparse solver (very slow here).
# 6 nm slab embeddings ~41k (the earlier 6 nm runs used 45000); 12 nm ~2x
L1DM_6nm=${L1DM_6nm:-45000}
L1DM_12nm=${L1DM_12nm:-90000}
export CUDA_PATH=/usr/local/cuda-13.3 PYTHONPATH=release/qtx_modespace/src
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=.venv-aarch64/bin/python
mkdir -p $R
for d in $DEVS; do
  case $d in
    6nm) NY=${NY_6nm:-4} ;;
    12nm) NY=${NY_12nm:-2} ;;
  esac
  S=/scratch/jiacao/mg_modespace/$d
  mkdir -p $S
  if [ ! -f $S/prep/prep.json ]; then
    echo "[$(date +%T)] prep $d"
    nice $PY -u -m qtx_modespace.prep examples/${d}_experiment --out $S/prep --gpu --sigma-cache $S/sigma \
      > $R/prep_$d.log 2>&1 || { echo "prep $d failed"; continue; }
  fi
  ex=EXACT_$d; EX=
  dm=L1DM_$d
  [ "${!ex}" = 1 ] && EX="--exact --psi-dir $S/psi"
  if [ $PART = tile ]; then
    t=${d}_tile${NY}y_l1_$CUT; P1="--l1-partition tile --tile-split $NY,1 --tile-xsplit 1 --l1-cache $S/level1_tile${NY}y"
  else
    t=${d}_l1_$CUT; P1="--l1-partition slab --l1-cache $S/level1_slab"
  fi
  echo "[$(date +%T)] $t (exact: ${!ex})"
  nice $PY -u -m qtx_modespace $S/prep --gpu --sigma-cache $S/sigma --out $S/$t --basis l1 $P1 --l1-boundary obc \
    --l1-cut $CUT --l1-store-cut 1e-10 --l1-dense-max ${!dm} \
    --transmission $EX --dE $DE > $R/$t.log 2>&1
  rc=$?; echo "[$(date +%T)] $t exit $rc"
  mkdir -p $R/$t
  cp $S/$t/transmission.json $S/$t/basis.json $R/$t/ 2>/dev/null
done
echo done
