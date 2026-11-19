#!/bin/bash
# Slab level 2 on y-cut tile level 1 (nested U = U1 V, "level 2 on level 1"), stand-alone MS T on Ec_lead + 0 ... 0.5 eV
# (10 meV), user 2026-11-18:
#   6 nm : tile level 1, 2 y strips, CAP+OBC, cut 1e-6 (new cache level1_tile2y)  + slab level 2, cut 1e-6;
#          folded lead (as the 6 nm runs of 11-17), exact T from the cached exact psi ($S/psi, 51 energies).
#   12 nm: tile level 1, 4 y strips, CAP+OBC, cut 1e-6 (from the level1_tile4y cache, store cut 1e-10) + slab level 2,
#          cut 1e-6; --obc beyn (as 12nm_tile4y_l1_1e-8 / 12nm_exactT_beyn); T error against those exact T json files.
# Step 1 (--basis l1, no transmission) only writes the level-1 modes; step 2 (--basis l2 --l1-modes) builds the global
# rho of the level-1 system by RGF and caches its eigenpairs in <out>/eig_l2on1_*.npz: another --l2-cut = rerun of step
# 2 with that file copied (only the MS solves).
# Waits for the 12 nm exact-T runs (GPU memory).
#   setsid nohup workflows/run_ms_l2onl1.sh > results_ms/gpu/ms_ycut/run_l2onl1.out 2>&1 &   (attelas2, repo root)
cd "$(dirname "$0")/.."
R=${R:-results_ms/gpu/ms_ycut}
DEVS=${DEVS:-6nm 12nm}
C1=${C1:-1e-6}
C2=${C2:-1e-6}
L1ARGS=${L1ARGS:-}            # extra level-1 options, e.g. "--l1-env-prec c64" (local rho in complex64, half the GPU memory)
DE=${DE:-0:0.5:0.01}
export CUDA_PATH=/usr/local/cuda-13.3 PYTHONPATH=release/qtx_modespace/src
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=.venv-aarch64/bin/python
while pgrep -u $USER -f "exactT_beyn|run_ms_ycut.sh" > /dev/null; do sleep 60; done
for d in $DEVS; do
  S=/scratch/jiacao/mg_modespace/$d
  case $d in
    6nm)  NY=2; DM=45000; OBC=;            EX="--exact --psi-dir $S/psi" ;;
    12nm) NY=4; DM=90000; OBC="--obc beyn --sigma-pass"; EX= ;;
  esac
  t1=${d}_tile${NY}y_l1_$C1
  t2=${d}_tile${NY}y_l1_${C1}_l2_$C2
  echo "[$(date +%T)] $t1"
  nice $PY -u -m qtx_modespace $S/prep --gpu --sigma-cache $S/sigma --out $S/$t1 --basis l1 \
    --l1-partition tile --tile-split $NY,1 --tile-xsplit 1 --l1-cache $S/level1_tile${NY}y --l1-boundary obc \
    --l1-cut $C1 --l1-store-cut 1e-10 --l1-dense-max $DM $OBC $L1ARGS > $R/$t1.log 2>&1
  rc=$?; echo "[$(date +%T)] $t1 exit $rc"
  mkdir -p $R/$t1; cp $S/$t1/basis.json $R/$t1/ 2>/dev/null
  [ $rc = 0 ] || continue
  echo "[$(date +%T)] $t2"
  nice $PY -u -m qtx_modespace $S/prep --gpu --sigma-cache $S/sigma --out $S/$t2 --basis l2 \
    --l1-modes $S/$t1/modes --l2-cut $C2 --transmission $EX --dE $DE $OBC > $R/$t2.log 2>&1
  rc=$?; echo "[$(date +%T)] $t2 exit $rc"
  mkdir -p $R/$t2; cp $S/$t2/transmission.json $S/$t2/basis.json $R/$t2/ 2>/dev/null
done
echo done
