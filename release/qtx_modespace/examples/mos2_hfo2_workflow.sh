#!/bin/bash
# qtx_modespace example workflow: MoS2 on HfO2, 2808 atoms (device_MoS2HfO2_2808a_29088o), every feature of 0.2.0.
#
#   A  slab level-1 basis, CAP + lead Sigma (CAP+OBC), cut 1e-8; mode-space T vs exact (exact psi kept in $OUT/psi)
#   B  slab level-1 basis, CAP only, cut 1e-8; mode-space T
#   C  tile level-1 basis (~2500 orbitals per tile), CAP+OBC, cut 1e-6; mode-space T
#   D  open system on basis A: FGMRES + CAP-Schwarz, two-level (mode-space coarse level) and one-level
#   E  closed system on basis A: A0 = E S - H without Sigma, unit right-hand sides on the first 128 lead-face
#      orbitals of slab 0, two-level and one-level, vs a direct solve
#
# Usage: examples/mos2_hfo2_workflow.sh DEVICE_FOLDER [OUT]     (steps: STEPS="A B C D E", default all)
# Environment: THREADS (BLAS threads per process, default 6), WORKERS (processes / threads for the contour points and
# the Schwarz windows, default 4), DE (energies - Ec for T, default 0.05,0.15,0.2), QTTOOLS_SRC (QuaTrEx src folder:
# real-axis OBC from qttools; without it the folded lead at E + 1e-6 i is used), PY (python, default python).
# Cost on a 28-core node (THREADS 6, WORKERS 4): A / B 40 min each (local rho of the 4 slabs), C 70 min (8 tiles),
# real-axis lead Sigma ~8 min per new energy (cached in $OUT/sigma, shared by all steps), exact solve 4-9 min per
# energy, FGMRES 2-5 min per energy, E ~10 min per energy.
set -e
DEV=${1:?usage: $0 DEVICE_FOLDER [OUT]}
OUT=${2:-mos2_hfo2_workflow}
THREADS=${THREADS:-6}
WORKERS=${WORKERS:-4}
DE=${DE:-0.05,0.15,0.2}
PY=${PY:-python}
STEPS=${STEPS:-A B C D E}
SIG=${SIGMA_CACHE:-$OUT/sigma}
mkdir -p $OUT
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
run() { echo "[$(date +%T)] $*"; $PY -u -m qtx_modespace $DEV --sigma-cache $SIG "$@"; }
L1="--basis l1 --l1-workers $WORKERS --l1-threads $THREADS"

for s in $STEPS; do case $s in
A) run --out $OUT/A_slab_obc $L1 --l1-boundary obc --l1-cut 1e-8 --transmission --dE $DE --exact --psi-dir $OUT/psi \
       > $OUT/A_slab_obc.log 2>&1 ;;
B) run --out $OUT/B_slab_cap $L1 --l1-boundary cap --l1-cut 1e-8 --transmission --dE $DE > $OUT/B_slab_cap.log 2>&1 ;;
C) run --out $OUT/C_tile_obc $L1 --l1-partition tile --tile-orbs 2500 --l1-boundary obc --l1-cut 1e-6 \
       --transmission --dE $DE > $OUT/C_tile_obc.log 2>&1 ;;
D) for co in ms none; do
     run --out $OUT/D_fgmres_$co --modes $OUT/A_slab_obc/modes --transmission --exact --psi-dir $OUT/psi --dE $DE \
         --fgmres --fgmres-coarse $co --workers $WORKERS > $OUT/D_fgmres_$co.log 2>&1
   done ;;
E) for co in ms none; do
     run --out $OUT/E_closed_$co --modes $OUT/A_slab_obc/modes --system closed --fgmres --fgmres-coarse $co \
         --rhs-slabs 0 --rhs-face --rhs-max 128 --rhs-chunk 64 --dE 0.05,0.2 --exact --workers $WORKERS \
         > $OUT/E_closed_$co.log 2>&1
   done ;;
esac; done
# summary
grep -h -E "^level 1: N1|^E = Ec|FGMRES \(|closed: its|max \|" $OUT/*.log
