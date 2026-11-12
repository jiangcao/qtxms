#!/bin/bash
# qtx_modespace workflow on a QuaTrEx device folder, CPU or GPU.
#
#   0  (always) prepared inputs in $OUT/prep (qtx_modespace.prep), loaded by every step
#   A  slab level-1 basis, CAP + lead Sigma (CAP+OBC), cut $CUT_SLAB; mode-space T vs exact (exact psi in $OUT/psi)
#   B  slab level-1 basis, CAP only, cut $CUT_SLAB; mode-space T
#   C  tile level-1 basis (~$TILE_ORBS orbitals per tile), CAP+OBC, cut $CUT_TILE; mode-space T
#   D  open system on basis A: FGMRES + CAP-Schwarz, two-level (mode-space coarse level) and one-level
#   E  closed system on basis A: A0 = E S - H without Sigma, unit right-hand sides on the first $RHS_MAX lead-face
#      orbitals of slab 0, two-level and one-level, vs a direct solve
#   P  transmission.png of every run (matplotlib; qtx_modespace.plot)
#
# Usage: examples/workflow.sh DEVICE_FOLDER [OUT]
# Environment (defaults in brackets):
#   STEPS     steps to run ["A B C D E P"]
#   GPU       1: dense parts on the GPU (--gpu; cupy, see scripts/setup_env.sh) [0]
#   THREADS   BLAS threads per process [6];  WORKERS: processes / threads for contour points and windows (CPU) [4]
#   DE        energies - Ec for T [0.05,0.15,0.2];  DE_CLOSED: energies of step E [0.05,0.2]
#   EXACT     0: no exact solves in A / D (large devices on the CPU) [1]
#   CUT_SLAB, CUT_TILE  level-1 occupation cuts [1e-8, 1e-6];  TILE_ORBS [2500]
#   CAP, CAP_SPLIT      Schwarz windows: width,ramp,d [16,2,2] and transverse split, e.g. 2,1 (large cross sections) []
#   RHS_MAX   closed-system columns [128]
#   PREP_OPTS extra options of qtx_modespace.prep, e.g. "--cells 3 --trim" []
#   SIGMA_CACHE  lead-Sigma cache shared by all steps [$OUT/sigma];  QTTOOLS_SRC: real-axis qttools OBC (else folded)
#   PY        python [python]
# Steps already done are not skipped; every step reuses the Sigma cache, the level-1 block cache and $OUT/psi.
#
# Reference cost, MoS2/HfO2 2808 atoms (29 088 orbitals): 28-core CPU node (THREADS 6, WORKERS 4) A / B 40 min each,
# C 70 min, Sigma ~8 min per energy, exact 4-9 min per energy; GH200 (GPU=1) whole A + D 4.5 min.
# 6 nm MoS2/HfO2 (344 640 device orbitals, GH200): A basis 45 min (N1 38 054), exact solve ~2.5 min per energy
# (streaming block LU, ~100 GB host memory), MS solve 17 s.
set -e
DEV=${1:?usage: $0 DEVICE_FOLDER [OUT]}
OUT=${2:-qtx_workflow}
THREADS=${THREADS:-6}
WORKERS=${WORKERS:-4}
DE=${DE:-0.05,0.15,0.2}
DE_CLOSED=${DE_CLOSED:-0.05,0.2}
CUT_SLAB=${CUT_SLAB:-1e-8}
CUT_TILE=${CUT_TILE:-1e-6}
TILE_ORBS=${TILE_ORBS:-2500}
CAP=${CAP:-16,2,2}
RHS_MAX=${RHS_MAX:-128}
PY=${PY:-python}
STEPS=${STEPS:-A B C D E P}
SIG=${SIGMA_CACHE:-$OUT/sigma}
mkdir -p $OUT
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
G=; [ "${GPU:-0}" = 1 ] && G=--gpu
X=--exact; [ "${EXACT:-1}" = 0 ] && X=
SPLIT=; [ -n "$CAP_SPLIT" ] && SPLIT="--cap-split $CAP_SPLIT"
$PY -c "import qtx_modespace as q; print('qtx_modespace', q.__version__)"

# step 0: read the QuaTrEx folder once; every step below loads $OUT/prep instead of DEVICE_FOLDER
if [ ! -f $OUT/prep/prep.json ]; then
  echo "[$(date +%T)] prep $DEV -> $OUT/prep"
  $PY -u -m qtx_modespace.prep $DEV --out $OUT/prep --sigma-cache $SIG $G $PREP_OPTS > $OUT/prep.log 2>&1
fi
DEV=$OUT/prep
run() { echo "[$(date +%T)] $*"; $PY -u -m qtx_modespace $DEV --sigma-cache $SIG $G "$@"; }
L1="--basis l1 --l1-workers $WORKERS --l1-threads $THREADS"
need_A() { [ -f $OUT/A_slab_obc/modes/slab0.npy ] || { echo "step $1 needs the basis of step A" >&2; exit 1; }; }

for s in $STEPS; do case $s in
A) run --out $OUT/A_slab_obc $L1 --l1-boundary obc --l1-cut $CUT_SLAB --transmission --dE $DE $X --psi-dir $OUT/psi \
       > $OUT/A_slab_obc.log 2>&1 ;;
B) run --out $OUT/B_slab_cap $L1 --l1-boundary cap --l1-cut $CUT_SLAB --transmission --dE $DE > $OUT/B_slab_cap.log 2>&1 ;;
C) run --out $OUT/C_tile_obc $L1 --l1-partition tile --tile-orbs $TILE_ORBS --l1-boundary obc --l1-cut $CUT_TILE \
       --transmission --dE $DE > $OUT/C_tile_obc.log 2>&1 ;;
D) need_A D
   for co in ms none; do
     run --out $OUT/D_fgmres_$co --modes $OUT/A_slab_obc/modes --transmission $X --psi-dir $OUT/psi --dE $DE \
         --fgmres --fgmres-coarse $co --cap $CAP $SPLIT --workers $WORKERS > $OUT/D_fgmres_$co.log 2>&1
   done ;;
E) need_A E
   for co in ms none; do
     run --out $OUT/E_closed_$co --modes $OUT/A_slab_obc/modes --system closed --fgmres --fgmres-coarse $co \
         --cap $CAP $SPLIT --rhs-slabs 0 --rhs-face --rhs-max $RHS_MAX --rhs-chunk 64 --dE $DE_CLOSED --exact \
         --workers $WORKERS > $OUT/E_closed_$co.log 2>&1
   done ;;
P) runs=$(ls -d $OUT/*/transmission.json 2>/dev/null | xargs -r -n1 dirname)
   [ -z "$runs" ] || $PY -m qtx_modespace.plot $runs 2>/dev/null || echo "plots skipped (matplotlib missing?)" ;;
*) echo "unknown step $s" >&2; exit 1 ;;
esac; done
# summary
grep -h -E "^level 1: N1|^E = Ec|FGMRES \(|closed: its|^max \|" $OUT/*.log || true
