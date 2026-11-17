#!/bin/bash
# OFF state of the run_ms_cuts.sh sweep (user, 2026-11-17): 0.4 eV gate barrier over the middle third of the channel
# (workflows/gate_device.py, w 2 A; V = 0 on the leads, so the lead Sigma cache of the flat run is reused), same grid
# (Ec_lead + 0 ... 0.5 eV, 10 meV), slab level 1 cuts 1e-8 / 1e-6 / 1e-4 on 2808a, 9072a, 21600a, two bases per cut:
#   <dev>_l1_<cut>_gate  basis rebuilt at the gate potential (level-1 cache $S/<dev>/gate/level1)
#   <dev>_l1_<cut>_flat  flat-band basis of the run_ms_cuts.sh run reused (--modes $S/<dev>/l1_<cut>/modes)
# Exact psi of the gated device in $S/<dev>/gate/psi (computed by the first run, reused by the others). Same host as
# run_ms_cuts.sh (its modes / Sigma caches are in its /scratch). Run data in results_ms/gpu/ms_cuts_gate, summary,
# figure and json copies in reports/2026-11-17-slab-level1-cuts-gate.
#   nohup workflows/run_ms_cuts_gate.sh > results_ms/gpu/ms_cuts_gate/run.out 2>&1 &   (attelas2, repo root)
#   one process per device (GPU otherwise idle most of the time): DEVS=9072a nohup ... > run_9072a.out &, etc.
cd "$(dirname "$0")/.."
R=${R:-results_ms/gpu/ms_cuts_gate}
S=/scratch/jiacao/mg_modespace/ms_cuts
DEVS=${DEVS:-2808a 9072a 21600a}
CUTS=${CUTS:-1e-8 1e-6 1e-4}
DE=${DE:-0:0.5:0.01}
export CUDA_PATH=/usr/local/cuda-13.3 PYTHONPATH=release/qtx_modespace/src
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=.venv-aarch64/bin/python
mkdir -p $R
for d in $DEVS; do
  case $d in
    2808a) SRC=examples/device_MoS2HfO2_2808a_29088o ;;
    9072a) SRC=examples/device_MoS2HfO2_9072a_78912o ;;
    21600a) SRC=examples/device_MoS2HfO2_21600a_178560o ;;
  esac
  G=$S/$d/gate
  PREP=$G/prep
  if [ ! -f $PREP/prep.json ]; then
    echo "[$(date +%T)] prep $d (gate)"
    { $PY workflows/gate_device.py $SRC $G/device &&
      nice $PY -u -m qtx_modespace.prep $G/device --out $PREP --potential --gpu --sigma-cache $S/$d/sigma
    } > $R/prep_$d.log 2>&1 || { echo "prep $d failed"; continue; }
  fi
  for c in $CUTS; do
    for b in gate flat; do
      t=${d}_l1_${c}_$b
      O=$G/l1_${c}_$b
      if [ $b = gate ]; then B="--basis l1 --l1-boundary obc --l1-cut $c --l1-cache $G/level1"
      else B="--modes $S/$d/l1_$c/modes"; fi
      echo "[$(date +%T)] $t"
      nice $PY -u -m qtx_modespace $PREP --gpu --sigma-cache $S/$d/sigma --out $O $B --transmission --exact \
        --psi-dir $G/psi --dE $DE > $R/$t.log 2>&1
      rc=$?; echo "[$(date +%T)] $t exit $rc"
      mkdir -p $R/$t
      cp $O/transmission.json $O/basis.json $R/$t/ 2>/dev/null
    done
  done
done
$PY workflows/ms_cuts_summary.py $R reports/2026-11-17-slab-level1-cuts-gate
echo done
