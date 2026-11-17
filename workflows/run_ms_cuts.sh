#!/bin/bash
# Stand-alone mode-space T and T error vs the exact T over the 2808a energy grid (Ec_lead + 0 ... 0.5 eV, 10 meV,
# 51 points) for slab level 1 (CAP + lead Sigma, 24-point local rho) at cuts 1e-8 / 1e-6 / 1e-4, on 2808a, 9072a and
# 21600a (user, 2026-11-17). Release CLI (qtx_modespace), GPU, folded lead at E + 1e-6 i.
# 2808a / 9072a reuse the prep + level-1 caches of results_ms/gpu/wf_gh200{,_9072a} (stored to 1e-10: any cut >= 1e-10
# is a truncation); 21600a is prepared and its level 1 built here. Everything big (prep, level 1, Sigma, exact psi,
# modes) goes to /scratch on the host (repo NFS nearly full); run data (logs, transmission.json / basis.json per run)
# to results_ms/gpu/ms_cuts; summary table, figure and copies of the json files to reports/2026-11-17-slab-level1-cuts.
# The first cut per device computes the exact psi (--psi-dir), the others reuse it.
# CLEAN=1: same runs with the end-slab lead-band cleanup (--clean-slabs 0,-1, spurious-band window Ec - 0.6 ... Ec + 0.55
# to cover the grid; clean.RMS), as <dev>_l1_<cut>_clean next to the raw runs (user, 2026-11-17).
#   nohup workflows/run_ms_cuts.sh > results_ms/gpu/ms_cuts/run_ms_cuts.out 2>&1 &   (attelas2, repo root)
cd "$(dirname "$0")/.."
R=${R:-results_ms/gpu/ms_cuts}
S=/scratch/jiacao/mg_modespace/ms_cuts
DEVS=${DEVS:-2808a 9072a 21600a}
CUTS=${CUTS:-1e-8 1e-6 1e-4}
DE=${DE:-0:0.5:0.01}
SUF=; CL=
[ -n "$CLEAN" ] && { SUF=_clean; CL="--clean-slabs 0,-1 --win=-0.6,0.55"; }
export CUDA_PATH=/usr/local/cuda-13.3 PYTHONPATH=release/qtx_modespace/src
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=.venv-aarch64/bin/python
mkdir -p $R $S
for d in $DEVS; do
  case $d in
    2808a) PREP=results_ms/gpu/wf_gh200/prep; L1=results_ms/gpu/wf_gh200/A_slab_obc/level1 ;;
    9072a) PREP=results_ms/gpu/wf_gh200_9072a/prep; L1=results_ms/gpu/wf_gh200_9072a/A_slab_obc/level1 ;;
    21600a) PREP=$S/21600a/prep; L1=$S/21600a/level1
       if [ ! -f $PREP/prep.json ]; then
         echo "[$(date +%T)] prep 21600a"
         nice $PY -u -m qtx_modespace.prep examples/device_MoS2HfO2_21600a_178560o --out $PREP --gpu \
           --sigma-cache $S/21600a/sigma > $R/prep_21600a.log 2>&1 || { echo "prep 21600a failed"; continue; }
       fi ;;
  esac
  for c in $CUTS; do
    t=${d}_l1_$c$SUF
    O=$S/$d/l1_$c$SUF
    echo "[$(date +%T)] $t"
    nice $PY -u -m qtx_modespace $PREP --gpu --sigma-cache $S/$d/sigma --out $O --basis l1 --l1-boundary obc \
      --l1-cut $c --l1-cache $L1 $CL --transmission --exact --psi-dir $S/$d/psi --dE $DE > $R/$t.log 2>&1
    rc=$?; echo "[$(date +%T)] $t exit $rc"
    mkdir -p $R/$t
    cp $O/transmission.json $O/basis.json $R/$t/ 2>/dev/null
  done
done
$PY workflows/ms_cuts_summary.py $R
echo done
