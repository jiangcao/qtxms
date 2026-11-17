#!/bin/bash
# Rebuild the 2808a direct level-2 bases results_ms/mos2_hfo2_l2direct/modes (slab level 2 = global rho per slab, top
# n = 100 / 200 / 400 / 800 per slab, each with and without the end-slab lead-band RMS cleanup: files
# mu-0.15_0.6_n<n>[_lead]_slab<s>.npy, read by ~30 qtx_ms scripts, e.g. qtbm_cmp/*, span_bench/*, run_gate_off.sh)
# with the recorded command (qtx_ms.run_l2_clean, args in results_clean_mu-0.15_0.6_n100_n200_n400_n800.json), then
# compare N per stage and T per energy with that json. CPU (mont-fort15, qttools through QTTOOLS_SRC); global rho
# ~1.5 h + eigendecomposition ~25 min with 8 threads.
#   OUT=<dir> workflows/bases/rebuild_l2direct.sh     (repo root; default OUT results_ms/verify_rebuild/cpu/data/l2direct)
#   OUT=results_ms/mos2_hfo2_l2direct restores the deleted modes in place (rms_*.npz / bands_25.npz there are reused)
cd "$(dirname "$0")/../.."
OUT=${OUT:-results_ms/verify_rebuild/cpu/data/l2direct}
REF=results_ms/mos2_hfo2_l2direct/results_clean_mu-0.15_0.6_n100_n200_n400_n800.json
export QTTOOLS_SRC=${QTTOOLS_SRC:-/home/jiacao/quatrex_project/quatrex/src}
export OMP_NUM_THREADS=${THREADS:-8} OPENBLAS_NUM_THREADS=${THREADS:-8} MKL_NUM_THREADS=${THREADS:-8}
mkdir -p $OUT
[ "$OUT" != results_ms/mos2_hfo2_l2direct ] && cp -n $REF $OUT/results_clean.ref.json
echo "[$(date +%T)] run_l2_clean -> $OUT"
nice -n 10 .venv/bin/python -u -m qtx_ms.run_l2_clean --inputs inputs/mos2_hfo2 --out $OUT --nmodes 100,200,400,800 \
  --no-res --dE 0.02,0.05,0.15,0.3 --exact-from results_ms/mos2_hfo2_leadcms/results.json \
  --sigma-cache ${SIGMA:-$OUT/sigma} > $OUT/rebuild.log 2>&1
echo "[$(date +%T)] exit $?"
[ "$OUT" = results_ms/mos2_hfo2_l2direct ] && exit 0
.venv/bin/python workflows/bases/compare_l2_clean.py $OUT/results_clean.ref.json \
  $OUT/results_clean_mu-0.15_0.6_n100_n200_n400_n800.json
