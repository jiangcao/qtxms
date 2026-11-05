#!/bin/bash
# Residual enrichment from spurious device resonances (qtx_ms.spurious_enrich), slab level 2 cut 5e-14 (n400):
# corr raw / slab x source flat band / gated device, 3 rounds, evaluated on both devices after every round.
cd /usr/scratch/mont-fort15/jiacao/mg_modespace
D=results_ms/gate_off/enrich; mkdir -p $D
EC=0.5043489720738419
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
L2=results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead
EV=inputs/mos2_hfo2:results_ms/exact/mos2_hfo2.json,inputs/mos2_hfo2_gate0.4:results_ms/exact/mos2_hfo2_gate0.4.json
for corr in raw slab; do
  for src in flat:inputs/mos2_hfo2 gate:inputs/mos2_hfo2_gate0.4; do
    s=${src%%:*}; I=${src#*:}
    nice -n 10 uv run python -u -m qtx_ms.spurious_enrich --modes $L2 --source $I --eval $EV --Ec $EC --corr $corr \
      --rounds 3 --out $D/l2_${corr}_$s > $D/l2_${corr}_$s.log 2>&1 &
  done
done
wait
