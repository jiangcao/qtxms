#!/bin/bash
# Spurious device-state absorption (first run: deflation, logs *_defl.log) (qtx_ms.ms_deflate): gated (0.4 eV) and flat-band device, slab level 2 cut 5e-14 (n400)
# and slab level 1 cut 1e-8; resonances searched on the 10 meV grid (Sigma cached), thr 5 eV.
cd /usr/scratch/mont-fort15/jiacao/mg_modespace
D=results_ms/gate_off/deflate; mkdir -p $D
EC=0.5043489720738419
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
L2=results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead
L1=results_ms/mos2_hfo2_slabl1_tobc24/modes/t1e-8
for b in l2:$L2 l1_1e-8:$L1; do
  tag=${b%%:*}; m=${b#*:}
  for dev in gate:inputs/mos2_hfo2_gate0.4:results_ms/exact/mos2_hfo2_gate0.4.json flat:inputs/mos2_hfo2:results_ms/exact/mos2_hfo2.json; do
    IFS=: read d I X <<< "$dev"
    nice -n 10 uv run python -u -m qtx_ms.ms_deflate --inputs $I --modes $m --Ec $EC --find 0:0.5:0.01 \
      --exact $X --out $D/${tag}_$d.json --reuse --no-defl --absorb 1e-3,1e-2,0.1,1 > $D/${tag}_$d.log 2>&1 &
  done
done
wait
