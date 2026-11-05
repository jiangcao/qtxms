#!/bin/bash
# Bases built AT the gate potential (inputs/mos2_hfo2_gate0.4), for the OFF-state test (flat-band bases: run_gate_off.sh):
#  1. slab level 1 (CAP+OBC, 24-pt exact local rho, accelerated path, columns down to 1e-10), cuts 1e-6 / 1e-8 / 1e-10
#  2. slab level 2 (direct, top 400 per slab + end-slab lead RMS), contour Sigma shared with mos2_hfo2 (same leads)
# then ms_scan on Ec + 0:0.5:0.01 vs the gated exact json.
cd /usr/scratch/mont-fort15/jiacao/mg_modespace
I=inputs/mos2_hfo2_gate0.4
EC=0.5043489720738419
SIG=results_ms/sigma/mos2_hfo2
X=results_ms/exact/mos2_hfo2_gate0.4.json
D=results_ms/gate_off/gate_basis; mkdir -p $D
base=results_ms/mos2_hfo2_gate0.4_slabl1_tobc24
odir=${base}/tiles_obc_1e-10
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
# 1. slab level 1 tiles (4 x 6 threads over the contour points)
nice -n 10 uv run python -u -m qtx_ms.run_ms --inputs $I --no-cores --tile-orbs 100000 --obc folded --mu=-0.15,0.6 \
    --dE 0.0 --mg= --exact-from $X --tile-contour 24 --tile-obc 24 --tile-obc-dir $odir --sigma-cache $SIG \
    --tile-rho-thr 1e-10 --out ${base}_t1e-10 --tile-worker 0,1 \
    --tile-fast schur --tile-env-prec double --tile-pts-workers 4 --tile-pts-threads 6 > $D/slabl1_tiles.log 2>&1
for t in 1e-6 1e-8 1e-10; do
  uv run python -m qtx_ms.slab_l1_modes --inputs $I --tiles $odir --cut $t --out ${base}/modes/t$t >> $D/slabl1_tiles.log 2>&1
done
# 2. slab level 2 n400 (rho eigendecomposition ~25 min) in parallel with the level-1 scans
nice -n 10 uv run python -u -m qtx_ms.run_l2_clean --inputs $I --out results_ms/mos2_hfo2_gate0.4_l2direct --nmodes 400 \
    --no-res --sigma-cache $SIG --obc folded --Ec $EC --dE 0.0,0.1,0.2,0.3 --exact-from $X > $D/l2direct.log 2>&1 &
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for t in 1e-6 1e-8 1e-10; do
  nice -n 10 uv run python -u -m qtx_ms.ms_scan --inputs $I --obc folded --Ec $EC --sigma-cache $SIG --modes ${base}/modes/t$t \
     --dE 0:0.5:0.01 --no-exact --exact $X --out $D/scan_slabl1_t$t.json > $D/scan_slabl1_t$t.log 2>&1 &
done
wait
M=$(ls results_ms/mos2_hfo2_gate0.4_l2direct/modes/*n400_lead_slab0.npy | sed 's/_slab0.npy//')
nice -n 10 uv run python -u -m qtx_ms.ms_scan --inputs $I --obc folded --Ec $EC --sigma-cache $SIG --modes $M \
   --dE 0:0.5:0.01 --no-exact --exact $X --out $D/scan_l2_n400.json > $D/scan_l2_n400.log 2>&1
