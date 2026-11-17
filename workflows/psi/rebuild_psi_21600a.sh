#!/bin/bash
# Exact psi of 21600a for qtx_ms.mlevel --psi-dir (the 8 energies of results_ms/gpu/wf_gh200_21600a/psi, deleted
# 2026-11-17), release solver (workflows/psi/ensure_psi.sh: block LU on the GPU, streaming with the LU factors on the
# host when the GPU is short of memory), checked against the T_exact the mlevel runs printed (6 decimals). Replaces
# results_ms/gpu/mlevel/run_exact_21600a.sh for all energies.
#   workflows/psi/rebuild_psi_21600a.sh   (repo root, a GH200 host; psi in /scratch/jiacao/mg_modespace/21600a/psi)
#   WAIT_GB=n: wait until n GB of the GPU are free (at most WAIT_H hours, default 24)
cd "$(dirname "$0")/../.."
OUT=${OUT:-results_ms/verify_rebuild/psi_21600a}
export CUDA_PATH=/usr/local/cuda-13.3 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
if [ -n "$WAIT_GB" ]; then
  end=$(( $(date +%s) + ${WAIT_H:-24} * 3600 ))
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge $(( WAIT_GB * 1024 )) ] && break
    [ "$(date +%s)" -ge "$end" ] && { echo "$(date): GPU still busy ($free MiB free), not started"; exit 1; }
    sleep 60
  done
  echo "$(date): $free MiB free on the GPU"
fi
PSI=$(workflows/psi/ensure_psi.sh 21600a 0.02,0.05,0.1,0.15,0.3,0.45,0.58,0.7)
PYTHONPATH=release/qtx_modespace/src nice .venv-aarch64/bin/python -u workflows/psi/rebuild_psi.py --stack release --gpu \
  --inputs /scratch/jiacao/mg_modespace/21600a/prep_release --sigma-cache /scratch/jiacao/mg_modespace/21600a/sigma_release \
  --psi-dir $PSI --dE 0.02,0.05,0.1,0.15,0.3,0.45,0.58,0.7 \
  --ref-T 0.000000,0.000000,1.490400,0.937877,0.482522,5.952482,1.886626,2.159828 --ref-abs 5e-7 --out $OUT.json
