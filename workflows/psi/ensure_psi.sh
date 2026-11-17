#!/bin/bash
# Exact psi for qtx_ms.mlevel --psi-dir, kept in /scratch on the GPU host (host-local, not the repo NFS; user 2026-11-17):
#   PSI=$(workflows/psi/ensure_psi.sh DEV DE_LIST)      DEV = 2808a | 9072a | 21600a, DE_LIST e.g. 0.05,0.15
# prints the directory /scratch/jiacao/mg_modespace/<DEV>/psi and computes the energies that are missing there with the
# release solver (qtx_modespace.core.exact_solve via workflows/psi/rebuild_psi.py --stack release: block-tridiagonal LU
# on the GPU, streaming variant with the LU factors on the host when the dense blocks do not fit; folded lead, no
# qttools on the GH200 hosts) into psi_<E>_0.00e+00.npy, which qtx_ms.mlevel.load_psi reads (checked: 21600a Ec+0.15,
# T 0.9378768573 = recorded; residual in the qtx_ms system 2e-10 -> mlevel T / psi errors have a ~1e-10 floor).
# Existing files are reused. On failure it warns on stderr and still prints the directory (mlevel then runs without
# T / psi errors for the missing energies).
# Release prep per device: 2808a / 9072a from results_ms/gpu/wf_gh200{,_9072a}/prep (repo NFS); 21600a from
# /scratch/jiacao/mg_modespace/21600a/prep_release, made from examples/ when missing (~90 s, 3.4 GB on /scratch).
cd "$(dirname "$0")/../.."
DEV=$1; DES=$2
D=/scratch/jiacao/mg_modespace
case $DEV in
  2808a)  PREP=results_ms/gpu/wf_gh200/prep ;;
  9072a)  PREP=results_ms/gpu/wf_gh200_9072a/prep ;;
  21600a) PREP=$D/21600a/prep_release ;;
  *) echo "ensure_psi.sh: unknown device $DEV" >&2; exit 1 ;;
esac
PSI=$D/$DEV/psi
mkdir -p $PSI
export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-13.3} PYTHONPATH=release/qtx_modespace/src
PY=.venv-aarch64/bin/python
if [ ! -f $PREP/prep.json ]; then
  [ $DEV = 21600a ] && nice $PY -u -m qtx_modespace.prep examples/device_MoS2HfO2_21600a_178560o --out $PREP --gpu \
    --sigma-cache $D/$DEV/sigma_release >> $PSI/ensure_psi.log 2>&1
fi
if ! nice $PY -u workflows/psi/rebuild_psi.py --stack release --gpu --inputs $PREP --sigma-cache $D/$DEV/sigma_release \
     --psi-dir $PSI --dE $DES >> $PSI/ensure_psi.log 2>&1; then
  echo "ensure_psi.sh: exact psi of $DEV at $DES not (all) computed, see $PSI/ensure_psi.log" >&2
fi
echo $PSI
