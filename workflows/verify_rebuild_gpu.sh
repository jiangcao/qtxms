#!/bin/bash
# Check (2026-11-17) that the npy / npz files deleted from results_ms/ (list: results_ms/deleted_2026-11-17.txt) can be
# rebuilt with workflows/{sigma,psi,bases} and match the recorded runs. GPU part, attelas1 (2808a + 9072a, < 30 GB):
#   Sigma + psi of 2808a / 9072a against the recorded T_exact / channels, then every release-CLI GPU run whose modes/
#   were deleted, rerun from its config.toml with fresh caches (level 1 rebuilt) and compared with its transmission.json.
# Big rebuilt files: /scratch/jiacao/mg_modespace/verify (host-local); logs + json: results_ms/verify_rebuild/gpu.
#   nohup workflows/verify_rebuild_gpu.sh > results_ms/verify_rebuild/gpu.out 2>&1 &     (repo root, attelas1)
cd "$(dirname "$0")/.."
S=${S:-/scratch/jiacao/mg_modespace/verify}
L=results_ms/verify_rebuild/gpu
mkdir -p $S $L
export CUDA_PATH=/usr/local/cuda-13.3 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=.venv-aarch64/bin/python
step() { echo "[$(date +%T)] $*"; }
res() { echo "$1: $(tail -1 $2)" >> $L/summary.txt; }
: > $L/summary.txt
# ---- 2808a: Sigma, psi (energies of results_ms/gpu/npts: 0.02 ... 0.4), recorded in npts/A_n24_c1e-8
R8=results_ms/gpu/npts/A_n24_c1e-8/transmission.json
step "sigma 2808a"
nice $PY -u workflows/sigma/rebuild_sigma.py --stack release --gpu --inputs results_ms/gpu/wf_gh200/prep \
  --cache $S/2808a/sigma --dE 0.02,0.05,0.1,0.15,0.2,0.3,0.4 --ref $R8 --out $L/sigma_2808a.json > $L/sigma_2808a.log 2>&1
res sigma_2808a $L/sigma_2808a.log
step "psi 2808a"
nice $PY -u workflows/psi/rebuild_psi.py --stack release --gpu --inputs results_ms/gpu/wf_gh200/prep \
  --sigma-cache $S/2808a/sigma --psi-dir $S/2808a/psi --dE 0.02,0.05,0.1,0.15,0.2,0.3,0.4 --ref $R8 \
  --out $L/psi_2808a.json > $L/psi_2808a.log 2>&1
res psi_2808a $L/psi_2808a.log
# ---- 2808a bases: all GPU release-CLI runs with deleted modes/ (shared caches: level 1 rebuilt by the first run)
for r in gpu/wf_gh200/A_slab_obc gpu/npts/A_n24_c1e-8 gpu/npts/A_n24_c1e-6 gpu/npts/A_n24_c1e-4 gpu/npts/A_n24_c1e-2 \
         gpu/npts/A_n48_c1e-8 gpu/npts/A_n48_c1e-6 gpu/npts/A_n48_c1e-4 gpu/clean/A_n24_c1e-8_clean \
         gpu/clean/A_n24_c1e-6_clean gpu/prof/A; do
  n=$(echo $r | tr / _)
  step "bases $r"
  F=; [ $r = gpu/prof/A ] && F=results_ms/gpu/wf_gh200/prep          # its prep_2808a folder no longer exists
  FOLDER=$F CACHE=$S/2808a workflows/bases/rebuild_run.sh results_ms/$r $S/runs/$n > $L/bases_$n.log 2>&1
  cp $S/runs/$n/rebuild.log $L/bases_${n}_run.log 2>/dev/null
  res bases_$n $L/bases_$n.log
done
# ---- 9072a: Sigma, psi, slab level 1 1e-8 basis (results_ms/gpu/wf_gh200_9072a/A_slab_obc)
R9=results_ms/gpu/wf_gh200_9072a/A_slab_obc/transmission.json
step "sigma 9072a"
nice $PY -u workflows/sigma/rebuild_sigma.py --stack release --gpu --inputs results_ms/gpu/wf_gh200_9072a/prep \
  --cache $S/9072a/sigma --dE 0.05,0.15,0.2 --ref $R9 --out $L/sigma_9072a.json > $L/sigma_9072a.log 2>&1
res sigma_9072a $L/sigma_9072a.log
step "psi 9072a"
nice $PY -u workflows/psi/rebuild_psi.py --stack release --gpu --inputs results_ms/gpu/wf_gh200_9072a/prep \
  --sigma-cache $S/9072a/sigma --psi-dir $S/9072a/psi --dE 0.05,0.15,0.2 --ref $R9 --out $L/psi_9072a.json \
  > $L/psi_9072a.log 2>&1
res psi_9072a $L/psi_9072a.log
step "bases gpu/wf_gh200_9072a/A_slab_obc"
CACHE=$S/9072a workflows/bases/rebuild_run.sh results_ms/gpu/wf_gh200_9072a/A_slab_obc $S/runs/wf_gh200_9072a_A \
  > $L/bases_wf_gh200_9072a_A.log 2>&1
cp $S/runs/wf_gh200_9072a_A/rebuild.log $L/bases_wf_gh200_9072a_A_run.log 2>/dev/null
res bases_wf_gh200_9072a_A $L/bases_wf_gh200_9072a_A.log
# ---- 9072a slab level 2 n400 of release 0.1.0 (results_ms/ship010, CPU 6 h then; here GPU, folded lead at eta 1e-6)
step "bases ship010 9072a"
FOLDER=examples/device_MoS2HfO2_9072a_78912o CACHE=$S/9072a_ship workflows/bases/rebuild_run.sh \
  results_ms/ship010/out_device_MoS2HfO2_9072a_78912o $S/runs/ship010_9072a --gpu > $L/bases_ship010_9072a.log 2>&1
cp $S/runs/ship010_9072a/rebuild.log $L/bases_ship010_9072a_run.log 2>/dev/null
res bases_ship010_9072a $L/bases_ship010_9072a.log
step done; cat $L/summary.txt
