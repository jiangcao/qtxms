#!/bin/bash
# qtx_ms.hier, 12 nm (714 240 orbitals; 43 x 8 cores), two nodes (8 GPUs, as mn_test.sbatch): level 3 solved exactly (--bottom bt, user, 2026-11-25), cuts 1e-5.
# Slabs of level 3 twice as wide as at 6 nm (~4500 orbitals: ~22 GB of factors on every GPU). Bases and Sigma are cached: a run cut off by the limit resumes.
#   sbatch envs/daint/hier_12nm.sbatch        (repo root, after store_12nm.sbatch; code copy dist_dev13)
# OBC: folded (default) | beyn-real (Beyn Sigma of the periodized lead at the real energies, contour and bases unchanged) | beyn; MAXIT
#   OBC=beyn-real MAXIT=200 CODE=$PWD/dist_dev15 sbatch envs/daint/hier_12nm.sbatch
# OBC=closed: no lead, RHS evenly spaced columns of the identity on the left contact face (8256 in all; RHS=0: all); bases of the closed device
#   OBC=closed MAXIT=200 CODE=$PWD/dist_dev16 sbatch envs/daint/hier_12nm.sbatch
# SIG: folder of the Sigma files (default hier_sigma)
#   SIG=$PWD/cache_daint/12nm/hier_sigma_new OBC=beyn-real MAXIT=200 CODE=$PWD/dist_dev19 sbatch envs/daint/hier_12nm.sbatch
# Nodes: sbatch --nodes=4 (4 ranks per node; srun takes the job size)
# EXTRA: more arguments of qtx_ms.hier.run, e.g. EXTRA=--profile
# CUT2, CUT3: cuts of the merges y (level 2) and x (level 3), default 1e-5 (level 1: --cut 1e-5)
# DEV: folder in cache_daint (prep, store, Sigma, bases; default 12nm; OFF state: DEV=12nm_off after prep_off_12nm.sbatch)
#SBATCH --job-name=hier-12nm
#SBATCH --account=c33
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --uenv=prgenv-gnu/24.11:v1
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
R=$PWD; CODE=${CODE:-$R/dist_dev13}; export R
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
source envs/daint/mn_env.sh
PY=$R/.venv-daint/bin/python; O=$R/results_ms/daint; C=$R/cache_daint/${DEV:-12nm}; J=$SLURM_JOB_ID
export QTX_NCCL=/user-environment/env/default/lib/libnccl.so.2
OBC=${OBC:-folded}; L=$O/hier_l3_${DEV:-12nm}_$J; cd $CODE
echo "$(hostname) job $J, $SLURM_NNODES nodes, obc $OBC, code $CODE"
( while sleep 5; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' '; done ) > $L.mon 2>/dev/null & MON=$!
QTX_RANK_LOG=$L.rank srun -n $SLURM_NTASKS -N $SLURM_NNODES --ntasks-per-node=4 --mpi=cray_shasta --cpu-bind=none --gpu-bind=none $R/envs/daint/rank.sh $PY -u -m qtx_ms.hier.run \
  --inputs $C/prep --dev-store $C/store --sigma-cache ${SIG:-$C/hier_sigma} --cache $C/hier --cap 16,2,2 --cap-split 8,1 --cut 1e-5 --up y:${CUT2:-1e-5},x:${CUT3:-1e-5} \
  --chunk 10 --maxit ${MAXIT:-100} --obc $OBC --rhs-cols ${RHS:-80} --fgmres-log 25 --gpu --dist nccl --dE ${DE:-0.15,0.30,0.50} --out $L.json $EXTRA > $L.log 2>&1
rc=$?
kill $MON
echo "exit $rc after $SECONDS s; GPU MiB peak of this node $(tr ' ' '\n' < $L.mon | sort -n | tail -1)"
grep -E "^ *level|^setup|^obc|memory of|^Ec|Error|done" $L.log | cut -c1-330
exit $rc
