#!/bin/bash
# Gate sweep on the 12 nm OFF-state device (cache_daint/12nm_off: prep / store / Sigma / bases of Vb 0.2, Vds 0.4; job 5461684):
# for every barrier VB in VBS, the potential change dV = V(VB) - V(0.2) (qtx_ms.prep.offstate, same gate, ramps and drain) is added
# to the fine H as (dV S + S dV) / 2 (qtx_ms.hier.run --dV); dV = 0 on the leads, so the lead Sigma are reused.
#   MODE=reuse (default): the Vb 0.2 bases (cache_daint/12nm_off/hier); the Galerkin blocks project dV into them.
#   MODE=build: new bases of H + dV in cache_daint/12nm_off/hier_vb$VB (reference for the iteration counts).
# VDSS: drain biases (default 0.4): dV = V(VB, VDS) - V(0.2, 0.4) is -(VDS - 0.4) on the drain and the right lead face -> Sigma_R of the
# 0.4 lead at E + VDS - 0.4 (OBC shift_R; new Beyn Sigma_R files in the same Sigma folder); the left lead and the injection reused.
# Same arguments as 5461684 (cache tag), 6 nodes:
#   VBS="0.1 0.3 0.0" CODE=$PWD/dist_dev27 sbatch envs/daint/hier_gate_12nm.sbatch
#   MODE=build VBS="0.1 0.3" CODE=$PWD/dist_dev27 sbatch envs/daint/hier_gate_12nm.sbatch
#   VBS=0.2 VDSS="0.1 0.2 0.3 0.5" CODE=$PWD/dist_dev28 sbatch --job-name=hier-bias envs/daint/hier_gate_12nm.sbatch
#SBATCH --job-name=hier-gate
#SBATCH --account=c33
#SBATCH --partition=debug
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --uenv=prgenv-gnu/24.11:v1
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
R=$PWD; CODE=${CODE:-$R/dist_dev27}; export R
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
source envs/daint/mn_env.sh
PY=$R/.venv-daint/bin/python; O=$R/results_ms/daint; C=$R/cache_daint/12nm_off; J=$SLURM_JOB_ID; MODE=${MODE:-reuse}
export QTX_NCCL=/user-environment/env/default/lib/libnccl.so.2
REF=$R/examples/12nm_off/inputs/potential.npy
echo "$(hostname) job $J, $SLURM_NNODES nodes, mode $MODE, barriers ${VBS:-0.1 0.3 0.0}, drain ${VDSS:-0.4}, code $CODE"
for VB in ${VBS:-0.1 0.3 0.0}; do for VDS in ${VDSS:-0.4}; do
  T=vb$VB; [ $VDS = 0.4 ] || T=${T}_vds$VDS
  D=$R/examples/12nm_off_$T
  (cd $CODE && $PY -m qtx_ms.prep.offstate $R/examples/12nm_experiment --out $D --Vb $VB --Vds $VDS)
  B=$C/hier; [ $MODE = build ] && B=$C/hier_$T
  L=$O/hier_gate_${T}_${MODE}_$J
  (cd $CODE && QTX_RANK_LOG=$L.rank srun -n $SLURM_NTASKS -N $SLURM_NNODES --ntasks-per-node=4 --mpi=cray_shasta --cpu-bind=none --gpu-bind=none \
    $R/envs/daint/rank.sh $PY -u -m qtx_ms.hier.run \
    --inputs $C/prep --dev-store $C/store --sigma-cache $C/hier_sigma --cache $B --cap 16,2,2 --cap-split 8,1 --cut 1e-5 --up y:1e-7,x:1e-7 \
    --chunk 10 --maxit 300 --obc beyn --rhs-cols 80 --fgmres-log 25 --gpu --dist nccl --dE ${DE:-0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50} \
    --dV $D/inputs/potential.npy,$REF --dV-bases $MODE --out $L.json $EXTRA > $L.log 2>&1)
  echo "Vb $VB, Vds $VDS: exit $? after $SECONDS s"
  grep -E "^dV|^ *level|^setup|^obc: [0-9]|^Ec|Error|done" $L.log | cut -c1-200
done; done
