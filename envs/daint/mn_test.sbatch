#!/bin/bash
# qtx_ms --dist on two nodes (8 ranks, 8 GPUs): MPI through srun + cray-mpich of the uenv, NCCL over Slingshot.
# Probe of the libraries, strip tests (cpu, host-staged MPI, NCCL), then 9072a on 1 and 2 nodes: --bottom repl without
# shared points (as in dist_test.sbatch), and the defaults (domain bottom solve, shared contour points).
# Code: dist_dev2/ (copy of qtx_ms + tests). Queued jobs keep dist_dev/.
#SBATCH --job-name=mn-test
#SBATCH --account=lp82
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
R=$PWD; CODE=${CODE:-$R/dist_dev2}; export R
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
source envs/daint/mn_env.sh
PY=$R/.venv-daint/bin/python
O=$R/results_ms/daint; C=$R/cache_daint; J=$SLURM_JOB_ID; mkdir -p $O
export QTX_TEST_SIGMA=$R/.tools/dist_sigma
cd $CODE
run() { local n=$1 N=$2; shift 2; srun -n $n -N $N --ntasks-per-node=4 --mpi=cray_shasta --cpu-bind=none --gpu-bind=none \
  $R/envs/daint/rank.sh "$@"; }
echo "$(hostname) job $J, code $CODE, $SLURM_NNODES nodes"
$PY -c "import qtx_ms; print(qtx_ms.__file__)"
T="from mpi4py import MPI; import socket; c=MPI.COMM_WORLD; v=MPI.Get_library_version().split(chr(10))[0]; print(c.rank,c.size,c.allreduce(1),socket.gethostname(),v[:90],flush=True); assert c.size == 8 and c.allreduce(1) == 8 and len(set(c.allgather(socket.gethostname()))) == 2, 'ranks do not see each other'"
hello() { timeout 120 bash -c "$(declare -f run); run 8 2 $PY -c \"$T\"" > .hello_$J 2>&1; local rc=$?; sort -n .hello_$J | tail -12; rm -f .hello_$J
  return $rc; }
echo "=== mpi4py, 8 ranks on 2 nodes"
if ! hello; then
  echo "=== wheel failed: mpi4py from source against cray-mpich"; mn_build_mpi4py 2>&1 | tail -5
  export PYTHONPATH=$R/.tools/mpi4py_cray:$PYTHONPATH; unset MPI4PY_LIBMPI MPI4PY_MPIABI
  hello || { echo "mpi4py does not run on two nodes"; exit 1; }
fi
U=/user-environment/env/default/lib/libnccl.so.2
NCCL_OK=
probe() {  # name, VAR=value ... (environment of the run)
  local n=$1 f=$O/mn_probe_$1_$J.log; shift
  timeout 75 bash -c "$(declare -f run); env NCCL_DEBUG=WARN $* srun -n 8 -N 2 --ntasks-per-node=4 --mpi=cray_shasta --cpu-bind=none \
    --gpu-bind=none $R/envs/daint/rank.sh $PY -u tests/nccl_probe.py" > $f 2>&1; local rc=$?
  echo "=== probe $n: rc $rc  ($*)"; grep -E "^init|^sum|MB|PASS" $f; [ $rc = 0 ] || grep -E "WARN|Error" $f | sed 's/^.*NCCL WARN/WARN/' | sort | uniq -c | sort -rn | head -4 | cut -c1-200
  if [ $rc = 0 ] && grep -q PASS $f && [ -z "$NCCL_OK" ]; then NCCL_OK="$*"; echo "-> selected"; fi
}
F=/opt/cray/libfabric
probe uenv QTX_NCCL=$U
[ -n "$NCCL_OK" ] || probe uenv_lf122 QTX_NCCL=$U LD_LIBRARY_PATH=$F/1.22.0/lib64:$LD_LIBRARY_PATH
[ -n "$NCCL_OK" ] || probe uenv_lf231 QTX_NCCL=$U LD_LIBRARY_PATH=$F/2.3.1/lib64:$LD_LIBRARY_PATH
[ -n "$NCCL_OK" ] || probe pip_lf231 LD_LIBRARY_PATH=$F/2.3.1/lib64:$LD_LIBRARY_PATH
[ -n "$NCCL_OK" ] || probe pip_tcp NCCL_NET=Socket NCCL_NET_PLUGIN=none
if [ -n "$NCCL_OK" ]; then
  echo "=== strip nccl, 8 ranks on 2 nodes"
  timeout 200 bash -c "$(declare -f run); export $NCCL_OK; export OMP_NUM_THREADS=2; run 8 2 $PY -u tests/dist_mpi.py nccl" > $O/mn_strip_nccl_$J.log 2>&1
  rc=$?; echo "rc $rc"; grep -E "ranks|PASS|FAILED|Error" $O/mn_strip_nccl_$J.log | cut -c1-200 | head -8
  [ $rc = 0 ] || NCCL_OK=
fi
ml() {  # ranks nodes label options
  local n=$1 N=$2 L=$O/mn_$3_n${1}_$J K=nccl X="$NCCL_OK"; shift 3
  # one node: the NCCL of the venv without the network plugin (as in dist_test.sbatch); two nodes: the probe's choice, else MPI
  [ $N = 1 ] && X="NCCL_NET=Socket NCCL_NET_PLUGIN=none"; [ $N = 2 ] && [ -z "$X" ] && K=mpi X="QTX_NONE=1"
  QTX_RANK_LOG=$L.rank timeout 420 bash -c "$(declare -f run); export $X; run $n $N $PY -u -m qtx_ms.mlevel $* --dist $K --gpu --exact \
    --inputs $R/inputs/mos2_hfo2_9072a --dE 0.15,0.50 --sigma-cache $C/9072a/sigma --dm-cache $C/9072a/dmw \
    --up y:1e-2,x:1e-2,x:1e-2,x:1e-2,x:1e-2 --cap 16,2,2 --cap-split 2,1 --cap-tov 6 --cap-solver rows64 --l1 dm:1e-2 \
    --halo 1 --halo-smooth full --sweep exact --exact-max 100000 --chunk 10 --maxit 60 --host-blocks --basis-cache none --out $L.json" > $L.log 2>&1
  local rc=$?
  echo "=== $L exit $rc (--dist $K, $X)"; grep -E "domain decomp|windows per rank|contour points shared|levels at|^Ec|exact level|peak per rank|Error|done," $L.log | cut -c1-220
}
ml 4 1 repl --bottom repl --no-share-points
ml 4 1 new
ml 8 2 new
ml 8 2 repl --bottom repl --no-share-points
