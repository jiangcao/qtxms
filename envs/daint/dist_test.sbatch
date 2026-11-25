#!/bin/bash
# Domain decomposition (qtx_ms.mlevel --dist) on the four GPUs of one node, debug queue: installs mpi4py / mpich / NCCL if
# missing, the strip tests under MPI (cpu, host-staged GPU halos, NCCL), then 2808a and 9072a with bases built from the
# stored window rho (--basis-cache none) at Ec + 0.15, 0.50 on 4 ranks against the one-GPU runs of 2026-11-24.
# The code is taken from $CODE (default: dist_dev/, a copy of qtx_ms + tests; the queued jobs keep the repo's qtx_ms).
#SBATCH --job-name=dist-test
#SBATCH --account=lp82
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
R=$PWD; CODE=${CODE:-$R/dist_dev}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
.venv-daint/bin/python -c "import mpi4py" 2>/dev/null || bash envs/daint/dist_setup.sh || exit 1
O=$R/results_ms/daint; C=$R/cache_daint; J=$SLURM_JOB_ID; mkdir -p $O
export QTX_TEST_SIGMA=$R/.tools/dist_sigma
MP="$R/.venv-daint/bin/mpiexec -launcher fork"
cd $CODE
echo "$(hostname) job $J, code $CODE"; $R/.venv-daint/bin/python -c "import qtx_ms; print(qtx_ms.__file__)"
for t in cpu mpi nccl; do
  echo "=== strip $t"; OMP_NUM_THREADS=2 timeout 240 $MP -n 4 $R/envs/daint/rank.sh python -u tests/dist_mpi.py $t 2>&1 | grep -v "^CapSchwarz" | tail -8
done
ml() {  # ranks label options
  local n=$1 L=$O/dist_$2_n${1}_$J; shift 2
  local D="--dist nccl"; [ $n = 1 ] && D=
  QTX_RANK_LOG=$L.rank timeout 600 $MP -n $n $R/envs/daint/rank.sh python -u -m qtx_ms.mlevel "$@" $D --gpu --exact --cap 16,2,2 \
    --cap-split 2,1 --cap-tov 6 --cap-solver rows64 --l1 dm:1e-2 --halo 1 --halo-smooth full --sweep exact \
    --exact-max 100000 --chunk 10 --maxit 60 --host-blocks --basis-cache none --out $L.json > $L.log 2>&1
  local rc=$?
  echo "=== $L exit $rc"; grep -E "domain decomp|windows per rank|levels at|^Ec|exact level|peak per rank|Error|done," $L.log | cut -c1-200
}
ml 1 2808a --inputs $R/inputs/mos2_hfo2 --dE 0.15,0.50 --sigma-cache $C/sigma_2808a --dm-cache $C/dmw --up y:1e-2,x:1e-2
ml 4 2808a --inputs $R/inputs/mos2_hfo2 --dE 0.15,0.50 --sigma-cache $C/sigma_2808a --dm-cache $C/dmw --up y:1e-2,x:1e-2
ml 2 2808a --inputs $R/inputs/mos2_hfo2 --dE 0.15,0.50 --sigma-cache $C/sigma_2808a --dm-cache $C/dmw --up y:1e-2,x:1e-2
ml 4 9072a --inputs $R/inputs/mos2_hfo2_9072a --dE 0.15,0.50 --sigma-cache $C/9072a/sigma --dm-cache $C/9072a/dmw \
  --up y:1e-2,x:1e-2,x:1e-2,x:1e-2,x:1e-2
ml 4 9072a_free2 --inputs $R/inputs/mos2_hfo2_9072a --dE 0.15 --sigma-cache $C/9072a/sigma --dm-cache $C/9072a/dmw \
  --up y:1e-2,x:1e-2,x:1e-2,x:1e-2,x:1e-2 --free-levels 2
