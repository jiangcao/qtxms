#!/bin/bash
# Device store (qtx_ms.devstore, mlevel --dev-store) on 2808a (user, 2026-11-25): the dmc 1e-5 + Schur settings of
# dmc_dist.sbatch on 4 GPUs, window rho and bases built fresh in both runs (own dm cache each): full matrices (ref) and
# sliced store (store). Same N, its, T / psi errors expected; earlier run: results_ms/daint/dmc1e-5_schur_2808a_4g_5448557.log.
#   sbatch envs/daint/devstore_test.sbatch   (repo root; code copy dist_dev3)
#SBATCH --job-name=devstore-test
#SBATCH --account=lp82
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
R=$PWD; CODE=${CODE:-$R/dist_dev3}; O=$R/results_ms/daint; C=$R/cache_daint; J=$SLURM_JOB_ID
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
IN=$R/inputs/mos2_hfo2; SG=$C/sigma_2808a; W=$C/devstore_2808a; ST=$W/store
mkdir -p $O $W; cd $CODE
echo "$(hostname) job $J, code $CODE"; $R/.venv-daint/bin/python -c "import qtx_ms; print(qtx_ms.__file__)"
test -f $ST/store.json || $R/.venv-daint/bin/python -m qtx_ms.devstore --inputs $IN --out $ST || exit 1
for v in ref store; do
  X=; test $v = store && X="--dev-store $ST"
  L=$O/devstore_2808a_${v}_$J; DM=$W/dmw_${v}_$J; mkdir -p $DM; t0=$SECONDS
  QTX_RANK_LOG=$L.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.mlevel \
    --dist nccl --inputs $IN --dE ${DE:-0.15,0.50} --gpu --obc folded --sigma-cache $SG --dm-cache $DM --exact --cap 16,2,2 \
    --cap-split 2,1 --cap-tov 6 --cap-solver rows64 --l1 dmc:1e-5 --dm-keep 1e-6 --schur --schur-prec double --up y:1e-5,x:1e-5 \
    --halo 1 --halo-smooth full --sweep exact --exact-max 100000 --chunk 10 --maxit 60 --host-blocks $X --out $L.json > $L.log 2>&1
  echo "=== $v exit $? after $(( SECONDS - t0 )) s"
  grep -E "numa|domain decomp|device store|^Ec|^  level|levels at|exact level|peak per rank|Error|done," $L.log | cut -c1-220
done
