#!/bin/bash
# --fine-blocks (S_k, H_k formed from the sparse fine core-pair blocks on the GPU, never stored) on daint, 4 GPUs (user,
# 2026-11-25): (1) tests/dist_mpi.py nccl (the strip, all cases incl. fine1, fine2, fine4); (2) 2808a with the device store,
# settings of devstore_test.sbatch, Ec + 0.15, 0.50: K = 0, 1, 2, fresh window rho and bases each; same N, its, T expected.
#   sbatch envs/daint/fine_blocks_test.sbatch   (repo root; code copy dist_dev4)
#SBATCH --job-name=fine-blocks-test
#SBATCH --account=lp82
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
R=$PWD; CODE=${CODE:-$R/dist_dev4}; O=$R/results_ms/daint; C=$R/cache_daint; J=$SLURM_JOB_ID
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
IN=$R/inputs/mos2_hfo2; SG=$C/sigma_2808a; W=$C/devstore_2808a; ST=$W/store
cd $CODE
echo "$(hostname) job $J, code $CODE"; $R/.venv-daint/bin/python -c "import qtx_ms; print(qtx_ms.__file__)"
QTX_TEST_SIGMA=$W/strip_sigma_$J timeout 420 $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python tests/dist_mpi.py nccl 2>&1 | tail -25
for K in 0 1 2; do
  L=$O/fine_blocks_2808a_k${K}_$J; DM=$W/dmw_fb${K}_$J; mkdir -p $DM; t0=$SECONDS
  QTX_RANK_LOG=$L.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.mlevel \
    --dist nccl --inputs $IN --dev-store $ST --fine-blocks $K --dE 0.15,0.50 --gpu --sigma-cache $SG --dm-cache $DM --exact --cap 16,2,2 \
    --cap-split 2,1 --cap-tov 6 --cap-solver rows64 --l1 dmc:1e-5 --dm-keep 1e-6 --schur --schur-prec double --up y:1e-5,x:1e-5 \
    --halo 1 --halo-smooth full --sweep exact --exact-max 100000 --chunk 10 --maxit 60 --host-blocks --no-share-points --out $L.json > $L.log 2>&1
  echo "=== K=$K exit $? after $(( SECONDS - t0 )) s"
  grep -E "core-pair|device store|^Ec|levels at|exact level|peak per rank|Error|done," $L.log | cut -c1-200
done
