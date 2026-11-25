#!/bin/bash
# dmc selection with Schur folding on 2808a, 9072a, 21600a (user, 2026-11-24), 4 GPUs of one node (--dist nccl, shared
# contour points, distributed bottom solve): the settings of results_ms/gpu/mlevel/run_dmc_grid.sh CUT=1e-5 DMKEEP=1e-6
# (attelas2 logs dmc1e-5_k1e-6_<dev>_y2*.log) with --schur-prec double; dm 1e-2 reference: results_ms/daint/grid_*, dist_*.
# Energies Ec + 0.15, 0.30, 0.45, 0.50; exact T and psi (--exact). Bases built at the first energy, then reused; window
# rho in new folders (tag _core_d). Code: dist_dev2/ (one node: no uenv, NCCL of the venv).
#   sbatch [--dependency=afterany:<job>] envs/daint/dmc_dist.sbatch [devices...]
#SBATCH --job-name=dmc-dist
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
R=$PWD; CODE=${CODE:-$R/dist_dev2}; O=$R/results_ms/daint; C=$R/cache_daint; J=$SLURM_JOB_ID
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
mkdir -p $O; cd $CODE
echo "$(hostname) job $J, code $CODE"; $R/.venv-daint/bin/python -c "import qtx_ms; print(qtx_ms.__file__)"
X=x:1e-5,x:1e-5,x:1e-5,x:1e-5
for d in ${@:-2808a 9072a 21600a}; do
  case $d in
    2808a)  IN=$R/inputs/mos2_hfo2;        SG=$C/sigma_2808a;    DM=$C/dmw;        UP=y:1e-5,x:1e-5 ;;
    9072a)  IN=$R/inputs/mos2_hfo2_9072a;  SG=$C/9072a/sigma;    DM=$C/9072a/dmw;  UP=y:1e-5,$X ;;
    21600a) IN=$R/inputs/mos2_hfo2_21600a; SG=$C/21600a/sigma;   DM=$C/21600a/dmw; UP=y:1e-5,$X ;;
    *) echo "unknown device $d"; exit 2 ;;
  esac
  L=$O/dmc1e-5_schur_${d}_4g_$J; t0=$SECONDS
  QTX_RANK_LOG=$L.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.mlevel \
    --dist nccl --inputs $IN --dE ${DE:-0.15,0.30,0.45,0.50} --gpu --sigma-cache $SG --dm-cache $DM --exact --cap 16,2,2 \
    --cap-split 2,1 --cap-tov 6 --cap-solver rows64 --l1 dmc:1e-5 --dm-keep 1e-6 --schur --schur-prec double --up $UP \
    --halo 1 --halo-smooth full --sweep exact --exact-max 100000 --chunk 10 --maxit 60 --host-blocks --out $L.json > $L.log 2>&1
  rc=$?
  echo "=== $d exit $rc after $(( SECONDS - t0 )) s"
  grep -E "domain decomp|windows per rank|^Ec|^  level|levels at|exact level|peak per rank|Error|done," $L.log | cut -c1-200
done
