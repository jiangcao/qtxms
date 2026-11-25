#!/bin/bash
# 6 nm / 12 nm MoS2/HfO2, density-matrix hierarchy on the four GPUs of one node (mlevel --dist nccl): 16 A windows,
# 4 / 8 y strips of 16.5 A, dmc 1e-5, keep 1e-6, Schur, 6 / 7 merges, A1 / A2 by projection, Ec + 0.15, no exact psi.
# Operators and smoothers resident on the GPUs (--stream-gb 40). Window rho, Sigma and the bases of every finished level
# are stored in cache_daint/<dev>_store: a job stopped by its limit is resumed by the next. H, S come from the device store
# (qtx_ms.devstore, built by the job when missing): every rank holds the rows of its own windows.
# Code: dist_dev3/ (copy of qtx_ms). .mon: CPU-only memory of the node (NUMA nodes with CPUs, HBM excluded) and RSS of the ranks. T reference: 6 nm 0.907406 (attelas2 exact psi), 12 nm 1.868165 (--obc beyn).
#   sbatch -p debug -t 00:30:00 envs/daint/mlevel_wide_dist.sbatch 6nm
#   sbatch --dependency=afterany:<job> envs/daint/mlevel_wide_dist.sbatch 12nm
#SBATCH --job-name=mlevel-wide-4g
#SBATCH --account=lp82
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=.tools/%x-%j.out
cd "$SLURM_SUBMIT_DIR"
D=${1:?device 6nm | 12nm}
R=$PWD; O=$R/results_ms/daint; J=$SLURM_JOB_ID
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
case $D in
  6nm)  C=$R/cache_daint/6nm_store; SG=$R/cache_daint/6nm_dist/sigma; IN=$R/inputs/mos2_hfo2_6nm; NY=4; UP=y:1e-5,x:1e-5,y:1e-5,x:1e-5,x:1e-5,x:1e-5 ;;
  12nm) C=$R/cache_daint/12nm_store; SG=$R/cache_daint/12nm/sigma; IN=$R/cache_daint/12nm/prep; NY=8; UP=y:1e-5,x:1e-5,y:1e-5,x:1e-5,y:1e-5,x:1e-5,x:1e-5 ;;
  *) echo "unknown device $D"; exit 2 ;;
esac
test -d $IN || { echo "no inputs $IN"; exit 1; }
mkdir -p $SG $C/dmw $O; ST=$C/store; CODE=${CODE:-$R/dist_dev3}
L=$O/mlevel_${D}_4g_$J
( cd $CODE; exec $R/.venv-daint/bin/python -u - <<'PY'
import os, time, subprocess
from qtx_ms.devstore import hostmem
while True:
    g = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' '", shell=True, capture_output=True, text=True).stdout.strip()
    r = []
    for p in os.listdir("/proc"):
        try:
            if p.isdigit() and b"qtx_ms.mlevel" in open(f"/proc/{p}/cmdline", "rb").read() and os.path.basename(open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")[0]).startswith(b"python"):
                d = {l.split(":")[0]: int(l.split()[1]) for l in open(f"/proc/{p}/status") if l.startswith("Rss")}
                r.append((d["RssAnon"] / 2**20, d["RssFile"] / 2**20))
        except OSError:
            pass
    print(time.strftime("%T"), "CPU memory free %.0f GiB" % hostmem()["cpu"], "ranks anon/file GiB", " ".join("%.1f/%.1f" % v for v in r), "sum anon %.1f" % sum(v[0] for v in r), "GPU MiB", g, flush=True)
    time.sleep(10)
PY
) > $L.mon 2>&1 &
MON=$!
cd $CODE
echo "$(hostname) job $J, code $CODE"; $R/.venv-daint/bin/python -c "import qtx_ms; print(qtx_ms.__file__)"
test -f $ST/store.json || { time $R/.venv-daint/bin/python -u -m qtx_ms.devstore --inputs $IN --out $ST; } || exit 1
QTX_RANK_LOG=$L.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.mlevel \
  --dist nccl --inputs $IN --dev-store $ST --no-share-points --dE ${DE:-0.15} --gpu --sigma-cache $SG --dm-cache $C/dmw --cap 16,2,2 \
  --cap-split $NY,1 --cap-tov 6 --cap-solver rows64 --l1 dmc:1e-5 --dm-keep 1e-6 --schur --schur-prec double \
  --up $UP --halo 1 --halo-smooth full --sweep exact --exact-max 100000 \
  --chunk 20 --maxit 60 --free-levels 2 --stream-gb 40 --mu=-0.15,0.8 --out $L.json > $L.log 2>&1
rc=$?
kill $MON
echo "[$(date +%T)] $(hostname) exit $rc"; exit $rc
