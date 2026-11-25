#!/bin/bash
# 6 nm on 4 GPUs with the single-purpose drivers (user, 2026-11-25): MODE=basis: mlbasis (window rho, 6 merges) into a fresh cache;
# MODE=solve: mlsolve with it at 8 energies that have an exact T (results_ms/gpu/ms_ycut/6nm_l1_1e-8/transmission.json),
# --fine-blocks 2, smoothers streamed from pinned host memory. Options of mlevel_wide_dist.sbatch 6nm (T 0.9074056942 at
# Ec + 0.15, its 31-37 at level 7, N 124468 .. 15357, job 5454090). Lead Sigma of the new energies is computed (~10 s each).
#   A=$(MODE=basis sbatch --parsable envs/daint/drivers_6nm.sbatch); MODE=solve sbatch --dependency=afterok:$A envs/daint/drivers_6nm.sbatch
# MODE=rerun: the energies that did not converge in job 5454678: mlsolve at Ec + 0.40, 0.50 with --maxit 150, then the study driver (mlevel, exact level 7
# only, smoothers on the GPU) at 0.50 on the same bases: driver or method?   MODE=rerun sbatch envs/daint/drivers_6nm.sbatch
#SBATCH --job-name=drivers-6nm
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
MODE=${MODE:?basis | solve}
R=$PWD; O=$R/results_ms/daint; J=$SLURM_JOB_ID; CODE=${CODE:-$R/dist_dev6}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
C=$R/cache_daint/6nm_store; SG=$R/cache_daint/6nm_dist/sigma; IN=$R/inputs/mos2_hfo2_6nm; ST=$C/store; DMW=${DMW:-$C/dmw_drv}
L=$O/drivers_6nm_${MODE}_$J; mkdir -p $DMW $O
( cd $CODE; exec $R/.venv-daint/bin/python -u - <<'PY'
import os, time, subprocess
from qtx_ms.devstore import hostmem
while True:
    g = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' '", shell=True, capture_output=True, text=True).stdout.strip()
    r = []
    for p in os.listdir("/proc"):
        try:
            if p.isdigit() and b"qtx_ms.ml" in open(f"/proc/{p}/cmdline", "rb").read() and os.path.basename(open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")[0]).startswith(b"python"):
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
echo "$(hostname) job $J, $MODE, code $CODE, dm cache $DMW"; $R/.venv-daint/bin/python -c "import qtx_ms; print(qtx_ms.__file__)"
B="--inputs $IN --gpu --sigma-cache $SG --dm-cache $DMW --dev-store $ST --dist nccl --no-share-points --cap-split 4,1 \
  --up y:1e-5,x:1e-5,y:1e-5,x:1e-5,x:1e-5,x:1e-5 --mu=-0.15,0.8 --fine-blocks 2"
if [ $MODE = basis ]; then
  X="mlbasis $B"
elif [ $MODE = rerun ]; then
  X="mlsolve $B --stream-smoother-gb 0 --stream-gb 40 --dE 0.40,0.50 --chunk 20 --maxit 150 --fgmres-log 10"
else
  X="mlsolve $B --stream-smoother-gb 0 --stream-gb 40 --dE ${DE:-0.15,0.05,0.50,0.30,0.20,0.10,0.40,0.25} --chunk 20 --maxit 60"
fi
QTX_RANK_LOG=$L.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.$X --out $L.json > $L.log 2>&1
rc=$?
if [ $MODE = rerun ]; then
  echo "mlsolve exit $rc after $SECONDS s"
  QTX_RANK_LOG=${L}_study.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.mlevel $B \
    --cap 16,2,2 --cap-tov 6 --cap-solver rows64 --l1 dmc:1e-5 --dm-keep 1e-6 --schur --schur-prec double --halo 1 --halo-smooth full \
    --sweep exact --exact-max 16000 --stream-gb 40 --dE 0.50 --chunk 20 --maxit 150 --fgmres-log 10 --out ${L}_study.json > ${L}_study.log 2>&1
  echo "study driver exit $? after $SECONDS s"
  grep -E "^Ec|exact level|mode space|levels at|residual est|Error|^done" ${L}_study.log | cut -c1-170
  grep -E "residual est" $L.log | cut -c1-150
fi
kill $MON
echo "exit $rc"
grep -E "levels|^Ec|NOT CONV|peak per rank|Error|^done|  N " $L.log | cut -c1-200
awk '/CPU memory free/ {if (m == "" || $5 < m) {m = $5; l = $0}} END {print "least CPU memory free:", l}' $L.mon
exit $rc
