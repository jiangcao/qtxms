#!/bin/bash
# Memory test of mlevel_12nm.sbatch in the debug queue (30 min): same settings with 2 contour points instead of 24 (rho
# cost / 12; level sizes only approximate, bases not usable). GPU 0 = --free-levels 2 --stream-gb 8, GPU 1 = everything
# streamed (--free-levels 0 --stream-gb 2). Own sigma / dm caches per mode in cache_daint/12nm/memtest (delete after).
# The GPU peaks are in the "mem |" lines of the logs; host / GPU use every 15 s in the .mon file. Killed by the limit:
# the level reached is the answer. Needs cache_daint/12nm/prep.
#   sbatch --dependency=afterok:<prep> envs/daint/memtest_12nm.sbatch
#SBATCH --job-name=mem-12nm
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
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
C=$PWD/cache_daint/12nm; T=$C/memtest; O=$PWD/results_ms/daint; J=$SLURM_JOB_ID; mkdir -p $T $O
test -f $C/prep/prep.json || { echo "no prep"; exit 1; }
UP=y:1e-5,x:1e-5,y:1e-5,x:1e-5,y:1e-5,x:1e-5,x:1e-5
run() {  # gpu tag options
  local g=$1 L=$O/memtest_12nm_${J}_$2 D=$T/$2; shift 2
  mkdir -p $D/sigma $D/dmw
  CUDA_VISIBLE_DEVICES=$g taskset -c $(( g * 72 ))-$(( g * 72 + 71 )) python -u -m qtx_ms.mlevel --inputs $C/prep \
    --dE 0.15 --gpu --sigma-cache $D/sigma --dm-cache $D/dmw --npts 2 \
    --cap 16,2,2 --cap-split 8,1 --cap-tov 6 --cap-solver rows64 --l1 dmc:1e-5 --dm-keep 1e-6 --schur \
    --schur-prec double --up $UP --halo 1 --halo-smooth full --sweep exact --exact-max 100000 --chunk 20 --maxit 60 \
    --mu=-0.15,0.8 "$@" --out $L.json > $L.log 2>&1
  rc=$?
  echo "[$(date +%T)] GPU $g $L exit $rc"
}
echo "memtest 12nm on $(hostname), job $J"
( while :; do echo "$(date +%T) host free $(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo) GiB, GPU used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ') MiB"; sleep 15; done ) > $O/memtest_12nm_$J.mon &
MON=$!
run 0 free2  --free-levels 2 --stream-gb 8 & A=$!
run 1 stream --free-levels 0 --host-blocks --stream-gb 2 & B=$!
wait $A $B
kill $MON
