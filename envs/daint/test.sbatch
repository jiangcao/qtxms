#!/bin/bash
# Env check on one daint compute node (nothing runs on the login node): sbatch envs/daint/test.sbatch  (repo root)
#SBATCH --job-name=dmbase-test
#SBATCH --account=lp82
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --output=.tools/test-%j.out
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=16
source envs/daint/activate.sh
hostname; nvidia-smi -L
python - <<'PY'
import time, cupy as cp
for k in range(cp.cuda.runtime.getDeviceCount()):
    with cp.cuda.Device(k):
        a = cp.random.rand(4000, 4000) + 1j * cp.random.rand(4000, 4000) + 60 * cp.eye(4000)
        t = time.time(); x = cp.linalg.solve(a, a); cp.cuda.Stream.null.synchronize()
        print("GPU", k, "solve 4000: %.2f s" % (time.time() - t), "err %.1e" % float(abs(x - cp.eye(4000)).max()))
PY
python -m pytest -q -rs tests; r1=$?
python -m pytest -q -rs release/qtx_modespace/tests; r2=$?
echo "rc tests $r1, release $r2"; exit $((r1 | r2))
