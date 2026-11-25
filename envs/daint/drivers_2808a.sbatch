#!/bin/bash
# Single-purpose drivers (mlbasis, mlsolve, mlmode, mlexact) on 2808a, 4 GPUs, over the exact-T grid Ec + 0.01 .. 0.50 (50 energies;
# user, 2026-11-25: "more energies, where we have comparison with old result"). Device store; fresh window rho and bases.
#  exact    mlexact, one GPU                                    against results_ms/exact/mos2_hfo2.json (T_exact)
#  dm       dm 1e-2, y2 x (options of mlevel_grid.sbatch, job 5446160: T, its, mode-space T of every level at 50 energies):
#           mlbasis, mlsolve (stored blocks, --fine-blocks 0), mlmode --levels all
#  dmc      dmc 1e-5, Schur (6 nm settings): mlbasis, mlsolve K = 1, mlsolve K = 2 with streamed smoothers, mlmode K = 1
# The comparison tables are printed by the Python part at the end.
# Two debug jobs: PART=a exact + dm, PART=b dmc.
#   A=$(PART=a sbatch --parsable envs/daint/drivers_2808a.sbatch); PART=b sbatch --dependency=afterany:$A envs/daint/drivers_2808a.sbatch   (repo root; code copy dist_dev6)
#SBATCH --job-name=drivers-2808a
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
R=$PWD; CODE=${CODE:-$R/dist_dev6}; O=$R/results_ms/daint; C=$R/cache_daint; J=$SLURM_JOB_ID
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
source envs/daint/activate.sh
IN=$R/inputs/mos2_hfo2; SG=$C/sigma_2808a; W=$C/devstore_2808a; ST=$W/store; P=$O/drivers_2808a_$J
PY=$R/.venv-daint/bin/python; GRID=${GRID:-$(LC_ALL=C seq -f %.2f 0.01 0.01 0.50 | paste -sd,)}
PART=${PART:-ab}; mkdir -p $O; cd $CODE
echo "$(hostname) job $J, code $CODE"; $PY -c "import qtx_ms; print(qtx_ms.__file__)"
COM="--inputs $IN --gpu --sigma-cache $SG --dev-store $ST --cap-split 2,1 --dist nccl --no-share-points"
DM="$COM --dm-cache $W/dmw_drvdm_$J --l1 dm:1e-2 --up y:1e-2,x:1e-2 --no-schur --dm-keep 1e-4"
DMC="$COM --dm-cache $W/dmw_drvdmc_$J --up y:1e-5,x:1e-5"
run() {  # name, module, options
  local n=$1 m=$2 t0=$SECONDS; shift 2
  QTX_RANK_LOG=${P}_$n.rank $R/.venv-daint/bin/mpiexec -launcher fork -n 4 $R/envs/daint/rank.sh python -u -m qtx_ms.$m "$@" > ${P}_$n.log 2>&1
  echo "=== $n ($m) exit $? after $(( SECONDS - t0 )) s"
  grep -E "levels 1\.\.|levels 2\.\.|N 7335|N 9297|NOT CONV|Error|^done," ${P}_$n.log | tail -4 | cut -c1-160
}
if [[ $PART == *a* ]]; then
t0=$SECONDS
$PY -u -m qtx_ms.mlexact --inputs $IN --gpu --sigma-cache $SG --dE $GRID --out ${P}_exact.json > ${P}_exact.log 2>&1
echo "=== exact exit $? after $(( SECONDS - t0 )) s"; grep -E "Error|done" ${P}_exact.log
run dm_basis mlbasis $DM --out ${P}_dm_basis.json
run dm_solve mlsolve $DM --dE $GRID --maxit 60 --chunk 10 --out ${P}_dm_solve.json
run dm_mode mlmode $DM --dE $GRID --levels all --out ${P}_dm_mode.json
fi
if [[ $PART == *b* ]]; then
run dmc_basis mlbasis $DMC --fine-blocks 1 --out ${P}_dmc_basis.json
run dmc_solve_k1 mlsolve $DMC --fine-blocks 1 --free-levels 2 --dE $GRID --maxit 60 --chunk 10 --out ${P}_dmc_solve_k1.json
run dmc_solve_k2s mlsolve $DMC --fine-blocks 2 --stream-smoother-gb 0 --dE $GRID --maxit 60 --chunk 10 --out ${P}_dmc_solve_k2s.json
run dmc_mode mlmode $DMC --fine-blocks 1 --levels all --dE $GRID --out ${P}_dmc_mode.json
run dmc_basis2 mlbasis $DMC --fine-blocks 1
fi
echo "##### comparison"
$PY - $P $R <<'PY'
import json, sys
import numpy as np
P, R = sys.argv[1:]
j = lambda f: json.load(open(f))
ex = j(f"{R}/results_ms/exact/mos2_hfo2.json")
old = j(f"{R}/results_ms/daint/grid_2808a_5446160.json")
try:
    ex2 = j(P + "_exact.json")
except FileNotFoundError:
    ex2 = None
def rel(a, b): return abs(a - b) / max(abs(b), 1e-12)
if ex2:
    ks = [k for k in sorted(ex2["E"]) if k in ex["E"]]
    print(f"exact: {len(ks)} energies, mlexact vs results_ms/exact: max rel dT {max(rel(ex2['E'][k]['T_exact'], ex['E'][k]['T_exact']) for k in ks):.1e}; "
          f"vs job 5446160: {max(rel(ex2['E'][k]['T_exact'], old['E'][k]['T_exact']) for k in ks if k in old['E']):.1e}; Ec {ex2['Ec']} / {ex['Ec']}")
def solved(tag, withold):
    try:
        r = j(f"{P}_{tag}.json")
    except FileNotFoundError:
        return
    ks = sorted(r["E"])
    dt = [rel(r["E"][k]["T"], ex["E"][k]["T_exact"]) for k in ks]
    its = [max(r["E"][k]["its"]) for k in ks]
    print(f"{tag}: {len(ks)} energies, converged {sum(r['E'][k]['converged'] for k in ks)}, N {r['E'][ks[0]]['N']}, max rel dT vs exact {max(dt):.1e} "
          f"(at {ks[int(np.argmax(dt))]}), its max {max(its)} mean {np.mean(its):.1f}, energy time mean {np.mean([r['E'][k]['t']['energy'] for k in ks]):.1f} s, peak {r['E'][ks[-1]].get('peak_gb_per_rank')}")
    if withold:
        d = [(k, max(r["E"][k]["its"]) - max(old["E"][k]["runs"][-1]["its"]), rel(r["E"][k]["T"], old["E"][k]["runs"][-1]["T"])) for k in ks if k in old["E"]]
        print(f"    vs job 5446160 (same basis options): max |d its| {max(abs(v[1]) for v in d)}, max rel dT {max(v[2] for v in d):.1e}, N old {[l['N'] for l in old['E'][ks[0]]['levels']]}")
def mode(tag, withold):
    try:
        r = j(f"{P}_{tag}.json")
    except FileNotFoundError:
        return
    ks = sorted(r["E"])
    for q, l in enumerate(r["E"][ks[0]]["levels"]):
        dt = [rel(r["E"][k]["levels"][q]["T"], ex["E"][k]["T_exact"]) for k in ks]
        line = f"{tag} level {l['level']} (N {l['N']}): rel T error vs exact median {np.median(dt):.1e} max {max(dt):.1e}"
        if withold:
            m = [(rel(r["E"][k]["levels"][q]["T"], [w["T"] for w in old["E"][k]["mode_space"] if w["level"] == l["level"]][0]), k) for k in ks
                 if k in old["E"] and any(w["level"] == l["level"] for w in old["E"][k]["mode_space"])]
            line += f"; vs mode-space T of job 5446160: max {max(m)[0]:.1e} over {len(m)} energies"
        print(line)
    print(f"    {tag}: energy time mean {np.mean([r['E'][k]['t']['energy'] for k in ks]):.1f} s, levels once {r['E'][ks[0]]['t_levels']:.0f} s")
solved("dm_solve", True); mode("dm_mode", True)
solved("dmc_solve_k1", False); solved("dmc_solve_k2s", False); mode("dmc_mode", False)
PY
