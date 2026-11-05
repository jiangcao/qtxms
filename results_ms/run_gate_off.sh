#!/bin/bash
# OFF-state test: 0.4 eV gate barrier on device slabs 1-2 (inputs/mos2_hfo2_gate0.4, prep.gate_barrier), bases built
# WITHOUT the potential (flat band) reused: slab level 1 cuts 1e-6 / 1e-8 / 1e-10, tile level 1 tobc_t1e-6_lc, slab level 2 cut 5e-14 (n400)
# lead. Exact T on Ec + 0:0.5:0.02 (4 workers x 6 threads, lead Sigma cache of mos2_hfo2 reused), then MS T on the
# 10 meV grid (Sigma cached) vs exact.
cd /usr/scratch/mont-fort15/jiacao/mg_modespace
I=inputs/mos2_hfo2_gate0.4
D=results_ms/gate_off; mkdir -p $D
EC=0.5043489720738419
SIG=results_ms/sigma/mos2_hfo2
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
for w in 0 1 2 3; do
  dE=$(python3 -c "print(','.join(f'{0.02*i:.2f}' for i in range($w, 26, 4)))")
  nice -n 10 uv run python -u -m qtx_ms.exact_T --inputs $I --obc folded --Ec $EC --sigma-cache $SIG --dE $dE \
     --json $D/exact_w$w.json > $D/exact_w$w.log 2>&1 &
done
wait
python3 - <<'PY'
import json
out = None
for w in range(4):
    z = json.load(open(f"results_ms/gate_off/exact_w{w}.json"))
    out = z if out is None else (out["E"].update(z["E"]) or out)
out["E"] = dict(sorted(out["E"].items(), key=lambda kv: float(kv[0])))
json.dump(out, open("results_ms/exact/mos2_hfo2_gate0.4.json", "w"), indent=1)
print(len(out["E"]), "exact energies")
PY
M=results_ms/mos2_hfo2_slabl1_tobc24/modes
for t in 1e-6 1e-8 1e-10; do
  uv run python -m qtx_ms.slab_l1_modes --inputs inputs/mos2_hfo2 --tiles results_ms/mos2_hfo2_slabl1_tobc24/tiles_obc_1e-10 --cut $t --out $M/t$t
done
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for b in $M/t1e-6 $M/t1e-8 $M/t1e-10 results_ms/mos2_hfo2_vcycle/tobc_t1e-6_lc_U1 results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead; do
  tag=$(echo $b | tr '/' '_')
  nice -n 10 uv run python -u -m qtx_ms.ms_scan --inputs $I --obc folded --Ec $EC --sigma-cache $SIG --modes $b \
     --dE 0:0.5:0.01 --no-exact --exact results_ms/exact/mos2_hfo2_gate0.4.json --out $D/scan_$tag.json > $D/scan_$tag.log 2>&1 &
done
wait
