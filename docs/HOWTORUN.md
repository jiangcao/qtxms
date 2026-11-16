# How to run

Moved verbatim from CLAUDE.md on 2026-11-15. Update when adding modules or making a release.

## How to run
Environment: `uv sync` (Python 3.12, numpy, scipy, pypardiso/MKL on x86-64; `qtx_ms` and `qtmode` installed
editable). Prefix commands with `uv run`, or activate `.venv`.
```
python -m qtx_ms.run 20     # EM sweep (~5 min)
python -m qtx_ms.scaling    # EM length scaling (~5 min)
python -m qtx_ms.make_test_tb # writes test.xyz, test_H.txt, test_S.txt
python -m qtx_ms.run_tb --xyz test.xyz --H test_H.txt --S test_S.txt --norb "Si:2,H:1" --dE 0.02,0.08,0.15,0.25 --Eb 0.6
python -m qtx_ms.prep.si_sheet --model sp3d5s --xsec 3x5 --out inputs/si3x5_sp3d5s   # ~2 min, writes generic inputs
# density-matrix hierarchy (multilevel FGMRES preconditioner); GPU hosts: .venv-aarch64/bin/python, caches on /scratch
python -m qtx_ms.mlevel --inputs inputs/mos2_hfo2 --dE 0.05 --gpu --sigma-cache /scratch/jiacao/mg_modespace/sigma_2808a \
    --psi-dir results_ms/gpu/wf_gh200/psi --dm-cache /scratch/jiacao/mg_modespace/geneo/dmw --cap 16,2,2 --cap-split 2,1 \
    --cap-tov 6 --cap-solver rows64 --l1 dm:1e-2 --up y:1e-2,x:1e-2 --halo 1 --sweep exact   # 2808a, ~3 min on a GH200
```
