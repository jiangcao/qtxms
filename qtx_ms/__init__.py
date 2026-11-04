"""Mode space as the coarse level of a multigrid transport solver.

Library modules
  device    - effective-mass rough-sheet model, analytic leads, RGF
  em        - effective-mass bases (CMS, LDM) and two-level solver
  abtb      - ab initio / TB readers, TBDevice, Sancho-Rubio leads, RGF, bases
  twolevel  - generic two-level preconditioner + flexible GMRES
  prep      - input preparation: generic input set (xyz, norb, H, S, lead cell, meta),
              Si nanosheet structure + sp3s* / sp3d5s* TB  (python -m qtx_ms.prep.si_sheet)

Drivers (run from the repo root with `python -m qtx_ms.<name>`)
  run, scaling, make_test_tb, run_tb
"""
