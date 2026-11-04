"""Input preparation: build structures and TB Hamiltonians, write the generic input set (inputs.py).

  inputs       - write_inputs / read_inputs: xyz, norb, blocks, H, S, lead cell, meta.json
  si_sheet     - CLI: rough Si nanosheet + oxide, sp3s* or sp3d5s*  (python -m qtx_ms.prep.si_sheet)
  structure    - Si / oxide / Au atomic structures (open device, rings, periodic references)
  hamiltonian  - sp3s* Si + Harrison oxide + Au;  hamiltonian5, sp3d5s - sp3d5s* Si
  model        - model switch (QT_MODEL, XSEC), lead cell, band edges
Ab initio models enter at the same point: write their H, S, xyz, norb with inputs.write_inputs.
"""
