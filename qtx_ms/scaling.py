import sys, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from qtx_ms.device import Device
from qtx_ms.em import basis_ldm, assemble_U, TwoLevel, transmission_from_psi
for L in (10.0, 20.0, 40.0):
    dev = Device(L=L); H = dev.full_H()
    eps, _ = dev.lead_modes(); e1 = eps[0] - 2 * dev.t
    energies = e1 + np.array([0.04, 0.10, 0.16])
    beta = 1.05 * spla.eigsh(H, k=1, which="LA", return_eigenvectors=False)[0] - energies.min()
    U = assemble_U(basis_ldm(dev, e1 + 1.0, thr=1e-3, Hfull=H))
    for name, UU, al in (("none", sp.csr_matrix((dev.n, 0)), e1 + 0.3), ("LDM Eb=1", U, e1 + 1.0)):
        its, mv = [], []
        for E in energies:
            B, Sig = dev.gamma_inj(E)
            tl = TwoLevel(dev, H, UU, E, Sig, cheb_deg=8, Emax=al, beta=beta)
            for c in range(B.shape[1]):
                x, h, _ = tl.solve(B[:, c].copy()); its.append(len(h) - 1)
            mv.append(tl.nmv / B.shape[1])
        print(f"L={L:4.0f} nm n={dev.n:6d} {name:9s} N={UU.shape[1]:5d} its mean {np.mean(its):5.1f} max {max(its):3d} matvec/rhs {np.mean(mv):5.0f}", flush=True)
