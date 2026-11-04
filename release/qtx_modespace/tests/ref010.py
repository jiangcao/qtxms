"""The 0.1.0 contour recursion, kept as the reference for the regression tests (no pytest needed)."""
import numpy as np


def g_diag_010(blk, z, SL, SR, iL, iR):
    """The 0.1.0 recursion (6 block products per slab), kept as the reference."""
    ns = len(blk["HD"])
    PL, PR = np.eye(len(blk["HD"][0]))[iL], np.eye(len(blk["HD"][-1]))[iR]
    D = [z * blk["SD"][s] - blk["HD"][s] for s in range(ns)]
    D[0] = D[0] - PL.T @ SL @ PL
    D[-1] = D[-1] - PR.T @ SR @ PR
    Up = [z * blk["SU"][s] - blk["HU"][s] for s in range(ns - 1)]
    Lo = [z * blk["SU"][s].conj().T - blk["HU"][s].conj().T for s in range(ns - 1)]
    gl = [None] * ns
    for s in range(ns):
        gl[s] = np.linalg.inv(D[s] if s == 0 else D[s] - Lo[s - 1] @ gl[s - 1] @ Up[s - 1])
    out = [None] * ns
    out[-1] = G = gl[-1]
    for s in range(ns - 2, -1, -1):
        out[s] = G = gl[s] + gl[s] @ Up[s] @ G @ Lo[s] @ gl[s]
    return out
