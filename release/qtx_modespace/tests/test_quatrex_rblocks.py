"""read_h5 returns the Gamma-point matrix sum_R M_R when the transverse images are stored as separate R blocks.

    PYTHONPATH=src python -m pytest tests -q
"""
import numpy as np
import scipy.sparse as sp
import pytest

from qtx_modespace.quatrex import read_h5

h5py = pytest.importorskip("h5py")


def write_blocks(f, blocks):
    with h5py.File(f, "w") as h:
        for k, M in blocks.items():
            c = sp.coo_matrix(M)
            g = h.create_group(k)
            g.attrs["shape"] = c.shape
            g["row"], g["col"], g["data"] = c.row, c.col, c.data


def test_sum_of_R_blocks(tmp_path):
    rng = np.random.default_rng(0)
    n = 12
    M0 = sp.random(n, n, 0.3, random_state=1) + 0j
    Mp = sp.random(n, n, 0.1, random_state=2) * (1 + 0.5j)
    Mm = Mp.conj().T                                       # M_{-R} = M_R^H
    f = tmp_path / "hamiltonian.h5"
    write_blocks(f, {"[0, 0, 0]": M0, "[0, 1, 0]": Mp, "[0, -1, 0]": Mm})
    got = read_h5(f, log=lambda s: None)
    assert abs(got - (M0 + Mp + Mm)).max() < 1e-14


def test_single_block_unchanged(tmp_path):
    M0 = sp.random(8, 8, 0.4, random_state=3)
    f = tmp_path / "overlap.h5"
    write_blocks(f, {"[0, 0, 0]": M0})
    assert abs(read_h5(f, log=lambda s: None) - M0).max() == 0


def test_missing_gamma_block(tmp_path):
    f = tmp_path / "h.h5"
    write_blocks(f, {"[0, 1, 0]": sp.eye(3)})
    with pytest.raises(AssertionError):
        read_h5(f, log=lambda s: None)
