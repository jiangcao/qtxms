"""Prepared inputs (quatrex.save_inputs / load_inputs, python -m qtx_modespace.prep): round trip of a read_quatrex
dict, option checks, and read_quatrex accepting the prepared folder.

    PYTHONPATH=src python -m pytest tests -q
"""
import numpy as np
import scipy.sparse as sp
import pytest

from qtx_modespace.quatrex import read_quatrex, save_inputs, load_inputs, is_prepared

quiet = lambda *s: None


def fake_inp(complex_=False, bias=False, seed=0):
    rng = np.random.default_rng(seed)
    nat, nb = 24, 4
    norb = rng.integers(1, 4, nat)
    n = int(norb.sum())
    def herm():
        M = sp.random(n, n, 0.2, random_state=rng.integers(1 << 30)).tocsr()
        if complex_:
            M = M + 1j * sp.random(n, n, 0.2, random_state=rng.integers(1 << 30)).tocsr()
        return (M + M.conj().T).tocsr()
    H, S = herm(), herm()
    lead = dict(H00=H[:5, :5], H01=H[:5, 5:10], S00=S[:5, :5], S01=S[:5, 5:10])
    meta = dict(name="fake", axis=0, periodic=[dict(axis=1, length=10.0)], block_length=3.0, cells_per_block=2,
                nblocks=nb, coupling_range=2.5, trimmed=None, lead_checks={},
                transport=dict(device_blocks=[1, 2], lead_left_block=0, lead_right_block=3),
                edges=dict(Ev_lead=-1.0, Ec_lead=0.5, mid_gap=0.0, right_lead_shift=0.1 if bias else 0.0))
    return dict(labels=np.array(["Mo", "S"] * (nat // 2)), xyz=rng.random((nat, 3)), norb=norb,
                blocks=np.repeat(np.arange(nb), nat // nb), H=H, S=S, lead=lead,
                lead_right={k: 2 * v for k, v in lead.items()} if bias else None, perm=rng.permutation(n), meta=meta)


@pytest.mark.parametrize("complex_,bias", [(False, False), (True, True)])
def test_round_trip(tmp_path, complex_, bias):
    inp = fake_inp(complex_, bias)
    save_inputs(inp, tmp_path / "prep", options=dict(cells=2, trim=False, potential=bias), log=quiet)
    assert is_prepared(tmp_path / "prep")
    got = read_quatrex(tmp_path / "prep", log=quiet)                    # the prepared folder in place of the device
    for k in ("labels", "xyz", "norb", "blocks", "perm"):
        assert np.array_equal(got[k], inp[k]), k
    for k in ("H", "S"):
        assert got[k].dtype == inp[k].dtype and abs(got[k] - inp[k]).max() == 0, k
    for pre in ("lead", "lead_right"):
        if inp[pre] is None:
            assert got[pre] is None
            continue
        for k, M in inp[pre].items():
            assert got[pre][k].shape == M.shape and abs(got[pre][k] - M).max() == 0, (pre, k)
    assert got["meta"] == inp["meta"]


def test_option_mismatch(tmp_path):
    save_inputs(fake_inp(), tmp_path / "prep", options=dict(cells=None, trim=False, potential=False), log=quiet)
    load_inputs(tmp_path / "prep", cells=2, log=quiet)                  # the stored cells per block: accepted
    for kw in (dict(cells=3), dict(trim=True), dict(potential=True)):
        with pytest.raises(ValueError):
            load_inputs(tmp_path / "prep", log=quiet, **kw)


def test_partial_folder_is_not_prepared(tmp_path):
    d = tmp_path / "prep"
    save_inputs(fake_inp(), d, log=quiet)
    (d / "prep.json").unlink()
    assert not is_prepared(d)
