"""qtx_ms.prep.offstate: OFF-state profile (source 0, barrier Vb on the gate, drain -Vds, flat ends) and the device folder."""
import json
import numpy as np
import pytest

from qtx_ms.prep import offstate


def test_profile():
    x = np.linspace(0.0, 380.0, 3801)
    V = offstate.profile(x, 0.2, 0.4, 126.7, 253.4, 31.7)
    assert np.all(V[x <= 95.0] == 0.0) and np.all(V[x >= 285.1] == -0.4)
    gate = (x >= 126.7) & (x <= 253.4)
    assert np.allclose(V[gate], 0.2) and V.max() == pytest.approx(0.2)
    assert np.all(np.diff(V[x <= 126.7]) >= 0) and np.all(np.diff(V[x >= 253.4]) <= 0)


def test_folder(tmp_path):
    src = tmp_path / "dev"
    (src / "inputs").mkdir(parents=True)
    x = np.arange(0.0, 380.0, 2.0)
    with open(src / "inputs" / "structure.xyz", "w") as f:
        f.write(f"{len(x)}\nLattice\n" + "".join(f"S {v:.3f} 0.0 0.0\n" for v in x))
    for g in ("hamiltonian.h5", "overlap.h5"):
        (src / "inputs" / g).write_text("")
    (src / "quatrex_config.toml").write_text("")
    offstate.main([str(src), "--out", str(tmp_path / "off"), "--Vb", "0.2", "--Vds", "0.4"])
    V = np.load(tmp_path / "off" / "inputs" / "potential.npy")
    assert len(V) == len(x) and V[0] == 0.0 and V[-1] == -0.4 and V.max() == pytest.approx(0.2)
    assert (tmp_path / "off" / "inputs" / "hamiltonian.h5").is_symlink()
    info = json.load(open(tmp_path / "off" / "potential.json"))
    assert info["V_drain"] == -0.4 and info["Vb"] == 0.2
    with pytest.raises(AssertionError):                                   # ramps into the lead ends
        offstate.main([str(src), "--out", str(tmp_path / "bad"), "--Vb", "0.2", "--Vds", "0.4", "--x1", "50"])
