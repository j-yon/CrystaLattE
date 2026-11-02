import numpy as np
import pytest

import crystalatte as cle
from crystalatte.compute import ComputeSpec, InputFileBackend
from crystalatte.core.multimer import Monomer, Multimer
from crystalatte.core.sym_ops import SymOp, SymOpList

SPEC = ComputeSpec(program="psi4", method="hf", basis="sto-3g")


def _hand_built():
    def mono(x):
        cart = np.array([[x, 0.0, 0.0], [x + 0.74, 0.0, 0.0]])
        return Monomer(_symbols=["H", "H"], _frac_coords=np.zeros_like(cart), _cart_coords=cart,
                       _centroid_frac=np.zeros(3), _centroid_cart=cart.mean(axis=0))

    return Multimer([mono(0.0), mono(4.0)], SymOpList([SymOp.identity()] * 2, 1))


def test_submit_rejects_multimers_without_generation(tmp_path):
    with pytest.raises(ValueError, match="crystalatte.generate"):
        InputFileBackend(tmp_path).submit([_hand_built()], SPEC)


def test_submit_rejects_empty(tmp_path):
    with pytest.raises(ValueError, match="no multimers"):
        InputFileBackend(tmp_path).submit([], SPEC)


def test_submit_rejects_mixed_crystals(tmp_path, benzene, ammonia):
    ms = cle.generate(benzene, N=2, R=5.0, verbose=0)[:1] + cle.generate(ammonia, N=2, R=5.0, verbose=0)[:1]
    with pytest.raises(ValueError, match="fingerprint"):
        InputFileBackend(tmp_path).submit(ms, SPEC)


def test_submit_rejects_mixed_runs_of_one_order(tmp_path, benzene):
    ms = cle.generate(benzene, N=2, R=5.0, verbose=0)[:1] + cle.generate(benzene, N=2, R=6.0, verbose=0)[-1:]
    with pytest.raises(ValueError, match="different generate"):
        InputFileBackend(tmp_path).submit(ms, SPEC)


def test_submit_rejects_duplicate_ids(tmp_path, benzene):
    m = cle.generate(benzene, N=2, R=5.0, verbose=0)[0]
    with pytest.raises(ValueError, match=m.id):
        InputFileBackend(tmp_path).submit([m, m], SPEC)


def test_input_backend_writes_by_id(tmp_path, benzene):
    ms = cle.generate(benzene, N=2, R=6.0, verbose=0)
    InputFileBackend(tmp_path).submit(ms, SPEC)
    assert sorted(p.stem for p in (tmp_path / "dimers").glob("*.in")) == sorted(m.id for m in ms)


def test_input_backend_status_collect_not_implemented(tmp_path):
    b = InputFileBackend(tmp_path)
    with pytest.raises(NotImplementedError):
        b.status()
    with pytest.raises(NotImplementedError):
        b.collect()


def test_psi4_input_contains_tight_defaults(tmp_path, benzene):
    ms = cle.generate(benzene, N=2, R=5.0, verbose=0)
    InputFileBackend(tmp_path).submit(ms[:1], SPEC)
    text = (tmp_path / "dimers" / f"{ms[0].id}.in").read_text()
    assert "'e_convergence': 10" in text
    assert "'freeze_core': 'true'" in text
