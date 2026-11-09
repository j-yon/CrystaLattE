from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

import crystalatte as cle
from crystalatte.compute import ComputeSpec, InputFileBackend
from crystalatte.compute.backends.base import manifest_entry
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


COCRYSTAL = Path(__file__).parents[1] / "other/salicylic_hydroxybenzamide/salicylic_hydroxybenzamide.cif"


def _closest_atoms(multimer) -> float:
    return min(
        float(np.linalg.norm(a - b))
        for m1, m2 in combinations(multimer.monomers, 2)
        for a in m1.cart_coords
        for b in m2.cart_coords
    )


def test_min_separation_of_known_geometry():
    assert _hand_built().min_separation == pytest.approx(4.0 - 0.74)


def test_min_separation_is_closest_atom_pair_across_monomers(ammonia):
    for m in cle.generate(ammonia, N=3, R=4.0, verbose=0):
        assert m.min_separation == pytest.approx(_closest_atoms(m))


def test_min_separation_respects_cutoff(benzene):
    assert all(m.min_separation <= 5.0 for m in cle.generate(benzene, N=2, R=5.0, verbose=0))


def test_manifest_entry_describes_the_multimer(ammonia):
    m = cle.generate(ammonia, N=2, R=4.0, verbose=0)[0]
    assert manifest_entry(m) == {
        "N": 2,
        "multiplicity": m.multiplicity,
        "weight": m.weight,
        "geometric_mean": m.geometric_mean,
        "min_separation": m.min_separation,
        "formulas": ["H3N", "H3N"],
    }


def test_manifest_entry_tells_cocrystal_monomers_apart():
    crystal = next(cle.from_cif(str(COCRYSTAL)))
    pairs = {tuple(sorted(manifest_entry(m)["formulas"])) for m in cle.generate(crystal, N=2, R=4.0, verbose=0)}
    assert pairs == {
        ("C7H6O3", "C7H6O3"),
        ("C7H6O3", "C7H7NO2"),
        ("C7H7NO2", "C7H7NO2"),
    }
