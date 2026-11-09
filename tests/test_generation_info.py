"""Generation provenance, stable ids and lattice weights on Multimer"""
import json
import warnings

import numpy as np
import pydantic
import pytest

import crystalatte as cle
from conftest import cif, load
from crystalatte.core.generation import GenerationInfo, crystal_fingerprint
from crystalatte.core.multimer import Monomer, Multimer, order_name
from crystalatte.core.sym_ops import SymOp, SymOpList


@pytest.fixture
def benzene():
    return load("benzene")


@pytest.fixture
def salicylic_hydroxybenzamide():
    return load("salicylic_hydroxybenzamide")


@pytest.fixture
def ice_states():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*ordered state.*")
        return list(cle.from_cif(cif("ice")))


def test_order_name():
    assert [order_name(n) for n in (1, 2, 3, 6, 7)] == ["monomer", "dimer", "trimer", "hexamer", "7mer"]


def test_ids_stable_across_runs(benzene):
    a = [m.id for m in cle.generate(benzene, N=2, R=8.0, verbose=0)]
    b = [m.id for m in cle.generate(benzene, N=2, R=8.0, verbose=0)]
    assert sorted(a) == sorted(b) and len(set(a)) == len(a)
    assert all(i.startswith("dimer-") and len(i) == len("dimer-") + 16 for i in a)


def test_weight_is_multiplicity_over_N_for_single_reference(benzene):
    for m in cle.generate(benzene, N=3, R=6.0, verbose=0):
        assert m.weight == pytest.approx(m.multiplicity / 3)


def test_generation_info_round_trips(benzene):
    g = cle.generate(benzene, N=2, R=8.0, verbose=0)[0].generation
    assert GenerationInfo.model_validate(json.loads(json.dumps(g.model_dump(mode="json")))) == g
    with pytest.raises(pydantic.ValidationError):
        GenerationInfo.model_validate({**g.model_dump(mode="json"), "z_fu": 0})
    assert (g.N, g.R, g.z_fu) == (2, 8.0, 4)


def test_generation_info_rejects_bad_values(benzene):
    good = cle.generate(benzene, N=2, R=8.0, verbose=0)[0].generation.model_dump(mode="json")
    for bad in ({"N": 1}, {"site_counts": [4, 4]}, {"extra": 1}):
        with pytest.raises(pydantic.ValidationError):
            GenerationInfo.model_validate({**good, **bad})


def test_generation_records_input_crystal_fingerprint(benzene):
    g = cle.generate(benzene, N=2, R=8.0, verbose=0)[0].generation
    assert g.crystal_fingerprint == crystal_fingerprint(benzene)
    assert g.species == ("C6H6",) and g.site_counts == (4,)
    assert g.options == {"use_origin": False, "use_com": False}


def test_cocrystal_z_fu_and_species(salicylic_hydroxybenzamide):
    g = cle.generate(salicylic_hydroxybenzamide, N=2, R=6.0, verbose=0)[0].generation
    # P2_1/c with one salicylic acid and one hydroxybenzamide in the ASU: 4 of each per cell
    assert len(set(g.species)) == 2 and g.z_fu == 4


def _pairs_per_cell(crystal, R):
    """Brute-force count of contact pairs (closest atoms within R) per unit cell"""
    crystal = crystal.detect_symmetry(1e-3)
    cell = [
        op.apply(ref.frac_coords)
        for ref in crystal.get_references()
        for op in crystal.space_group.get_unique_sym_ops(ref.centroid_frac)
    ]
    shifts = np.array(list(np.ndindex(7, 7, 7))) - 3
    degree = 0
    for i, home in enumerate(cell):
        a = crystal.to_cartesian(home)
        for j, other in enumerate(cell):
            for t in shifts:
                if i == j and not t.any():
                    continue
                b = crystal.to_cartesian(other + t)
                if np.linalg.norm(a[:, None] - b[None], axis=-1).min() < R:
                    degree += 1
    return degree / 2


def test_cocrystal_dimer_weights_sum_to_pairs_per_formula_unit(salicylic_hydroxybenzamide):
    multimers = cle.generate(salicylic_hydroxybenzamide, N=2, R=6.0, verbose=0)
    z_fu = multimers[0].generation.z_fu
    assert sum(m.weight for m in multimers) == pytest.approx(
        _pairs_per_cell(salicylic_hydroxybenzamide, 6.0) / z_fu
    )


def test_fingerprint_differs_between_disorder_states(ice_states):
    s0, s1 = ice_states
    assert crystal_fingerprint(s0) != crystal_fingerprint(s1)
    assert crystal_fingerprint(s0) == crystal_fingerprint(s0)


def test_weight_without_generation_raises():
    op = SymOp.identity()
    mons = [
        Monomer(["He"], np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(3), np.zeros(3), op),
        Monomer(["He"], np.full((1, 3), 0.5), np.full((1, 3), 3.0), np.full(3, 0.5), np.full(3, 3.0), op),
    ]
    ops = SymOpList([op, op], 1)
    assert Multimer(mons, ops, 1).generation is None
    with pytest.raises(ValueError, match="generate"):
        Multimer(mons, ops, 1).weight


def test_z_fu_override_scales_weights(benzene):
    auto = {m.id: m for m in cle.generate(benzene, N=2, R=6.0, verbose=0)}
    over = cle.generate(benzene, N=2, R=6.0, z_fu=2, verbose=0)
    assert over[0].generation.z_fu == 2 and auto[over[0].id].generation.z_fu == 4
    assert {m.id for m in over} == set(auto)
    for m in over:
        assert m.weight == pytest.approx(2 * auto[m.id].weight)


@pytest.mark.parametrize("z_fu", [0, -4, 3, 8, 2.0, True])
def test_z_fu_override_rejects_bad_values(benzene, z_fu):
    # benzene has 4 C6H6 per cell, so only 1, 2 and 4 are valid formula-unit counts
    with pytest.raises(ValueError, match="z_fu"):
        cle.generate(benzene, N=2, R=6.0, z_fu=z_fu, verbose=0)
