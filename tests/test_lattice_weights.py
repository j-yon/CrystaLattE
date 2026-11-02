"""Per-formula-unit weight vs an independent P1 brute force, using a toy nonadditive energy"""
import warnings

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import crystalatte as cle
from conftest import load
from crystalatte.core.crystal import Crystal, expand_cell
from crystalatte.core.multimer import ASU, Multimer
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp

P1 = SpaceGroup.from_sym_ops([SymOp.identity()], number=1)


def toy_energy(m: Multimer) -> float:
    """Permutation- and rigid-motion-invariant stand-in for a nonadditive energy

    Sum over inter-monomer atom pairs of 1/d**6 (Cartesian, Å)
    """
    coords = [mono.cart_coords for mono in m.monomers]
    total = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            total += float(np.sum(cdist(coords[i], coords[j]) ** -6))
    return total


def symmetric_total(crystal, N, R):
    multimers = cle.generate(crystal, N=N, R=R, verbose=0)
    total = sum(m.weight * toy_energy(m) for m in multimers)
    return total, multimers[0].generation.z_fu


def p1_total(crystal, N, R, z_fu):
    """Energy per formula unit from a symmetry-free run over the whole cell"""
    detected = crystal.detect_symmetry(1e-3)
    frac, symbols, _ = expand_cell(
        detected._asu.positions, detected._asu.atoms, detected.space_group.sym_ops
    )
    p1 = Crystal(*detected.lattice_parameters, P1, ASU([str(s) for s in symbols], frac), detected.state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        multimers = cle.generate(p1, N=N, R=R, detect_symmetry=False, verbose=0)
    return sum(m.multiplicity / N * toy_energy(m) for m in multimers) / z_fu


@pytest.mark.slow
@pytest.mark.parametrize("fixture, N, R", [
    ("benzene", 2, 6.0), ("benzene", 3, 4.5),  # Z'=1/2, molecule on an inversion centre
    ("pyrazole", 2, 6.0),  # Z'>1
    ("salicylic_hydroxybenzamide", 2, 6.0),  # cocrystal
])
def test_per_formula_unit_matches_p1_brute_force(fixture, N, R):
    c = load(fixture)
    sym, z_fu = symmetric_total(c, N, R)
    assert sym == pytest.approx(p1_total(c, N, R, z_fu), rel=1e-9)


def test_weight_with_unequal_site_counts():
    """No x23/other fixture has unequal site_counts, so build one by hand"""
    m = cle.generate(load("benzene"), N=2, R=4.0, verbose=0)[0]
    m.sym_ops.anchor_counts = {0: 1, 1: 3}
    m._generation = m.generation.model_copy(
        update={"N": 2, "site_counts": (2, 4), "species": ("A", "B"), "z_fu": 2}
    )
    assert m.weight == (2 * 1 + 4 * 3) / (2 * 2)
