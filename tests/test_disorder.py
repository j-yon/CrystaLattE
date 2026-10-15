"""Ordered-state enumeration for disordered CIFs (see docs/superpowers/specs/2026-10-15-disorder-states-design.md)"""
import numpy as np
import pytest

from conftest import load
from crystalatte.core.crystal import Crystal, DisorderState, expand_cell
from crystalatte.core.multimer import ASU
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp

P1 = SpaceGroup.from_sym_ops([SymOp.identity()], number=1)


def _benzene_cell():
    crystal, _ = load("benzene")
    return crystal, expand_cell(crystal._asu.positions, crystal._asu.atoms, crystal.space_group.sym_ops)


def test_expand_cell_returns_every_atom_of_the_cell_once():
    _, (frac, symbols, site) = _benzene_cell()
    # Pbca, 6-atom ASU (half a molecule on an inversion centre) -> 4 molecules x 12 atoms
    assert len(frac) == 48
    assert symbols.tolist().count("C") == 24
    assert sorted(set(site.tolist())) == list(range(6))
    assert np.all((frac >= 0.0) & (frac < 1.0))


def test_ordered_crystal_has_no_state():
    crystal, _ = load("benzene")
    assert crystal.state is None


def test_state_survives_symmetry_detection():
    crystal, (frac, symbols, _) = _benzene_cell()
    state = DisorderState("config 1", 0.5)
    p1 = Crystal(*crystal.lattice_parameters, P1, ASU([str(s) for s in symbols], frac), state)
    detected = p1.detect_symmetry()
    assert len(detected.space_group.sym_ops) == 8, "P1 benzene must be lifted back to Pbca"
    assert detected.state is state
