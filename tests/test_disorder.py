"""Ordered-state enumeration for disordered CIFs (see docs/superpowers/specs/2026-10-15-disorder-states-design.md)"""
import numpy as np
import pytest

from conftest import load
from crystalatte.core.crystal import Crystal, DisorderState, expand_cell
from crystalatte.core.multimer import ASU
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp
from crystalatte.io.disorder import (
    _periodic_distances,
    contacts,
    explicit_choices,
    normalize_tag,
    symmetry_choices,
)

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


@pytest.mark.parametrize("raw,expected", [
    (1.0, "1"), (-1.0, "-1"), (2.5, "2.5"), (".", None), ("?", None), ("", None), (None, None), ("A", "A"),
])
def test_normalize_tag(raw, expected):
    assert normalize_tag(raw) == expected


def test_explicit_choices_take_one_group_per_assembly():
    occ = [1.0, 0.7, 0.3, 0.6, 0.4]
    asm = [None, "A", "A", "B", "B"]
    grp = [None, "1", "2", "1", "2"]
    choices = explicit_choices(occ, asm, grp)
    assert [c.label for c in choices] == ["A1 B1", "A1 B2", "A2 B1", "A2 B2"]
    assert [round(c.weight, 3) for c in choices] == [0.42, 0.28, 0.18, 0.12]
    assert choices[1].keep.tolist() == [True, True, False, False, True]
    assert choices[1].groups == (("A", "1"), ("B", "2"))
    assert choices[1].settled.tolist() == [False, True, True, True, True]


def test_tagged_but_fully_occupied_site_is_in_every_choice():
    # ritonavir's H11 is tagged B/1 with occupancy 1; selecting by tag alone loses it
    choices = explicit_choices([0.5, 0.5, 1.0], ["B", "B", "B"], ["1", "2", "1"])
    assert len(choices) == 2
    assert all(c.keep[2] for c in choices)


def test_unnamed_assembly_is_labelled_by_group_only():
    choices = explicit_choices([0.6, 0.4], [None, None], ["1", "2"])
    assert [c.label for c in choices] == ["1", "2"]


def test_groups_sort_numerically():
    choices = explicit_choices([0.5, 0.5], ["A", "A"], ["10", "2"])
    assert [c.label for c in choices] == ["A2", "A10"]


def test_single_partial_group_also_admits_absence():
    choices = explicit_choices([1.0, 0.5], [None, "S"], [None, "1"])
    assert [(c.label, c.weight) for c in choices] == [("S1", 0.5), ("S-", 0.5)]
    assert choices[1].keep.tolist() == [True, False]


def test_untagged_and_negative_partial_sites_are_left_for_the_symmetry_stage():
    choices = explicit_choices([0.5, 0.5, 0.6, 0.4], [None, None, "A", "A"], [None, "-1", "1", "2"])
    assert len(choices) == 2
    assert all(c.keep[0] and c.keep[1] for c in choices)
    assert all(not c.settled[0] and not c.settled[1] for c in choices)


def test_block_without_groups_is_a_single_trivial_choice():
    (only,) = explicit_choices([1.0, 0.5], [None, None], [None, None])
    assert (only.label, only.weight, only.groups) == ("", 1.0, ())
    assert only.keep.all() and not only.settled.any()


def _ice_inputs():
    crystal, _ = load("ice")  # conftest reads ice with disorder="ignore"
    asu = crystal._asu
    occupancies = [1.0 if s == "O" else 0.5 for s in asu.atoms]
    labels = [f"site {i}" for i in range(len(asu.atoms))]
    return asu.positions, asu.atoms, occupancies, labels, crystal.space_group.sym_ops, crystal.lattice_vectors


def test_two_hydrogens_are_never_bonded():
    # ice: the two H sites on one O...O contact are 0.76 A apart, just above the 0.74 A H-H cutoff
    bonded, clash = contacts(["H", "H"], np.array([[0.0, 0.7], [0.7, 0.0]]))
    assert not bonded[0, 1] and clash[0, 1]


def test_ice_has_two_ordered_states_in_its_own_cell():
    choices = symmetry_choices(*_ice_inputs())
    # 18 Bernal-Fowler configurations of the 4-molecule cell fall into 12 + 6 under P6_3/mmc
    assert [c.label for c in choices] == ["config 1", "config 2"]
    assert [c.weight for c in choices] == pytest.approx([12 / 18, 6 / 18])
    for c in choices:
        assert sorted(c.symbols.tolist()) == ["H"] * 8 + ["O"] * 4


def test_ice_states_obey_the_ice_rules():
    *_, lattice = _ice_inputs()
    for c in symmetry_choices(*_ice_inputs()):
        d = _periodic_distances(c.frac, lattice)
        oxygen, hydrogen = c.symbols == "O", c.symbols == "H"
        assert ((d[np.ix_(oxygen, hydrogen)] < 1.2).sum(axis=1) == 2).all(), "two H per O"
        hh = d[np.ix_(hydrogen, hydrogen)] + np.eye(hydrogen.sum()) * 9
        assert hh.min() > 1.2, "one H per O...O contact"


def test_ice_states_recover_cc_and_cmc21():
    lattice_parameters = load("ice")[0].lattice_parameters
    numbers = []
    for c in symmetry_choices(*_ice_inputs()):
        crystal = Crystal(*lattice_parameters, P1, ASU([str(s) for s in c.symbols], c.frac)).detect_symmetry()
        refs = crystal.get_references()
        assert len(refs) == 2 and all(sorted(r.symbols) == ["H", "H", "O"] for r in refs)
        numbers.append(crystal.space_group.number)
    assert numbers == [9, 36]


def test_half_an_atom_per_cell_has_no_ordered_state():
    with pytest.raises(ValueError, match="H1.*whole number"):
        symmetry_choices(
            np.array([[0.10, 0.20, 0.30], [0.15, 0.25, 0.35]]), ["O", "H"], [1.0, 0.5],
            ["O1", "H1"], [SymOp.identity()], np.diag([5.0, 6.0, 7.0]),
        )


def test_whole_molecule_disorder_is_refused():
    ops = [SymOp.identity(), SymOp("-x,-y,-z")]
    with pytest.raises(ValueError, match="not bonded to any fully occupied atom"):
        symmetry_choices(np.array([[0.1, 0.1, 0.1]]), ["O"], [0.5], ["O1"], ops, np.diag([5.0, 5.0, 5.0]))
