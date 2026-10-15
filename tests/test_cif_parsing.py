"""Pytest suite for CIF parsing across diverse space groups and settings.

Checks space group number, Hall symbol, lattice parameters, and ASU atom count
for each crystal. Lattice parameter tolerance is 1e-3 Å / 1e-3 deg to match
CIF reported precision.
"""

import numpy as np
import pytest

import crystalatte as cle
from conftest import cif, load as parse_first


# ---------------------------------------------------------------------------
# Triclinic — SG 1 (P 1)
# ---------------------------------------------------------------------------

def test_ammonia_sg1_p1():
    crystal, reference = parse_first("ammonia")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 1
    assert crystal.space_group.symbol["Hall"] == "P 1"
    assert (a, b, c) == pytest.approx((5.1305, 5.1305, 5.1305), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-3)
    assert len(reference.symbols) == 4  # NH3


# ---------------------------------------------------------------------------
# Triclinic — SG 2 (P -1)
# ---------------------------------------------------------------------------

def test_ethyl_carbamate_sg2():
    crystal, reference = parse_first("ethyl_carbamate")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 2
    assert crystal.space_group.symbol["Hall"] == "-P 1"
    assert (a, b, c) == pytest.approx((5.051, 7.011, 7.543), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((101.37, 104.58, 76.65), abs=1e-3)
    assert len(reference.symbols) == 13  # C3H7NO2


# ---------------------------------------------------------------------------
# Monoclinic — SG 4 (P 21, unique b axis)
# ---------------------------------------------------------------------------

def test_14_cyclohexanedione_sg4():
    crystal, reference = parse_first("14_cyclohexanedione")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 4
    assert crystal.space_group.symbol["Hall"] == "P 2yb"
    assert (a, b, c) == pytest.approx((6.65, 6.21, 6.87), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 99.82, 90.0), abs=1e-3)
    assert len(reference.symbols) == 16  # C6H8O2


# ---------------------------------------------------------------------------
# Monoclinic — SG 14 in three different settings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cif_rel,expected_hall,abc,angles,n_atoms", [
    (
        "uracil",
        "-P 2yab",
        (11.938, 12.376, 3.6552),
        (90.0, 120.9, 90.0),
        12,   # C4H4N2O2
    ),
    (
        "anthracene",
        "-P 2yab",
        (8.4144, 5.9903, 11.0953),
        (90.0, 125.293, 90.0),
        24,   # C14H10
    ),
    (
        "succinic_acid",
        "-P 2yab",
        (5.0205, 8.768, 5.4662),
        (90.0, 93.043, 90.0),
        14,   # C4H6O4
    ),
    (
        "imidazole",
        "-P 2ybc",
        (7.582, 5.371, 9.79),
        (90.0, 118.98, 90.0),
        9,    # C3H4N2
    ),
    (
        "oxalic_acid_beta",
        "-P 2ybc",
        (5.33, 6.015, 5.436),
        (90.0, 115.83, 90.0),
        8,    # C2H2O4
    ),
    (
        "formamide",
        "-P 2yn",
        (3.604, 9.041, 6.994),
        (90.0, 100.5, 90.0),
        6,    # CH3NO
    ),
    (
        "hydroxyurea",
        "-P 2ybc",
        (8.393, 4.907, 8.798),
        (90.0, 121.2, 90.0),
        9,    # CH4N2O2
    ),
])
def test_sg14_setting(cif_rel, expected_hall, abc, angles, n_atoms):
    crystal, reference = parse_first(cif_rel)
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 14
    assert crystal.space_group.symbol["Hall"] == expected_hall
    assert len(crystal.space_group.sym_ops) == 4
    assert (a, b, c) == pytest.approx(abc, abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx(angles, abs=1e-3)
    assert len(reference.symbols) == n_atoms


# ---------------------------------------------------------------------------
# Orthorhombic — SG 19 (P 21 21 21)
# ---------------------------------------------------------------------------

def test_cytosine_sg19():
    crystal, reference = parse_first("cytosine")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 19
    assert crystal.space_group.symbol["Hall"] == "P 2ac 2ab"
    assert (a, b, c) == pytest.approx((13.044, 9.496, 3.814), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-3)
    assert len(reference.symbols) == 13  # C4H5N3O


# ---------------------------------------------------------------------------
# Orthorhombic — SG 29 (no Hall symbol in CIF, number-only fallback)
# ---------------------------------------------------------------------------

def test_fomepizole_sg29_no_hall():
    """CIF lacks a Hall symbol — code falls back to the standard setting for SG 29."""
    crystal, reference = parse_first("fomepizole")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 29
    assert (a, b, c) == pytest.approx((14.831, 16.848, 5.4962), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-3)
    assert len(reference.symbols) == 12  # C4H6N2


# ---------------------------------------------------------------------------
# Orthorhombic — SG 33 in two different settings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cif_rel,expected_hall,abc,n_atoms", [
    (
        "pyrazole",
        "P -2n 2a",
        (8.19, 12.588, 6.773),
        9,    # C3H4N2
    ),
    (
        "acetic_acid",
        "P 2c -2n",
        (13.31, 4.09, 5.769),
        8,    # C2H4O2
    ),
])
def test_sg33_setting(cif_rel, expected_hall, abc, n_atoms):
    crystal, reference = parse_first(cif_rel)
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 33
    assert crystal.space_group.symbol["Hall"] == expected_hall
    assert (a, b, c) == pytest.approx(abc, abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-3)
    assert len(reference.symbols) == n_atoms


# ---------------------------------------------------------------------------
# Orthorhombic — SG 58 (P nnm)
# ---------------------------------------------------------------------------

def test_pyrazine_sg58():
    crystal, reference = parse_first("pyrazine")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 58
    assert crystal.space_group.symbol["Hall"] == "-P 2n 2"
    assert (a, b, c) == pytest.approx((9.325, 5.85, 3.733), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-3)
    assert len(reference.symbols) == 10  # C4H4N2


# ---------------------------------------------------------------------------
# Orthorhombic — SG 61 (P bca) in two settings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cif_rel,expected_hall,abc,n_atoms", [
    (
        "benzene",
        "-P 2ac 2ab",
        (7.39, 9.42, 6.81),
        12,   # C6H6
    ),
    (
        "cyanamide",
        "-P 2ac 2ab",
        (6.856, 6.628, 9.147),
        5,    # CH2N2
    ),
    (
        "oxalic_acid_alpha",
        "-P 2bc 2ac",
        (6.548, 7.844, 6.086),
        8,    # C2H2O4
    ),
])
def test_sg61_setting(cif_rel, expected_hall, abc, n_atoms):
    crystal, reference = parse_first(cif_rel)
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 61
    assert crystal.space_group.symbol["Hall"] == expected_hall
    assert len(crystal.space_group.sym_ops) == 8
    assert (a, b, c) == pytest.approx(abc, abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-3)
    assert len(reference.symbols) == n_atoms


# ---------------------------------------------------------------------------
# Tetragonal — SG 113 and 114
# ---------------------------------------------------------------------------

def test_urea_sg113():
    crystal, reference = parse_first("urea")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 113
    assert crystal.space_group.symbol["Hall"] == "P -4 2ab"
    assert (a, b, c) == pytest.approx((5.589, 5.589, 4.6947), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-3)
    assert len(reference.symbols) == 8   # CH4N2O


def test_adamantane_sg114():
    crystal, reference = parse_first("adamantane")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 114
    assert crystal.space_group.symbol["Hall"] == "P -4 2n"
    assert (a, b, c) == pytest.approx((6.639, 6.639, 8.918), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-3)
    assert len(reference.symbols) == 26  # C10H16


# ---------------------------------------------------------------------------
# Trigonal / rhombohedral — SG 161 and 167
# ---------------------------------------------------------------------------

def test_trioxane_sg161():
    crystal, reference = parse_first("trioxane")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 161
    assert crystal.space_group.symbol["Hall"] == 'R 3 -2"c'
    assert (a, b, c) == pytest.approx((9.32, 9.32, 8.196), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 120.0), abs=1e-3)
    assert len(reference.symbols) == 12  # C3H6O3


def test_triazine_sg167():
    crystal, reference = parse_first("triazine")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 167
    assert crystal.space_group.symbol["Hall"] == '-R 3 2"c'
    assert (a, b, c) == pytest.approx((9.647, 9.647, 7.281), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 120.0), abs=1e-3)
    assert len(reference.symbols) == 9   # C3H3N3


# ---------------------------------------------------------------------------
# Hexagonal — SG 194
# ice.cif uses oxidation-state labels ("O-2") that the atom-label parser
# cannot resolve to an element symbol — pre-existing issue.
# ---------------------------------------------------------------------------

def test_ice_sg194():
    crystal, reference = parse_first("ice")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 194
    assert crystal.space_group.symbol["Hall"] == "-P 6c 2c"
    assert (a, b, c) == pytest.approx((4.506, 4.506, 7.346), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 120.0), abs=1e-3)
    # 4 half-occupied H positions per O due to proton disorder in ice Ih
    assert len(reference.symbols) == 5  # O + 4H


# ---------------------------------------------------------------------------
# Cubic — SG 205 and 217
# ---------------------------------------------------------------------------

def test_carbon_dioxide_sg205():
    crystal, reference = parse_first("carbon_dioxide")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 205
    assert crystal.space_group.symbol["Hall"] == "-P 2ac 2ab 3"
    assert (a, b, c) == pytest.approx((5.624, 5.624, 5.624), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-3)
    assert len(reference.symbols) == 3   # CO2


def test_hexamine_sg217():
    crystal, reference = parse_first("hexamine")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 217
    assert crystal.space_group.symbol["Hall"] == "I -4 2 3"
    assert (a, b, c) == pytest.approx((7.021, 7.021, 7.021), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-3)
    assert len(reference.symbols) == 22  # C6H12N4


# ---------------------------------------------------------------------------
# Edge case — naphthalene has only _symmetry_space_group_name_H-M,
# no SG number and no Hall symbol — needs H-M→number lookup to parse.
# ---------------------------------------------------------------------------

def test_naphthalene_no_number_field():
    # naphthalene.cif names its group ONLY by Hermann-Mauguin symbol ("P21/a"):
    # no International number, no Hall symbol. Resolving it therefore exercises
    # the H-M -> Hall route.
    #
    # The setting is the point, not the number. P2_1/c, P2_1/n and P2_1/a are
    # all space group 14 with different operations, so asserting `number == 14`
    # alone would pass just as happily against the STANDARD setting -- which is
    # P2_1/c, and wrong for this file. The Hall symbol and the operations are
    # what pin it down.
    crystal, reference = parse_first("naphthalene")
    assert crystal.space_group.number == 14
    assert crystal.space_group.symbol["Hall"] == "-P 2yab"
    assert crystal.space_group.symbol["short"] == "P21/a"
    assert sorted(op.op for op in crystal.space_group.sym_ops) == [
        "-x+1/2,y+1/2,-z", "-x,-y,-z", "x+1/2,-y+1/2,z", "x,y,z"
    ]


# ---------------------------------------------------------------------------
# Field-name flexibility
#
# CIF gives most symmetry items two spellings -- the 1997 `_symmetry_*` family
# and the 2012 `_space_group_*` family -- and depositors emit either, or both,
# or describe the group only by symbol, or only by an explicit operation list.
# The fixtures above happen to cover several of these combinations, but only
# incidentally; the tests below vary ONE field at a time so a regression names
# the field it broke.
# ---------------------------------------------------------------------------

_MINIMAL_CELL = """
_cell_length_a     5.0
_cell_length_b     6.0
_cell_length_c     7.0
_cell_angle_alpha  90.0
_cell_angle_beta   90.0
_cell_angle_gamma  90.0
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1  0.10  0.20  0.30
O1  0.40  0.15  0.05
"""


def _write_cif(tmp_path, symmetry: str, name: str = "probe"):
    """Write a minimal orthorhombic CIF whose only variable is its symmetry block."""
    path = tmp_path / f"{name}.cif"
    path.write_text(f"data_{name}\n{symmetry}\n{_MINIMAL_CELL}")
    return str(path)


def test_accepts_cif1_symmetry_field_names(tmp_path):
    crystal, _ = next(cle.from_cif(_write_cif(tmp_path, """
_symmetry_int_tables_number       19
_symmetry_space_group_name_Hall   'P 2ac 2ab'
""")))
    assert crystal.space_group.number == 19


def test_accepts_cif2_symmetry_field_names(tmp_path):
    # Same group, 2012 spellings throughout.
    crystal, _ = next(cle.from_cif(_write_cif(tmp_path, """
_space_group_IT_number      19
_space_group_name_Hall      'P 2ac 2ab'
""")))
    assert crystal.space_group.number == 19


def test_hall_symbol_alone_is_enough(tmp_path):
    # No number anywhere: the Hall symbol names the setting and the number is
    # recovered from it.
    crystal, _ = next(cle.from_cif(_write_cif(tmp_path, """
_space_group_name_Hall  '-P 2yab'
""")))
    assert crystal.space_group.number == 14
    assert crystal.space_group.symbol["Hall"] == "-P 2yab"


@pytest.mark.parametrize("written", ["P21/a", "P 21/a", "P 1 21/a 1", "p21/a"])
def test_hermann_mauguin_symbol_is_matched_regardless_of_punctuation(tmp_path, written):
    # Depositors punctuate H-M symbols freely and inconsistently. All four of
    # these name the same setting and must resolve to it.
    crystal, _ = next(cle.from_cif(_write_cif(tmp_path, f"""
_symmetry_space_group_name_H-M  '{written}'
""")))
    assert crystal.space_group.symbol["Hall"] == "-P 2yab"


def test_hermann_mauguin_resolves_the_setting_not_just_the_number(tmp_path):
    # The regression this whole path exists to prevent. P2_1/a and P2_1/c are
    # both space group 14; resolving "P21/a" to the number alone would hand back
    # the STANDARD setting, whose operations are different and wrong for a file
    # that said P2_1/a.
    a_setting, _ = next(cle.from_cif(_write_cif(tmp_path, """
_symmetry_space_group_name_H-M  'P 21/a'
""", name="a_setting")))
    c_setting, _ = next(cle.from_cif(_write_cif(tmp_path, """
_symmetry_space_group_name_H-M  'P 21/c'
""", name="c_setting")))

    assert a_setting.space_group.number == c_setting.space_group.number == 14
    assert a_setting.space_group.symbol["Hall"] == "-P 2yab"
    assert c_setting.space_group.symbol["Hall"] == "-P 2ybc"
    assert sorted(o.op for o in a_setting.space_group.sym_ops) != \
           sorted(o.op for o in c_setting.space_group.sym_ops), \
        "the two settings must not collapse to the same operations"


def test_explicit_operation_list_alone_is_enough(tmp_path):
    # Nothing names the group -- no number, no Hall, no H-M -- but the file
    # spells the operations out. That is exact for this cell.
    crystal, _ = next(cle.from_cif(_write_cif(tmp_path, """
loop_
_space_group_symop_operation_xyz
'x, y, z'
'-x, -y, -z'
""")))
    # Assert the operations themselves, not their `.op` display strings: that
    # attribute preserves whatever the file wrote ("x, y, z" here, spaces and
    # all) and is explicitly not used in any calculation.
    ops = crystal.space_group.sym_ops
    assert len(ops) == 2
    identity = np.eye(3, dtype=np.int16)
    assert any(np.array_equal(o.rot, identity) for o in ops)
    assert any(np.array_equal(o.rot, -identity) for o in ops)
    assert all(np.array_equal(o.tr, np.zeros(3, dtype=np.int16)) for o in ops)


def test_listed_operations_override_a_contradictory_symbol(tmp_path):
    # A file whose declared symbol disagrees with its own operation list is
    # defective. The operations describe the coordinates actually in the file,
    # so they win -- but loudly, because silently trusting either half would
    # build the wrong crystal.
    path = _write_cif(tmp_path, """
_symmetry_int_tables_number  2
_symmetry_space_group_name_H-M  'P -1'
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
""")
    with pytest.warns(RuntimeWarning, match="do not match"):
        crystal, _ = next(cle.from_cif(path))
    assert len(crystal.space_group.sym_ops) == 1, "the listed operations must win"


def test_coset_representative_differences_are_not_a_disagreement(tmp_path, recwarn):
    # 'x-1/2' and 'x+1/2' are the same operation: they differ by a lattice
    # vector. Comparing the declared setting against the listed operations has
    # to reduce modulo the lattice or every such file warns spuriously.
    path = _write_cif(tmp_path, """
_symmetry_space_group_name_H-M  'P 21/a'
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
'-x+1/2, y+1/2, -z'
'-x, -y, -z'
'x-1/2, -y-1/2, z'
""")
    crystal, _ = next(cle.from_cif(path))
    assert crystal.space_group.symbol["Hall"] == "-P 2yab"
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)], \
        "differing coset representatives were mistaken for a disagreement"


def test_missing_cell_volume_is_not_an_error(tmp_path):
    # _cell_volume is a cross-check, not an input. Plenty of valid CIFs omit it.
    crystal, _ = next(cle.from_cif(_write_cif(tmp_path, """
_symmetry_int_tables_number  1
""")))
    assert crystal.volume == pytest.approx(5.0 * 6.0 * 7.0)


def test_block_with_no_symmetry_information_reports_what_was_missing(tmp_path):
    # The error has to name the fields actually looked for, otherwise a user
    # holding an unusual CIF has no way to tell what to add.
    with pytest.raises(ValueError, match="no usable symmetry information"):
        next(cle.from_cif(_write_cif(tmp_path, "_cell_formula_units_Z  4")))


def test_missing_cell_parameters_are_named(tmp_path):
    path = tmp_path / "nocell.cif"
    path.write_text("data_nocell\n_symmetry_int_tables_number 1\n"
                    "_cell_length_a 5.0\n_cell_angle_alpha 90.0\n")
    with pytest.raises(ValueError, match="missing cell parameters"):
        next(cle.from_cif(str(path)))


# ---------------------------------------------------------------------------
# Partial occupancy
#
# A site with occupancy below 1 is present only part of the time, averaged over
# the crystal. Reading such sites as ordinary atoms builds molecules that do not
# exist: ice Ih gives every oxygen FOUR half-occupied hydrogens at 0.85-1.00 A.
# Which subset is real is not recoverable from the file, so the reader refuses
# by default rather than guessing.
# ---------------------------------------------------------------------------

_DISORDERED_CIF = """
_symmetry_int_tables_number  1
_cell_length_a     5.0
_cell_length_b     6.0
_cell_length_c     7.0
_cell_angle_alpha  90.0
_cell_angle_beta   90.0
_cell_angle_gamma  90.0
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
O1  0.10  0.20  0.30  1.0
H1  0.15  0.25  0.35  0.5
H2  0.05  0.15  0.25  0.5
"""


@pytest.fixture
def disordered(tmp_path):
    path = tmp_path / "disordered.cif"
    path.write_text(f"data_disordered\n{_DISORDERED_CIF}")
    return str(path)


def test_default_refuses_a_cell_with_no_ordered_state(disordered):
    # The default enumerates ordered states; this synthetic P1 cell holds half a hydrogen
    # per site, which no ordered state can have, so it must still refuse rather than guess.
    with pytest.raises(ValueError, match="whole number"):
        next(cle.from_cif(disordered))


def test_partial_occupancy_error_names_the_offending_sites(disordered):
    # The message has to identify WHICH sites and at what occupancy, or the user
    # cannot tell whether the disorder is incidental or structural.
    with pytest.raises(ValueError) as excinfo:
        next(cle.from_cif(disordered, disorder="error"))
    message = str(excinfo.value)
    assert "H1=0.5" in message and "H2=0.5" in message
    assert "O1" not in message, "a fully occupied site must not be reported"


def test_disorder_ignore_keeps_every_site_and_warns(disordered):
    with pytest.warns(RuntimeWarning, match="partially occupied"):
        crystal, _ = next(cle.from_cif(disordered, disorder="ignore"))
    assert sorted(crystal._asu.atoms) == ["H", "H", "O"]


def test_disorder_drop_keeps_only_full_sites_and_warns(disordered):
    with pytest.warns(RuntimeWarning, match="dropping them"):
        crystal, _ = next(cle.from_cif(disordered, disorder="drop"))
    assert crystal._asu.atoms == ["O"], "the half-occupied hydrogens must be gone"
    assert len(crystal._asu.positions) == 1, "coordinates must be dropped in step"


def test_ordered_crystals_are_unaffected_by_the_default():
    # The guard must not cost anything for the 25 fixtures that are fully
    # ordered, including those that declare _atom_site_occupancy as all 1.0.
    for name in ("benzene", "urea", "naphthalene", "fomepizole", "succinic_acid"):
        crystal, reference = parse_first(name)
        assert len(reference.symbols) > 0


def test_unknown_disorder_policy_is_rejected(disordered):
    with pytest.raises(ValueError, match="unknown disorder policy"):
        next(cle.from_cif(disordered, disorder="whatever"))


def test_ice_is_the_real_motivating_case():
    # Not a synthetic construction: ice Ih is proton-disordered, both hydrogen
    # sites carry occupancy 0.5, and reading them all produces one oxygen with
    # four hydrogens at bonding distance. The default resolves it into ordered
    # states; 'error' still refuses and 'ignore' still builds the impossible molecule.
    with pytest.warns(RuntimeWarning, match="2 ordered state"):
        assert len(list(cle.from_cif(cif("ice")))) == 2

    with pytest.raises(ValueError, match="partially occupied"):
        next(cle.from_cif(cif("ice"), disorder="error"))

    with pytest.warns(RuntimeWarning):
        crystal, reference = next(cle.from_cif(cif("ice"), disorder="ignore"))
    assert reference.symbols.count("H") == 4, (
        "if this is no longer 4, the unphysical reading has changed and the "
        "disorder guard may no longer be guarding anything"
    )
