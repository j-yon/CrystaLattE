"""Pytest suite for CIF parsing across diverse space groups and settings.

Checks space group number, Hall symbol, lattice parameters, and ASU atom count
for each crystal. Lattice parameter tolerance is 1e-3 Å / 1e-3 deg to match
CIF reported precision.
"""

from pathlib import Path

import pytest

import crystalatte as cle

DATA = Path(__file__).parent


def parse_first(cif_path: Path):
    """Return (crystal, reference) for the first block in a CIF file."""
    return next(cle.from_cif(str(cif_path)))


# ---------------------------------------------------------------------------
# Triclinic — SG 1 (P 1)
# ---------------------------------------------------------------------------

def test_ammonia_sg1_p1():
    crystal, reference = parse_first(DATA / "x23/ammonia/ammonia.cif")
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
    crystal, reference = parse_first(DATA / "x23/ethyl_carbamate/ethyl_carbamate.cif")
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
    crystal, reference = parse_first(DATA / "x23/14_cyclohexanedione/14_cyclohexanedione.cif")
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
        "x23/uracil/uracil.cif",
        "-P 2yab",
        (11.938, 12.376, 3.6552),
        (90.0, 120.9, 90.0),
        12,   # C4H4N2O2
    ),
    (
        "x23/anthracene/anthracene.cif",
        "-P 2yab",
        (8.4144, 5.9903, 11.0953),
        (90.0, 125.293, 90.0),
        24,   # C14H10
    ),
    (
        "x23/succinic_acid/succinic_Acid.cif",
        "-P 2yab",
        (5.0205, 8.768, 5.4662),
        (90.0, 93.043, 90.0),
        14,   # C4H6O4
    ),
    (
        "x23/imidazole/imidazole.cif",
        "-P 2ybc",
        (7.582, 5.371, 9.79),
        (90.0, 118.98, 90.0),
        9,    # C3H4N2
    ),
    (
        "x23/oxalic_acid_beta/oxalic_acid_beta.cif",
        "-P 2ybc",
        (5.33, 6.015, 5.436),
        (90.0, 115.83, 90.0),
        8,    # C2H2O4
    ),
    (
        "x23/formamide/formamide.cif",
        "-P 2yn",
        (3.604, 9.041, 6.994),
        (90.0, 100.5, 90.0),
        6,    # CH3NO
    ),
    (
        "other/hydroxyurea/hydroxyurea.cif",
        "-P 2ybc",
        (8.393, 4.907, 8.798),
        (90.0, 121.2, 90.0),
        9,    # CH4N2O2
    ),
])
def test_sg14_setting(cif_rel, expected_hall, abc, angles, n_atoms):
    crystal, reference = parse_first(DATA / cif_rel)
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
    crystal, reference = parse_first(DATA / "x23/cytosine/cytosine.cif")
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
    crystal, reference = parse_first(DATA / "other/fomepizole/fomepizole.cif")
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
        "x23/pyrazole/pyrazole.cif",
        "P -2n 2a",
        (8.19, 12.588, 6.773),
        9,    # C3H4N2
    ),
    (
        "x23/acetic_acid/acetic_acid.cif",
        "P 2c -2n",
        (13.31, 4.09, 5.769),
        8,    # C2H4O2
    ),
])
def test_sg33_setting(cif_rel, expected_hall, abc, n_atoms):
    crystal, reference = parse_first(DATA / cif_rel)
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
    crystal, reference = parse_first(DATA / "x23/pyrazine/pyrazine.cif")
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
        "x23/benzene/benzene.cif",
        "-P 2ac 2ab",
        (7.39, 9.42, 6.81),
        12,   # C6H6
    ),
    (
        "x23/cyanamide/cyanamide.cif",
        "-P 2ac 2ab",
        (6.856, 6.628, 9.147),
        5,    # CH2N2
    ),
    (
        "x23/oxalic_acid_alpha/oxalic_acid_alpha.cif",
        "-P 2bc 2ac",
        (6.548, 7.844, 6.086),
        8,    # C2H2O4
    ),
])
def test_sg61_setting(cif_rel, expected_hall, abc, n_atoms):
    crystal, reference = parse_first(DATA / cif_rel)
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
    crystal, reference = parse_first(DATA / "x23/urea/urea.cif")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 113
    assert crystal.space_group.symbol["Hall"] == "P -4 2ab"
    assert (a, b, c) == pytest.approx((5.589, 5.589, 4.6947), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-3)
    assert len(reference.symbols) == 8   # CH4N2O


def test_adamantane_sg114():
    crystal, reference = parse_first(DATA / "x23/adamantane/adamantane.cif")
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
    crystal, reference = parse_first(DATA / "x23/trioxane/trioxane.cif")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 161
    assert crystal.space_group.symbol["Hall"] == 'R 3 -2"c'
    assert (a, b, c) == pytest.approx((9.32, 9.32, 8.196), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 120.0), abs=1e-3)
    assert len(reference.symbols) == 12  # C3H6O3


def test_triazine_sg167():
    crystal, reference = parse_first(DATA / "x23/triazine/triazine.cif")
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
    crystal, reference = parse_first(DATA / "x23/ice/ice.cif")
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
    crystal, reference = parse_first(DATA / "x23/carbon_dioxide/carbon_dioxide.cif")
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    assert crystal.space_group.number == 205
    assert crystal.space_group.symbol["Hall"] == "-P 2ac 2ab 3"
    assert (a, b, c) == pytest.approx((5.624, 5.624, 5.624), abs=1e-3)
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0), abs=1e-3)
    assert len(reference.symbols) == 3   # CO2


def test_hexamine_sg217():
    crystal, reference = parse_first(DATA / "x23/hexamine/hexamine.cif")
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

@pytest.mark.xfail(reason="naphthalene.cif has no SG number or Hall symbol — needs H-M→number lookup")
def test_naphthalene_no_number_field():
    crystal, reference = parse_first(DATA / "x23/naphthalene/naphthalene.cif")
    assert crystal.space_group.number == 14
