"""Integer dressing groups: the cell's own symmetry, and the exact self-isometries of each monomer in it.

See ``docs/superpowers/specs/2026-10-20-integer-extra-symmetry-design.md``.
"""

import numpy as np
import pytest
import qcelemental as qcel

from crystalatte.work.dedup import dressing

from conftest import crystal

FIXTURES = [
    "14_cyclohexanedione", "acetic_acid", "adamantane", "ammonia", "anthracene",
    "benzene", "carbon_dioxide", "cyanamide", "cytosine", "ethyl_carbamate",
    "formamide", "hexamine", "ice", "imidazole", "naphthalene", "oxalic_acid_alpha",
    "oxalic_acid_beta", "pyrazine", "pyrazole", "succinic_acid", "triazine",
    "trioxane", "uracil", "urea", "fomepizole", "hydroxyurea",
]

# order of the cell symmetry by crystal family: triclinic 2, monoclinic 4, orthorhombic 8,
# tetragonal 16, hexagonal axes 24, cubic 48
H_ORDER = {
    "ethyl_carbamate": 2,
    "adamantane": 16, "urea": 16,
    "ice": 24, "triazine": 24, "trioxane": 24,
    "ammonia": 48, "carbon_dioxide": 48, "hexamine": 48,
}
MONOCLINIC = {
    "14_cyclohexanedione", "anthracene", "formamide", "imidazole", "naphthalene",
    "oxalic_acid_beta", "succinic_acid", "uracil", "hydroxyurea",
}

# fixtures whose monomer has exact symmetry beyond its site symmetry, and the size of its dressing group
EXTRA = {"triazine": 12, "carbon_dioxide": 12}


def _Z(mon):
    return np.array([qcel.periodictable.to_Z(s) for s in mon.symbols], dtype=np.int64)


def _cell_symmetry(xtal):
    rotations = np.stack([op.rot for op in xtal.space_group.sym_ops]).astype(np.int64)
    return dressing.cell_symmetry(rotations, np.asarray(xtal.lattice_vectors, float))


@pytest.mark.parametrize("name", FIXTURES)
def test_cell_symmetry_order_and_contains_the_space_group(name):
    xtal = crystal(name).detect_symmetry(1e-3)
    H = _cell_symmetry(xtal)

    assert len(H) == H_ORDER.get(name, 4 if name in MONOCLINIC else 8)
    for op in xtal.space_group.sym_ops:
        assert any(np.array_equal(op.rot.astype(np.int64), h) for h in H)


def test_cell_symmetry_is_the_box_not_the_bravais_lattice():
    # R-3c on hexagonal axes: the R-centred lattice has -3m (12 operations), the hexagonal box 6/mmm
    # (24). Triazine's extra operations lie outside -3m, so taking the Bravais lattice's symmetry instead
    # would silently lose every merge they make.
    H = _cell_symmetry(crystal("triazine").detect_symmetry(1e-3))
    centring = np.array([2 / 3, 1 / 3, 1 / 3])

    def keeps_R_lattice(h):
        y = h @ centring
        return any(np.allclose(y - t, np.rint(y - t)) for t in (np.zeros(3), centring, 2 * centring))

    assert len(H) == 24
    assert sum(keeps_R_lattice(h) for h in H) == 12


def test_cell_symmetry_keeps_the_lookup_on_an_inconsistent_cell():
    # A tetragonal space group on a cell whose a and b disagree by 1e-3: the metric test sees only
    # orthorhombic symmetry, but the lookup still supplies the full tetragonal cell symmetry.
    rotations = np.array([
        np.eye(3),
        np.diag([-1, -1, 1]),
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    ], dtype=np.int64)
    H = dressing.cell_symmetry(rotations, np.diag([5.0, 5.005, 7.0]))
    assert len(H) == 16


def test_cell_symmetry_adds_a_cell_shape_more_symmetric_than_its_space_group():
    # P1 on a cubic cell: the lookup gives only +-I, the metric test adds the rest of m-3m.
    H = dressing.cell_symmetry(np.eye(3, dtype=np.int64)[None], 6.0 * np.eye(3))
    assert len(H) == 48


@pytest.mark.parametrize("name", FIXTURES)
def test_dressing_group(name):
    xtal = crystal(name).detect_symmetry(1e-3)
    refs = xtal.get_references()
    H = _cell_symmetry(xtal)
    A = np.asarray(xtal.lattice_vectors, float)

    for i, mon in enumerate(refs):
        W, W_tr = dressing.site_symmetry(xtal.space_group, mon.centroid_frac)
        D, D_tr = dressing.dressing_group(
            mon.frac_coords, mon.centroid_frac, _Z(mon), A, W, W_tr, H
        )

        assert len(D) == EXTRA.get(name, len(W))
        # W is always kept, even where it is inexact (ice's H atoms fit only to ~2.7e-4 A)
        for w in W:
            assert any(np.array_equal(w, d) for d in D)
        # D lies in the cell symmetry and is a group
        for d in D:
            assert any(np.array_equal(d, h) for h in H)
        assert {(a @ b).tobytes() for a in D for b in D} == {d.tobytes() for d in D}
        # W and its translations pass through untouched (ice's fix the centroid only to ~8.5e-5,
        # see test_special_position_tolerance_absorbs_cif_rounding); every extra dressing fixes it
        # exactly, with its translation in twelfths
        assert np.array_equal(D[: len(W)], np.stack(W)) and np.array_equal(D_tr[: len(W)], np.stack(W_tr))
        c = np.asarray(mon.centroid_frac, float)
        for d, t in zip(D[len(W):], D_tr[len(W):]):
            assert np.allclose(d @ c + t / 12.0, c, atol=1e-9)


def test_dressing_group_rejects_an_approximate_symmetry():
    # Benzene's molecular point group detects at 0.05 A with four operations, but only the two in the
    # site symmetry are exact; an approximate operation must not become a dressing.
    xtal = crystal("benzene").detect_symmetry(1e-3)
    mon = xtal.get_references()[0]
    D, _ = dressing.dressing_group(
        mon.frac_coords, mon.centroid_frac, _Z(mon), np.asarray(xtal.lattice_vectors, float),
        *dressing.site_symmetry(xtal.space_group, mon.centroid_frac), _cell_symmetry(xtal),
    )
    assert len(D) == 2
