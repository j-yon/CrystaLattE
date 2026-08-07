from pathlib import Path

import numpy as np

import crystalatte as cle
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp, SymOpList
from crystalatte.work import generate

DATA = Path(__file__).parent
BENZENE_DIMERS = 3  # captured on pre-Task-5 code: benzene, N=2, R=5.0, use_origin=True


def test_get_site_symmetry_inversion_center():
    # Pbca (space group 61): origin is an inversion centre => {E, i}.
    # SG 61 is one of the entries in the tracked space_groups.json, so this
    # test does not depend on the (untracked) full space-group table.
    sg = SpaceGroup(61)
    w0 = sg.get_site_symmetry(np.array([0.0, 0.0, 0.0]))
    rots = [op.rot for op in w0]
    assert len(w0) == 2
    assert any(np.array_equal(r, np.eye(3, dtype=r.dtype)) for r in rots)
    assert any(np.array_equal(r, -np.eye(3, dtype=r.dtype)) for r in rots)


def test_get_site_symmetry_general_position():
    sg = SpaceGroup(61)
    wg = sg.get_site_symmetry(np.array([0.13, 0.21, 0.37]))
    assert len(wg) == 1
    assert np.array_equal(wg[0].rot, np.eye(3, dtype=wg[0].rot.dtype))


def _translate(dx12):
    return SymOp.identity().translate(np.array(dx12, dtype=np.int16))


def test_relative_transforms_int_pure_translation():
    g = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    L, v = generate._relative_transforms_int(g)
    assert np.array_equal(L[0, 1], np.eye(3, dtype=np.int64))
    assert np.array_equal(v[0, 1], np.array([12, 0, 0]))
    assert np.array_equal(v[1, 0], np.array([-12, 0, 0]))


def test_relative_transforms_int_nonorthogonal_inverse():
    # hexagonal 6-fold: rot = [[1,-1,0],[1,0,0],[0,0,1]] is NOT orthogonal;
    # its inverse differs from its transpose.
    six = SymOp("x-y,x,z")
    g = SymOpList([SymOp.identity(), six], 1)
    L, v = generate._relative_transforms_int(g)
    R = six.rot.astype(np.int64)
    # L[1,0] = rot_inv[1] @ rot[0] = inv(R); verify it is the exact inverse
    assert np.array_equal(L[1, 0] @ R, np.eye(3, dtype=np.int64))
    # and that the wrong (transpose) formula would have differed
    assert not np.array_equal(L[1, 0], R.T)


def test_relative_transforms_int_is_cached():
    g = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    first = generate._relative_transforms_int(g)
    second = generate._relative_transforms_int(g)
    assert first[0] is second[0]  # same cached array object


def test_same_multimer_int_translation_swap():
    # Same dimer counted from each monomer: [I, T(+1)] ~ [I, T(-1)]. W = {I}.
    generate._W_lin = [np.eye(3, dtype=np.int64)]
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([-12, 0, 0])], 1)
    assert generate._same_multimer_int(G, H) is True


def test_same_multimer_int_non_match():
    generate._W_lin = [np.eye(3, dtype=np.int64)]
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([24, 0, 0])], 1)
    assert generate._same_multimer_int(G, H) is False


def test_same_multimer_int_inversion_needs_W():
    # G = [I, T(+1)];  H = [I, i0.T(+1)] with partner rot = -I, tr = (-1,0,0).
    # Same multimer only via s0 = -I, so it merges iff -I is in the dressing W.
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    h_partner = SymOp.from_components(
        -np.eye(3, dtype=np.int16), np.array([-12, 0, 0], dtype=np.int16)
    )
    H = SymOpList([SymOp.identity(), h_partner], 1)

    generate._W_lin = [np.eye(3, dtype=np.int64)]
    assert generate._same_multimer_int(G, H) is False

    generate._W_lin = [np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)]
    assert generate._same_multimer_int(G, H) is True


def test_bind_globals_sets_site_symmetry_and_gate():
    from pathlib import Path

    import crystalatte as cle

    DATA = Path(__file__).parent

    crystal, monomer = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    generate._bind_globals(crystal, crystal.get_references())

    # Benzene is on an inversion centre: identity + inversion in the dressing.
    assert any(np.array_equal(w, np.eye(3, dtype=np.int64)) for w in generate._W_lin)
    assert any(np.array_equal(w, -np.eye(3, dtype=np.int64)) for w in generate._W_lin)
    # Molecular point group (D6h-ish) is richer than the site symmetry.
    assert generate._check_congruence is True


def test_is_bijection_fallback_gated_by_check_congruence(monkeypatch):
    # Two clusters that the integer matcher rejects; the float fallback must run
    # only when _check_congruence is True.
    generate._W_lin = [np.eye(3, dtype=np.int64)]
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([24, 0, 0])], 1)

    calls = []
    monkeypatch.setattr(generate, "_fingerprints_match", lambda a, b, **k: True)
    monkeypatch.setattr(
        generate, "_same_multimer", lambda a, b, **k: calls.append(1) or False
    )

    generate._check_congruence = False
    assert generate._is_bijection(G, H) is False
    assert calls == []  # gated off: float fallback not called

    generate._check_congruence = True
    assert generate._is_bijection(G, H) is False
    assert calls == [1]  # gated on: float fallback called once


def test_is_bijection_integer_path_matches_without_fallback(monkeypatch):
    # A crystallographic merge must be decided by the integer path alone.
    generate._W_lin = [np.eye(3, dtype=np.int64)]
    generate._check_congruence = True
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([-12, 0, 0])], 1)

    def _boom(*a, **k):
        raise AssertionError("float fallback should not run for integer merges")

    monkeypatch.setattr(generate, "_fingerprints_match", lambda a, b, **k: True)
    monkeypatch.setattr(generate, "_same_multimer", _boom)
    assert generate._is_bijection(G, H) is True


def test_generate_dimer_count_regression():
    crystal, monomer = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    mers = generate.generate(crystal, 2, 5.0, use_origin=True, n_jobs=1)
    assert len(mers) == BENZENE_DIMERS   # from Step 5


def test_int_inv_rejects_non_unimodular():
    import pytest
    # det = 2 is not unimodular: no exact integer inverse exists.
    with pytest.raises(AssertionError):
        generate._int_inv(np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64))


