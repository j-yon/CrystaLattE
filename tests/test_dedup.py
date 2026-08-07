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
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([-12, 0, 0])], 1)
    assert generate._same_multimer_int(G, H, W=[np.eye(3, dtype=np.int64)]) is True


def test_same_multimer_int_non_match():
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([24, 0, 0])], 1)
    assert generate._same_multimer_int(G, H, W=[np.eye(3, dtype=np.int64)]) is False


def test_same_multimer_int_inversion_needs_W():
    # G = [I, T(+1)];  H = [I, i0.T(+1)] with partner rot = -I, tr = (-1,0,0).
    # Same multimer only via s0 = -I, so it merges iff -I is in the dressing W.
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    h_partner = SymOp.from_components(
        -np.eye(3, dtype=np.int16), np.array([-12, 0, 0], dtype=np.int16)
    )
    H = SymOpList([SymOp.identity(), h_partner], 1)

    assert generate._same_multimer_int(G, H, W=[np.eye(3, dtype=np.int64)]) is False

    assert generate._same_multimer_int(
        G, H, W=[np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)]
    ) is True


def test_bind_globals_sets_site_symmetry_and_gate():
    from pathlib import Path

    import crystalatte as cle

    DATA = Path(__file__).parent

    crystal, monomer = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    generate._bind_globals(crystal, crystal.get_references())

    # Benzene is on an inversion centre: identity + inversion in the dressing.
    assert any(np.array_equal(w, np.eye(3, dtype=np.int64)) for w in generate._W_lin_by_ref[0])
    assert any(np.array_equal(w, -np.eye(3, dtype=np.int64)) for w in generate._W_lin_by_ref[0])
    # Molecular point group (D6h-ish) is richer than the site symmetry.
    assert generate._check_congruence_by_ref[0] is True


def test_is_bijection_fallback_gated_by_check_congruence(monkeypatch):
    # Two clusters that the integer matcher rejects; the float fallback must run
    # only when _check_congruence_by_ref[0] is True. Both clusters default to
    # ref_indices = [0, 0] (mono-reference, base 0).
    generate._W_lin_by_ref = [[np.eye(3, dtype=np.int64)]]
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([24, 0, 0])], 1)

    calls = []
    monkeypatch.setattr(generate, "_fingerprints_match", lambda a, b, **k: True)
    monkeypatch.setattr(
        generate, "_same_multimer", lambda a, b, **k: calls.append(1) or False
    )

    generate._check_congruence_by_ref = [False]
    assert generate._is_bijection(G, H) is False
    assert calls == []  # gated off: float fallback not called

    generate._check_congruence_by_ref = [True]
    assert generate._is_bijection(G, H) is False
    assert calls == [1]  # gated on: float fallback called once


def test_is_bijection_integer_path_matches_without_fallback(monkeypatch):
    # A crystallographic merge must be decided by the integer path alone.
    generate._W_lin_by_ref = [[np.eye(3, dtype=np.int64)]]
    generate._check_congruence_by_ref = [True]
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




# --- dressing translations (isometries about the centroid, not the origin) ---

UREA_DIMERS_R10 = 18  # urea, N=2, R=10.0, use_origin=True; verified against the
                      # NRE+CHSEV oracle and by explicit Kabsch congruence


def test_self_isometry_linear_part_is_not_a_symmetry():
    # A self-isometry is an isometry about the monomer CENTROID. Applied as a
    # bare linear map about the origin it does NOT map the molecule onto
    # itself unless the centroid happens to be a fixed point. Urea is a case
    # where it does not: the linear part displaces the molecule by several
    # Angstroms, which is exactly why the affine translation part is required.
    crystal, _ = next(cle.from_cif(str(DATA / "x23/urea/urea.cif")))
    mon = crystal.get_references()[0]
    X, c = mon.cart_coords, mon.centroid_cart

    def max_mismatch(P):
        return max(np.min(np.linalg.norm(X - p, axis=1)) for p in P)

    worst_linear = 0.0
    for s in mon.self_isometries:
        worst_linear = max(worst_linear, max_mismatch(X @ s.T))
        # the affine form about the centroid IS a symmetry
        assert max_mismatch((X - c) @ s.T + c) < 1e-8
    assert worst_linear > 1.0  # linear part alone is badly wrong


def test_bind_globals_sets_dressing_translations():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/urea/urea.cif")))
    crystal = crystal.detect_symmetry(1e-3)
    generate._bind_globals(crystal, crystal.get_references())

    S = generate._self_iso_by_ref[0]
    D = generate._self_iso_tr_by_ref[0]
    W = generate._W_lin_by_ref[0]
    W_tr = generate._W_tr_by_ref[0]
    assert len(D) == len(S)
    assert len(W_tr) == len(W)

    c = generate._monomers[0].centroid_cart
    for s, d in zip(S, D):
        assert np.allclose(d, c - s @ c)
    # urea's centroid is NOT a fixed point of its point group, so some
    # dressing translations are nonzero (this is the case the bug missed)
    assert max(float(np.linalg.norm(d)) for d in D) > 1.0
    assert max(int(np.abs(t).max()) for t in W_tr) > 0

    # the integer dressings must fix the centroid EXACTLY (in twelfths)
    c_frac = generate._monomers[0].centroid_frac
    for r, t in zip(W, W_tr):
        assert np.allclose(r @ c_frac + t / 12.0, c_frac, atol=1e-6)


def test_bind_globals_dressing_translations_vanish_for_benzene():
    # Benzene's centroid is a fixed point of its point group, so every
    # dressing translation is zero and the pre-fix code was accidentally right.
    crystal, _ = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    generate._bind_globals(crystal, crystal.get_references())
    assert all(np.allclose(d, 0.0) for d in generate._self_iso_tr_by_ref[0])
    assert all(not np.any(t) for t in generate._W_tr_by_ref[0])


def test_same_multimer_int_needs_dressing_translation():
    # Two urea dimers related by a crystallographic isometry whose dressing
    # does not fix the centroid. They merge only when the dressing carries its
    # translation part; with W_tr omitted the guard is unsatisfiable.
    crystal, _ = next(cle.from_cif(str(DATA / "x23/urea/urea.cif")))
    crystal = crystal.detect_symmetry(1e-3)
    refs = crystal.get_references()
    generate._bind_globals(crystal, refs)
    W, W_tr = generate._W_lin_by_ref[0], generate._W_tr_by_ref[0]

    rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.int16)
    g = SymOpList([SymOp.identity(),
                   SymOp.from_components(rot, np.array([-12, -24, 0], dtype=np.int16))], 1)
    h = SymOpList([SymOp.identity(),
                   SymOp.from_components(rot, np.array([-12, 12, 0], dtype=np.int16))], 1)

    assert generate._same_multimer_int(g, h, W, W_tr) is True
    # without the translation parts the same pair is missed
    assert generate._same_multimer_int(g, h, W) is False


def test_urea_dimer_count_regression():
    from crystalatte import count_chemistry_classes

    crystal, _ = next(cle.from_cif(str(DATA / "x23/urea/urea.cif")))
    mers = generate.generate(crystal, 2, 10.0, use_origin=True, n_jobs=1, verbose=0)
    assert len(mers) == UREA_DIMERS_R10
    # dedup completeness: no two survivors share a chemistry class
    n_classes, _ = count_chemistry_classes(mers, atol=1e-8)
    assert n_classes == len(mers)


# --- Wyckoff special-position validation ---

def test_get_unique_sym_ops_ambiguous_wyckoff_position():
    # Parameterized Wyckoff positions are stored in the table with their free
    # parameters zeroed, so a centroid at the origin matches several letters at
    # once (in SG 217: g/e/c all collapse onto [0,0,0] alongside the genuine
    # fixed position a). The multiplicity check must not fire on such an
    # ambiguous match -- the orbit size is fixed by orbit-stabilizer.
    crystal, _ = next(cle.from_cif(str(DATA / "x23/hexamine/hexamine.cif")))
    crystal = crystal.detect_symmetry(1e-3)
    sg = crystal.space_group
    centroid = crystal.get_references()[0].centroid_frac

    ops = sg.get_unique_sym_ops(centroid)
    stab = sg.get_site_symmetry(centroid)
    assert len(ops) == len(sg.sym_ops) // len(stab)


def test_generate_runs_on_high_symmetry_crystals():
    # These three crash outright if the Wyckoff multiplicity check misfires.
    for name in ("adamantane", "carbon_dioxide", "hexamine"):
        crystal, _ = next(cle.from_cif(str(DATA / f"x23/{name}/{name}.cif")))
        mers = generate.generate(crystal, 2, 6.0, use_origin=True, n_jobs=1, verbose=0)
        assert len(mers) > 0


def test_linear_monomer_dedup_is_complete():
    # CO2 is linear, so its point group (D_inf_h) is CONTINUOUS. The finite
    # self-isometry sample cannot supply the dressing the relative-transform
    # matchers need, so such monomers must be decided by point-cloud
    # congruence instead.
    from crystalatte import count_chemistry_classes

    crystal, _ = next(cle.from_cif(str(DATA / "x23/carbon_dioxide/carbon_dioxide.cif")))
    mers = generate.generate(crystal, 2, 15.0, use_origin=True, n_jobs=1, verbose=0)
    n_classes, _ = count_chemistry_classes(mers, atol=1e-8)
    assert n_classes == len(mers)


def test_special_position_tolerance_absorbs_cif_rounding():
    # Ice Ih's CIF writes the 3-fold special position (1/3, 2/3, z) as the
    # rounded decimals 0.3333 / 0.6667, putting the monomer centroid ~8.5e-5
    # (fractional) off the exact axis. That is chemically nothing but far above
    # a 1e-6 site-symmetry tolerance, so the 3-fold used to be missed, the
    # molecule was replicated 3x essentially on top of itself, and the
    # resulting "dimers" had coincident atoms.
    from scipy.spatial.distance import pdist

    crystal, _ = next(cle.from_cif(str(DATA / "x23/ice/ice.cif")))
    mers = generate.generate(crystal, 2, 8.0, use_origin=True, n_jobs=1, verbose=0)
    assert mers

    worst = min(
        float(pdist(np.vstack([m.cart_coords for m in mer.monomers])).min())
        for mer in mers
    )
    assert worst > 0.1, f"coincident atoms in a generated multimer ({worst:.3e} A)"


def test_ice_dedup_is_complete():
    from crystalatte import count_chemistry_classes

    crystal, _ = next(cle.from_cif(str(DATA / "x23/ice/ice.cif")))
    mers = generate.generate(crystal, 2, 8.0, use_origin=True, n_jobs=1, verbose=0)
    n_classes, _ = count_chemistry_classes(mers, atol=1e-8)
    assert n_classes == len(mers)
