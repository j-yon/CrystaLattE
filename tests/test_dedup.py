import concurrent.futures as cf

import numpy as np
import pytest
import qcelemental as qcel
from scipy.spatial.distance import pdist

from crystalatte import count_chemistry_classes
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp, SymOpList
from crystalatte.work import generate, neighbors
from crystalatte.work.dedup import cloud, equivalence, context, invariants, matchers
from crystalatte.work.dedup.invariants import cluster_atoms_cart
from crystalatte.work.validate import _chsev, _nre

from conftest import GENERATE_KW, load

#: benzene, N = 2, R = 5.0, anchor-relative convention. Captured before the
#: integer matcher replaced the float one and unchanged by it, so it pins
#: agreement between the two implementations rather than just the current one.
BENZENE_DIMERS = 3
#: pyrazole is Z' = 2 (Pna2_1, and detection does not collapse it), so it is the
#: regression case for the cross-reference path. Captured while the point-cloud
#: matcher still decided those pairs, so it pins agreement between the two.
PYRAZOLE_DIMERS_R8 = 42
#: fomepizole is Z' = 3 (Pca2_1), the only fixture with three independent
#: molecules. `dedup/cloud.py` claims the integer matcher reproduces every merge
#: the point-cloud matcher used to make on BOTH multi-reference crystals; until
#: this constant existed, only the Z' = 2 half of that claim was tested.
FOMEPIZOLE_DIMERS_R7 = 34


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
    L, v = invariants.relative_transforms_int(g)
    assert np.array_equal(L[0, 1], np.eye(3, dtype=np.int64))
    assert np.array_equal(v[0, 1], np.array([12, 0, 0]))
    assert np.array_equal(v[1, 0], np.array([-12, 0, 0]))


def test_relative_transforms_int_nonorthogonal_inverse():
    # hexagonal 6-fold: rot = [[1,-1,0],[1,0,0],[0,0,1]] is NOT orthogonal;
    # its inverse differs from its transpose.
    six = SymOp("x-y,x,z")
    g = SymOpList([SymOp.identity(), six], 1)
    L, v = invariants.relative_transforms_int(g)
    R = six.rot.astype(np.int64)
    # L[1,0] = rot_inv[1] @ rot[0] = inv(R); verify it is the exact inverse
    assert np.array_equal(L[1, 0] @ R, np.eye(3, dtype=np.int64))
    # and that the wrong (transpose) formula would have differed
    assert not np.array_equal(L[1, 0], R.T)


def test_relative_transforms_int_is_cached():
    g = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    first = invariants.relative_transforms_int(g)
    second = invariants.relative_transforms_int(g)
    assert first[0] is second[0]  # same cached array object


def test_same_multimer_int_translation_swap():
    # Same dimer counted from each monomer: [I, T(+1)] ~ [I, T(-1)]. W = {I}.
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([-12, 0, 0])], 1)
    assert matchers.same_multimer_int(G, H, W=[np.eye(3, dtype=np.int64)]) is True


def test_same_multimer_int_non_match():
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([24, 0, 0])], 1)
    assert matchers.same_multimer_int(G, H, W=[np.eye(3, dtype=np.int64)]) is False


def test_same_multimer_int_inversion_needs_W():
    # G = [I, T(+1)];  H = [I, i0.T(+1)] with partner rot = -I, tr = (-1,0,0).
    # Same multimer only via s0 = -I, so it merges iff -I is in the dressing W.
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    h_partner = SymOp.from_components(
        -np.eye(3, dtype=np.int16), np.array([-12, 0, 0], dtype=np.int16)
    )
    H = SymOpList([SymOp.identity(), h_partner], 1)

    assert matchers.same_multimer_int(G, H, W=[np.eye(3, dtype=np.int64)]) is False

    assert matchers.same_multimer_int(
        G, H, W=[np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)]
    ) is True


# ---------------------------------------------------------------------------
# Multi-reference (Z' > 1) support in the exact integer matcher
#
# docs/dedup_correctness.tex Def. 2 is already multi-reference: it draws each
# dressing from D_{ref_G(a)} and requires pi to be reference-compatible. Only
# `anchored_match`'s signature was single-group, which forced every
# cross-reference pair out to the heuristic point-cloud matcher.
# ---------------------------------------------------------------------------


def _refs(sym_op_list, indices):
    sym_op_list.ref_indices = np.array(indices, dtype=np.int64)
    return sym_op_list


def test_same_multimer_int_requires_reference_compatibility():
    # Identical placement algebra in both halves; only the reference labels
    # differ. The permutation must map each member onto a member drawn from the
    # SAME base reference (Def. 2), so the first pair merges and the second --
    # a ref0-with-ref0 dimer against a ref1-with-ref1 dimer, i.e. two different
    # molecules -- must not.
    W_by_ref = [[np.eye(3, dtype=np.int64)], [np.eye(3, dtype=np.int64)]]

    def dimer(indices):
        return _refs(SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1), indices)

    assert matchers.same_multimer_int(dimer([0, 1]), dimer([0, 1]), W_by_ref) is True
    assert matchers.same_multimer_int(dimer([0, 0]), dimer([1, 1]), W_by_ref) is False


def test_same_multimer_int_reference_compatibility_binds_the_permutation():
    # The two assertions above are decided by the CHEAP GUARD, not by the
    # constraint they are named for: `sorted(refG) != sorted(refH)` rejects a
    # ref0-ref0 dimer against a ref1-ref1 dimer before any permutation is
    # considered. Deleting the reference-compatibility test inside
    # `anchored_match` leaves them both passing.
    #
    # This case gets past the guard. Both clusters draw on references {0, 1}
    # exactly once, and the placement algebra is IDENTICAL; only the order of
    # the labels differs. Reference-compatibility then forces pi(0) onto H's
    # member 1, so the matcher compares Delta^G_01 = (I, +t) against
    # Delta^H_10 = (I, -t) and correctly declines. Without it the identity
    # permutation matches trivially and two different hetero dimers are merged.
    W_by_ref = [[np.eye(3, dtype=np.int64)], [np.eye(3, dtype=np.int64)]]

    def dimer(indices):
        return _refs(SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1), indices)

    g, h = dimer([0, 1]), dimer([1, 0])
    assert sorted(g.ref_indices.tolist()) == sorted(h.ref_indices.tolist()), (
        "the cheap guard must NOT fire here, or this test degenerates into the "
        "one above"
    )
    assert matchers.same_multimer_int(g, h, W_by_ref) is False, (
        "a reference-incompatible permutation was accepted: member 0 of one "
        "cluster was matched to a member of the other drawn from a different "
        "base reference, which merges two genuinely different hetero dimers"
    )

    # A THREE-member case, because at n = 2 the constraint is enforced twice
    # over and either copy alone suffices: `anchored_match` screens the anchor
    # image (refH[h0] != r0) and each subsequent image (refH[h] != rb), and with
    # only one member left to place after the anchor, whichever screen survives
    # forces the same answer. Deleting just one of the two therefore changes
    # nothing above.
    #
    # With refG = [0, 0, 1] against refH = [0, 1, 0] there are two members to
    # place after the anchor, so the per-image screen is doing work the anchor
    # screen cannot: anchoring correctly on H's member 0 still leaves G's second
    # ref-0 member free to be paired with H's ref-1 member. The reference
    # multiset is unchanged, so again the cheap guard stays silent.
    def trimer(indices):
        return _refs(
            SymOpList(
                [SymOp.identity(), _translate([12, 0, 0]), _translate([0, 24, 0])], 1
            ),
            indices,
        )

    assert matchers.same_multimer_int(trimer([0, 0, 1]), trimer([0, 1, 0]), W_by_ref) \
        is False, (
        "a reference-incompatible permutation was accepted on a three-member "
        "cluster: the per-image screen in anchored_match is not binding"
    )

    # Note the ANCHOR screen has no case of its own here, and cannot have one:
    # given the reference-multiset guard in `same_multimer_int` and the
    # per-image screen, it is redundant. `pi` is a bijection, so if every image
    # from b = 1 on is reference-compatible and the two multisets agree, the one
    # reference left over on each side is the same one -- the anchor is
    # compatible by counting. It is kept as pruning (it fails a bad anchor
    # immediately instead of after a full descent), not as correctness, which is
    # why deleting it alone changes no answer in this suite.


def test_same_multimer_int_merges_a_cross_reference_pair():
    # The same hetero dimer written from each end: anchored on reference 0 with
    # reference 1 a cell over, versus anchored on reference 1 with reference 0 a
    # cell back. pi = [1, 0] is reference-compatible and no dressing is needed.
    G = _refs(SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1), [0, 1])
    H = _refs(SymOpList([SymOp.identity(), _translate([-12, 0, 0])], 1), [1, 0])
    W_by_ref = [[np.eye(3, dtype=np.int64)], [np.eye(3, dtype=np.int64)]]

    assert matchers.same_multimer_int(G, H, W_by_ref) is True


def test_same_multimer_int_draws_the_dressing_from_the_members_own_reference():
    # As test_same_multimer_int_inversion_needs_W, but split across two
    # references. Working through (dagger), the inversion has to be supplied by
    # the ANCHOR's reference (member 0), not the partner's: with s_0 = I the
    # translation guard wants (-12,0,0) to equal (12,0,0).
    G = _refs(SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1), [0, 1])
    h_partner = SymOp.from_components(
        -np.eye(3, dtype=np.int16), np.array([-12, 0, 0], dtype=np.int16)
    )
    H = _refs(SymOpList([SymOp.identity(), h_partner], 1), [0, 1])

    I = np.eye(3, dtype=np.int64)
    assert matchers.same_multimer_int(G, H, [[I, -I], [I]]) is True
    # the same operation offered on the wrong reference must not be usable
    assert matchers.same_multimer_int(G, H, [[I], [I, -I]]) is False


# Both multi-reference fixtures, because the claim in `dedup/cloud.py` is about
# both: pyrazole is Z' = 2 (Pna2_1) and fomepizole is Z' = 3 (Pca2_1), and
# symmetry detection collapses neither. Before `anchored_match` took one dressing
# group per base reference, every hetero pair in these crystals fell out to
# `same_multimer_geom`.
@pytest.mark.parametrize("name,zprime,R,expected", [
    ("pyrazole",   2, 8.0, PYRAZOLE_DIMERS_R8),
    ("fomepizole", 3, 7.0, FOMEPIZOLE_DIMERS_R7),
])
def test_cross_reference_dimers_no_longer_need_the_cloud_matcher(
    name, zprime, R, expected, monkeypatch
):
    calls = []
    real = cloud.same_multimer_geom
    monkeypatch.setattr(
        cloud, "same_multimer_geom",
        lambda g, h, **kw: (calls.append(1), real(g, h, **kw))[1],
    )
    crystal, _ = load(name)
    # Assert the premise rather than trusting the fixture: if detection ever
    # collapsed these to Z' = 1 the test would still pass while testing nothing.
    assert len(crystal.detect_symmetry(1e-3).get_references()) == zprime, (
        f"{name} is no longer Z' = {zprime}; this test no longer exercises the "
        "cross-reference path"
    )

    mers = generate.generate(crystal, 2, R, **GENERATE_KW)

    assert len(mers) == expected
    assert calls == [], (
        f"{name} (Z'={zprime}): the cloud matcher was still consulted "
        f"{len(calls)} times; the integer matcher should now decide "
        "cross-reference pairs"
    )


@pytest.mark.parametrize("name,N,R", [
    ("pyrazole",   2, 8.0), ("pyrazole",   3, 6.0),
    ("fomepizole", 2, 7.0), ("fomepizole", 3, 6.0), ("fomepizole", 4, 5.5),
])
def test_cross_reference_dedup_stays_complete(name, N, R):
    # Completeness of the multi-reference path, now across orders as well as
    # across Z'. A cross-reference cluster at N = 4 draws on three different
    # dressing groups at once, which is where a per-member indexing error would
    # show up and a dimer would not.

    crystal, _ = load(name)
    mers = generate.generate(crystal, N, R, **GENERATE_KW)
    n_classes, _ = count_chemistry_classes(mers, atol=1e-8)
    assert mers, f"{name} N={N}: nothing generated, test is vacuous"
    assert n_classes == len(mers)


def test_bind_sets_site_symmetry_and_gate():
    crystal, monomer = load("benzene")
    context.bind(crystal, crystal.get_references())

    # Benzene is on an inversion centre: identity + inversion in the dressing.
    assert any(np.array_equal(w, np.eye(3, dtype=np.int64)) for w in context.W_lin_by_ref[0])
    assert any(np.array_equal(w, -np.eye(3, dtype=np.int64)) for w in context.W_lin_by_ref[0])
    # The molecular point group IS richer than the site symmetry ...
    assert len(context.self_iso_by_ref[0]) > len(context.W_lin_by_ref[0])
    # ... but that alone no longer opens the Cartesian congruence gate: benzene
    # is neither linear nor in a Sohncke group, and the branch was measured to
    # yield zero merges for it. See test_float_branch_gate_is_minimal.
    assert context.check_congruence_by_ref[0] is False


def test_are_equivalent_fallback_gated_by_check_congruence(monkeypatch):
    # Two clusters that the integer matcher rejects; the float fallback must run
    # only when check_congruence_by_ref[0] is True. Both clusters default to
    # ref_indices = [0, 0] (mono-reference, base 0).
    context.W_lin_by_ref = [[np.eye(3, dtype=np.int64)]]
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([24, 0, 0])], 1)

    calls = []
    monkeypatch.setattr(equivalence, "fingerprints_match", lambda a, b, **k: True)
    monkeypatch.setattr(
        matchers, "same_multimer", lambda a, b, **k: calls.append(1) or False
    )

    context.check_congruence_by_ref = [False]
    assert equivalence.are_equivalent(G, H) is False
    assert calls == []  # gated off: float fallback not called

    context.check_congruence_by_ref = [True]
    assert equivalence.are_equivalent(G, H) is False
    assert calls == [1]  # gated on: float fallback called once


def test_are_equivalent_integer_path_matches_without_fallback(monkeypatch):
    # A crystallographic merge must be decided by the integer path alone.
    context.W_lin_by_ref = [[np.eye(3, dtype=np.int64)]]
    context.check_congruence_by_ref = [True]
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([-12, 0, 0])], 1)

    def _boom(*a, **k):
        raise AssertionError("float fallback should not run for integer merges")

    monkeypatch.setattr(equivalence, "fingerprints_match", lambda a, b, **k: True)
    monkeypatch.setattr(matchers, "same_multimer", _boom)
    assert equivalence.are_equivalent(G, H) is True


def test_generate_dimer_count_regression():
    crystal, monomer = load("benzene")
    mers = generate.generate(crystal, 2, 5.0, use_origin=True, n_jobs=1)
    assert len(mers) == BENZENE_DIMERS


def test_int_inv_rejects_non_unimodular():
    # det = 2 is not unimodular: no exact integer inverse exists.
    with pytest.raises(AssertionError):
        invariants.int_inv(np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64))


# --- dressing translations (isometries about the centroid, not the origin) ---

UREA_DIMERS_R10 = 18  # urea, N=2, R=10.0, use_origin=True; verified against the
                      # NRE+CHSEV oracle and by explicit Kabsch congruence


def test_self_isometry_linear_part_is_not_a_symmetry():
    # A self-isometry is an isometry about the monomer CENTROID. Applied as a
    # bare linear map about the origin it does NOT map the molecule onto
    # itself unless the centroid happens to be a fixed point. Urea is a case
    # where it does not: the linear part displaces the molecule by several
    # Angstroms, which is exactly why the affine translation part is required.
    crystal, _ = load("urea")
    mon = crystal.get_references()[0]
    X, c = mon.cart_coords, mon.centroid_cart

    def max_mismatch(P):
        return max(np.min(np.linalg.norm(X - p, axis=1)) for p in P)

    worst_linear = 0.0
    for s in mon.point_group:
        worst_linear = max(worst_linear, max_mismatch(X @ s.T))
        # the affine form about the centroid IS a symmetry
        assert max_mismatch((X - c) @ s.T + c) < 1e-8
    assert worst_linear > 1.0  # linear part alone is badly wrong


def test_bind_sets_dressing_translations():
    crystal, _ = load("urea")
    crystal = crystal.detect_symmetry(1e-3)
    context.bind(crystal, crystal.get_references())

    S = context.self_iso_by_ref[0]
    D = context.self_iso_tr_by_ref[0]
    W = context.W_lin_by_ref[0]
    W_tr = context.W_tr_by_ref[0]
    assert len(D) == len(S)
    assert len(W_tr) == len(W)

    c = context.references[0].centroid_cart
    for s, d in zip(S, D):
        assert np.allclose(d, c - s @ c)
    # urea's centroid is NOT a fixed point of its point group, so some
    # dressing translations are nonzero (this is the case the bug missed)
    assert max(float(np.linalg.norm(d)) for d in D) > 1.0
    assert max(int(np.abs(t).max()) for t in W_tr) > 0

    # the integer dressings must fix the centroid EXACTLY (in twelfths)
    c_frac = context.references[0].centroid_frac
    for r, t in zip(W, W_tr):
        assert np.allclose(r @ c_frac + t / 12.0, c_frac, atol=1e-6)


def test_bind_dressing_translations_vanish_for_benzene():
    # Benzene's centroid is a fixed point of its point group, so every
    # dressing translation is zero and the pre-fix code was accidentally right.
    crystal, _ = load("benzene")
    context.bind(crystal, crystal.get_references())
    assert all(np.allclose(d, 0.0) for d in context.self_iso_tr_by_ref[0])
    assert all(not np.any(t) for t in context.W_tr_by_ref[0])


def test_same_multimer_int_needs_dressing_translation():
    # Two urea dimers related by a crystallographic isometry whose dressing
    # does not fix the centroid. They merge only when the dressing carries its
    # translation part; with W_tr omitted the guard is unsatisfiable.
    crystal, _ = load("urea")
    crystal = crystal.detect_symmetry(1e-3)
    refs = crystal.get_references()
    context.bind(crystal, refs)
    W, W_tr = context.W_lin_by_ref[0], context.W_tr_by_ref[0]

    rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.int16)
    g = SymOpList([SymOp.identity(),
                   SymOp.from_components(rot, np.array([-12, -24, 0], dtype=np.int16))], 1)
    h = SymOpList([SymOp.identity(),
                   SymOp.from_components(rot, np.array([-12, 12, 0], dtype=np.int16))], 1)

    assert matchers.same_multimer_int(g, h, W, W_tr) is True
    # without the translation parts the same pair is missed
    assert matchers.same_multimer_int(g, h, W) is False


def test_urea_dimer_count_regression():

    crystal, _ = load("urea")
    mers = generate.generate(crystal, 2, 10.0, **GENERATE_KW)
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
    crystal, _ = load("hexamine")
    crystal = crystal.detect_symmetry(1e-3)
    sg = crystal.space_group
    centroid = crystal.get_references()[0].centroid_frac

    ops = sg.get_unique_sym_ops(centroid)
    stab = sg.get_site_symmetry(centroid)
    assert len(ops) == len(sg.sym_ops) // len(stab)


def test_generate_runs_on_high_symmetry_crystals():
    # These three crash outright if the Wyckoff multiplicity check misfires.
    for name in ("adamantane", "carbon_dioxide", "hexamine"):
        crystal, _ = load(name)
        mers = generate.generate(crystal, 2, 6.0, **GENERATE_KW)
        assert len(mers) > 0


@pytest.mark.xfail(
    strict=True,
    reason="linear monomers are out of scope; deduplication is knowingly "
           "incomplete for them. Kept as the target so a future fix shows up "
           "as XPASS. See the limitation note in crystalatte.work.dedup",
)
def test_linear_monomer_dedup_is_complete():
    # CO2 is linear, so every atom lies ON the molecular axis and a twist about
    # that axis moves nothing. But a placement stores a full 3x3 rotation, so
    # that physically meaningless twist is still compared, and two placements of
    # the identical molecule can fail to match.
    #
    # This used to be absorbed by the tolerant Cartesian matcher, dressed by a
    # finite SAMPLE of CO2's continuous point group. That was removed: the
    # dressings did no physical work (one displaced no atom by more than 0.000
    # A) and it only worked when the needed twist fell inside the sample.
    #
    # Consequence, and what this test pins: 3 duplicate pairs now survive, so
    # n_classes (36) < len(mers) (39). The fix is to compare atoms rather than
    # rotations for linear monomers -- cloud.same_multimer_geom already merges
    # all 108 affected pairs.

    crystal, _ = load("carbon_dioxide")
    mers = generate.generate(crystal, 2, 15.0, **GENERATE_KW)
    n_classes, _ = count_chemistry_classes(mers, atol=1e-8)
    assert n_classes == len(mers)


def test_linear_monomer_emits_a_warning():
    # The one remaining linearity test in the codebase is purely diagnostic: it
    # must warn, and must not change any result.
    with pytest.warns(RuntimeWarning, match="linear molecule"):
        crystal, _ = load("carbon_dioxide")
        generate.generate(crystal, 2, 6.0, **GENERATE_KW)


def test_nonlinear_monomer_emits_no_warning(recwarn):
    crystal, _ = load("benzene")
    generate.generate(crystal, 2, 6.0, **GENERATE_KW)
    assert not [w for w in recwarn if "linear molecule" in str(w.message)]


def test_float_branch_gate_is_minimal():
    # The Cartesian congruence branch is provably needed in ONE case: a Sohncke
    # group with an achiral monomer, where enantiomers force an improper dressing
    # that W_lin, being entirely proper, cannot supply. `len(S) > len(W)` alone is
    # far too weak a gate -- it holds for benzene, urea, imidazole, ice and
    # hexamine, where the branch costs 32-44% of trimer runtime for ZERO merges.
    def _gate(name, **kw):
        crystal, _ = load(name)
        crystal = crystal.detect_symmetry(1e-3)
        context.bind(crystal, crystal.get_references(), **kw)
        return context.check_congruence_by_ref[0]

    assert _gate("ammonia") is True           # P2_13 is Sohncke, NH3 achiral
    # Rich molecular symmetry is NOT on its own a reason to open the gate.
    assert _gate("benzene") is False          # |S| > |W| but not Sohncke
    assert _gate("urea") is False
    assert _gate("ice") is False
    # Linear monomers deliberately no longer open it -- they are out of scope.
    assert _gate("carbon_dioxide") is False


def test_full_congruence_check_is_unconditional():
    # The escape hatch must run the branch for EVERY reference, including those
    # whose detected S is deficient. `len(S) > len(W)` belongs inside the narrow
    # arm: S is detected numerically and under-detects (adamantane finds 2 of 24
    # operations, ice 2 of 6), so guarding the hatch with it made the hatch
    # silently inoperative for exactly the crystals most likely to need it.
    def _gate(name, **kw):
        crystal, _ = load(name)
        crystal = crystal.detect_symmetry(1e-3)
        context.bind(crystal, crystal.get_references(), **kw)
        return context.check_congruence_by_ref[0]

    for name in ("benzene", "carbon_dioxide", "ammonia",
                 "adamantane", "ice", "urea", "hexamine"):
        assert _gate(name, full_congruence_check=True) is True, (
            f"escape hatch inoperative for {name}"
        )


def test_special_position_tolerance_absorbs_cif_rounding():
    # Ice Ih's CIF writes the 3-fold special position (1/3, 2/3, z) as the
    # rounded decimals 0.3333 / 0.6667, putting the monomer centroid ~8.5e-5
    # (fractional) off the exact axis. That is chemically nothing but far above
    # a 1e-6 site-symmetry tolerance, so the 3-fold used to be missed, the
    # molecule was replicated 3x essentially on top of itself, and the
    # resulting "dimers" had coincident atoms.

    crystal, _ = load("ice")
    mers = generate.generate(crystal, 2, 8.0, **GENERATE_KW)
    assert mers

    worst = min(
        float(pdist(np.vstack([m.cart_coords for m in mer.monomers])).min())
        for mer in mers
    )
    assert worst > 0.1, f"coincident atoms in a generated multimer ({worst:.3e} A)"


def test_ice_dedup_is_complete():

    crystal, _ = load("ice")
    mers = generate.generate(crystal, 2, 8.0, **GENERATE_KW)
    n_classes, _ = count_chemistry_classes(mers, atol=1e-8)
    assert n_classes == len(mers)


# --- Kabsch conditioning -----------------------------------------------------

def _random_rotation(seed, improper=False):
    rng = np.random.default_rng(seed)
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if (np.linalg.det(R) < 0) != improper:
        R[:, 0] *= -1
    return R


def test_kabsch_recovers_a_known_rotation():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(12, 3)); P -= P.mean(axis=0)
    R = _random_rotation(1)
    assert np.allclose(cloud.kabsch(P, P @ R.T), R, atol=1e-10)


def test_kabsch_allows_reflection():
    # Deliberate deviation from textbook Kabsch, which forces det = +1: a
    # mirror-image cluster has the same interaction energy, so enantiomers
    # must merge. Reflections have to be recoverable, not corrected away.
    rng = np.random.default_rng(2)
    P = rng.normal(size=(12, 3)); P -= P.mean(axis=0)
    R = _random_rotation(3, improper=True)
    fitted = cloud.kabsch(P, P @ R.T)
    assert np.linalg.det(fitted) < 0
    assert np.allclose(fitted, R, atol=1e-10)


def test_kabsch_returns_none_on_non_finite_input():
    P = np.zeros((3, 3)); P[0, 0] = np.nan
    assert cloud.kabsch(P, np.eye(3)) is None


def test_kabsch_full_correspondence_beats_a_thin_anchor_triple():
    # A triple that only just clears the non-collinearity threshold is a poor
    # basis to fit a rotation from: the error is amplified by the lever arm out
    # to distant atoms, and can exceed the atol the fit is judged against.
    # Re-fitting on the full correspondence removes that dependence.
    # The property is about the WORST case over configurations, not any single
    # draw: a thin triple is sometimes accidentally well-oriented.
    worst_triple = worst_full = 0.0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        A = rng.normal(scale=7.0, size=(24, 3))
        A[0], A[1], A[2] = [0, 0, 0], [1.0, 0, 0], [0, 2e-3, 0]  # near-degenerate
        A -= A.mean(axis=0)
        R = _random_rotation(seed + 100)
        B = A @ R.T + rng.normal(scale=1e-6, size=A.shape)
        B -= B.mean(axis=0)
        worst_triple = max(worst_triple, np.abs(A @ cloud.kabsch(A[:3], B[:3]).T - B).max())
        worst_full = max(worst_full, np.abs(A @ cloud.kabsch(A, B).T - B).max())

    assert worst_full < worst_triple / 10
    assert worst_full < 1e-4        # comfortably inside same_multimer_geom's atol


# ---------------------------------------------------------------------------
# Over-merging
#
# The dedup can fail in two directions and they are NOT symmetric:
#
#   under-merge  a class survives as several representatives. Harmless to the
#                lattice energy (multiplicities are conserved and the pieces are
#                congruent, so sum(m_i/N * E_i) is unchanged) and visible as
#                n_classes < len(mers).
#   over-merge   two genuinely distinct clusters are declared identical, one is
#                discarded, and the survivor absorbs its multiplicity. The
#                lattice sum is then WRONG by (m_B/N)(E_A - E_B), and the usual
#                `n_classes == len(mers)` check CANNOT see it -- a false merge
#                lowers both sides together.
#
# The exact-count rows in test_generate.py do catch over-merging indirectly (a
# spurious merge drops the count below the locked value), but only for those
# crystals at those radii. The tests below check the property directly and
# independently of the dedup's own logic.
# ---------------------------------------------------------------------------


def _cluster_referee(g_N):
    """``(NRE, CHSEV)`` for a cluster, computed from its reconstructed atoms."""
    # Both are isometry invariants the dedup path never consults. NRE depends
    # only on the Z-tagged distance multiset that fingerprints_match already
    # gates on, so it's a cheap first check; CHSEV also sees how distances are
    # arranged and catches over-merges past the gate (see
    # test_chsev_separates_a_homometric_pair_that_nre_cannot).
    atoms = cluster_atoms_cart(g_N)
    symbols = np.concatenate(
        [context.references[g_N.ref_indices[k]].symbols for k in range(len(g_N))]
    )
    bohr = atoms / 0.52917721067
    return float(_nre(symbols, bohr)), _chsev(symbols, bohr)


def test_chsev_separates_a_homometric_pair_that_nre_cannot():
    # Locks in the property that makes CHSEV the load-bearing half of the
    # referee above. {0,1,4,10,12,17} and {0,1,8,11,13,17} are homometric: their
    # difference multisets coincide, so the two clouds have identical
    # interatomic distances -- identical distance fingerprints, and therefore
    # identical NRE. Any over-merge between such a pair is invisible to an
    # NRE-only referee. CHSEV separates them.
    #
    # If this ever fails, the referee in the over-merge test has quietly lost
    # its teeth and is no stronger than the gate it is supposed to be checking.

    A = np.array([[0.0, 0, 0], [1, 0, 0], [4, 0, 0], [10, 0, 0], [12, 0, 0], [17, 0, 0]])
    B = np.array([[0.0, 0, 0], [1, 0, 0], [8, 0, 0], [11, 0, 0], [13, 0, 0], [17, 0, 0]])
    sym = ["C"] * 6

    def distances(P):
        i, j = np.triu_indices(len(P), 1)
        return np.sort(np.linalg.norm(P[i] - P[j], axis=1))

    assert np.allclose(distances(A), distances(B), atol=1e-12), \
        "the pair is no longer homometric, so it no longer tests anything"
    assert abs(_nre(sym, A) - _nre(sym, B)) < 1e-9, \
        "NRE must be blind here -- it is a function of the distance multiset"
    assert np.abs(_chsev(sym, A) - _chsev(sym, B)).max() > 1.0, \
        "CHSEV must separate them, or it adds nothing over NRE"


# Parametrized over ORDER as well as crystal. The direct over-merge check used
# to run only at N = 2, which left the regime where the anchored search works
# hardest -- its cost is O(|S| n^3) in cluster size and the permutation
# bookkeeping gets structurally harder with order -- entirely unrefereed. The
# higher-order rows are deliberately at tighter R so they stay cheap: the point
# is to exercise the matcher on 3- and 4-member permutations, not to enumerate
# a large shell.
@pytest.mark.parametrize("name,N,R", [
    ("benzene", 2, 8.0), ("urea", 2, 8.0), ("ice", 2, 7.0),
    ("adamantane", 2, 8.0), ("hexamine", 2, 9.0), ("ammonia", 2, 7.0),
    ("benzene", 3, 7.0), ("urea", 3, 7.0), ("pyrazole", 3, 6.0),
    ("benzene", 4, 6.0), ("ammonia", 4, 5.0),
    ("fomepizole", 2, 7.0), ("fomepizole", 3, 6.0),   # Z' = 3
    ("benzene", 5, 5.5), ("urea", 5, 5.5),
])
def test_no_over_merge_every_discarded_cluster_matches_a_survivor(name, N, R, monkeypatch):
    # Direct over-merge test. Every cluster the deduper THREW AWAY must be
    # genuinely equivalent to one it KEPT; otherwise a distinct interaction was
    # silently deleted and another double-counted. Refereed by the nuclear
    # repulsion energy, which no part of the dedup path consults.
    #
    # `generate_neighbors` returns only the survivors, so the discarded clusters
    # are collected one layer down, where they still exist: `deduplicate_bucket`
    # returns `(unique, duplicate)` per bucket. Wrapping it keeps this test at
    # full strength without asking the enumeration to carry the duplicates back
    # out, and the wrapper is pure observation -- the real function still does
    # the work and its result is passed through untouched.

    crystal, _ = load(name)
    crystal = crystal.detect_symmetry(1e-3)
    refs = crystal.get_references()
    bounds = np.array([[-1, 1], [-1, 1], [-1, 1]])

    duplicate = []
    real_dedup = neighbors.deduplicate_bucket

    def _capture(bucket):
        local_unique, local_dup = real_dedup(bucket)
        duplicate.extend(local_dup)
        return local_unique, local_dup

    monkeypatch.setattr(neighbors, "deduplicate_bucket", _capture)
    unique = neighbors.generate_neighbors(
        crystal, refs, bounds, R, N, **GENERATE_KW
    )
    assert duplicate, f"{name} N={N}: no duplicates generated, test is vacuous"

    survivors = [_cluster_referee(u) for u in unique]
    for d in duplicate:
        nre_d, chsev_d = _cluster_referee(d)
        near_nre = [c for n, c in survivors if abs(nre_d - n) < 1e-6]
        assert near_nre, (
            f"{name} N={N}: a discarded cluster (NRE={nre_d:.9f}) matches no "
            f"survivor on NRE (closest is "
            f"{min(abs(nre_d - n) for n, _ in survivors):.3e} away) -- this is "
            f"an OVER-MERGE: a distinct multimer was deleted and another's "
            f"multiplicity inflated"
        )
        gap = min(float(np.abs(chsev_d - c).max()) for c in near_nre)
        assert gap < 1e-6, (
            f"{name} N={N}: a discarded cluster matches a survivor on NRE but "
            f"NOT on CHSEV (closest spectrum is {gap:.3e} away) -- this is an "
            f"OVER-MERGE of two HOMOMETRIC clusters: same interatomic distances, "
            f"different structure. An NRE-only referee cannot see this case, "
            f"which is exactly why CHSEV is checked too"
        )


def test_no_over_merge_multiplicity_is_conserved(monkeypatch):
    # Whatever the dedup decides, every candidate must be accounted for exactly
    # once. A bug that dropped clusters outright would show here even if the
    # survivors happened to look self-consistent.
    #
    # Stated against the candidates going IN rather than the duplicates coming
    # out, so it needs nothing from `generate_neighbors` beyond the survivors:
    # every candidate is either a survivor or folded into one's multiplicity,
    # so the multiplicities must sum to the number of candidates enumerated.

    crystal, _ = load("benzene")
    crystal = crystal.detect_symmetry(1e-3)
    refs = crystal.get_references()

    candidates = 0
    real_dedup = neighbors.deduplicate_bucket

    def _count(bucket):
        nonlocal candidates
        candidates += len(bucket)
        return real_dedup(bucket)

    monkeypatch.setattr(neighbors, "deduplicate_bucket", _count)
    unique = neighbors.generate_neighbors(
        crystal, refs, np.array([[-1, 1], [-1, 1], [-1, 1]]), 8.0, 2,
        **GENERATE_KW,
    )
    assert candidates, "no candidates enumerated, test is vacuous"
    assert sum(u.multiplicity for u in unique) == candidates


@pytest.mark.parametrize("name", [
    "benzene", "uracil", "formamide", "cytosine", "ammonia", "acetic_acid",
    "imidazole", "urea", "adamantane", "carbon_dioxide", "hexamine", "ice",
])
def test_every_dressing_is_a_symmetry_of_its_monomer(name):
    # The ONLY protection against over-merging lives in how the dressing groups
    # are built. `anchored_match` trusts whatever group it is handed -- it solves
    # the relative-transform equations over that group and never asks whether a
    # dressing is a real symmetry. So the invariant that matters is checked here:
    # every stored dressing must actually map its monomer onto itself.
    #
    # W is derived from the space group, so it is EXACT (a site operation
    # necessarily permutes the atoms of the molecule it fixes).
    # S is detected numerically at atol=0.05, so its operations are only good to
    # that tolerance -- measured worst case 0.0482 A on imidazole. That gap is
    # the over-merge exposure; loosening detection widens it, and this test is
    # what would notice.

    crystal, _ = load(name)
    crystal = crystal.detect_symmetry(1e-3)
    refs = crystal.get_references()
    context.bind(crystal, refs)

    mon = refs[0]
    X = np.asarray(mon.cart_coords, float)
    pos = X - X.mean(axis=0)
    Z = np.array([qcel.periodictable.to_Z(s) for s in mon.symbols])
    A = np.asarray(crystal.lattice_vectors, float)

    def worst_displacement(ops):
        worst = 0.0
        for u in ops:
            Y = pos @ np.asarray(u, float).T
            for i in range(len(pos)):
                same_z = [j for j in range(len(pos)) if Z[j] == Z[i]]
                worst = max(worst, min(np.linalg.norm(pos[j] - Y[i]) for j in same_z))
        return worst

    W_cart = [A @ w.astype(float) @ np.linalg.inv(A)
              for w in context.W_lin_by_ref[0]]
    dW = worst_displacement(W_cart)
    dS = worst_displacement(context.self_iso_by_ref[0])

    assert dW < 1e-3, (
        f"{name}: a site-symmetry dressing displaces an atom by {dW:.4f} A. "
        "W is supposed to be exact by construction -- this means the site "
        "symmetry does not actually preserve the molecule."
    )
    assert dS <= 0.05 + 1e-9, (
        f"{name}: a molecular dressing displaces an atom by {dS:.4f} A, beyond "
        "the 0.05 A detection tolerance. Dressings this far from a real symmetry "
        "can merge genuinely distinct clusters (over-merge), which corrupts the "
        "lattice sum and is invisible to the n_classes == len(mers) check."
    )


@pytest.mark.parametrize("name", [
    "benzene", "uracil", "formamide", "cytosine", "ammonia", "acetic_acid",
    "imidazole", "urea", "adamantane", "carbon_dioxide", "hexamine", "ice",
])
def test_bound_dressing_groups_satisfy_the_containment_theorem(name):
    # W_lin <= S is a theorem, and `bind` is where the two groups meet, so this
    # is where the containment has to actually hold. It is the previous test's
    # companion: that one checks no dressing is spurious (no over-detection),
    # this one checks none is missing (no under-detection). The first protects
    # the lattice energy; the second protects against redundant QC jobs.
    crystal, _ = load(name)
    crystal = crystal.detect_symmetry(1e-3)
    refs = crystal.get_references()
    context.bind(crystal, refs)

    A = np.asarray(crystal.lattice_vectors, float)
    A_inv = np.linalg.inv(A)
    for i in range(len(refs)):
        S = context.self_iso_by_ref[i]
        for w in context.W_lin_by_ref[i]:
            w_cart = A @ np.asarray(w, float) @ A_inv
            assert any(np.allclose(w_cart, s, atol=1e-3) for s in S), (
                f"{name}: reference {i} has a site operation absent from S. "
                "Any merge that needs it as a dressing is lost."
            )


# ---------------------------------------------------------------------------
# Parallel dedup path
#
# generate_neighbors only spawns a pool when the estimated pairwise work clears
# PARALLEL_MIN_PAIR_WORK, because below that the pool costs more than it saves.
# No crystal in this suite is anywhere near that threshold, so without forcing
# it the parallel branch is never executed by the tests -- which is exactly how
# an ImportError in that branch reached a production-scale benchmark unnoticed.
# ---------------------------------------------------------------------------


def test_parallel_branch_executes_and_agrees_with_serial(monkeypatch):

    crystal, _ = load("benzene")
    crystal = crystal.detect_symmetry(1e-3)
    refs = crystal.get_references()
    bounds = np.array([[-1, 1], [-1, 1], [-1, 1]])

    serial_u = neighbors.generate_neighbors(
        crystal, refs, bounds, 8.0, 2, **GENERATE_KW
    )

    # force the pool on despite the small workload
    monkeypatch.setattr(neighbors, "PARALLEL_MIN_PAIR_WORK", 0)
    par_u = neighbors.generate_neighbors(
        crystal, refs, bounds, 8.0, 2, use_origin=True, n_jobs=2, verbose=0
    )

    assert len(par_u) == len(serial_u)
    # No separate duplicate-count assertion is needed: the multiplicities sum to
    # the candidate total either way, so equal multiplicity multisets already
    # pin the number of duplicates the two paths folded away.
    assert sorted(u.multiplicity for u in par_u) == \
           sorted(u.multiplicity for u in serial_u)


def test_parallel_is_skipped_below_the_work_threshold(monkeypatch):
    # The threshold must actually gate: a small run with n_jobs > 1 should never
    # reach the pool. Poison ProcessPoolExecutor so that touching it fails.

    def _boom(*a, **k):
        raise AssertionError("pool started for a workload below the threshold")

    monkeypatch.setattr(cf, "ProcessPoolExecutor", _boom)
    monkeypatch.setattr(neighbors, "PARALLEL_MIN_PAIR_WORK", 10**12)

    crystal, _ = load("benzene")
    crystal = crystal.detect_symmetry(1e-3)
    refs = crystal.get_references()
    unique = neighbors.generate_neighbors(
        crystal, refs, np.array([[-1, 1], [-1, 1], [-1, 1]]), 6.0, 2,
        use_origin=True, n_jobs=8, verbose=0,
    )
    assert unique
