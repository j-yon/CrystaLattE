import numpy as np
import pytest
import qcelemental as qcel
from scipy.spatial.distance import pdist

from crystalatte import count_chemistry_classes
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp, SymOpList
from crystalatte.work import generate, neighbors
from crystalatte.work.dedup import context, dressing, invariants
from crystalatte.work.validate import _chsev, _nre

from conftest import GENERATE_KW, load

#: benzene, N = 2, R = 5.0, anchor-relative convention. Captured before the
#: integer matcher replaced the float one and unchanged by it, so it pins
#: agreement between the two implementations rather than just the current one.
BENZENE_DIMERS = 3
#: pyrazole is Z' = 2 (Pna2_1, and detection does not collapse it), so it is the
#: regression case for the cross-reference path. Captured while the point-cloud
#: matcher (since removed) still decided those pairs, so it pins agreement with it.
PYRAZOLE_DIMERS_R8 = 42
#: fomepizole is Z' = 3 (Pca2_1), the only fixture with three independent
#: molecules; with pyrazole it pins both multi-reference crystals to the merges
#: the removed point-cloud matcher used to make.
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


def _same_label(G, H, W_by_ref, W_tr_by_ref=None):
    """Return whether ``orbit_labels`` puts ``G`` and ``H`` in one class under the given dressings

    Binds only what the orbit key reads: each reference's stacked ``(W, W^-1, W_tr)``. The label
    is exact, so it decides the same matching condition the removed ``same_multimer_int`` did.
    """
    stacked = []
    for r, W in enumerate(W_by_ref):
        W = np.stack([np.asarray(w, dtype=np.int64) for w in W])
        tr = (
            np.zeros((len(W), 3), dtype=np.int64)
            if W_tr_by_ref is None
            else np.stack([np.asarray(t, dtype=np.int64) for t in W_tr_by_ref[r]])
        )
        stacked.append((W, invariants.int_inv(W), tr))
    context.dress_stacked_by_ref = stacked
    labels = invariants.orbit_labels([G, H])
    return bool(labels[0] == labels[1])


def _site(i: int = 0):
    """``(W, W_tr)`` for bound reference ``i``, as :func:`context.bind` builds it"""
    return dressing.site_symmetry(
        context.active_crystal.space_group, context.references[i].centroid_frac
    )


def test_orbit_label_translation_swap():
    # Same dimer counted from each monomer: [I, T(+1)] ~ [I, T(-1)]. W = {I}.
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([-12, 0, 0])], 1)
    assert _same_label(G, H, [[np.eye(3, dtype=np.int64)]]) is True


def test_orbit_label_non_match():
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([24, 0, 0])], 1)
    assert _same_label(G, H, [[np.eye(3, dtype=np.int64)]]) is False


def test_orbit_label_inversion_needs_W():
    # G = [I, T(+1)];  H = [I, i0.T(+1)] with partner rot = -I, tr = (-1,0,0).
    # Same multimer only via s0 = -I, so it merges iff -I is in the dressing W.
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    h_partner = SymOp.from_components(
        -np.eye(3, dtype=np.int16), np.array([-12, 0, 0], dtype=np.int16)
    )
    H = SymOpList([SymOp.identity(), h_partner], 1)

    assert _same_label(G, H, [[np.eye(3, dtype=np.int64)]]) is False

    assert _same_label(
        G, H, [[np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)]]
    ) is True


# ---------------------------------------------------------------------------
# Multi-reference (Z' > 1) support in the exact orbit key
#
# docs/dedup_correctness.tex Def. 2 is multi-reference: it draws each dressing
# from D_{ref_G(a)} and requires pi to be reference-compatible. The orbit key
# realises both: members are sorted by base reference, anchors are drawn from
# the smallest reference and dressed by its W, and each partner is reduced to
# its coset under its OWN reference's W. These cases were written against the
# removed `same_multimer_int` and are kept as the label's specification.
# ---------------------------------------------------------------------------


def _refs(sym_op_list, indices):
    sym_op_list.ref_indices = np.array(indices, dtype=np.int64)
    return sym_op_list


def test_orbit_label_requires_reference_compatibility():
    # Identical placement algebra in both halves; only the reference labels
    # differ. The permutation must map each member onto a member drawn from the
    # SAME base reference (Def. 2), so the first pair merges and the second --
    # a ref0-with-ref0 dimer against a ref1-with-ref1 dimer, i.e. two different
    # molecules -- must not.
    W_by_ref = [[np.eye(3, dtype=np.int64)], [np.eye(3, dtype=np.int64)]]

    def dimer(indices):
        return _refs(SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1), indices)

    assert _same_label(dimer([0, 1]), dimer([0, 1]), W_by_ref) is True
    assert _same_label(dimer([0, 0]), dimer([1, 1]), W_by_ref) is False


def test_orbit_label_reference_compatibility_binds_the_permutation():
    # The second assertion above is decided by the reference multiset alone:
    # `orbit_labels` keys clusters in groups of equal sorted references, so a
    # ref0-ref0 dimer and a ref1-ref1 dimer are never even compared.
    #
    # This case shares the multiset. Both clusters draw on references {0, 1}
    # exactly once, and the placement algebra is IDENTICAL; only the order of
    # the labels differs. The key sorts members by reference, which pins the
    # ref-0 member as the anchor, so it compares Delta^G_01 = (I, +t) against
    # Delta^H_10 = (I, -t) and correctly keeps them apart. Keying in member
    # order instead would see (I, +t) twice and merge two different hetero dimers.
    W_by_ref = [[np.eye(3, dtype=np.int64)], [np.eye(3, dtype=np.int64)]]

    def dimer(indices):
        return _refs(SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1), indices)

    g, h = dimer([0, 1]), dimer([1, 0])
    assert sorted(g.ref_indices.tolist()) == sorted(h.ref_indices.tolist()), (
        "the reference multisets must agree here, or this test degenerates into "
        "the one above"
    )
    assert _same_label(g, h, W_by_ref) is False, (
        "a reference-incompatible permutation was accepted: member 0 of one "
        "cluster was matched to a member of the other drawn from a different "
        "base reference, which merges two genuinely different hetero dimers"
    )

    # A THREE-member case: with refG = [0, 0, 1] against refH = [0, 1, 0] there
    # are two members to place after the anchor, so a correctly chosen ref-0
    # anchor still leaves G's second ref-0 member free to be paired with H's
    # ref-1 member. The key must keep each partner in its own reference's slot.
    # The reference multiset is unchanged, so grouping alone cannot decide it.
    def trimer(indices):
        return _refs(
            SymOpList(
                [SymOp.identity(), _translate([12, 0, 0]), _translate([0, 24, 0])], 1
            ),
            indices,
        )

    assert _same_label(trimer([0, 0, 1]), trimer([0, 1, 0]), W_by_ref) is False, (
        "a reference-incompatible permutation was accepted on a three-member "
        "cluster: partners are not keyed by their own base reference"
    )


def test_orbit_label_merges_a_cross_reference_pair():
    # The same hetero dimer written from each end: anchored on reference 0 with
    # reference 1 a cell over, versus anchored on reference 1 with reference 0 a
    # cell back. pi = [1, 0] is reference-compatible and no dressing is needed.
    G = _refs(SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1), [0, 1])
    H = _refs(SymOpList([SymOp.identity(), _translate([-12, 0, 0])], 1), [1, 0])
    W_by_ref = [[np.eye(3, dtype=np.int64)], [np.eye(3, dtype=np.int64)]]

    assert _same_label(G, H, W_by_ref) is True


def test_orbit_label_draws_the_dressing_from_the_members_own_reference():
    # As test_orbit_label_inversion_needs_W, but split across two references.
    # Working through (dagger), the inversion has to be supplied by the ANCHOR's
    # reference (member 0), not the partner's: with s_0 = I the translations
    # (-12,0,0) and (12,0,0) disagree.
    G = _refs(SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1), [0, 1])
    h_partner = SymOp.from_components(
        -np.eye(3, dtype=np.int16), np.array([-12, 0, 0], dtype=np.int16)
    )
    H = _refs(SymOpList([SymOp.identity(), h_partner], 1), [0, 1])

    I = np.eye(3, dtype=np.int64)
    assert _same_label(G, H, [[I, -I], [I]]) is True
    # the same operation offered on the wrong reference must not be usable
    assert _same_label(G, H, [[I], [I, -I]]) is False


# Both multi-reference fixtures: pyrazole is Z' = 2 (Pna2_1) and fomepizole is
# Z' = 3 (Pca2_1), and symmetry detection collapses neither. Every hetero pair
# in these crystals once fell out to the point-cloud matcher; the exact orbit
# key now decides them alone, and must still reproduce the merges it made.
@pytest.mark.parametrize("name,zprime,R,expected", [
    ("pyrazole",   2, 8.0, PYRAZOLE_DIMERS_R8),
    ("fomepizole", 3, 7.0, FOMEPIZOLE_DIMERS_R7),
])
def test_cross_reference_dimer_counts(name, zprime, R, expected):
    crystal = load(name)
    # Assert the premise rather than trusting the fixture: if detection ever
    # collapsed these to Z' = 1 the test would still pass while testing nothing.
    assert len(crystal.detect_symmetry(1e-3).get_references()) == zprime, (
        f"{name} is no longer Z' = {zprime}; this test no longer exercises the "
        "cross-reference path"
    )

    mers = generate.generate(crystal, 2, R, **GENERATE_KW)

    assert len(mers) == expected


@pytest.mark.parametrize("name,N,R", [
    ("pyrazole",   2, 8.0), ("pyrazole",   3, 6.0),
    ("fomepizole", 2, 7.0), ("fomepizole", 3, 6.0), ("fomepizole", 4, 5.5),
])
def test_cross_reference_dedup_stays_complete(name, N, R):
    # Completeness of the multi-reference path, now across orders as well as
    # across Z'. A cross-reference cluster at N = 4 draws on three different
    # dressing groups at once, which is where a per-member indexing error would
    # show up and a dimer would not.

    crystal = load(name)
    mers = generate.generate(crystal, N, R, **GENERATE_KW)
    n_classes, _ = count_chemistry_classes(mers, atol=1e-8)
    assert mers, f"{name} N={N}: nothing generated, test is vacuous"
    assert n_classes == len(mers)


def test_bind_sets_site_symmetry():
    crystal = load("benzene")
    context.bind(crystal, crystal.get_references())

    # Benzene is on an inversion centre: identity + inversion in the dressing.
    assert any(np.array_equal(w, np.eye(3, dtype=np.int64)) for w in _site(0)[0])
    assert any(np.array_equal(w, -np.eye(3, dtype=np.int64)) for w in _site(0)[0])
    # Its molecular point group is richer than the site symmetry, but only
    # approximately: none of the extra operations is exact, so the dressing
    # group is the site symmetry alone.
    assert len(context.dress_stacked_by_ref[0][0]) == len(_site(0)[0])


def test_generate_dimer_count_regression():
    crystal = load("benzene")
    mers = generate.generate(crystal, 2, 5.0, use_origin=True)
    assert len(mers) == BENZENE_DIMERS


def test_int_inv_rejects_non_unimodular():
    # det = 2 is not unimodular: no exact integer inverse exists.
    with pytest.raises(AssertionError):
        invariants.int_inv(np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64))


# --- dressing translations (isometries about the centroid, not the origin) ---

UREA_DIMERS_R10 = 18  # urea, N=2, R=10.0, use_origin=True; verified against the
                      # NRE+CHSEV oracle and by explicit Kabsch congruence


def test_bind_sets_dressing_translations():
    crystal = load("urea")
    crystal = crystal.detect_symmetry(1e-3)
    context.bind(crystal, crystal.get_references())

    D, _, D_tr = context.dress_stacked_by_ref[0]
    W = _site(0)[0]
    W_tr = _site(0)[1]
    assert len(D_tr) == len(D)
    assert len(W_tr) == len(W)

    # urea's centroid is NOT a fixed point of its point group about the origin,
    # so some dressing translations are nonzero (this is the case the bug missed)
    assert max(int(np.abs(t).max()) for t in D_tr) > 0

    # the integer dressings must fix the centroid EXACTLY (in twelfths)
    c_frac = context.references[0].centroid_frac
    for r, t in zip(D, D_tr):
        assert np.allclose(r @ c_frac + t / 12.0, c_frac, atol=1e-6)


def test_bind_dressing_translations_vanish_for_benzene():
    # Benzene's centroid is a fixed point of its point group, so every
    # dressing translation is zero and the pre-fix code was accidentally right.
    crystal = load("benzene")
    context.bind(crystal, crystal.get_references())
    assert not np.any(context.dress_stacked_by_ref[0][2])
    assert all(not np.any(t) for t in _site(0)[1])


def test_orbit_label_needs_dressing_translation():
    # Two urea dimers related by a crystallographic isometry whose dressing
    # does not fix the centroid. They merge only when the dressing carries its
    # translation part; with W_tr zeroed the cosets cannot coincide.
    crystal = load("urea")
    crystal = crystal.detect_symmetry(1e-3)
    refs = crystal.get_references()
    context.bind(crystal, refs)
    W, W_tr = _site(0)

    rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.int16)
    g = SymOpList([SymOp.identity(),
                   SymOp.from_components(rot, np.array([-12, -24, 0], dtype=np.int16))], 1)
    h = SymOpList([SymOp.identity(),
                   SymOp.from_components(rot, np.array([-12, 12, 0], dtype=np.int16))], 1)

    assert _same_label(g, h, [W], [W_tr]) is True
    # without the translation parts the same pair is missed
    assert _same_label(g, h, [W]) is False


def test_urea_dimer_count_regression():

    crystal = load("urea")
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
    crystal = load("hexamine")
    crystal = crystal.detect_symmetry(1e-3)
    sg = crystal.space_group
    centroid = crystal.get_references()[0].centroid_frac

    ops = sg.get_unique_sym_ops(centroid)
    stab = sg.get_site_symmetry(centroid)
    assert len(ops) == len(sg.sym_ops) // len(stab)


def test_generate_runs_on_high_symmetry_crystals():
    # These three crash outright if the Wyckoff multiplicity check misfires.
    for name in ("adamantane", "carbon_dioxide", "hexamine"):
        crystal = load(name)
        mers = generate.generate(crystal, 2, 6.0, **GENERATE_KW)
        assert len(mers) > 0


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
    # That left 3 duplicate pairs surviving at R = 15 (n_classes 36 < 39). They
    # are merged now by the dressing group: CO2 lies on the cubic cell's [111]
    # axis, so the 12 cell-symmetry operations fixing that axis are exact
    # self-isometries and the orbit key uses them (see dedup.dressing). Linear
    # dedup is still incomplete in general -- a twist about the axis that is not
    # a cell symmetry is never a dressing, and one duplicate pair survives at
    # R = 20 -- but at this radius the result is complete, and this pins it.

    crystal = load("carbon_dioxide")
    mers = generate.generate(crystal, 2, 15.0, **GENERATE_KW)
    n_classes, _ = count_chemistry_classes(mers, atol=1e-8)
    assert n_classes == len(mers)


def test_linear_monomer_emits_a_warning():
    # The one remaining linearity test in the codebase is purely diagnostic: it
    # must warn, and must not change any result.
    with pytest.warns(RuntimeWarning, match="linear molecule"):
        crystal = load("carbon_dioxide")
        generate.generate(crystal, 2, 6.0, **GENERATE_KW)


def test_nonlinear_monomer_emits_no_warning(recwarn):
    crystal = load("benzene")
    generate.generate(crystal, 2, 6.0, **GENERATE_KW)
    assert not [w for w in recwarn if "linear molecule" in str(w.message)]


def test_special_position_tolerance_absorbs_cif_rounding():
    # Ice Ih's CIF writes the 3-fold special position (1/3, 2/3, z) as the
    # rounded decimals 0.3333 / 0.6667, putting the monomer centroid ~8.5e-5
    # (fractional) off the exact axis. That is chemically nothing but far above
    # a 1e-6 site-symmetry tolerance, so the 3-fold used to be missed, the
    # molecule was replicated 3x essentially on top of itself, and the
    # resulting "dimers" had coincident atoms.

    crystal = load("ice")
    mers = generate.generate(crystal, 2, 8.0, **GENERATE_KW)
    assert mers

    worst = min(
        float(pdist(np.vstack([m.cart_coords for m in mer.monomers])).min())
        for mer in mers
    )
    assert worst > 0.1, f"coincident atoms in a generated multimer ({worst:.3e} A)"


def test_ice_dedup_is_complete():

    crystal = load("ice")
    mers = generate.generate(crystal, 2, 8.0, **GENERATE_KW)
    n_classes, _ = count_chemistry_classes(mers, atol=1e-8)
    assert n_classes == len(mers)


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
    """``(NRE, CHSEV)`` for a cluster, computed from its reconstructed atoms.

    Both are isometry invariants that no part of the dedup path consults, so
    they can referee it. They are NOT equally strong, and the difference is the
    whole reason CHSEV is here:

    - NRE is ``sum_{i<j} Z_i Z_j / r_ij`` -- a pure function of the multiset of
      Z-tagged interatomic distances, so it agrees to machine precision on ANY
      two clusters with the same distances, including homometric ones that are
      structurally different. It cannot see an over-merge between such a pair.
    - CHSEV is the sorted eigenvalue spectrum of the Coulomb matrix. The same
      distances appear in it, but the eigenvalues depend on how they are
      ARRANGED in the matrix, not merely on their multiset, so CHSEV sees inside
      the gate. See
      :func:`test_chsev_separates_a_homometric_pair_that_nre_cannot`, which
      exhibits two clouds with identical fingerprints and identical NRE whose
      CHSEV differs by 4.7.

    NRE is kept alongside it as a cheap first discriminator; CHSEV is the one
    that closes the gap.
    """

    parts = [
        op.apply(context.references[r].frac_coords) for r, op in zip(g_N.ref_indices, g_N)
    ]
    atoms = context.active_crystal.to_cartesian(np.vstack(parts))
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


#: enumeration box half-width (cells) for referee rows whose merges sit beyond one cell
_REFEREE_BOX = {("carbon_dioxide", 2, 15.0): 2}


# Parametrized over ORDER as well as crystal. The direct over-merge check used
# to run only at N = 2, which left the regime where the anchored search works
# hardest -- its cost is O(|S| n^3) in cluster size and the permutation
# bookkeeping gets structurally harder with order -- entirely unrefereed. The
# higher-order rows are deliberately at tighter R so they stay cheap: the point
# is to exercise the matcher on 3- and 4-member permutations, not to enumerate
# a large shell. The pentamer rows sit at R = 8 because every pair of members
# must lie within R (2242715); at the old 5.5 no pentamer qualifies at all.
@pytest.mark.parametrize("name,N,R", [
    ("benzene", 2, 8.0), ("urea", 2, 8.0),
    pytest.param(
        "ice", 2, 7.0,
        marks=pytest.mark.xfail(
            strict=True,
            reason="the exact orbit key over-merges proton-disordered ice "
                   "(see the ice row in test_generate.py); open decision",
        ),
    ),
    ("adamantane", 2, 8.0), ("hexamine", 2, 9.0), ("ammonia", 2, 7.0),
    ("benzene", 3, 7.0), ("urea", 3, 7.0), ("pyrazole", 3, 6.0),
    ("benzene", 4, 6.0), ("ammonia", 4, 5.0),
    ("fomepizole", 2, 7.0), ("fomepizole", 3, 6.0),   # Z' = 3
    ("benzene", 5, 8.0), ("urea", 5, 8.0),
    # extra dressings beyond the site symmetry: each row has merges that only the
    # exact cell-symmetry self-isometries make (triazine's sigma_h, CO2's D3d)
    ("triazine", 3, 10.0), ("carbon_dioxide", 2, 15.0),
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

    crystal = load(name)
    crystal = crystal.detect_symmetry(1e-3)
    refs = crystal.get_references()
    box = _REFEREE_BOX.get((name, N, R), 1)
    bounds = np.array([[-box, box]] * 3)

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

    crystal = load("benzene")
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
    # are built. The orbit key trusts whatever group it is handed -- it
    # canonicalises over that group and never asks whether a dressing is a real
    # symmetry. So the invariant that matters is checked here: every stored
    # dressing must actually map its monomer onto itself.
    #
    # W is derived from the space group, so it is exact up to the CIF's own
    # rounding (ice's reconstructed H atoms fit to ~2.7e-4 A). The extra
    # dressings are accepted only when exact to 1e-6 A (dedup.dressing), so the
    # bound below holds for all of D.

    crystal = load(name)
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

    D_cart = [A @ d.astype(float) @ np.linalg.inv(A)
              for d in context.dress_stacked_by_ref[0][0]]
    dD = worst_displacement(D_cart)

    assert dD < 1e-3, (
        f"{name}: a dressing displaces an atom by {dD:.4f} A. Dressings this far "
        "from a real symmetry can merge genuinely distinct clusters (over-merge), "
        "which corrupts the lattice sum and is invisible to the "
        "n_classes == len(mers) check."
    )


@pytest.mark.parametrize("name", [
    "benzene", "uracil", "formamide", "cytosine", "ammonia", "acetic_acid",
    "imidazole", "urea", "adamantane", "carbon_dioxide", "hexamine", "ice",
])
def test_bound_dressing_groups_satisfy_the_containment_theorem(name):
    # W <= D by construction, and `bind` is where the two groups meet, so this
    # is where the containment has to actually hold. It is the previous test's
    # companion: that one checks no dressing is spurious (no over-detection),
    # this one checks no site operation is missing (no under-detection). The
    # first protects the lattice energy; the second against redundant QC jobs.
    crystal = load(name)
    crystal = crystal.detect_symmetry(1e-3)
    refs = crystal.get_references()
    context.bind(crystal, refs)

    for i in range(len(refs)):
        D = context.dress_stacked_by_ref[i][0]
        for w in _site(i)[0]:
            assert any(np.array_equal(w, d) for d in D), (
                f"{name}: reference {i} has a site operation absent from D. "
                "Any merge that needs it as a dressing is lost."
            )
