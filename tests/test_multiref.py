"""Multi-reference (Z' > 1) enumeration and deduplication.

Covers reference discovery, per-member `ref_indices` plumbing, per-reference
binding, and the physical dimers of a Z' = 4 cell. See tests/README.md.
"""

import numpy as np

from crystalatte.core.sym_ops import SymOp, SymOpList
from crystalatte.work import generate
from crystalatte.work.dedup import cloud, context, equivalence, invariants

from conftest import GENERATE_KW, load

def test_get_references_z1_benzene():
    crystal, _ = load("benzene")
    refs = crystal.get_references()
    assert len(refs) == 1
    assert len(refs[0].symbols) == 12  # full benzene


def test_get_references_z4_ammonia_p1():
    crystal, _ = load("ammonia")
    refs = crystal.get_references()
    assert len(refs) == 4                      # 4 independent NH3 in P1
    assert all(len(r.symbols) == 4 for r in refs)  # each NH3 = N + 3H
    # distinct molecules => distinct centroids
    cens = np.array([r.centroid_frac % 1.0 for r in refs])
    assert len({tuple(np.round(c, 3)) for c in cens}) == 4


def test_get_reference_is_first_of_references():
    crystal, _ = load("benzene")
    assert np.allclose(crystal.get_reference().centroid_frac,
                       crystal.get_references()[0].centroid_frac)


def test_symoplist_ref_indices_default_zeros():
    s = SymOpList([SymOp.identity(), SymOp.identity()], 1)
    assert list(s.ref_indices) == [0, 0]


def test_symoplist_ref_indices_explicit():
    s = SymOpList([SymOp.identity(), SymOp.identity()], 1)
    s.ref_indices = np.array([0, 1])
    assert list(s.ref_indices) == [0, 1]
    fc = SymOpList.from_components(
        s.rot_cache, s.tr_cache, 1, ref_indices=np.array([1, 0])
    )
    assert list(fc.ref_indices) == [1, 0]


def test_bind_multi_reference_state():
    crystal, _ = load("ammonia")
    refs = crystal.get_references()
    context.bind(crystal, refs)
    assert len(context.references) == 4
    assert len(context.Z_by_ref) == 4
    assert len(context.W_lin_by_ref) == 4
    assert len(context.check_congruence_by_ref) == 4
    # NH3 Z tags = [7,1,1,1] per reference
    assert list(context.Z_by_ref[0]) == [7, 1, 1, 1]


def test_fingerprint_uses_per_member_reference():
    # A 2-member cluster whose members come from different NH3 references must
    # reconstruct atoms from each base reference; fingerprint is well-defined.
    crystal, _ = load("ammonia")
    refs = crystal.get_references()
    context.bind(crystal, refs)
    g = SymOpList([SymOp.identity(), SymOp.identity()], 1)
    g.ref_indices = np.array([0, 1])   # member 0 <- ref 0, member 1 <- ref 1
    zz, d = invariants.distance_fingerprint(g)
    assert len(zz) == len(d) > 0


def is_mono_reference(g):
    return len(set(g.ref_indices)) == 1


def test_cloud_congruence_matches_translated_and_mirror():
    crystal, _ = load("ammonia")
    refs = crystal.get_references()
    context.bind(crystal, refs)
    # Same physical pair described two ways: (ref0 @ I, ref1 @ I) vs itself.
    g = SymOpList([SymOp.identity(), SymOp.identity()], 1); g.ref_indices = np.array([0, 1])
    h = SymOpList([SymOp.identity(), SymOp.identity()], 1); h.ref_indices = np.array([0, 1])
    assert cloud.same_multimer_geom(g, h) is True
    # A DIFFERENT PARTNER REFERENCE, same placement. This must match, and the
    # assertion is `is True` rather than a shrug: all four ammonia molecules in
    # the P1 cell lie in a single symmetry orbit (the true group is P2_1 3), so
    # the (ref0, ref1) and (ref0, ref2) pairs really are congruent clouds.
    # Confirmed independently of the matcher by the chemistry referee:
    # |NRE(0,1) - NRE(0,2)| = 0.000e+00 and max|CHSEV diff| = 4.3e-14.
    h2 = SymOpList([SymOp.identity(), SymOp.identity()], 1); h2.ref_indices = np.array([0, 2])
    assert cloud.same_multimer_geom(g, h2) is True

    # THE NEGATIVE CASE. The two assertions above only ever ask the matcher to
    # say yes, so on their own they are equally happy with a matcher that never
    # says no. Pushing the partner one cell along c gives a genuinely different
    # arrangement -- the referee puts it 1.69 hartree away in NRE, far outside
    # any tolerance here -- and the matcher must reject it.
    far = SymOpList(
        [SymOp.identity(),
         SymOp.identity().translate(np.array([0, 0, 12], dtype=np.int16))], 1
    )
    far.ref_indices = np.array([0, 1])
    assert cloud.same_multimer_geom(g, far) is False


def test_are_equivalent_routes_mono_reference_to_integer(monkeypatch):
    crystal, _ = load("ammonia")
    refs = crystal.get_references()
    context.bind(crystal, refs)
    # two mono-reference (base 0) clusters -> must go through the integer matcher
    g = SymOpList([SymOp.identity(), _tr([12, 0, 0])], 1)   # ref_indices default [0,0]
    h = SymOpList([SymOp.identity(), _tr([-12, 0, 0])], 1)
    called = {"geom": 0}
    monkeypatch.setattr(cloud, "same_multimer_geom",
                        lambda a, b, **k: called.__setitem__("geom", called["geom"] + 1) or False)
    monkeypatch.setattr(equivalence, "fingerprints_match", lambda a, b, **k: True)
    equivalence.are_equivalent(g, h)
    assert called["geom"] == 0        # mono-reference never uses the cloud matcher


def _tr(dx12):
    return SymOp.identity().translate(np.array(dx12, dtype=np.int16))


#: Ammonia's CIF is written in P1 with four NH3 in the cell, so it is Z' = 4
#: before symmetry detection: the reference's true nearest neighbours are the
#: other three ASU molecules at 3.94 A, which single-reference enumeration never
#: formed at all -- it saw only the reference's own lattice translates at 5.13 A.
#:
#: Counted under the anchor-relative `use_origin` convention, where the distance
#: is measured from the ANCHOR monomer's centroid rather than the global origin.
#: The two coincide only at Z' = 1, which is why this crystal pins the
#: distinction and benzene cannot.
AMMONIA_DIMERS = 3
# Same physical ammonia dimers, but under the DEFAULT closest-contact
# convention (no use_origin/use_com) — captured independently so a future
# use_origin regression can't silently pass this suite.
AMMONIA_DIMERS_DEFAULT = 4


def test_benzene_regression_unchanged():
    crystal, _ = load("benzene")
    mers = generate.generate(crystal, 2, 5.0, **GENERATE_KW)
    assert len(mers) == 3          # Z'=1 path unchanged


def test_ammonia_physical_dimers():
    crystal, _ = load("ammonia")
    mers = generate.generate(crystal, 2, 5.0, **GENERATE_KW)
    # now includes the real ~3.94 A hetero-molecular contacts, not just self-translates
    assert len(mers) == AMMONIA_DIMERS
    assert AMMONIA_DIMERS > 2
    # every returned dimer has 2 monomers
    assert all(len(m.monomers) == 2 for m in mers)


def test_ammonia_physical_dimers_default_convention():
    # Anchor-relative behavior locked independently of use_origin: the
    # default closest-contact convention must also find more than the old
    # self-translate-only artifact count.
    crystal, _ = load("ammonia")
    mers = generate.generate(crystal, 2, 5.0, n_jobs=1)
    assert len(mers) == AMMONIA_DIMERS_DEFAULT
    assert AMMONIA_DIMERS_DEFAULT > 2
    assert all(len(m.monomers) == 2 for m in mers)
