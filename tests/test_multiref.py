"""Multi-reference (Z' > 1) enumeration and deduplication.

Covers reference discovery, per-member `ref_indices` plumbing, per-reference
binding, and the physical dimers of a Z' = 4 cell. See tests/README.md.
"""

import numpy as np

from crystalatte.core.sym_ops import SymOp, SymOpList
from crystalatte.work import generate
from crystalatte.work.dedup import context

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
    assert len(context.mass_by_ref) == 4
    assert len(context.dress_stacked_by_ref) == 4
    # NH3: one N and three H per reference
    assert list(context.references[0].symbols) == ["N", "H", "H", "H"]


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
    mers = generate.generate(crystal, 2, 5.0)
    assert len(mers) == AMMONIA_DIMERS_DEFAULT
    assert AMMONIA_DIMERS_DEFAULT > 2
    assert all(len(m.monomers) == 2 for m in mers)
