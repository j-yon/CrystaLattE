"""Tests for the chemistry-equivalence oracle in `crystalatte.work.validate`.

WHY THIS FILE EXISTS
--------------------
`count_chemistry_classes` is not a feature; it is the REFEREE. The assertion
`count_chemistry_classes(mers)[0] == len(mers)` appears throughout the dedup
suite, and every locked multimer count in `test_generate.py` rests on the claim
that the oracle agreed with the deduper when those counts were captured.

An untested referee is worth nothing. Replacing the body of
`count_chemistry_classes` with the trivial answer -- every multimer is its own
class -- used to leave the entire suite green apart from a single xfail, which
means none of those `n_classes == len(mers)` assertions was testing anything.
The tests below pin the oracle down from both sides: it must MERGE things that
are genuinely the same (identity, rigid motion, reflection) and SEPARATE things
that are not (different clusters, a perturbed atom).

The two directions matter for different reasons. If the oracle wrongly
separates, the dedup suite reports duplicates that do not exist and the locked
counts become unreachable -- noisy, but visible. If the oracle wrongly merges,
every "no duplicates remain" assertion in the suite passes vacuously and a real
over-merge in the deduper goes unreported. That is the direction these tests
exist to close.
"""

import numpy as np
import pytest

import crystalatte as cle
from crystalatte.core.multimer import Monomer, Multimer
from crystalatte.core.sym_ops import SymOp, SymOpList
from crystalatte.work.validate import _nre_chsev, count_chemistry_classes

from conftest import GENERATE_KW, crystal as load_crystal


@pytest.fixture(scope="module")
def dimers():
    """A handful of genuinely distinct benzene dimers."""
    return cle.generate(load_crystal("benzene"), 2, 6.0, **GENERATE_KW)


def _remap(mer, fn):
    """Rebuild `mer` with `fn` applied to every monomer's Cartesian coordinates.

    The oracle reads only `symbols` and `cart_coords`, so the remaining fields
    are carried over untouched; this is a geometry transform, not a re-derivation
    of the multimer.
    """
    mons = [
        Monomer(
            mo.symbols,
            mo.frac_coords,
            fn(np.asarray(mo.cart_coords, dtype=float)),
            mo.centroid_frac,
            mo.centroid_cart,
            mo.symop,
        )
        for mo in mer.monomers
    ]
    return Multimer(mons, mer.sym_ops, mer.multiplicity)


# ---------------------------------------------------------------------------
# The oracle must SEPARATE things that differ
# ---------------------------------------------------------------------------

def test_oracle_separates_two_distinct_multimers(dimers):
    # The load-bearing direction. An oracle that merges everything reports
    # n_classes < len(mers) and makes the dedup suite look like it is failing;
    # an oracle that separates everything reports n_classes == len(mers)
    # unconditionally and makes the dedup suite pass without testing anything.
    # This pins the first half.
    a, b = dimers[0], dimers[1]
    n_classes, partition = count_chemistry_classes([a, b], atol=1e-8)
    assert n_classes == 2, "two genuinely different dimers were merged"
    assert partition == [[0], [1]]


@pytest.mark.parametrize("displacement", [1e-3, 1e-5, 1e-7])
def test_oracle_detects_a_single_perturbed_atom(dimers, displacement):
    # Sensitivity floor. One atom moved along x by `displacement` Angstroms must
    # split the class. This is what stops a too-loose tolerance from quietly
    # turning the oracle into "everything is the same multimer", which is the
    # failure that would hide an over-merge in the deduper.
    def nudge(coords):
        out = coords.copy()
        out[0, 0] += displacement
        return out

    mer = dimers[0]
    moved = _remap(mer, nudge)
    n_classes, _ = count_chemistry_classes([mer, moved], atol=1e-8)
    assert n_classes == 2, (
        f"a {displacement:.0e} A displacement of one atom went undetected; the "
        "oracle is too blunt to referee the deduper"
    )


def test_oracle_is_not_trivially_the_identity(dimers):
    # A direct guard against the degenerate implementation
    # `return len(multimers), [[i] for i in range(len(multimers))]`, which
    # satisfies every `n_classes == len(mers)` assertion in the suite. Feeding
    # the oracle a list it MUST collapse makes that implementation fail here.
    mer = dimers[0]
    n_classes, _ = count_chemistry_classes([mer, mer, mer], atol=1e-8)
    assert n_classes == 1, (
        "three copies of one multimer were reported as three classes -- the "
        "oracle is not actually comparing anything"
    )


# ---------------------------------------------------------------------------
# The oracle must MERGE things that are the same
# ---------------------------------------------------------------------------

def test_oracle_groups_repeated_copies_of_one_multimer(dimers):
    mer = dimers[0]
    n_classes, partition = count_chemistry_classes([mer, mer], atol=1e-8)
    assert n_classes == 1
    assert partition == [[0, 1]], "the partition must name both members of the class"


def test_oracle_is_invariant_under_rigid_rotation(dimers):
    # NRE and CHSEV are built from interatomic distances, so a rotation must not
    # move them. If this fails the oracle is reading absolute coordinates
    # somewhere and would report spurious duplicates for symmetry-related
    # clusters -- which is exactly what it is used to rule out.
    theta = 0.7
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    mer = dimers[0]
    turned = _remap(mer, lambda x: x @ rot.T)
    n_classes, _ = count_chemistry_classes([mer, turned], atol=1e-8)
    assert n_classes == 1, "a rigidly rotated copy was reported as a new class"


def test_oracle_is_invariant_under_translation(dimers):
    shift = np.array([3.5, -1.25, 0.75])
    mer = dimers[0]
    moved = _remap(mer, lambda x: x + shift)
    n_classes, _ = count_chemistry_classes([mer, moved], atol=1e-8)
    assert n_classes == 1, "a translated copy was reported as a new class"


def test_oracle_is_invariant_under_reflection(dimers):
    # Enantiomers are energetically equivalent and the deduper merges them, so
    # the referee has to agree. If the oracle separated mirror images it would
    # flag every enantiomeric merge the deduper makes as an over-merge.
    mer = dimers[0]
    mirrored = _remap(mer, lambda x: x * np.array([1.0, 1.0, -1.0]))
    n_classes, _ = count_chemistry_classes([mer, mirrored], atol=1e-8)
    assert n_classes == 1, "a mirror image was reported as a new class"


# ---------------------------------------------------------------------------
# Structural guarantees of the returned partition
# ---------------------------------------------------------------------------

def test_oracle_partition_is_exhaustive_and_disjoint(dimers):
    # Every input index appears in exactly one class. A partition that dropped
    # or double-counted an index would make `n_classes == len(mers)` mean
    # something other than what the dedup suite reads it as.
    n_classes, partition = count_chemistry_classes(dimers, atol=1e-8)
    flat = [i for cls in partition for i in cls]
    assert len(partition) == n_classes
    assert sorted(flat) == list(range(len(dimers)))
    assert len(flat) == len(set(flat)), "an index was placed in two classes"


def test_oracle_needs_its_chsev_half_to_separate_homometric_clusters():
    # The oracle is (NRE, CHSEV) and the second half is not decoration. NRE is
    # sum Z_i Z_j / r_ij -- a pure function of the multiset of Z-tagged
    # interatomic distances -- so it cannot tell apart two clusters that share
    # that multiset. CHSEV is built from the same distances but depends on how
    # they are ARRANGED in the Coulomb matrix, so it can.
    #
    # {0,1,4,10,12,17} and {0,1,8,11,13,17} are homometric: identical difference
    # multisets, hence identical interatomic distances and identical NRE. Split
    # each into two 3-atom "monomers" and the oracle must still call them two
    # classes. Deleting the CHSEV comparison makes this test fail and nothing
    # else in the suite notice, which is precisely why it is here.
    def cluster(positions):
        ops = SymOpList([SymOp.identity(), SymOp.identity()], 1)
        mons = []
        for half in (positions[:3], positions[3:]):
            cart = np.array([[x, 0.0, 0.0] for x in half])
            mons.append(Monomer(["C"] * 3, cart, cart, cart[0], cart[0], None))
        return Multimer(mons, ops, 1)

    a = cluster([0.0, 1.0, 4.0, 10.0, 12.0, 17.0])
    b = cluster([0.0, 1.0, 8.0, 11.0, 13.0, 17.0])

    # Establish the premise rather than assuming it: NRE really is blind here.
    nre_a, chsev_a = _nre_chsev(a)
    nre_b, chsev_b = _nre_chsev(b)
    assert abs(nre_a - nre_b) < 1e-9, (
        "the pair is no longer homometric, so this no longer tests the CHSEV half"
    )
    assert np.abs(chsev_a - chsev_b).max() > 1.0, "CHSEV should separate them"

    n_classes, _ = count_chemistry_classes([a, b], atol=1e-8)
    assert n_classes == 2, (
        "two homometric but structurally different clusters were merged; the "
        "oracle has degenerated to an NRE comparison and can no longer referee "
        "an over-merge that clears the deduper's distance fingerprint"
    )


def test_oracle_handles_an_empty_list():
    assert count_chemistry_classes([]) == (0, [])


def test_oracle_compares_multimers_of_different_orders(dimers):
    # A dimer and a trimer have different-length CHSEV vectors. The NRE guard
    # must short-circuit before the spectra are compared, or this raises on a
    # shape mismatch instead of answering.
    trimer = cle.generate(load_crystal("benzene"), 3, 6.0, **GENERATE_KW)[0]
    n_classes, _ = count_chemistry_classes([dimers[0], trimer], atol=1e-8)
    assert n_classes == 2


# ---------------------------------------------------------------------------
# The oracle against the deduper it referees
# ---------------------------------------------------------------------------

def test_oracle_collapses_a_duplicate_appended_to_a_deduped_set(dimers):
    # End-to-end statement of the contract the dedup suite relies on: a deduped
    # set is already one-class-per-multimer, and re-introducing a duplicate must
    # be caught. Both halves are asserted, so this fails whether the oracle
    # over-merges or under-merges.
    n_clean, _ = count_chemistry_classes(dimers, atol=1e-8)
    assert n_clean == len(dimers), "the deduped set should be one class each"

    n_dirty, _ = count_chemistry_classes([*dimers, dimers[0]], atol=1e-8)
    assert n_dirty == n_clean, (
        "appending a copy of an existing multimer did not collapse into its "
        "class; the oracle cannot detect the duplicates it is used to rule out"
    )
