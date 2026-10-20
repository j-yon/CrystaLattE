"""Pytest suite for multimer generation and chemistry-class validation.

Each parametrized case verifies:
  1. `generate()` returns the expected number of unique multimers
  2. `count_chemistry_classes()` agrees with that count (no duplicates remain)

Expected counts are locked in as regression targets; they are only valid
because `count_chemistry_classes` agrees with them on every case.

Run with:
    python -m pytest tests/test_generate.py -v

See tests/README.md for the suite map.
"""

from collections import Counter

import pytest

import crystalatte as cle
from crystalatte.work.validate import count_chemistry_classes

from conftest import GENERATE_KW, crystal

# Tolerance used by the original CrystaLattE oracle
_ATOL = 1e-8


# ---------------------------------------------------------------------------
# Dimer (N=2) counts
# ---------------------------------------------------------------------------

# Counts captured once the deduper agreed with the NRE+CHSEV oracle on every
# crystal; each was additionally confirmed by explicit Kabsch congruence, so a
# change here means the dedup changed, not that the reference drifted.
#
# Two rows moved since, both because the enumeration box grew to cover multimers
# it used to miss (the candidate total rose with the count, so nothing merged
# differently): uracil 111 -> 113 when ecdf36e sized the box by plane spacing,
# and ice 205 -> 208 at the same commit.
@pytest.mark.parametrize("name,R,expected", [
    ("benzene",              15.0, 52),
    ("uracil",               15.0, 113),
    ("formamide",          15.0, 193),
    ("cytosine",            15.0, 89),
    ("ammonia",              15.0, 82),
    ("acetic_acid",      15.0, 120),
    ("imidazole",          15.0, 135),
    ("urea",                    15.0, 49),
    ("adamantane",        15.0, 17),
    pytest.param(
        "carbon_dioxide", 15.0, 36,
        marks=pytest.mark.xfail(
            strict=True,
            reason="linear monomer: deduplication is knowingly incomplete, so 3 "
                   "duplicate pairs survive and 39 are returned. 36 is the "
                   "correct answer and is kept here as the target; see the "
                   "limitation note in crystalatte.work.dedup",
        ),
    ),
    ("hexamine",            15.0, 9),
    pytest.param(
        "ice", 15.0, 208,
        marks=pytest.mark.xfail(
            strict=True,
            reason="the exact orbit key (15e6e6f) over-merges proton-disordered "
                   "ice: 208 -> 78 from the same 497 candidates, and "
                   "test_no_over_merge_every_discarded_cluster_matches_a_survivor "
                   "confirms a real over-merge. 208 is the count before it; how "
                   "to resolve ice is an open decision",
        ),
    ),
    # Z' = 3 (Pca2_1) -- three symmetry-independent molecules, so every dimer
    # here is drawn from a pair of possibly-different references. Lives under
    # other/ rather than x23/.
    ("fomepizole",      15.0, 276),
    # Named only by Hermann-Mauguin symbol -- no number, no Hall symbol -- so
    # this row also covers the H-M -> Hall resolution path end to end. The
    # setting matters: read as the standard P2_1/c instead of the P2_1/a the
    # file means, the operations differ and this count moves.
    ("naphthalene",          15.0, 49),
])
def test_dimer_count(name, R, expected):
    xtal = crystal(name)
    multimers = cle.generate(xtal, N=2, R=R, **GENERATE_KW)

    n_classes, _ = count_chemistry_classes(multimers, atol=_ATOL)

    # Dedup completeness: every multimer should be a distinct chemistry class
    assert n_classes == len(multimers), (
        f"Duplicate multimers detected: {len(multimers)} generated but only "
        f"{n_classes} distinct chemistry classes"
    )

    assert len(multimers) == expected, (
        f"Expected {expected} dimers, got {len(multimers)}"
    )


# ---------------------------------------------------------------------------
# Trimer (N=3) counts
#
# Recaptured after 2242715 required EVERY pair of members to lie within R (as
# in the original CrystaLattE), not just each partner within R of the anchor;
# the old anchor-only counts were 676 / 1173 / 3204.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,R,expected", [
    ("benzene",  10.0, 154),
    ("uracil",    10.0, 259),
    ("formamide", 10.0, 785),
])
def test_trimer_count(name, R, expected):
    xtal = crystal(name)
    multimers = cle.generate(xtal, N=3, R=R, **GENERATE_KW)

    n_classes, _ = count_chemistry_classes(multimers, atol=_ATOL)

    assert n_classes == len(multimers), (
        f"Duplicate multimers detected: {len(multimers)} generated but only "
        f"{n_classes} distinct chemistry classes"
    )

    assert len(multimers) == expected, (
        f"Expected {expected} trimers, got {len(multimers)}"
    )


# ---------------------------------------------------------------------------
# Tetramer (N=4) counts
#
# The suite ran only N=2 and N=3 for a long time. Order matters to the matcher:
# the anchored search is O(|S| n^3) in cluster size, the permutation it solves
# for has n! candidate images before reference-compatibility prunes them, and
# the dressing bookkeeping is per-member -- so an indexing error that is
# invisible on a pair can surface on a quadruple. Radii are tight so these stay
# under a second each; the point is the order, not the shell size.
#
# Each count was confirmed against the chemistry oracle at capture time (the
# assertion below re-checks it on every run) and verified stable across repeated
# runs.
#
# Radii were widened when 2242715 made the cutoff apply to every pair of
# members: at the old anchor-only radii (6.0 / 5.5) these shells shrank to
# 6-27 tetramers, too few to exercise the matcher.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,R,expected", [
    ("benzene",   8.0, 93),
    ("urea",         8.0, 279),
    ("pyrazole", 8.0, 655),   # Z' = 2: cross-reference tetramers
    ("fomepizole", 7.5, 465),  # Z' = 3: three groups at once
])
def test_tetramer_count(name, R, expected):
    xtal = crystal(name)
    multimers = cle.generate(xtal, N=4, R=R, **GENERATE_KW)

    n_classes, _ = count_chemistry_classes(multimers, atol=_ATOL)

    assert n_classes == len(multimers), (
        f"Duplicate multimers detected: {len(multimers)} generated but only "
        f"{n_classes} distinct chemistry classes"
    )

    assert len(multimers) == expected, (
        f"Expected {expected} tetramers, got {len(multimers)}"
    )


# ---------------------------------------------------------------------------
# Pentamer (N=5) counts
#
# The smallest order at which a cluster can hold five distinct dressing choices
# simultaneously. Kept to three short rows at tight R -- each under a second --
# because the value here is the arity, not the shell: past N=4 the enumeration
# grows as C(M, N-1) and a larger radius buys cost rather than coverage.
#
# pyrazole is included because a Z' = 2 pentamer mixes two dressing groups
# across five members, which is the most demanding shape the matcher currently
# sees anywhere in the suite.
#
# Radii were widened when 2242715 made the cutoff apply to every pair of
# members: at the old anchor-only radii (5.5 / 5.0) no pentamer qualifies.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,R,expected", [
    ("benzene",   8.0, 97),
    ("urea",         8.0, 619),
    ("pyrazole", 7.5, 335),  # Z' = 2
])
def test_pentamer_count(name, R, expected):
    xtal = crystal(name)
    multimers = cle.generate(xtal, N=5, R=R, **GENERATE_KW)

    n_classes, _ = count_chemistry_classes(multimers, atol=_ATOL)

    assert n_classes == len(multimers), (
        f"Duplicate multimers detected: {len(multimers)} generated but only "
        f"{n_classes} distinct chemistry classes"
    )

    assert len(multimers) == expected, (
        f"Expected {expected} pentamers, got {len(multimers)}"
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_N", [1, 0, -1, -3])
def test_generate_rejects_orders_below_two(bad_N):
    # A multimer is an interaction between monomers, so N < 2 has nothing to
    # mean. N = 1 was the dangerous case: it ran to completion and returned one
    # "multimer" per reference -- the anchor alone, because combinations(near, 0)
    # yields a single empty tuple -- so the caller got a confidently wrong answer
    # instead of an error.
    xtal = crystal("benzene")
    with pytest.raises(ValueError, match="N must be an integer >= 2"):
        cle.generate(xtal, N=bad_N, R=5.0, **GENERATE_KW)


@pytest.mark.parametrize("bad_N", [2.0, "2", None, 2.5])
def test_generate_rejects_non_integer_orders(bad_N):
    # N indexes a combination width; a float or a string cannot. Rejected at the
    # door rather than deep inside itertools, where the error names neither N
    # nor generate().
    xtal = crystal("benzene")
    with pytest.raises(ValueError, match="N must be an integer >= 2"):
        cle.generate(xtal, N=bad_N, R=5.0, **GENERATE_KW)


def test_generate_accepts_the_smallest_valid_order():
    # The gate must not have moved the boundary: N = 2 is valid and still works.
    xtal = crystal("benzene")
    assert cle.generate(xtal, N=2, R=5.0, **GENERATE_KW)


# ---------------------------------------------------------------------------
# Regression: R-cutoff sensitivity
# Verifies that tightening R monotonically reduces (or holds) the count.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["benzene", "uracil"])
def test_dimer_count_decreases_with_tighter_R(name):
    xtal = crystal(name)

    counts = []
    for R in [20.0, 15.0, 10.0, 7.0]:
        multimers = cle.generate(xtal, N=2, R=R, **GENERATE_KW)
        counts.append(len(multimers))

    assert counts == sorted(counts, reverse=True), (
        f"Dimer counts should be non-increasing as R decreases: {counts}"
    )


# ---------------------------------------------------------------------------
# Extra molecular symmetry
# ---------------------------------------------------------------------------

# Molecular symmetry beyond the site symmetry. Recorded from the exact +
# Cartesian-congruence pipeline (ddc4c41) before the congruence pass was folded
# into the integer orbit key; the unique count, the total multiplicity and the
# multiplicity histogram must all survive that change.
@pytest.mark.parametrize("name,N,R,n_unique,total,histogram", [
    ("triazine", 2, 20.0, 61, 418, {2: 5, 6: 44, 12: 12}),
    pytest.param("triazine", 2, 35.0, 266, 2082, {2: 9, 6: 178, 12: 75, 24: 4},
                 marks=pytest.mark.slow),
    ("triazine", 3, 12.0, 123, 2103, {3: 1, 6: 2, 9: 14, 18: 103, 36: 3}),
    ("ammonia", 2, 20.0, 183, 1075, {2: 2, 3: 5, 6: 176}),
    ("benzene", 2, 20.0, 116, 366, {2: 49, 4: 67}),
])
def test_extra_symmetry_regression(name, N, R, n_unique, total, histogram):
    mers = cle.generate(crystal(name), N, R, **GENERATE_KW)
    assert len(mers) == n_unique
    assert sum(m.multiplicity for m in mers) == total
    assert dict(Counter(m.multiplicity for m in mers)) == histogram
