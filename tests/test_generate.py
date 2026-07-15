"""Pytest suite for multimer generation and chemistry-class validation.

Each parametrized case verifies:
  1. `generate()` returns the expected number of unique multimers
  2. `count_chemistry_classes()` agrees with that count (no duplicates remain)

Expected counts are marked TODO — run the suite once, observe the printed
output, then replace each None with the observed value to lock it in as a
regression target.

Run with:
    conda run -n cle python -m pytest Tests/test_generate.py -v
"""

from pathlib import Path

import pytest

import crystalatte as cle
from crystalatte.work.validate import count_chemistry_classes

DATA = Path(__file__).parent

# kwargs shared across all generate calls
_GENERATE_KWARGS = {"use_origin": True, "n_jobs": 1, "verbose": 0}

# Tolerance used by the original CrystaLattE oracle
_ATOL = 1e-8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _load(cif_rel: str):
    """Parse the first block of a CIF file and return (crystal, reference)."""
    return next(cle.from_cif(str(DATA / cif_rel)))


# ---------------------------------------------------------------------------
# Dimer (N=2) counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cif_rel,R,expected", [
    # Fill in `expected` after an initial run (replace None with observed count)
    ("x23/benzene/benzene.cif",              15.0, None),
    ("x23/uracil/uracil.cif",               15.0, None),
    ("x23/formamide/formamide.cif",          15.0, None),
    ("x23/cytosine/cytosine.cif",            15.0, None),
    ("x23/ammonia/ammonia.cif",              15.0, None),
    ("x23/acetic_acid/acetic_acid.cif",      15.0, None),
    ("x23/imidazole/imidazole.cif",          15.0, None),
    ("x23/urea/urea.cif",                    15.0, None),
    ("x23/adamantane/adamantane.cif",        15.0, None),
    ("x23/carbon_dioxide/carbon_dioxide.cif", 15.0, None),
    ("x23/hexamine/hexamine.cif",            15.0, None),
    ("x23/ice/ice.cif",                      15.0, None),
    ("x23/naphthalene/naphthalene.cif",      15.0, None),
])
def test_dimer_count(cif_rel, R, expected):
    crystal, reference = _load(cif_rel)
    multimers = cle.generate(crystal, reference, N=2, R=R, **_GENERATE_KWARGS)

    n_classes, _ = count_chemistry_classes(multimers, atol=_ATOL)

    # Dedup completeness: every multimer should be a distinct chemistry class
    assert n_classes == len(multimers), (
        f"Duplicate multimers detected: {len(multimers)} generated but only "
        f"{n_classes} distinct chemistry classes"
    )

    if expected is not None:
        assert len(multimers) == expected, (
            f"Expected {expected} dimers, got {len(multimers)}"
        )
    else:
        pytest.skip(f"TODO: set expected dimer count (currently {len(multimers)})")


# ---------------------------------------------------------------------------
# Trimer (N=3) counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cif_rel,R,expected", [
    ("x23/benzene/benzene.cif",  10.0, None),
    ("x23/uracil/uracil.cif",    10.0, None),
    ("x23/formamide/formamide.cif", 10.0, None),
])
def test_trimer_count(cif_rel, R, expected):
    crystal, reference = _load(cif_rel)
    multimers = cle.generate(crystal, reference, N=3, R=R, **_GENERATE_KWARGS)

    n_classes, _ = count_chemistry_classes(multimers, atol=_ATOL)

    assert n_classes == len(multimers), (
        f"Duplicate multimers detected: {len(multimers)} generated but only "
        f"{n_classes} distinct chemistry classes"
    )

    if expected is not None:
        assert len(multimers) == expected
    else:
        pytest.skip(f"TODO: set expected trimer count (currently {len(multimers)})")


# ---------------------------------------------------------------------------
# Regression: R-cutoff sensitivity
# Verifies that tightening R monotonically reduces (or holds) the count.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cif_rel", [
    "x23/benzene/benzene.cif",
    "x23/uracil/uracil.cif",
])
def test_dimer_count_decreases_with_tighter_R(cif_rel):
    crystal, reference = _load(cif_rel)

    counts = []
    for R in [20.0, 15.0, 10.0, 7.0]:
        multimers = cle.generate(crystal, reference, N=2, R=R, **_GENERATE_KWARGS)
        counts.append(len(multimers))

    assert counts == sorted(counts, reverse=True), (
        f"Dimer counts should be non-increasing as R decreases: {counts}"
    )
