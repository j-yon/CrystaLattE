"""The monomer self-isometry set must be an actual group.

``docs/dedup_correctness.tex`` Thm 1/2 assume ``S`` is the monomer's isometry
group. The detector filters a fixed candidate library, and that library is not
closed under composition — it contains no rotoreflections beyond ``i`` and
``sigma`` — so a raw filter cannot return a group. These tests pin the group
properties the deduper relies on.
"""

from pathlib import Path

import numpy as np
import pytest

import crystalatte as cle

DATA = Path(__file__).parent

# Orders of the finite subgroups of O(3) that a molecule can realise.
VALID_ORDERS = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36,
                40, 48, 60, 72, 96, 120}

CRYSTALS = ["benzene", "uracil", "formamide", "cytosine", "ammonia",
            "acetic_acid", "imidazole", "urea", "adamantane",
            "carbon_dioxide", "hexamine", "ice"]


def _isometries(name):
    crystal, _ = next(cle.from_cif(str(DATA / f"x23/{name}/{name}.cif")))
    return crystal.get_references()[0].self_isometries


def _contains(G, X, atol=1e-3):
    return any(np.allclose(X, g, atol=atol) for g in G)


@pytest.mark.parametrize("name", CRYSTALS)
def test_self_isometries_is_closed_under_composition(name):
    G = _isometries(name)
    for a in G:
        for b in G:
            assert _contains(G, a @ b), f"{name}: product escapes the group"


@pytest.mark.parametrize("name", CRYSTALS)
def test_self_isometries_has_valid_group_order(name):
    G = _isometries(name)
    assert len(G) in VALID_ORDERS, f"{name}: |S| = {len(G)} is not a point-group order"


@pytest.mark.parametrize("name", CRYSTALS)
def test_self_isometries_contains_identity_and_inverses(name):
    G = _isometries(name)
    assert _contains(G, np.eye(3))
    for g in G:
        assert np.allclose(g @ g.T, np.eye(3), atol=1e-6), "not orthogonal"
        assert _contains(G, g.T), "inverse missing"


def test_hexamine_is_tetrahedral():
    # Hexamine (urotropine) is exactly T_d: 24 operations, of which 12 are
    # improper (6 S4 + 6 sigma_d). The raw filter returns 18 = 24 - 6, missing
    # precisely the S4 rotoreflections, because S4 = sigma . C4 is a product of
    # library members rather than a library member itself. 18 does not divide
    # 24, so the raw set provably is not a group.
    G = _isometries("hexamine")
    assert len(G) == 24
    improper = [g for g in G if np.linalg.det(g) < 0]
    assert len(improper) == 12

    def is_pure_mirror(g):
        return np.allclose(sorted(np.linalg.eigvalsh((g + g.T) / 2)), [-1, 1, 1], atol=1e-6)

    s4 = [g for g in improper
          if not is_pure_mirror(g) and not np.allclose(g, -np.eye(3), atol=1e-6)]
    assert len(s4) == 6, "the six S4 rotoreflections must be recovered by closure"


def test_distorted_geometry_is_not_forced_into_a_group():
    # Adamantane is T_d in principle, but this CIF's geometry (a plastic
    # crystal, orientationally disordered) deviates by ~0.11 A, far outside the
    # detection tolerance. The right answer is a small group, not a fabricated
    # T_d -- repair must not invent symmetry that the coordinates do not have.
    G = _isometries("adamantane")
    assert len(G) < 24
    assert len(G) in VALID_ORDERS
