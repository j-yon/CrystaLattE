"""The monomer self-isometry set must be an actual group.

``docs/dedup_correctness.tex`` Thm 1/2 assume ``S`` is the monomer's isometry
group. The detector filters a fixed candidate library, and that library is not
closed under composition — it contains no rotoreflections beyond ``i`` and
``sigma`` — so a raw filter cannot return a group. These tests pin the group
properties the deduper relies on.
"""


import numpy as np
import pytest
import qcelemental as qcel

from conftest import load

# Orders of the finite subgroups of O(3) that a molecule can realise.
VALID_ORDERS = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36,
                40, 48, 60, 72, 96, 120}

CRYSTALS = ["benzene", "uracil", "formamide", "cytosine", "ammonia",
            "acetic_acid", "imidazole", "urea", "adamantane",
            "carbon_dioxide", "hexamine", "ice"]


def _isometries(name):
    crystal, _ = load(name)
    return crystal.get_references()[0].point_group


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


# ---------------------------------------------------------------------------
# Seeding S with the site symmetry W_lin
#
# `W_lin <= S` is a theorem (docs/dedup_correctness.tex C2): a space-group
# operation that fixes the monomer's centroid necessarily permutes that
# monomer's own atoms, so its linear part is a molecular self-isometry. The
# detector violated it for two of the twelve X23 references, and NOT because of
# a tolerance: the missing operations are rotoreflections, which the candidate
# library does not contain at any tolerance, and which closure cannot reach
# (S4 = (-I) C4^3 and neither factor is a symmetry of adamantane). Supplying
# them from the space group is the only route.
# ---------------------------------------------------------------------------


def _crystal(name):
    crystal, _ = load(name)
    return crystal.detect_symmetry(1e-3)


def _site_symmetry_cartesian(crystal, mon):
    """The monomer's site-symmetry linear parts, pushed into Cartesian."""
    A = np.asarray(crystal.lattice_vectors, float)
    A_inv = np.linalg.inv(A)
    out = []
    for op in crystal.space_group.get_site_symmetry(mon.centroid_frac):
        R_cart = A @ op.rot.astype(float) @ A_inv
        if not any(np.allclose(R_cart, u, atol=1e-6) for u in out):
            out.append(R_cart)
    return out


@pytest.mark.parametrize("name", CRYSTALS)
def test_site_symmetry_is_contained_in_seeded_self_isometries(name):
    # The containment the deduper's theory assumes, enforced.
    crystal = _crystal(name)
    mon = crystal.get_references()[0]
    W_cart = _site_symmetry_cartesian(crystal, mon)
    S = mon.point_group_seeded(W_cart)
    for w in W_cart:
        assert _contains(S, w), f"{name}: a site operation is missing from S"


def test_seeding_recovers_adamantanes_rotoreflections():
    # Adamantane sits on a -4 site: W = {E, C2, S4, S4^3}, all four of which
    # move its atoms by 0.00000 A. The detector finds the first two only.
    crystal = _crystal("adamantane")
    mon = crystal.get_references()[0]
    assert len(mon.point_group) == 2, "unseeded detection finds E and C2 only"

    S = mon.point_group_seeded(_site_symmetry_cartesian(crystal, mon))
    assert len(S) == 4
    rotoreflections = [
        g for g in S
        if np.linalg.det(g) < 0 and not np.allclose(g, -np.eye(3), atol=1e-6)
        and abs(np.trace(g) - 1.0) > 1e-6
    ]
    assert len(rotoreflections) == 2, "the two S4 operations must be recovered"


def test_seeding_recovers_ices_site_symmetry():
    crystal = _crystal("ice")
    mon = crystal.get_references()[0]
    assert len(mon.point_group) == 2, "unseeded detection is deficient for ice"
    S = mon.point_group_seeded(_site_symmetry_cartesian(crystal, mon))
    assert len(S) == 6


def test_a_seed_that_is_not_a_symmetry_is_rejected():
    # The seed is exact *given* that the site symmetry was determined correctly,
    # and that determination is itself made at a tolerance (SITE_ATOL). Since
    # over-detection is the unrecoverable direction -- it merges genuinely
    # distinct clusters and the NRE/CHSEV oracle cannot see it -- every seed
    # operation is verified against the coordinates before being trusted.
    crystal = _crystal("benzene")
    mon = crystal.get_references()[0]
    angle = np.deg2rad(37.0)
    bogus = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                      [np.sin(angle), np.cos(angle), 0.0],
                      [0.0, 0.0, 1.0]])

    S = mon.point_group_seeded([bogus])

    assert not _contains(S, bogus), "a non-symmetry was admitted into S"
    assert len(S) == len(mon.point_group), "a rejected seed must change nothing"


@pytest.mark.parametrize("name", CRYSTALS)
def test_an_empty_seed_reproduces_the_unseeded_answer(name):
    crystal = _crystal(name)
    mon = crystal.get_references()[0]
    assert len(mon.point_group_seeded([])) == len(mon.point_group)


@pytest.mark.parametrize("name", CRYSTALS)
def test_seeded_self_isometries_is_still_a_group(name):
    # Seeding must not buy the containment at the cost of Prop. 4 (closure),
    # which `deduplicate_bucket` needs for transitivity.
    crystal = _crystal(name)
    mon = crystal.get_references()[0]
    S = mon.point_group_seeded(_site_symmetry_cartesian(crystal, mon))
    assert len(S) in VALID_ORDERS, f"{name}: |S| = {len(S)} is not a point-group order"
    for a in S:
        assert _contains(S, a.T), f"{name}: inverse missing"
        for b in S:
            assert _contains(S, a @ b), f"{name}: product escapes the group"


# ---------------------------------------------------------------------------
# External point-group solver
#
# A dedicated solver identifies the IDEALISED group -- pymatgen names benzene
# D6h and adamantane T_d where the built-in filter reaches a fraction of each.
# Its operations are consumed through the same `seed` channel as the
# crystallographic site symmetry, so every one is re-verified against the actual
# coordinates. pymatgen is a REQUIRED dependency -- detection consults it by
# default, so making it optional would mean the same structure answering
# differently on two machines. These tests therefore import it outright: if it
# is missing, that is a broken install and should fail loudly, not skip quietly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("benzene", 24), ("adamantane", 24), ("hexamine", 24),
    ("ammonia", 6), ("urea", 4),
])
def test_solver_names_the_idealised_group(name, expected):
    # Establishes what the solver is for. These are the orders the built-in
    # filter cannot reach for benzene (4) and adamantane (2).
    from crystalatte.core.point_group import PointGroup

    mon = load(name)[0].detect_symmetry(1e-3).get_references()[0]
    assert len(PointGroup.from_solver(mon.symbols, mon.cart_coords)) == expected


@pytest.mark.parametrize("name", CRYSTALS)
def test_the_solver_never_shrinks_the_detected_group(name):
    # The guarantee the integration rests on. Adding the solver may only help.
    #
    # This is not free: an operation can verify individually and still break the
    # CLOSURE, since composing two operations each off by eps gives one off by
    # ~2 eps. Measured on benzene, 12 of the solver's 24 operations verify at
    # rung 0.05 and their closure then escapes -- which, before the per-rung
    # fallback, discarded the whole rung and dropped detection from 4 to 2.
    from crystalatte.core.point_group import PointGroup

    mon = load(name)[0].detect_symmetry(1e-3).get_references()[0]
    without = len(PointGroup.detect(mon.symbols, mon.cart_coords, use_solver=False))
    with_solver = len(PointGroup.detect(mon.symbols, mon.cart_coords, use_solver=True))
    assert with_solver >= without, (
        f"{name}: consulting the solver REDUCED the detected group "
        f"({without} -> {with_solver}); the per-rung fallback is not working"
    )


@pytest.mark.parametrize("name", CRYSTALS)
def test_solver_seeded_result_is_still_a_group(name):
    # Whatever the solver contributes must survive the same coherence checks as
    # everything else -- a spurious operation admitted here would merge
    # genuinely different clusters, which the NRE/CHSEV oracle cannot see.
    from crystalatte.core.point_group import PointGroup

    mon = load(name)[0].detect_symmetry(1e-3).get_references()[0]
    G = PointGroup.detect(mon.symbols, mon.cart_coords, use_solver=True)
    assert len(G) in VALID_ORDERS
    for a in G:
        for b in G:
            assert _contains(G, a @ b), f"{name}: not closed under composition"


def test_seed_larger_than_the_group_cap_is_rejected_not_crashed():
    # Regression: `_Frame.close` bounds-checked its CLOSURE loop but not the
    # loop that inserts the seed, so a seed larger than _MAX_GROUP_ORDER walked
    # off the end of its preallocated stack with an IndexError. Reachable from
    # ordinary use -- a loose atol makes the raw filter accept hundreds of
    # operations (214 for adamantane at 1.0 A) and they all arrive that way.
    #
    # The right answer is None, meaning "incoherent", which sends detection to a
    # tighter rung.
    from crystalatte.core.point_group import (
        _MAX_GROUP_ORDER, PointGroup, _Frame,
    )

    mon = load("benzene")[0].detect_symmetry(1e-3).get_references()[0]
    pos = np.asarray(mon.cart_coords, float)
    pos = pos - pos.mean(axis=0)
    Z = np.array([qcel.periodictable.to_Z(s) for s in mon.symbols])

    oversized = [_rotation_about_z(2 * np.pi * k / (_MAX_GROUP_ORDER + 10))
                 for k in range(_MAX_GROUP_ORDER + 10)]
    assert _Frame(pos, Z).close(oversized, 0.05) is None

    # and the same situation reached the ordinary way
    for tol in (0.2, 0.5, 1.0):
        assert len(PointGroup.detect(mon.symbols, mon.cart_coords, atol=tol)) >= 1


def _rotation_about_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_report_sweep_shows_whether_loosening_would_help():
    # The dependency-free half of the report. Benzene's missing operations are
    # NOT hiding behind the tolerance: it sits at 4 from 0.05 A upward, so
    # loosening buys nothing and only trades toward over-detection. uracil is the
    # contrast -- its molecular mirror really is just past the default.
    from crystalatte.core.point_group import point_group_report

    benzene = load("benzene")[0].detect_symmetry(1e-3).get_references()[0]
    sweep = point_group_report(benzene.symbols, benzene.cart_coords)["sweep"]
    assert len(set(sweep.values())) == 1, (
        f"benzene should be flat across the sweep, got {sweep}"
    )

    uracil = load("uracil")[0].detect_symmetry(1e-3).get_references()[0]
    sweep = point_group_report(uracil.symbols, uracil.cart_coords)["sweep"]
    assert max(sweep.values()) > min(sweep.values()), (
        f"uracil's mirror should appear as the tolerance loosens, got {sweep}"
    )


def test_report_names_the_site_symmetry_without_pymatgen():
    # spglib is already a hard dependency and can NAME an integer rotation set,
    # so the site-symmetry label costs nothing. It cannot detect a molecular
    # point group from coordinates -- that is what the pymatgen solver is for.
    from crystalatte.core.point_group import point_group_report
    from crystalatte.work.dedup import context

    xtal = load("hexamine")[0].detect_symmetry(1e-3)
    refs = xtal.get_references()
    context.bind(xtal, refs)
    report = point_group_report(
        refs[0].symbols, refs[0].cart_coords,
        site_symmetry=context.W_lin_by_ref[0],
    )
    assert report["site_symmetry"] == "-43m"


def test_report_exposes_the_completeness_gap():
    # The question the report exists to answer: how much of the idealised group
    # do the coordinates actually support? For benzene the answer is "4 of 24",
    # and nothing else in the API surfaces that.
    from crystalatte.core.point_group import point_group_report

    mon = load("benzene")[0].detect_symmetry(1e-3).get_references()[0]
    report = point_group_report(mon.symbols, mon.cart_coords)
    assert report["schoenflies"] == "D6h"
    assert report["idealised"] == 24
    assert report["detected"] < report["idealised"]
    assert report["complete"] is False


def test_report_says_complete_when_it_is():
    from crystalatte.core.point_group import point_group_report

    mon = load("hexamine")[0].detect_symmetry(1e-3).get_references()[0]
    report = point_group_report(mon.symbols, mon.cart_coords)
    assert report["detected"] == report["idealised"] == 24
    assert report["complete"] is True
