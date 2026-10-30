import dataclasses

import numpy as np
import pytest
import qcelemental as qcel
from crystalatte.core.crystal import Crystal
from crystalatte.core.multimer import ASU
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOpList
from crystalatte.work import neighbors
from crystalatte.work.dedup import context, dotkey, dressing, invariants, linear
from conftest import GENERATE_KW, crystal, load

_CELL = ((6.0, 7.0, 10.0), (90, 100, 90))


def _Z(mon):
    return np.array([qcel.periodictable.to_Z(s) for s in mon.symbols], dtype=np.int64)


def _hcn_cart(centre):
    axis = np.array([0.48, 0.60, 0.64])
    axis /= np.linalg.norm(axis)
    x = np.array([0.0, 1.07, 1.07 + 1.16])
    return ["H", "C", "N"], centre + (x - x.mean())[:, None] * axis


def _water_cart(centre):
    half = np.radians(104.5) / 2
    o = np.zeros(3)
    h1 = 0.96 * np.array([np.sin(half), np.cos(half), 0.0])
    h2 = 0.96 * np.array([-np.sin(half), np.cos(half), 0.0])
    pos = np.array([o, h1, h2])
    # a generic orientation, so that no cell symmetry is an accidental self-isometry
    a, b = 0.2, 2.5
    tilt = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]) @ np.array(
        [[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]]
    )
    return ["O", "H", "H"], centre + (pos - pos.mean(0)) @ tilt.T


def _p21c(*builders):
    A = Crystal(*_CELL, SpaceGroup(14), ASU(["H"], np.zeros((1, 3)))).lattice_vectors
    symbols, frac = [], []
    for build, centre in builders:
        s, cart = build(A @ np.asarray(centre, float))
        symbols += s
        frac.append(np.linalg.solve(A, cart.T).T)
    return Crystal(*_CELL, SpaceGroup(14), ASU(symbols, np.concatenate(frac)))


@pytest.fixture
def hcn_crystal():
    return _p21c((_hcn_cart, (0.23, 0.31, 0.17)))


@pytest.fixture
def mixed_crystal():
    return _p21c((_hcn_cart, (0.23, 0.31, 0.17)), (_water_cart, (0.71, 0.62, 0.58)))


@pytest.fixture
def argon_crystal():
    return Crystal((5.26,) * 3, (90,) * 3, SpaceGroup(225), ASU(["Ar"], np.zeros((1, 3))))


def _H(cry):
    rots = np.stack([op.rot for op in cry.space_group.sym_ops]).astype(np.int64)
    return dressing.cell_symmetry(rots, np.asarray(cry.lattice_vectors, float))

def test_integer_nullspace():
    N = linear.integer_nullspace(np.array([[1, 1, 1], [0, 2, -2]]))
    assert N.shape == (1, 3) and abs(np.gcd.reduce(N[0])) == 1
    assert np.array_equal(np.array([[1, 1, 1], [0, 2, -2]]) @ N.T, np.zeros((2, 1)))
    assert np.array_equal(linear.integer_nullspace(np.zeros((0, 4), np.int64)), np.eye(4, dtype=np.int64))

@pytest.mark.parametrize("name,k", [
    ("carbon_dioxide", 1), ("triazine", 2), ("adamantane", 2),
    ("benzene", 3), ("naphthalene", 4), ("ethyl_carbamate", 6),
])
def test_metric_basis_dimension(name, k):
    cry = crystal(name)
    B = linear.metric_basis(_H(cry))
    assert B.shape == (k, 3, 3) and B.dtype == np.int64
    for h in _H(cry):
        assert np.array_equal(np.einsum("ji,njk,kl->nil", h, B, h), B)

@pytest.mark.parametrize("name", ["carbon_dioxide", "triazine", "naphthalene", "ethyl_carbamate"])
def test_actual_metric_lies_in_the_basis(name):
    cry = crystal(name)
    A = np.asarray(cry.lattice_vectors, float)
    B = linear.metric_basis(_H(cry)).reshape(-1, 9).T.astype(float)
    g, *_ = np.linalg.lstsq(B, (A.T @ A).ravel(), rcond=None)
    assert np.abs(B @ g - (A.T @ A).ravel()).max() < 1e-8


def _ref_setup(name):
    cry = crystal(name).detect_symmetry(1e-3)
    mons = cry.get_references()
    context.bind(cry, mons)
    return cry, mons

def test_is_linear():
    _, (co2,) = _ref_setup("carbon_dioxide")
    _, (bz,) = _ref_setup("benzene")
    assert linear.is_linear(co2.cart_coords) and not linear.is_linear(bz.cart_coords)
    assert linear.is_linear(np.zeros((1, 3)))

def test_line_params_carbon_dioxide():
    cry, (co2,) = _ref_setup("carbon_dioxide")
    D, _, D_tr = context.dress_stacked_by_ref[0]
    p = linear.line_params(co2.frac_coords, co2.centroid_frac, _Z(co2),
                           np.asarray(cry.lattice_vectors, float), D, D_tr)
    assert p.q == 1 and not p.c0.any() and p.B.shape == (3, 0)
    assert p.U.shape == (3, 1) and abs(p.U[:, 0]).tolist() == [1, 1, 1] and len(set(p.U[:, 0])) == 1
    assert p.symmetric_ends

def test_line_params_general_position_polar(hcn_crystal):
    # one HCN per asymmetric unit on a general position of P2_1/c
    mons = hcn_crystal.get_references()
    context.bind(hcn_crystal, mons)
    D, _, D_tr = context.dress_stacked_by_ref[0]
    p = linear.line_params(mons[0].frac_coords, mons[0].centroid_frac, _Z(mons[0]),
                           np.asarray(hcn_crystal.lattice_vectors, float), D, D_tr)
    assert p.B.shape == (3, 3) and p.U.shape == (3, 3) and not p.symmetric_ends

def test_line_params_rejects_an_axis_the_dressings_do_not_keep():
    # a fake 4-fold about z tilts CO2's [111] axis, so the setup must refuse it
    cry, (co2,) = _ref_setup("carbon_dioxide")
    D, _, D_tr = context.dress_stacked_by_ref[0]
    four = np.array([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]], dtype=np.int64)
    with pytest.warns(RuntimeWarning, match="treating it as non-linear"):
        assert linear.line_params(co2.frac_coords, co2.centroid_frac, _Z(co2),
                                  np.asarray(cry.lattice_vectors, float),
                                  np.concatenate((D, four)),
                                  np.concatenate((D_tr, np.zeros((1, 3), np.int64)))) is None


def test_bind_records_linear_setup():
    _ref_setup("carbon_dioxide")
    assert context.linear_by_ref == [True] and context.line_by_ref[0].symmetric_ends
    assert context.metric_pieces.shape == (1, 3, 3)
    _ref_setup("benzene")
    assert context.linear_by_ref == [False] and context.line_by_ref == [None]


def test_anchor_rank_puts_nonlinear_first(mixed_crystal):
    refs = mixed_crystal.get_references()[::-1]  # HCN first, so the rule must reorder
    context.bind(mixed_crystal, refs)
    assert context.linear_by_ref == [True, False]
    assert context.anchor_rank.tolist() == [1, 0]


def _dimer(t1_twelfths, rot1=np.eye(3, dtype=np.int16)):
    return SymOpList.from_components(
        np.stack([np.eye(3, dtype=np.int16), rot1]),
        np.stack([np.zeros(3, np.int16), np.array(t1_twelfths, np.int16)]), 1, np.zeros(2, np.int64))


def _candidates(cry, refs, R, monkeypatch):
    """Return every candidate cluster ``generate_neighbors`` hands to ``orbit_labels``"""
    seen = []

    def _record(cands):
        seen.extend(cands)
        return np.arange(len(cands))

    monkeypatch.setattr(neighbors, "orbit_labels", _record)
    neighbors.generate_neighbors(cry, refs, np.array([[-2, 2]] * 3), R, 2, **GENERATE_KW)
    return seen

def test_dot_key_merges_the_surviving_co2_pair():
    _ref_setup("carbon_dioxide")
    k = dotkey.dot_keys([_dimer((-36, 0, 0)), _dimer((-24, -24, 12))], (0, 0))
    assert np.array_equal(k[0], k[1])

def test_dot_key_separates_equal_length_offsets_at_different_angles():
    _ref_setup("carbon_dioxide")
    k = dotkey.dot_keys([_dimer((12, 12, 12)), _dimer((12, 12, -12))], (0, 0))
    assert not np.array_equal(k[0], k[1])

def test_dot_key_ignores_member_order_and_inversion():
    _ref_setup("carbon_dioxide")
    a = _dimer((-36, 0, 0))
    swapped = SymOpList.from_components(a.rot_cache[::-1].copy(), a.tr_cache[::-1].copy(), 1, a.ref_indices)
    inverted = _dimer((-36, 0, 0), -np.eye(3, dtype=np.int16))    # second molecule flipped end to end
    k = dotkey.dot_keys([a, swapped, inverted], (0, 0))
    assert np.array_equal(k[0], k[1]) and np.array_equal(k[0], k[2])

def test_dot_key_agrees_with_every_existing_merge(monkeypatch):
    # each existing-key class must have one dot key: the spec's consistency argument
    cry = crystal("carbon_dioxide").detect_symmetry(1e-3)
    refs = cry.get_references()
    cands = _candidates(cry, refs, 15.0, monkeypatch)
    frame = invariants.orbit_labels(cands)
    dk = dotkey.dot_keys(cands, (0, 0))
    for lab in np.unique(frame):
        rows = dk[frame == lab]
        assert (rows == rows[0]).all()

def test_dot_key_single_atoms(argon_crystal):
    context.bind(argon_crystal, argon_crystal.get_references())
    k = dotkey.dot_keys([_dimer((6, 6, 0)), _dimer((0, 6, 6)), _dimer((12, 0, 0))], (0, 0))
    assert k.shape[1] == 1                      # one o·o entry, no axis rows
    assert np.array_equal(k[0], k[1]) and not np.array_equal(k[0], k[2])


def test_dot_key_guards_the_form_construction(monkeypatch):
    # S * c0 = 12 * 2**62 wraps to 0 in int64, so only a bound taken before the forms are built sees it
    _ref_setup("carbon_dioxide")
    huge = dataclasses.replace(context.line_by_ref[0], c0=np.array([2**62, 0, 0], dtype=np.int64))
    monkeypatch.setattr(context, "line_by_ref", [huge])
    with pytest.raises(OverflowError):
        dotkey.dot_keys([_dimer((12, 0, 0))], (0, 0))
