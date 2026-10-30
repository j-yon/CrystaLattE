import numpy as np
import pytest
from crystalatte.work.dedup import context, dressing, linear
from conftest import GENERATE_KW, crystal, load

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
