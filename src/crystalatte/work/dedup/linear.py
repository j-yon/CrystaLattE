"""Exact integer linear algebra and cell-shape pieces for the linear-monomer dot-product key.

Everything here is exact: nullspaces are computed over the rationals and returned as primitive
integer rows, and the metric basis is built from integer sums over the cell symmetry ``H``.
"""

from fractions import Fraction
from math import gcd, lcm

import numpy as np
from numpy.typing import NDArray


def integer_nullspace(M: NDArray) -> NDArray:
    """Return primitive integer rows spanning ``{x : M x = 0}`` over the rationals

    :param M: integer matrix of shape ``(r, m)``
    :returns: int64 array of shape ``(m - rank, m)``; the identity when ``r == 0``
    """
    M = np.asarray(M)
    m = M.shape[1]
    rows = [[Fraction(int(v)) for v in row] for row in M]
    pivots: list[int] = []
    r = 0
    for c in range(m):
        p = next((i for i in range(r, len(rows)) if rows[i][c] != 0), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        rows[r] = [v / rows[r][c] for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] != 0:
                f = rows[i][c]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        pivots.append(c)
        r += 1
    out = []
    for free in (c for c in range(m) if c not in pivots):
        x = [Fraction(0)] * m
        x[free] = Fraction(1)
        for i, pc in enumerate(pivots):
            x[pc] = -rows[i][free]
        den = lcm(*(v.denominator for v in x))
        ints = [int(v * den) for v in x]
        g = gcd(*ints)
        out.append([v // g for v in ints])
    return np.array(out, dtype=np.int64).reshape(len(out), m)


def metric_basis(H: NDArray) -> NDArray:
    """Return a basis of the symmetric metrics preserved by every ``h`` in ``H``

    :param H: integer cell-symmetry matrices of shape ``(n, 3, 3)``
    :returns: int64 array of shape ``(k, 3, 3)`` of symmetric ``G_k`` with ``h^T G_k h = G_k``
    """
    H = np.asarray(H, dtype=np.int64)
    basis: list[NDArray] = []
    for i in range(3):
        for j in range(i, 3):
            E = np.zeros((3, 3), dtype=np.int64)
            E[i, j] = E[j, i] = 1
            S = np.einsum("nji,jk,nkl->il", H, E, H)
            if np.linalg.matrix_rank(np.array([*basis, S]).reshape(len(basis) + 1, 9)) > len(basis):
                basis.append(S // np.gcd.reduce(S.ravel()))
    return np.array(basis, dtype=np.int64).reshape(len(basis), 3, 3)
