"""Exact integer linear algebra and cell-shape pieces for the linear-monomer dot-product key.

Everything here is exact: nullspaces are computed over the rationals and returned as primitive
integer rows, and the metric basis is built from integer sums over the cell symmetry ``H``.
"""

import warnings
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm

import numpy as np
from numpy.typing import NDArray

from .dressing import EXACT_ATOL

#: coarse pre-filter on a dressing keeping the molecular axis; it only gives a
#: more specific fallback message, the SELF_CHECK_ATOL self-check is binding
AXIS_RTOL = 1e-3

#: largest least-squares residual, in fractional units, of the self-check
SELF_CHECK_ATOL = 1e-6


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


def is_linear(cart_coords: NDArray) -> bool:
    """Return whether a monomer's atoms lie on one line; true for a single atom

    :param cart_coords: ``(n, 3)`` Cartesian atom positions
    :returns: whether the centred positions have rank at most one
    """
    pos = np.asarray(cart_coords, dtype=float)
    return bool(np.linalg.matrix_rank(pos - pos.mean(axis=0), tol=1e-6) <= 1)


def symmetric_ends(
    frac_coords: NDArray, centroid_frac: NDArray, Z: NDArray, lattice: NDArray
) -> bool:
    """Return whether inversion through the centroid maps the monomer onto itself

    :param frac_coords: ``(n, 3)`` fractional atom positions
    :param centroid_frac: the centroid to invert through
    :param Z: ``(n,)`` atomic numbers
    :param lattice: ``(3, 3)`` fractional-to-Cartesian matrix, cell vectors as columns
    :returns: whether every inverted atom lies within :data:`~.dressing.EXACT_ATOL` of an atom of
        the same element
    """
    X = np.asarray(frac_coords, dtype=float)
    image = 2.0 * np.asarray(centroid_frac, dtype=float) - X
    dist = np.linalg.norm((image[:, None] - X[None]) @ np.asarray(lattice, dtype=float).T, axis=2)
    same_element = np.asarray(Z)[:, None] == np.asarray(Z)[None]
    return bool(np.where(same_element, dist, np.inf).min(axis=1).max() <= EXACT_ATOL)


@dataclass(frozen=True)
class LineParams:
    """The unknowns of a linear reference monomer under its dressing group

    The centroid is ``c0 / q + B @ xi`` and the axis is ``U @ eta`` for real ``xi`` and ``eta``.
    """

    c0: NDArray  # (3,) int64, fixed part of the centroid times q
    q: int  # > 0
    B: NDArray  # (3, n_xi) int64, free centroid directions
    U: NDArray | None  # (3, n_eta) int64, free axis directions; None for a single atom
    symmetric_ends: bool


def _residual(M: NDArray, x: NDArray) -> float:
    """Return the largest entry of the least-squares residual of ``x`` against the columns of ``M``"""
    if M.shape[1] == 0:
        return float(np.abs(x).max())
    coef, *_ = np.linalg.lstsq(M.astype(float), x, rcond=None)
    return float(np.abs(M @ coef - x).max())


def _reject(reason: str) -> None:
    warnings.warn(
        f"linear monomer setup failed ({reason}); treating it as non-linear",
        RuntimeWarning,
        stacklevel=3,
    )


def line_params(
    frac_coords: NDArray,
    centroid_frac: NDArray,
    Z: NDArray,
    lattice: NDArray,
    D: NDArray,
    D_tr: NDArray,
) -> LineParams | None:
    """Return the centroid and axis unknowns of a linear reference monomer

    The centroid ranges over the points fixed by every dressing ``x -> h x + t / 12``; the axis over
    the vectors that every ``h`` maps to plus or minus themselves.

    :param frac_coords: ``(n, 3)`` fractional atom positions of a linear reference monomer
    :param centroid_frac: its centroid, fixed by every dressing
    :param Z: ``(n,)`` atomic numbers
    :param lattice: ``(3, 3)`` fractional-to-Cartesian matrix, cell vectors as columns
    :param D: ``(d, 3, 3)`` int64 dressing linear parts
    :param D_tr: ``(d, 3)`` int64 dressing translations, in twelfths
    :returns: the :class:`LineParams`, or ``None`` with a :class:`RuntimeWarning` when the dressings
        do not fix a centroid or keep the axis, or the actual monomer fails the self-check
    """
    X = np.asarray(frac_coords, dtype=float)
    c = np.asarray(centroid_frac, dtype=float)
    A = np.asarray(lattice, dtype=float)
    D = np.asarray(D, dtype=np.int64)
    D_tr = np.asarray(D_tr, dtype=np.int64)
    eye = np.eye(3, dtype=np.int64)

    # centroid: (x q, q) with 12 (I - h) x q - t q = 0 for every dressing
    N = integer_nullspace(
        np.concatenate([np.hstack((12 * (eye - h), -t[:, None])) for h, t in zip(D, D_tr)])
    )
    i = next((i for i, v in enumerate(N) if v[3] != 0), None)
    if i is None:
        _reject("no centroid is fixed by every dressing")
        return None
    star = N[i]
    rest = [star[3] * w - w[3] * star for j, w in enumerate(N) if j != i]
    rest = [w // np.gcd.reduce(w) for w in rest if w.any()]
    sign = 1 if star[3] > 0 else -1
    c0, q = sign * star[:3], int(abs(star[3]))
    B = np.array([w[:3] for w in rest], dtype=np.int64).reshape(len(rest), 3).T

    U = None
    u0 = None
    if len(X) > 1:
        cart = X @ A.T
        centred = cart - cart.mean(axis=0)
        principal = np.linalg.eigh(centred.T @ centred)[1][:, -1]
        proj = centred @ principal
        u0 = X[np.argmax(proj)] - X[np.argmin(proj)]
        Au0 = A @ u0
        rows = []
        for h in D:
            eps = int(np.sign((A @ h @ u0) @ Au0))
            if np.linalg.norm(A @ (h @ u0 - eps * u0)) > AXIS_RTOL * np.linalg.norm(Au0):
                _reject("a dressing does not keep the molecular axis")
                return None
            rows.append(h - eps * eye)
        U = integer_nullspace(np.concatenate(rows)).T

    centroid_off = _residual(B, c - c0 / q) > SELF_CHECK_ATOL
    if centroid_off or (U is not None and _residual(U, u0) > SELF_CHECK_ATOL):
        _reject("the actual centroid or axis is outside the derived unknowns")
        return None

    return LineParams(c0, q, B, U, symmetric_ends(X, c, Z, A))
