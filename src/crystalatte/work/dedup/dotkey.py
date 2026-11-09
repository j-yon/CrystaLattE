"""Exact dot-product key for clusters whose members are all linear monomers or single atoms.

A linear member is described by its centroid ``p`` and axis ``u``, neither of which carries the twist about
the axis that the orbit key cannot absorb. Two clusters are congruent iff, for some reference-compatible
relabelling and end-to-end flips, the dot products between the centroid offsets ``p_b - p_0`` and the axes
agree. Each dot product is kept as an exact integer polynomial in the cell-shape pieces ``g_k`` and the
reference unknowns ``xi`` and ``eta`` (:class:`~.linear.LineParams`), so equal tables mean equal numbers for
the real crystal with no tolerance.
"""

from itertools import permutations, product
from math import lcm
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ...core.sym_ops import SymOpList
from . import context, invariants

#: int64 entries of the variant stack materialised per chunk of clusters; bounds it to ~128 MB
DOT_CHUNK_ENTRIES: Final[int] = 1 << 24


def _n_tri(m: int) -> int:
    """Return the number of distinct monomials ``x_i x_j`` with ``i <= j`` over ``m`` unknowns"""
    return m * (m + 1) // 2


def _collapse(C: NDArray) -> NDArray:
    """Return the coefficients of the distinct monomials of a square coefficient matrix

    ``x^T C x`` has coefficient ``C_ii`` on ``x_i^2`` and ``C_ij + C_ji`` on ``x_i x_j`` for ``i < j``.

    :param C: integer array of shape ``(..., m, m)``
    :returns: integer array of shape ``(..., m (m + 1) / 2)`` in upper-triangle order
    """
    i, j = np.triu_indices(C.shape[-1])
    return C[..., i, j] + np.where(i != j, C[..., j, i], 0)


def _forms(rot: NDArray, tr: NDArray, refs: tuple[int, ...]) -> tuple[NDArray, NDArray, list[bool]]:
    """Return every member's centroid and axis as integer forms in the global unknowns

    :param rot: ``(N, n, 3, 3)`` member rotations, members ordered like ``refs``
    :param tr: ``(N, n, 3)`` member translations, in twelfths
    :param refs: each member's reference index
    :returns: centroid forms ``(N, n, 3, 1 + Xi)`` over ``[1, xi]`` scaled by ``S``, axis forms
        ``(N, n, 3, Eta)`` over ``eta``, and whether each member has an axis
    """
    distinct = list(dict.fromkeys(refs))
    line = {i: context.line_by_ref[i] for i in distinct}
    S = 12 * lcm(*(line[i].q for i in distinct))
    xi_at, eta_at, n_xi, n_eta = {}, {}, 0, 0
    for i in distinct:
        xi_at[i], eta_at[i] = n_xi, n_eta
        n_xi += line[i].B.shape[1]
        n_eta += 0 if line[i].U is None else line[i].U.shape[1]

    N, n = rot.shape[:2]
    P = np.zeros((N, n, 3, 1 + n_xi), dtype=np.int64)
    U = np.zeros((N, n, 3, n_eta), dtype=np.int64)
    has_axis = []
    for a, i in enumerate(refs):
        p, R = line[i], rot[:, a]
        P[:, a, :, 0] = R @ ((S // p.q) * p.c0) + (S // 12) * tr[:, a]
        lo = 1 + xi_at[i]
        P[:, a, :, lo : lo + p.B.shape[1]] = S * (R @ p.B)
        has_axis.append(p.U is not None)
        if p.U is not None:
            U[:, a, :, eta_at[i] : eta_at[i] + p.U.shape[1]] = R @ p.U
    return P, U, has_axis


def _check_overflow(rot: NDArray, tr: NDArray, refs: tuple[int, ...], G: NDArray) -> None:
    """Raise if building the forms or their dot-product tables could leave int64

    The bound is taken in Python integers before any form is built, so a wrapped intermediate cannot hide.
    Entries stay below ``2**62``, so the int64 sentinel of :func:`~.invariants.segmented_lexmin` exceeds them all.

    :param rot: ``(N, n, 3, 3)`` member rotations
    :param tr: ``(N, n, 3)`` member translations, in twelfths
    :param refs: each member's reference index
    :param G: ``(k, 3, 3)`` metric pieces
    :raises OverflowError: if any form or table entry could reach ``2**62``
    """
    line = [context.line_by_ref[i] for i in dict.fromkeys(refs)]
    S = 12 * lcm(*(p.q for p in line))
    r, t = int(np.abs(rot).max(initial=0)), int(np.abs(tr).max(initial=0))
    big = 0
    for p in line:
        big = max(
            big,
            3 * r * (S // p.q) * int(np.abs(p.c0).max(initial=0)) + (S // 12) * t,
            3 * r * S * int(np.abs(p.B).max(initial=0)),
            0 if p.U is None else 3 * r * int(np.abs(p.U).max(initial=0)),
        )
    # each entry sums at most 9 products per form, doubled by the collapse and four-fold by the offsets
    if 72 * big * big * int(np.abs(G).max(initial=0)) >= 2**62 or big >= 2**62:
        raise OverflowError("dot-product formulas would overflow int64")


def _relabellings(refs: tuple[int, ...], has_axis: list[bool]) -> list[tuple[NDArray, NDArray]]:
    """Return every listing order within reference blocks and every end-to-end flip of symmetric members

    :param refs: each member's reference index, equal references contiguous
    :param has_axis: whether each member has an axis
    :returns: ``(pi, eps)`` pairs; position ``b`` of the relabelled cluster is member ``pi[b]``, whose axis
        is multiplied by ``eps[b]``
    """
    blocks: list[list[int]] = []
    for a, i in enumerate(refs):
        if blocks and refs[blocks[-1][0]] == i:
            blocks[-1].append(a)
        else:
            blocks.append([a])
    flippable = [
        has_axis[a] and context.line_by_ref[i].symmetric_ends for a, i in enumerate(refs)
    ]
    out = []
    for block_perms in product(*(permutations(b) for b in blocks)):
        pi = np.array([a for perm in block_perms for a in perm], dtype=np.int64)
        for signs in product(*((1, -1) if flippable[a] else (1,) for a in pi)):
            out.append((pi, np.array(signs, dtype=np.int64)))
    return out


def _variant_rows(
    PP: NDArray, PU: NDArray, UU: NDArray, pi: NDArray, eps: NDArray, axis_pos: NDArray
) -> NDArray:
    """Return one relabelling's flattened dot-product table for each cluster

    :param PP: ``(N, k, n, n, T_P)`` collapsed centroid-centroid products
    :param PU: ``(N, k, n, n, 1 + Xi, Eta)`` centroid-axis products
    :param UU: ``(N, k, n, n, T_U)`` collapsed axis-axis products
    :param pi: member at each relabelled position
    :param eps: axis sign at each relabelled position
    :param axis_pos: relabelled positions whose member has an axis, ascending
    :returns: ``(N, L)`` int64 rows, ``k``-major
    """
    N, k, n = PP.shape[:3]
    PP = PP[:, :, pi][:, :, :, pi]
    PU = PU[:, :, pi][:, :, :, pi] * eps[None, None, None, :, None, None]
    UU = UU[:, :, pi][:, :, :, pi] * (eps[:, None] * eps[None, :])[None, None, :, :, None]

    b, c = np.triu_indices(n - 1)
    b, c = b + 1, c + 1
    oo = PP[:, :, b, c] - PP[:, :, b, 0] - PP[:, :, 0, c] + PP[:, :, 0, 0][:, :, None]
    ou = PU[:, :, 1:][:, :, :, axis_pos] - PU[:, :, :1][:, :, :, axis_pos]
    bu, cu = np.triu_indices(len(axis_pos))
    uu = UU[:, :, axis_pos[bu], axis_pos[cu]]
    return np.concatenate(
        (oo.reshape(N, k, -1), ou.reshape(N, k, -1), uu.reshape(N, k, -1)), axis=2
    ).reshape(N, -1)


def dot_keys(cands: list[SymOpList], refs: tuple[int, ...]) -> NDArray:
    """Return an exact congruence key for clusters whose members are all linear

    Two clusters get equal rows iff some listing order (within each reference) and end-to-end flip of
    symmetric members gives equal tables of dot-product formulas between the centroid offsets from member
    ``0`` and the axes. Each formula is an integer polynomial in the metric pieces ``g_k`` and the reference
    unknowns ``xi`` and ``eta``; the key is the lexicographically smallest table over every variant.

    :param cands: clusters whose base-reference indices, sorted like ``refs``, equal ``refs``; the dedup
        context must be bound and every reference in ``refs`` must have line unknowns
    :param refs: the clusters' reference indices sorted by :data:`~.context.anchor_rank`
    :returns: an ``(N, L)`` int64 array; ``L`` depends only on ``refs``
    :raises ValueError: if a reference in ``refs`` has no line unknowns or a cluster's references differ
    :raises OverflowError: if the integer formulas could exceed int64
    """
    n = len(refs)
    if not cands:
        return dot_keys_arrays(
            np.zeros((0, n, 3, 3), np.int64), np.zeros((0, n, 3), np.int64), np.zeros((0, n), np.int64), refs
        )
    return dot_keys_arrays(
        np.stack([g.rot_cache for g in cands]),
        np.stack([g.tr_cache for g in cands]),
        np.stack([g.ref_indices for g in cands]),
        refs,
    )


def dot_keys_arrays(rot: NDArray, tr: NDArray, ref_idx: NDArray, refs: tuple[int, ...]) -> NDArray:
    """:func:`dot_keys` of clusters given as stacked member arrays

    :param rot: ``(N, n, 3, 3)`` member rotations
    :param tr: ``(N, n, 3)`` member translations, in twelfths
    :param ref_idx: ``(N, n)`` member base references
    :param refs: the clusters' reference indices sorted by :data:`~.context.anchor_rank`
    :returns: an ``(N, L)`` int64 array, as :func:`dot_keys`
    :raises ValueError: if a reference in ``refs`` has no line unknowns or a cluster's references differ
    :raises OverflowError: if the integer formulas could exceed int64
    """
    refs = tuple(int(i) for i in refs)
    n = len(refs)
    if any(context.line_by_ref[i] is None for i in refs):
        raise ValueError(f"references {refs} are not all linear")

    order = np.argsort(context.anchor_rank[ref_idx], axis=1, kind="stable")
    if not (np.take_along_axis(ref_idx, order, axis=1) == np.array(refs, dtype=np.int64)).all():
        raise ValueError(f"every cluster must have references {refs}")
    rot = np.take_along_axis(rot.astype(np.int64), order[:, :, None, None], axis=1)
    tr = np.take_along_axis(tr.astype(np.int64), order[:, :, None], axis=1)

    G = context.metric_pieces
    _check_overflow(rot, tr, refs, G)
    P, U, has_axis = _forms(rot, tr, refs)
    variants = _relabellings(refs, has_axis)
    axis_pos = np.flatnonzero(has_axis)

    n_ax = len(axis_pos)
    width = len(G) * (
        (n - 1) * n // 2 * _n_tri(P.shape[-1]) + (n - 1) * n_ax * P.shape[-1] * U.shape[-1]
        + n_ax * (n_ax + 1) // 2 * _n_tri(U.shape[-1])
    )
    if width == 0:
        return np.zeros((len(rot), 0), dtype=np.int64)

    step = max(1, DOT_CHUNK_ENTRIES // (len(variants) * width))
    out = np.empty((len(rot), width), dtype=np.int64)
    for lo in range(0, len(rot), step):
        Pc, Uc = P[lo : lo + step], U[lo : lo + step]
        PP = _collapse(np.einsum("naxi,kxy,nbyj->nkabij", Pc, G, Pc))
        PU = np.einsum("naxi,kxy,nbyj->nkabij", Pc, G, Uc)
        UU = _collapse(np.einsum("naxi,kxy,nbyj->nkabij", Uc, G, Uc))
        stack = np.stack([_variant_rows(PP, PU, UU, pi, eps, axis_pos) for pi, eps in variants], axis=1)
        out[lo : lo + step] = invariants.segmented_lexmin(stack)
    return out

