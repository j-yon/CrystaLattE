from collections import defaultdict
from functools import cache
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ...core.sym_ops import SymOpList
from . import context


def int_inv(M: NDArray) -> NDArray:
    """Return the exact integer inverse of a unimodular integer matrix

    :param M: a ``(3, 3)`` or ``(n, 3, 3)`` integer matrix with ``det = ±1``
    :returns: exact ``int64`` inverse, same shape as ``M``
    """
    inv = np.rint(np.linalg.inv(M)).astype(np.int64)

    # assert unimodularity, otherwise the integer inverse is not exact
    identity = np.broadcast_to(np.eye(3, dtype=np.int64), np.shape(M))
    assert np.array_equal(inv @ np.asarray(M, dtype=np.int64), identity), (
        "int_inv called on a non-unimodular matrix"
    )

    return inv


def relative_transforms(g_N: SymOpList) -> tuple[NDArray, NDArray]:
    """Return the cluster's pairwise relative transforms ``Delta[a,b] = T_a^-1 T_b = (L[a,b], v[a,b])``

    Exact integers in the lattice basis (translations in twelfths), cached on the cluster

    :param g_N: cluster whose relative transforms are requested
    :returns: A tuple ``(L, v)`` of shapes ``(n, n, 3, 3)`` and ``(n, n, 3)``
    """
    # check for existing cache
    cached = getattr(g_N, "_rel_int_cache", None)
    if cached is not None:
        return cached

    R = g_N.rot_cache.astype(np.int64)  # (n, 3, 3)
    t = g_N.tr_cache.astype(np.int64)  # (n, 3), twelfths
    R_inv = int_inv(R)

    # contract rotations
    L = np.einsum("aij,bjk->abik", R_inv, R)  # (n, n, 3, 3)

    # contract translations
    dt = t[None, :, :] - t[:, None, :]  # (n, n, 3): t_b - t_a
    v = np.einsum("aij,abj->abi", R_inv, dt)  # (n, n, 3)

    # cache and return transform
    g_N._rel_int_cache = (L, v)
    return L, v


#: dressed rows materialised per chunk in :func:`_dressed_minima`; bounds its intermediates to ~100 MB
DRESS_CHUNK_ROWS: Final[int] = 1 << 19


def lex_codes(rows: NDArray) -> NDArray:
    """Return order-preserving int64 codes for the rows of an integer array

    Each column is shifted to start at zero and packed big-endian into as few bits as its observed range needs,
    so comparing codes compares rows lexicographically. If the ranges need more than 63 bits, falls back to each
    row's rank among the distinct rows, which also preserves order but costs a sort.

    :param rows: a ``(..., m)`` integer array
    :returns: a ``(...)`` int64 array of codes, comparable only with codes from the same call
    """
    flat = rows.reshape(-1, rows.shape[-1])
    if len(flat) == 0:
        return np.zeros(rows.shape[:-1], dtype=np.int64)
    lo = flat.min(axis=0)
    bits = [int(span).bit_length() for span in flat.max(axis=0) - lo]
    if sum(bits) > 63:
        _, rank = np.unique(flat, axis=0, return_inverse=True)
        return rank.reshape(rows.shape[:-1]).astype(np.int64)
    code = np.zeros(len(flat), dtype=np.int64)
    for j, b in enumerate(bits):
        if b:
            code = (code << b) | (flat[:, j] - lo[j])
    return code.reshape(rows.shape[:-1])


def _dressed_minima(D: NDArray, ref_a: int, ref_b: int) -> NDArray:
    """Return each relative transform's canonical partner row under every anchor dressing

    Row ``u`` of ``D`` is a relative transform ``Delta = (L, v)`` from an anchor of reference ``ref_a`` to a
    partner of reference ``ref_b``. For every ``s`` in the anchor's dressing group ``D_a``, the result holds the
    lexicographically smallest row of the coset ``s^-1 Delta D_b``: the partner's label once the anchor is
    relabelled by ``s``.

    :param D: ``(U, 12)`` int64 relative transforms flattened as ``(L (9), v (3))``
    :param ref_a: anchor base reference
    :param ref_b: partner base reference
    :returns: ``(U, |D_a|, 12)`` int64 canonical rows
    """
    _, Wa_inv, da = context.dress_stacked_by_ref[ref_a]
    Wb, _, db = context.dress_stacked_by_ref[ref_b]
    S, K = len(Wa_inv), len(Wb)
    out = np.empty((len(D), S, 12), dtype=np.int64)
    step = max(1, DRESS_CHUNK_ROWS // (S * K))
    for i in range(0, len(D), step):
        L = D[i : i + step, :9].reshape(-1, 1, 1, 3, 3)
        v = D[i : i + step, 9:].reshape(-1, 1, 1, 3, 1)
        # s^-1 (Delta w): linear part s^-1 L W, translation s^-1 (L d_w + v - d_s); axes (u, s, w)
        Lx = Wa_inv[None, :, None] @ L @ Wb[None, None]
        vx = Wa_inv[None, :, None] @ (
            L @ db[None, None, :, :, None] + v - da[None, :, None, :, None]
        )
        rows = np.concatenate(
            (Lx.reshape(*Lx.shape[:3], 9), vx[..., 0]), axis=3
        )  # (u, S, K, 12)
        k = lex_codes(rows).argmin(axis=2)
        out[i : i + step] = np.take_along_axis(rows, k[..., None, None], axis=2)[
            :, :, 0
        ]
    return out


def _group_keys(cands: list[SymOpList], refs: tuple[int, ...]) -> NDArray:
    """Return the orbit keys of clusters that share one sorted base-reference tuple

    :param cands: clusters whose sorted ``ref_indices`` equal ``refs``
    :param refs: the shared base references, ascending
    :returns: an ``(N, n - 1)`` int64 array; two clusters have equal keys iff their rows are equal
    """
    N, n = len(cands), len(refs)
    if n == 1:
        return np.zeros((N, 1), dtype=np.int64)

    # sort members by base reference so every cluster's members line up; the key ignores member order
    order = np.argsort(np.stack([g.ref_indices for g in cands]), axis=1, kind="stable")
    R = np.take_along_axis(
        np.stack([g.rot_cache for g in cands]).astype(np.int64),
        order[..., None, None],
        axis=1,
    )
    t = np.take_along_axis(
        np.stack([g.tr_cache for g in cands]).astype(np.int64), order[..., None], axis=1
    )

    # relative transforms Delta[a,b] = T_a^-1 T_b of every cluster at once
    R_inv = int_inv(R)
    L = R_inv[:, :, None] @ R[:, None]  # (N, n, n, 3, 3)
    v = (R_inv[:, :, None] @ (t[:, None] - t[:, :, None])[..., None])[
        ..., 0
    ]  # (N, n, n, 3)

    # an anchor's base reference leads its key, so only anchors of the smallest reference can win.
    # After sorting they are the first c members, and each sees its partners' references as refs[1:]
    c = refs.count(refs[0])
    partner_refs = refs[1:]
    S = len(context.dress_stacked_by_ref[refs[0]][0])
    M = np.empty(
        (N, c, S, n - 1), dtype=np.int64
    )  # (cluster, anchor, anchor dressing, partner slot)
    for ref_b in sorted(set(partner_refs)):
        cols = [j for j, r in enumerate(partner_refs) if r == ref_b]
        pairs = [
            (a, j, j + (j >= a)) for a in range(c) for j in cols
        ]  # (anchor, slot, member)
        D = np.concatenate(
            [
                np.concatenate((L[:, a, b].reshape(N, 9), v[:, a, b]), axis=1)
                for a, _, b in pairs
            ]
        )

        # many clusters share a relative transform: dress each distinct one once.
        # Rows are packed into int64 codes first, a 1-D unique being far faster than unique(axis=0)
        _, first, inverse = np.unique(
            lex_codes(D), return_index=True, return_inverse=True
        )
        U = D[first]
        codes = lex_codes(_dressed_minima(U, refs[0], ref_b))[inverse.ravel()].reshape(
            len(pairs), N, S
        )
        for p, (a, j, _) in enumerate(pairs):
            M[:, a, :, j] = codes[p]

        # partners of one reference are unordered; their slots are contiguous
        M[..., cols[0] : cols[-1] + 1].sort(axis=3)

    # lexicographic minimum over (anchor, anchor dressing) for each cluster
    flat = M.reshape(N * c * S, n - 1)
    order = np.lexsort((*flat.T[::-1], np.repeat(np.arange(N), c * S)))
    return flat[order[:: c * S]]


def orbit_labels(candidates: list[SymOpList]) -> NDArray:
    """Label clusters so that two share a label iff they are the same multimer under an isometry, modulo each
    member's dressing group

    The label is the class of an exact orbit key. Seen from an anchor ``a``, each partner ``b`` is the coset
    ``Delta[a,b] D_b`` (a dressing relabels the same molecule), reduced to its lexicographic minimum. The key
    is the minimum, over every anchor ``a`` and anchor dressing ``s`` in ``D_a``, of the sorted cosets
    ``s^-1 Delta[a,b] D_b``, prefixed with base-reference indices. ``D`` is the site symmetry plus each exact
    cell-symmetry self-isometry of the monomer (:mod:`.dressing`), so non-crystallographic congruences such as
    triazine's sigma_h merge here too.

    All clusters are keyed in one batch: each distinct relative transform is dressed once
    (:func:`_dressed_minima`) and rows are compared as packed int64 codes (:func:`lex_codes`).

    :param candidates: clusters to label; the dedup context must be bound
    :returns: an ``(N,)`` int64 array of labels
    """
    labels = np.empty(len(candidates), dtype=np.int64)
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, g in enumerate(candidates):
        groups[tuple(sorted(g.ref_indices.tolist()))].append(i)

    n_labels = 0
    for refs, idx in groups.items():
        _, inverse = np.unique(
            _group_keys([candidates[i] for i in idx], refs), axis=0, return_inverse=True
        )
        labels[idx] = inverse.ravel() + n_labels
        n_labels += int(inverse.max()) + 1
    return labels


def cluster_atoms_cart(g_N: SymOpList) -> NDArray:
    """Reconstruct the cluster's Cartesian atom positions from the module-level context bindings

    :param g_N: The cluster whose atom positions are requested
    :returns: An ``(n_atoms, 3)`` array of Cartesian coordinates
    """
    ref = g_N.ref_indices
    parts = [
        op.apply(context.references[ref[k]].frac_coords) for k, op in enumerate(g_N)
    ]
    return context.active_crystal.to_cartesian(np.vstack(parts))


@cache
def _triu_indices(n: int) -> tuple[NDArray, NDArray]:
    """Cache np.triu_indices for efficiency

    :param n: Number of atoms in the cluster
    :returns: The ``(i, j)`` index arrays for pairs with ``i < j``
    """
    return np.triu_indices(n, k=1)


def distance_fingerprint(g_N: SymOpList) -> tuple[NDArray, NDArray]:
    """Return the cluster's cached Z-tagged sorted pairwise atom-distance fingerprint

    :param g_N: cluster to fingerprint
    :returns: tuple ``(zz, distances)`` of equal-length 1-D arrays, sorted jointly by ``(zz, distance)``
    """
    # if cached, return
    cached = getattr(g_N, "_dist_fp_cache", None)
    if cached is not None:
        return cached

    # not cached, compute here
    atoms = cluster_atoms_cart(g_N)  # (n_atoms, 3)
    ref = g_N.ref_indices  # (n_ops,)
    Z = np.concatenate(
        [context.Z_by_ref[ref[k]] for k in range(len(g_N))]
    )  # (n_atoms,)
    i_idx, j_idx = _triu_indices(len(atoms))  # (n_atom_pairs,)
    dists = np.linalg.norm(atoms[i_idx] - atoms[j_idx], axis=1)
    zz = Z[i_idx] * Z[j_idx]
    order = np.lexsort((dists, zz))  # lexsort by (zz, dists)
    fp = (zz[order], dists[order])

    # cache and return
    g_N._dist_fp_cache = fp
    return fp
