import warnings
from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ...core.sym_ops import SymOpList
from . import context, dotkey


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


def relative_rows(rot: NDArray, tr: NDArray, p: NDArray, q: NDArray) -> NDArray:
    """Return the relative transforms ``Delta = T_p^-1 T_q`` of member pairs

    :param rot: ``(P, 3, 3)`` integer rotations of the members
    :param tr: ``(P, 3)`` integer translations of the members, in twelfths
    :param p: ``(E,)`` first member of each pair
    :param q: ``(E,)`` second member of each pair
    :returns: ``(E, 12)`` int64 rows flattened as ``(L (9), v (3))``
    """
    R = rot.astype(np.int64)
    t = tr.astype(np.int64)
    R_inv = int_inv(R)
    L = R_inv[p] @ R[q]
    v = (R_inv[p] @ (t[q] - t[p])[..., None])[..., 0]
    return np.concatenate((L.reshape(-1, 9), v), axis=1)


def pair_codes(rows: NDArray, ref_a: NDArray, ref_b: NDArray) -> NDArray:
    """Return the dressed code of every member pair under every dressing of its first member

    Code ``[e, s]`` stands for the canonical row of the coset ``s^-1 Delta_e D_b`` (:func:`_dressed_minima`),
    ranked among the rows of every pair with the same ``(ref_a, ref_b)``, so such codes compare like the rows
    they stand for. Many pairs share a relative transform, so each distinct one is dressed once.

    :param rows: ``(E, 12)`` relative transforms from :func:`relative_rows`
    :param ref_a: ``(E,)`` base reference of each pair's first member
    :param ref_b: ``(E,)`` base reference of each pair's second member
    :returns: ``(E, S_max)`` int32 codes; columns past the first member's dressing count are ``-1``
    """
    n_refs = len(context.dress_stacked_by_ref)
    S_max = max(len(W) for W, _, _ in context.dress_stacked_by_ref)
    out = np.full((len(rows), S_max), -1, dtype=np.int32)
    pair = ref_a * n_refs + ref_b
    for ab in np.unique(pair).tolist():
        sel = np.flatnonzero(pair == ab)
        _, first, inverse = np.unique(
            lex_codes(rows[sel]), return_index=True, return_inverse=True
        )
        dressed = lex_codes(_dressed_minima(rows[sel[first]], ab // n_refs, ab % n_refs))
        _, rank = np.unique(dressed, return_inverse=True)
        out[sel, : dressed.shape[1]] = rank.reshape(dressed.shape)[inverse.ravel()]
    return out


def _ref_tuple(group: int, n: int) -> tuple[int, ...]:
    """Decode a group code from :func:`chunk_keys` into its base references, sorted by ``context.anchor_rank``"""
    B = len(context.anchor_rank)
    rank_to_ref = np.argsort(context.anchor_rank)
    return tuple(int(rank_to_ref[(group // B ** (n - 1 - k)) % B]) for k in range(n))


def chunk_keys(codes: NDArray, edges: NDArray, refs: NDArray) -> tuple[NDArray, NDArray]:
    """Return the orbit keys of a chunk of clusters

    Members are sorted by anchor rank. An anchor's base reference leads its key, so only anchors of the
    lowest-ranked reference can win: after sorting they are the first ``c`` members, and each sees its
    partners' references as ``refs[1:]``.

    :param codes: ``(E, S_max)`` pair codes from :func:`pair_codes`
    :param edges: ``(C, n, n)`` row of ``codes`` for each ordered member pair of each cluster; the diagonal is
        never read
    :param refs: ``(C, n)`` base reference of each member
    :returns: ``(group, key)``: the ``(C,)`` int64 code of each cluster's ranked base-reference tuple
        (decoded by :func:`_ref_tuple`) and ``(C, n - 1)`` int32 keys. Two clusters are the same multimer, up to
        the linear merge, iff both agree
    """
    C, n = refs.shape
    rank = context.anchor_rank[refs]
    order = np.argsort(rank, axis=1, kind="stable")
    srank = np.take_along_axis(rank, order, axis=1)
    group = srank @ (len(context.anchor_rank) ** np.arange(n - 1, -1, -1, dtype=np.int64))
    if n == 1:
        return group, np.zeros((C, 1), dtype=np.int32)

    keys = np.empty((C, n - 1), dtype=np.int32)
    for g in np.unique(group).tolist():
        rows = np.flatnonzero(group == g)
        ref_tuple = _ref_tuple(g, n)
        o = order[rows]
        e = edges[rows[:, None, None], o[:, :, None], o[:, None, :]]

        c = ref_tuple.count(ref_tuple[0])
        partner_refs = ref_tuple[1:]
        S = len(context.dress_stacked_by_ref[ref_tuple[0]][0])
        K = np.empty((len(rows), c, S, n - 1), dtype=np.int32)  # (cluster, anchor, anchor dressing, slot)
        for a in range(c):
            for j in range(n - 1):
                K[:, a, :, j] = codes[e[:, a, j + (j >= a)], :S]

        # partners of one reference are unordered; their slots are contiguous
        for ref_b in sorted(set(partner_refs)):
            cols = [j for j, r in enumerate(partner_refs) if r == ref_b]
            K[..., cols[0] : cols[-1] + 1].sort(axis=3)

        # lexicographic minimum over (anchor, anchor dressing) for each cluster
        flat = K.reshape(-1, n - 1)
        pick = np.lexsort((*flat.T[::-1], np.repeat(np.arange(len(rows)), c * S)))
        keys[rows] = flat[pick[:: c * S]]
    return group, keys


def _list_keys(cands: list[SymOpList]) -> tuple[NDArray, NDArray]:
    """Return :func:`chunk_keys` for a list of clusters of one size, each its own partner table"""
    n = len(cands[0])
    rot = np.concatenate([g.rot_cache for g in cands])
    tr = np.concatenate([g.tr_cache for g in cands])
    ref = np.concatenate([g.ref_indices for g in cands]).astype(np.int64)

    a, b = np.nonzero(~np.eye(n, dtype=bool))
    base = np.arange(len(cands))[:, None] * n
    p, q = (base + a).ravel(), (base + b).ravel()
    codes = pair_codes(relative_rows(rot, tr, p, q), ref[p], ref[q])
    edges = np.zeros((len(cands), n, n), dtype=np.int64)
    edges[:, a, b] = np.arange(len(p)).reshape(len(cands), len(a))
    return chunk_keys(codes, edges, ref.reshape(len(cands), n))


def _group_keys(cands: list[SymOpList], refs: tuple[int, ...]) -> NDArray:
    """Return the orbit keys of clusters that share one ranked base-reference tuple

    :param cands: clusters whose ``ref_indices``, sorted by ``context.anchor_rank``, equal ``refs``
    :param refs: the shared base references, sorted by ``context.anchor_rank`` (non-linear first)
    :returns: an ``(N, n - 1)`` int32 array; two clusters have equal keys iff their rows are equal
    """
    if len(refs) == 1:
        return np.zeros((len(cands), 1), dtype=np.int32)
    return _list_keys(cands)[1]


def labels_from_keys(group: NDArray, keys: NDArray, cluster: Callable[[int], SymOpList]) -> NDArray:
    """Label clusters from their :func:`chunk_keys`, merging all-linear groups by the dot-product key

    :param group: ``(N,)`` group codes
    :param keys: ``(N, n - 1)`` orbit keys
    :param cluster: returns cluster ``k`` as a :class:`SymOpList`, for the dot-product key
    :returns: an ``(N,)`` int64 array of labels
    """
    labels = np.empty(len(group), dtype=np.int64)
    n_labels = 0
    for g in np.unique(group).tolist():
        idx = np.flatnonzero(group == g)
        refs = _ref_tuple(g, keys.shape[1] + 1)
        _, inverse = np.unique(keys[idx], axis=0, return_inverse=True)
        inverse = inverse.ravel()
        if all(context.linear_by_ref[r] for r in refs):
            inverse = _merge_by_dot_key(idx, refs, inverse, cluster)
        labels[idx] = inverse + n_labels
        n_labels += int(inverse.max()) + 1
    return labels


def orbit_labels(candidates: list[SymOpList]) -> NDArray:
    """Label clusters so that two share a label iff they are the same multimer under an isometry, modulo each
    member's dressing group

    The label is the class of an exact orbit key. Seen from an anchor ``a``, each partner ``b`` is the coset
    ``Delta[a,b] D_b`` (a dressing relabels the same molecule), reduced to its lexicographic minimum. The key
    is the minimum, over every anchor ``a`` and anchor dressing ``s`` in ``D_a``, of the sorted cosets
    ``s^-1 Delta[a,b] D_b``, prefixed with base-reference indices. ``D`` is the site symmetry plus each exact
    cell-symmetry self-isometry of the monomer (:mod:`.dressing`), so non-crystallographic congruences such as
    triazine's sigma_h merge here too.

    Each distinct relative transform is dressed once (:func:`pair_codes`), and a cluster's key is gathered from
    the codes of its member pairs (:func:`chunk_keys`). ``generate_neighbors`` builds the same codes from its
    partner tables instead of from :class:`SymOpList` objects.

    A group whose members are all linear is then merged further by the exact dot-product key
    (:mod:`.dotkey`), which also absorbs twists about a molecular axis that no dressing contains.

    :param candidates: clusters of one size to label; the dedup context must be bound
    :returns: an ``(N,)`` int64 array of labels
    """
    if not candidates:
        return np.empty(0, dtype=np.int64)
    group, keys = _list_keys(candidates)
    return labels_from_keys(group, keys, candidates.__getitem__)


def _merge_by_dot_key(
    idx: NDArray, refs: tuple[int, ...], frame: NDArray, cluster: Callable[[int], SymOpList]
) -> NDArray:
    """Coarsen the frame labels of an all-linear group to its dot-key classes

    Every frame class is one multimer, so only its first cluster is keyed (:func:`.dotkey.dot_keys`).
    If the key could overflow int64, the frame labels are kept and a :class:`RuntimeWarning` is raised.

    :param idx: ``(N,)`` indices of the group's clusters, passed to ``cluster``
    :param refs: their base references sorted by ``context.anchor_rank``, every one linear
    :param frame: ``(N,)`` frame labels of the group, numbered ``0..m-1``
    :param cluster: returns cluster ``k`` as a :class:`SymOpList`
    :returns: ``(N,)`` labels numbered ``0..k-1``, ``k <= m``
    """
    _, first = np.unique(frame, return_index=True)
    try:
        keys = dotkey.dot_keys([cluster(int(idx[i])) for i in first], refs)
    except OverflowError as err:
        warnings.warn(
            f"{err} for references {refs}; keeping their frame key, which may leave duplicates",
            RuntimeWarning,
            stacklevel=3,
        )
        return frame
    _, merged = np.unique(keys, axis=0, return_inverse=True)
    return merged.ravel()[frame]
