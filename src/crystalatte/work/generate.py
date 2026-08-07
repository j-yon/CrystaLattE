"""Generate unique N-mers from a crystal and a reference monomer.

Pipeline:

1. :func:`_generate_neighbors` enumerates symmetry-related neighbor monomers
   within a cutoff (filter convention selected by keyword arguments:
   ``use_origin``, ``use_com``, or the default closest-contact), forms all
   ``C(M, N-1)`` :class:`~crystalatte.core.sym_ops.SymOpList` candidates with
   the reference identity prepended, groups them by ``translation_fp`` (a
   structural hash), and runs :func:`_process_bucket` per bucket to deduplicate
   via :func:`_is_bijection`.
2. :func:`_is_bijection` tests equivalence cheapest-first:

   a. distance-fingerprint gate (:func:`_fingerprints_match`) — a cached,
      vectorized necessary condition that rejects most non-equivalent pairs;
   b. exact-integer matcher (:func:`_same_multimer_int`) — the complete,
      roundoff-free decision for every crystallographic (same-orbit) merge,
      modding out the site symmetry (twelfths-scale integer arithmetic; no
      floating-point tolerance involved);
   c. float congruence fallback (:func:`_same_multimer`), gated by
      ``_check_congruence`` — runs only when the integer matcher fails and the
      molecular symmetry is richer than the site symmetry. It matches the two
      clusters in relative-transform space using the FULL monomer
      self-isometry group (:attr:`~crystalatte.core.multimer.Monomer.self_isometries`),
      so it merges every energetically-equivalent congruence: proper
      molecular-symmetry congruences AND mirror-image enantiomers alike, with
      no chirality gate and no chemistry oracle. The dedup target is
      energetic equivalence, not point-group-restricted congruence.
3. :func:`generate` post-processes each unique
   :class:`~crystalatte.core.sym_ops.SymOpList` into a
   :class:`~crystalatte.core.multimer.Multimer`.
"""

import math
from itertools import combinations, product
from collections import defaultdict

import numpy as np
import qcelemental as qcel
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm
from numpy.typing import NDArray

from ..core.crystal import Crystal
from ..core.multimer import Monomer, Multimer
from ..core.sym_ops import SymOp, SymOpList


# Module-level bindings set by `_bind_globals` (in main process) and by the
# worker initializer (`_init_bucket_worker`, in ProcessPoolExecutor children).
# Per-reference lists indexed by reference index (one entry per independent monomer).
_crystal: Crystal | None = None
_monomers: list[Monomer] = []
_Z_by_ref: list[NDArray] = []
_self_iso_by_ref: list[list[NDArray]] = [[np.eye(3)]]
_W_lin_by_ref: list[list[NDArray]] = [[np.eye(3, dtype=np.int64)]]
_check_congruence_by_ref: list[bool] = [False]
_A: NDArray | None = None      # fractional -> Cartesian matrix (cart = _A @ frac)
_A_inv: NDArray | None = None  # Cartesian -> fractional matrix


def _bind_globals(crystal: Crystal, monomers: list[Monomer]) -> None:
    """Bind the deduper's per-run inputs at module scope.

    Sets the module globals for per-reference state: per-reference Z arrays
    (atomic numbers), self-isometries, site-symmetry dressings, and the
    congruence gates. Also sets the lattice vectors ``_A`` / ``_A_inv`` for
    fractional<->Cartesian conversion.

    :param crystal: The crystal supplying the lattice and Cartesian conversion
    :param monomers: The reference monomers (one per crystal asymmetric unit)
    :returns: ``None``
    """
    global _crystal, _monomers, _Z_by_ref, _self_iso_by_ref
    global _W_lin_by_ref, _check_congruence_by_ref, _A, _A_inv
    _crystal = crystal
    _monomers = list(monomers)
    _A = np.asarray(crystal.lattice_vectors, dtype=float)
    _A_inv = np.linalg.inv(_A)

    _Z_by_ref, _self_iso_by_ref, _W_lin_by_ref, _check_congruence_by_ref = [], [], [], []
    for mon in _monomers:
        _Z_by_ref.append(
            np.array([qcel.periodictable.to_Z(s) for s in mon.symbols], dtype=np.int64)
        )
        S = mon.self_isometries
        _self_iso_by_ref.append(S)
        site_ops = crystal.space_group.get_site_symmetry(mon.centroid_frac)
        W = []
        for op in site_ops:
            r = op.rot.astype(np.int64)
            if not any(np.array_equal(r, w) for w in W):
                W.append(r)
        if not W:
            W = [np.eye(3, dtype=np.int64)]
        _W_lin_by_ref.append(W)
        _check_congruence_by_ref.append(len(S) > len(W))


def _init_bucket_worker(crystal: Crystal, monomers: list[Monomer]) -> None:
    """Process-pool initializer: bind the deduper globals in a worker process.

    :param crystal: The crystal passed to :func:`_bind_globals`
    :param monomers: The reference monomers passed to :func:`_bind_globals`
    :returns: ``None``
    """
    _bind_globals(crystal, monomers)


def _cart_frames(g_N: SymOpList) -> tuple[NDArray, NDArray]:
    """Return the Cartesian rigid placement of each monomer copy in the cluster.

    A crystal sym op ``x' = rot @ x + tr/12`` (fractional) becomes, in
    Cartesian, a linear part ``Rc = A @ rot @ A^-1`` (orthogonal, since crystal
    sym ops are isometries) and a translation ``tc = A @ (tr/12)``, with ``A``
    the fractional->Cartesian matrix.

    The result is not cached: this is called only by :func:`_relative_transforms`,
    which caches the derived ``(L, v)``, so a per-object cache here would never
    be re-read.

    :param g_N: The cluster whose monomer placements are requested
    :returns: A tuple ``(Rc, tc)`` of shapes ``(n, 3, 3)`` and ``(n, 3)``; copy
        ``k``'s atoms are ``Rc[k] @ X + tc[k]`` for reference Cartesian
        coordinates ``X``
    """
    rot = g_N.rot_cache.astype(float)          # (n, 3, 3)
    tr = g_N.tr_cache.astype(float) / 12.0      # (n, 3)
    Rc = _A @ rot @ _A_inv                       # (n, 3, 3)
    tc = (_A @ tr.T).T                           # (n, 3)
    return Rc, tc


def _relative_transforms(g_N: SymOpList) -> tuple[NDArray, NDArray]:
    """Return the pairwise relative Cartesian transforms of the cluster.

    ``L[a, b] = Rc_a^-1 Rc_b`` and ``v[a, b] = Rc_a^-1 (tc_b - tc_a)`` are
    invariant under any global isometry applied to the whole cluster (it cancels
    in ``T_a^-1 T_b``), including reflection — which is exactly why matching them
    merges identical clusters, lattice translates, and enantiomers without ever
    choosing the global map explicitly. Cached on the
    :class:`~crystalatte.core.sym_ops.SymOpList` since :func:`_same_multimer`
    re-reads them across many comparisons.

    :param g_N: The cluster whose relative transforms are requested
    :returns: A tuple ``(L, v)`` of shapes ``(n, n, 3, 3)`` and ``(n, n, 3)``
    """
    cached = getattr(g_N, "_rel_cache", None)
    if cached is not None:
        return cached
    Rc, tc = _cart_frames(g_N)
    Rc_inv = np.linalg.inv(Rc)                        # (n, 3, 3)
    L = np.einsum("aij,bjk->abik", Rc_inv, Rc)        # (n, n, 3, 3)
    dt = tc[None, :, :] - tc[:, None, :]              # (n, n, 3): tc_b - tc_a
    v = np.einsum("aij,abj->abi", Rc_inv, dt)         # (n, n, 3)
    g_N._rel_cache = (L, v)
    return L, v


def _int_inv(M: NDArray) -> NDArray:
    """Return the exact integer inverse of a unimodular integer matrix.

    Lattice-basis rotations are integer with ``det = ±1`` but are not
    orthogonal, so the inverse is recovered via the adjugate (``round`` of the
    float inverse), never the transpose.

    :param M: A ``(3, 3)`` integer matrix with ``det = ±1``
    :returns: The exact ``(3, 3)`` ``int64`` inverse
    """
    inv = np.rint(np.linalg.inv(M)).astype(np.int64)
    assert np.array_equal(inv @ M.astype(np.int64), np.eye(3, dtype=np.int64)), (
        "_int_inv called on a non-unimodular matrix (exact integer inverse does not exist)"
    )
    return inv


def _relative_transforms_int(g_N: SymOpList) -> tuple[NDArray, NDArray]:
    """Return the pairwise relative transforms of the cluster in the integer
    lattice basis: ``L[a,b] = rot_a^-1 rot_b`` (integer) and
    ``v[a,b] = rot_a^-1 (tr_b - tr_a)`` (integer, in twelfths).

    Exact at all distances (no Cartesian conversion, no roundoff). Cached on the
    :class:`~crystalatte.core.sym_ops.SymOpList` since the matcher re-reads them
    across many comparisons.

    :param g_N: The cluster whose integer relative transforms are requested
    :returns: A tuple ``(L, v)`` of shapes ``(n, n, 3, 3)`` and ``(n, n, 3)``
    """
    cached = getattr(g_N, "_rel_int_cache", None)
    if cached is not None:
        return cached
    rot = g_N.rot_cache.astype(np.int64)          # (n, 3, 3)
    tr = g_N.tr_cache.astype(np.int64)            # (n, 3), twelfths
    rot_inv = np.empty_like(rot)
    for i in range(len(rot)):
        rot_inv[i] = _int_inv(rot[i])
    L = np.einsum("aij,bjk->abik", rot_inv, rot)           # (n, n, 3, 3)
    dt = tr[None, :, :] - tr[:, None, :]                   # (n, n, 3): tr_b - tr_a
    v = np.einsum("aij,abj->abi", rot_inv, dt)            # (n, n, 3)
    g_N._rel_int_cache = (L, v)
    return L, v


def _same_multimer_int(g_N: SymOpList, h_N: SymOpList, W: list[NDArray] | None = None) -> bool:
    """Return True iff ``g_N`` and ``h_N`` are the same multimer under a
    crystallographic isometry, working entirely in the exact integer lattice
    basis and modding out the site-symmetry dressing ``W``.

    This is the exact-integer form of the anchored relative-transform search
    (see ``docs/dedup_correctness.tex``, Thm 2). It subsumes the old exact
    matrix branch and is complete for every same-orbit (multiplicity-required)
    merge, including centrosymmetric site-inversion cases. Non-crystallographic
    congruences (dressing outside ``_W_lin``) are left to the float fallback in
    :func:`_is_bijection`.

    :param g_N: The first cluster
    :param h_N: The second cluster
    :returns: True iff the clusters are the same multimer via a crystallographic
        isometry
    """
    n = len(g_N)
    if n != len(h_N):
        return False

    LG, vG = _relative_transforms_int(g_N)
    LH, vH = _relative_transforms_int(h_N)
    S = _W_lin_by_ref[0] if W is None else W

    def in_S(M: NDArray) -> bool:
        return any(np.array_equal(M, s) for s in S)

    def verify(pi: list[int], s: list[NDArray]) -> bool:
        for a in range(n):
            sa_inv = _int_inv(s[a])
            for b in range(n):
                if not np.array_equal(vG[a, b], sa_inv @ vH[pi[a], pi[b]]):
                    return False
                if not np.array_equal(
                    LG[a, b], sa_inv @ LH[pi[a], pi[b]] @ s[b]
                ):
                    return False
        return True

    for h0 in range(n):
        for s0 in S:
            pi = [-1] * n
            s: list[NDArray] = [None] * n
            used = [False] * n
            pi[0], s[0], used[h0] = h0, s0, True

            def assign(b: int) -> bool:
                if b == n:
                    return verify(pi, s)
                target = s0 @ vG[0, b]  # want vH[h0, pi(b)] == target
                for h in range(n):
                    if used[h] or not np.array_equal(vH[h0, h], target):
                        continue
                    sb = _int_inv(LH[h0, h]) @ s0 @ LG[0, b]
                    if not in_S(sb):
                        continue
                    pi[b], s[b], used[h] = h, sb, True
                    if assign(b + 1):
                        return True
                    pi[b], s[b], used[h] = -1, None, False
                return False

            if assign(1):
                return True
    return False


def _base(g_N: SymOpList) -> int:
    """Return the (single) base reference index of a mono-reference cluster.

    :param g_N: The cluster whose base reference index is requested
    :returns: The base reference index (``g_N.ref_indices[0]``)
    """
    return int(g_N.ref_indices[0])


def _cluster_atoms_cart(g_N: SymOpList) -> NDArray:
    """Reconstruct the cluster's Cartesian atom positions from the module-level
    ``_monomers`` / ``_crystal`` bindings, using each member's base reference.

    The result is not cached: this is called only by
    :func:`_distance_fingerprint`, which caches the derived fingerprint, so a
    per-object cache here would never be re-read.

    :param g_N: The cluster whose atom positions are requested
    :returns: An ``(n_atoms, 3)`` array of Cartesian coordinates
    """
    ref = g_N.ref_indices
    parts = [op.apply(_monomers[ref[k]].frac_coords) for k, op in enumerate(g_N)]
    return _crystal.to_cartesian(np.vstack(parts))


def _distance_fingerprint(g_N: SymOpList) -> tuple[NDArray, NDArray]:
    """Return the cluster's cached Z-tagged sorted pairwise atom-distance
    fingerprint: interatomic distances sorted lexicographically by
    ``(Z_i * Z_j, distance)``.

    Any isometry — rotation, translation, or reflection — preserves every
    interatomic distance, and the monomer's internal symmetry only permutes
    atoms within a monomer, so two clusters that are the same multimer have
    identical fingerprints. It is therefore a strong necessary condition, and
    unlike the relative-transform invariants it distinguishes pure-translation
    copies whose separation vectors lie in different point-group orbits (the
    benzene case). Cached per candidate, so the dedup pays ``O(#candidates)``
    fingerprint builds rather than ``O(#comparisons)``.

    :param g_N: The cluster to fingerprint
    :returns: A tuple ``(zz, distances)`` of equal-length 1-D arrays, sorted
        jointly by ``(zz, distance)``
    """
    cached = getattr(g_N, "_dist_fp_cache", None)
    if cached is not None:
        return cached
    atoms = _cluster_atoms_cart(g_N)
    ref = g_N.ref_indices
    Z = np.concatenate([_Z_by_ref[ref[k]] for k in range(len(g_N))])
    i_idx, j_idx = np.triu_indices(len(atoms), k=1)
    dists = np.linalg.norm(atoms[i_idx] - atoms[j_idx], axis=1)
    zz = Z[i_idx] * Z[j_idx]
    order = np.lexsort((dists, zz))
    fp = (zz[order], dists[order])
    g_N._dist_fp_cache = fp
    return fp


def _fingerprints_match(g_N: SymOpList, h_N: SymOpList, atol: float = 1e-6) -> bool:
    """Return True iff the two clusters' Z-tagged distance fingerprints coincide.

    This is the cheap necessary gate run before the exact and anchored branches.
    The tight ``atol`` is safe because equivalent clusters share
    exact-arithmetic distances; homometric false positives (rare) are caught
    downstream by :func:`_same_multimer`.

    :param g_N: The first cluster
    :param h_N: The second cluster
    :param atol: Absolute tolerance in Angstroms for the distance comparison
    :returns: True iff both fingerprints match
    """
    if len(g_N) != len(h_N):
        return False
    zz_g, d_g = _distance_fingerprint(g_N)
    zz_h, d_h = _distance_fingerprint(h_N)
    if zz_g.shape != zz_h.shape or not np.array_equal(zz_g, zz_h):
        return False
    return np.allclose(d_g, d_h, atol=atol, rtol=0.0)


def _same_multimer(g_N: SymOpList, h_N: SymOpList, atol: float = 1e-3) -> bool:
    """Return True iff ``g_N`` and ``h_N`` are the same multimer up to a global
    isometry (rotation, translation, or reflection).

    The matching condition (derived by cancelling the global map ``M`` in
    ``T_a^-1 T_b``) is: there is a monomer permutation ``pi`` and a per-monomer
    internal-symmetry choice ``s_a`` in
    :attr:`~crystalatte.core.multimer.Monomer.self_isometries` such that, for
    every ordered pair ``(a, b)``,
    ``L^G[a,b] = s_a^-1 @ L^H[pi a, pi b] @ s_b`` and
    ``v^G[a,b] = s_a^-1 @ v^H[pi a, pi b]``.

    Solved by anchoring node 0: enumerate its image ``h0 = pi(0)`` and dressing
    ``s0`` in ``S`` (only ``n * |S|`` choices). For each, ``pi`` is forced by
    matching the relative position vectors ``v^G[0,b] == s0^-1 v^H[h0, pi b]``,
    then each ``s_b = (L^H[h0, pi b])^-1 s0 L^G[0,b]`` is solved and checked for
    membership in ``S``, and finally all pairs are verified. This is
    ``O(|S| * n^3)`` instead of the ``O(n! * |S|^n)`` brute force, while
    exploring the identical condition (so completeness is preserved).

    Callers (:func:`_is_bijection`) gate this behind the cheap
    :func:`_fingerprints_match` pre-filter, so it only ever runs on clusters
    that already share a distance fingerprint (true duplicates plus rare
    homometric collisions).

    :param g_N: The first cluster
    :param h_N: The second cluster
    :param atol: Absolute tolerance in Angstroms. It can be tight when the
        relevant self-isometries are exact (identity / crystallographic), and
        must be looser when the merge relies on an approximate molecular
        operation recovered from distorted CIF coordinates.
    :returns: True iff the two clusters are the same multimer
    """
    n = len(g_N)
    if n != len(h_N):
        return False

    LG, vG = _relative_transforms(g_N)
    LH, vH = _relative_transforms(h_N)
    S = _self_iso_by_ref[_base(g_N)]

    def in_S(M: NDArray) -> bool:
        return any(np.allclose(M, s, atol=atol) for s in S)

    def verify(pi: list[int], s: list[NDArray]) -> bool:
        for a in range(n):
            sa_inv = s[a].T  # orthogonal -> inverse is transpose
            for b in range(n):
                if not np.allclose(vG[a, b], sa_inv @ vH[pi[a], pi[b]], atol=atol):
                    return False
                if not np.allclose(
                    LG[a, b], sa_inv @ LH[pi[a], pi[b]] @ s[b], atol=atol
                ):
                    return False
        return True

    for h0 in range(n):
        for s0 in S:
            pi = [-1] * n
            s = [None] * n
            used = [False] * n
            pi[0], s[0], used[h0] = h0, s0, True

            def assign(b: int) -> bool:
                if b == n:
                    return verify(pi, s)
                target = s0 @ vG[0, b]  # want vH[h0, pi(b)] == target
                for h in range(n):
                    if used[h] or not np.allclose(vH[h0, h], target, atol=atol):
                        continue
                    sb = LH[h0, h].T @ s0 @ LG[0, b]
                    if not in_S(sb):
                        continue
                    pi[b], s[b], used[h] = h, sb, True
                    if assign(b + 1):
                        return True
                    pi[b], s[b], used[h] = -1, None, False
                return False

            if assign(1):
                return True
    return False


def _same_multimer_geom(g_N: SymOpList, h_N: SymOpList, atol: float = 1e-3) -> bool:
    """Return True iff the two clusters' reconstructed, Z-labeled atom clouds are
    congruent under some isometry (rotation, translation, or reflection).

    Anchor-based: center both clouds, pick a non-collinear anchor triple in G,
    match it to every Z-compatible, distance-compatible triple in H, solve the
    (possibly improper) rigid map by Kabsch, and accept iff it carries every
    atom of G onto a same-Z atom of H within ``atol``. Reflection is allowed
    (enantiomers are energetically equivalent).

    :param g_N: The first cluster (may span multiple base references)
    :param h_N: The second cluster (may span multiple base references)
    :param atol: Absolute tolerance in Angstroms
    :returns: True iff the two clusters' atom clouds are congruent
    """
    A = _cluster_atoms_cart(g_N)
    B = _cluster_atoms_cart(h_N)
    if A.shape != B.shape:
        return False
    ZA = np.concatenate([_Z_by_ref[g_N.ref_indices[k]] for k in range(len(g_N))])
    ZB = np.concatenate([_Z_by_ref[h_N.ref_indices[k]] for k in range(len(h_N))])
    if sorted(ZA) != sorted(ZB):
        return False
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)

    n = len(A)
    # anchor triple in A: first three mutually non-collinear atoms
    def _triple(P):
        i0 = 0
        for i1 in range(1, n):
            for i2 in range(i1 + 1, n):
                v1, v2 = P[i1] - P[i0], P[i2] - P[i0]
                if np.linalg.norm(np.cross(v1, v2)) > 1e-3:
                    return (i0, i1, i2)
        return None
    tri = _triple(A)
    if tri is None:          # (near-)linear cluster: fall back to full-pair distance check
        from scipy.spatial.distance import cdist
        DA = np.sort(cdist(A, A).ravel())
        DB = np.sort(cdist(B, B).ravel())
        return bool(np.allclose(DA, DB, atol=atol))
    ia, ib, ic = tri
    dref = np.array([
        np.linalg.norm(A[ia] - A[ib]),
        np.linalg.norm(A[ia] - A[ic]),
        np.linalg.norm(A[ib] - A[ic]),
    ])
    from scipy.spatial.distance import cdist
    DB_full = cdist(B, B)
    same = {z: np.where(ZB == z)[0] for z in np.unique(ZB)}
    for jb in same[ZA[ia]]:
        for jc in same[ZA[ic]]:
            for jd in same[ZA[ib]]:
                if len({jb, jc, jd}) != 3:
                    continue
                dcand = np.array([DB_full[jb, jd], DB_full[jb, jc], DB_full[jd, jc]])
                if not np.allclose(dref, dcand, atol=atol):
                    continue
                P = A[[ia, ib, ic]]
                Q = B[[jb, jd, jc]]
                R = _kabsch(P, Q)         # allows reflection
                if R is None:
                    continue
                mapped = A @ R.T
                # every mapped atom must coincide with a same-Z atom of B
                ok = True
                used = np.zeros(n, dtype=bool)
                for k in range(n):
                    cand = same[ZA[k]]
                    dists = np.linalg.norm(B[cand] - mapped[k], axis=1)
                    hit = cand[dists < atol]
                    hit = [h for h in hit if not used[h]]
                    if not hit:
                        ok = False
                        break
                    used[hit[0]] = True
                if ok:
                    return True
    return False


def _kabsch(P: NDArray, Q: NDArray) -> NDArray | None:
    """Return the orthogonal R (det +/-1) minimizing ||P R^T - Q|| for centered
    corresponding point sets, or None if degenerate.

    :param P: An ``(k, 3)`` array of centered source points
    :param Q: An ``(k, 3)`` array of centered target points
    :returns: The orthogonal map ``R``, or None on an SVD failure
    """
    H = P.T @ Q
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    R = (Vt.T @ U.T)
    return R


def _mono(g_N: SymOpList) -> bool:
    """Return True iff every member of ``g_N`` shares the same base reference.

    :param g_N: The cluster to test
    :returns: True iff ``g_N`` is mono-reference
    """
    return len(set(g_N.ref_indices.tolist())) == 1


def _is_bijection(g_N: SymOpList, h_N: SymOpList) -> bool:
    """Return True iff ``g_N`` and ``h_N`` represent the same multimer.

    Stages, cheapest first:

    0. Distance-fingerprint gate (:func:`_fingerprints_match`) — a cached,
       vectorized necessary condition rejecting most within-bucket pairs.
    1. Mono-reference (both clusters drawn from the same single base
       reference): exact-integer matcher (:func:`_same_multimer_int`) — the
       complete, roundoff-free decision for every crystallographic
       (same-orbit) merge, modding out that reference's site symmetry
       (``_W_lin_by_ref[i]``). Subsumes the former exact matrix branch.
       Falls back to the float congruence matcher (:func:`_same_multimer`)
       only when the integer matcher fails and that reference's molecular
       symmetry is richer than its site symmetry
       (``_check_congruence_by_ref[i]``). It uses the FULL molecular
       self-isometry group, so it merges every energetically-equivalent
       congruence: proper molecular-symmetry congruences AND mirror-image
       enantiomers. This is intentional and goes beyond an enantiomer-only
       check; the dedup target is energetic equivalence.
    2. Cross-reference (members drawn from different base references, or the
       two clusters' base references differ): labeled point-cloud congruence
       (:func:`_same_multimer_geom`) — reflection-allowed Kabsch matching on
       the reconstructed, Z-labeled Cartesian atom clouds.

    :param g_N: The first cluster
    :param h_N: The second cluster
    :returns: True iff the clusters represent the same multimer
    """
    if not _fingerprints_match(g_N, h_N):
        return False
    if _mono(g_N) and _mono(h_N) and _base(g_N) == _base(h_N):
        i = _base(g_N)
        if _same_multimer_int(g_N, h_N, _W_lin_by_ref[i]):
            return True
        if _check_congruence_by_ref[i]:
            return _same_multimer(g_N, h_N)
        return False
    return _same_multimer_geom(g_N, h_N)


def _process_bucket(
    bucket: list[SymOpList],
) -> tuple[list[SymOpList], list[SymOpList]]:
    """Deduplicate one bucket of clusters that share the same ``translation_fp``.

    For each new candidate, compare it to every existing unique representative
    via :func:`_is_bijection`. On a match, the candidate with the smaller summed
    translation norm is kept as the representative (and the other is reclassified
    as a duplicate). This keeps the representative closer to the reference, which
    matters when a downstream radius filter is applied per multimer.

    :param bucket: The clusters sharing a single ``translation_fp``
    :returns: A tuple ``(unique, duplicate)`` of cluster lists
    """
    local_unique: list[SymOpList] = []
    local_dup: list[SymOpList] = []
    for g_t in bucket:
        for u in local_unique:
            if not _is_bijection(g_t, u):
                continue
            if (
                np.linalg.norm(g_t.translations, axis=1).sum()
                < np.linalg.norm(u.translations, axis=1).sum()
            ):
                local_dup.append(u)
                local_unique.remove(u)
                local_unique.append(g_t)
                g_t.multiplicity = u.multiplicity + 1
            else:
                local_dup.append(g_t)
                u.multiplicity += 1
            break
        else:
            local_unique.append(g_t)
    return local_unique, local_dup


def _neighbor_distance(cart_mon: NDArray, monomer: Monomer, kwargs: dict) -> float:
    """Return the cutoff distance for a partner monomer.

    The convention is selected by ``kwargs``:

    - ``use_origin=True``: closest atom of the partner to the anchor monomer's
      centroid.
    - ``use_com=True``: distance between the partner's center of mass and the
      reference monomer's centroid.
    - default: closest atom-pair contact between the reference and partner
      monomers (``cdist`` minimum).

    :param cart_mon: The partner monomer's Cartesian coordinates
    :param monomer: The reference monomer
    :param kwargs: Keyword arguments selecting the distance convention
    :returns: The partner-to-reference distance in Angstroms
    """
    if kwargs.get("use_origin"):
        return float(np.linalg.norm(cart_mon - monomer.centroid_cart, axis=1).min())
    if kwargs.get("use_com", False):
        masses = np.array([qcel.periodictable.to_mass(s) for s in monomer.symbols])
        com_mon = np.average(cart_mon, axis=0, weights=masses)
        return float(np.linalg.norm(com_mon - monomer.centroid_cart))
    return float(cdist(monomer.cart_coords, cart_mon).min())


def _generate_neighbors(
    crystal: Crystal,
    monomers: list[Monomer],
    cutoff: NDArray,
    R: float,
    N: int,
    **kwargs,
) -> tuple[list[SymOpList], list[SymOpList]]:
    """Generate all unique N-mers in the supercell defined by ``cutoff`` by
    anchoring each reference monomer in turn, then deduplicate via a
    centroid-distance bucketing and :func:`_is_bijection`.

    :param crystal: The crystal containing the monomers
    :param monomers: The reference monomers (one per crystal asymmetric unit);
        each is anchored in turn as the multimer's first member
    :param cutoff: Per-axis ``(low, high)`` integer lattice translation bounds
    :param R: Distance cutoff in Angstroms. The distance convention is selected
        by ``kwargs`` (see :func:`_neighbor_distance`).
    :param N: N-mer order (``N=2`` for dimers, ``N=3`` for trimers, ...)
    :param kwargs: Additional options (``debug``, ``n_jobs``, and the distance
        convention flags consumed by :func:`_neighbor_distance`)
    :returns: A tuple ``(unique, duplicate)`` of cluster lists
    """
    if kwargs.get("debug", False):
        translations = list(product([-1, 0, 1], repeat=3))
    else:
        translations = list(product(*[range(cm, cp + 1) for cm, cp in cutoff]))

    # Bind per-reference state (workers below get the same via _init_bucket_worker).
    _bind_globals(crystal, monomers)

    # Full partner pool across ALL references: (base_index j, SymOp placing monomers[j]).
    pool = []
    for j, mon in enumerate(monomers):
        sym_ops = crystal.space_group.get_unique_sym_ops(mon.centroid_frac)
        for op, tr in product(sym_ops, translations):
            pool.append((j, op.translate(np.array(tr, dtype=np.int16) * 12)))

    # Anchor each reference i; partners = pool members within R of anchor i
    # (excluding anchor i's own zero-translation identity).
    candidates: list[SymOpList] = []
    for i, anchor in enumerate(monomers):
        near = []
        for (j, op) in pool:
            if j == i and op == SymOp.identity():
                continue
            partner_cart = crystal.to_cartesian(op.apply(monomers[j].frac_coords))
            if _neighbor_distance(partner_cart, anchor, kwargs) <= R:
                near.append((j, op))
        print(f"Reference {i}: {len(near)} monomers within {R} Angstroms.")
        for partner_combo in combinations(near, N - 1):
            members = [(i, SymOp.identity()), *partner_combo]
            ops = [op for (_, op) in members]
            refs = np.array([b for (b, _) in members], dtype=np.int64)
            member_cens = np.array(
                [crystal.to_cartesian(op.apply(monomers[b].centroid_frac))
                 for (b, op) in members]
            )
            centroid_frac = np.mean(
                [op.apply(monomers[b].centroid_frac) for (b, op) in members], axis=0
            )
            sop = SymOpList(ops, 1)
            sop.ref_indices = refs
            sop.centroid_frac = centroid_frac
            sop.centroid_cart = crystal.to_cartesian(centroid_frac)
            # bucket key = sorted rounded pairwise member-centroid distances
            # (an isometry invariant of the cluster; valid across references,
            # unlike translation_fp which only reflects SymOp translations).
            # Deliberately coarse (Euclidean Angstroms, 2 dp) and over-inclusive:
            # equivalent clusters' centroid distances agree to ~1e-9, so a 0.01 A
            # bin never straddles a rounding boundary in practice, while any
            # extra within-bucket collisions are harmless — correctness is
            # decided by _is_bijection, not by the bucket key itself.
            if len(member_cens) > 1:
                dfp = np.sort(np.round(pdist(member_cens), 2))
            else:
                dfp = np.array([0.0])
            sop._bucket_key = hash(dfp.tobytes())
            candidates.append(sop)

    buckets: dict[int, list[SymOpList]] = defaultdict(list)
    for g_t in candidates:
        buckets[g_t._bucket_key].append(g_t)

    unique: list[SymOpList] = []
    duplicate: list[SymOpList] = []

    if "n_jobs" in kwargs and kwargs["n_jobs"] > 1:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import cpu_count

        n_jobs = min(kwargs["n_jobs"], len(buckets), cpu_count() - 1)
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_bucket_worker,
            initargs=(crystal, monomers),
        ) as ex:
            results = list(
                tqdm(
                    ex.map(_process_bucket, buckets.values()),
                    total=len(buckets),
                    desc="Processing buckets",
                    leave=False,
                )
            )

        for local_unique, local_dup in results:
            unique.extend(local_unique)
            duplicate.extend(local_dup)

    else:
        for bucket in tqdm(
            buckets.values(),
            total=len(buckets),
            desc="Processing buckets",
            leave=False,
        ):
            local_unique, local_dup = _process_bucket(bucket)
            unique.extend(local_unique)
            duplicate.extend(local_dup)

    print(
        f"Found {len(duplicate)} equivalent multimers from neighboring unit cell search."
    )

    return unique, duplicate


def _init_multimers(crystal_: Crystal, monomers_: list[Monomer]) -> None:
    """Bind globals used by :func:`_process_multimer` so it can run from a
    :class:`~concurrent.futures.ProcessPoolExecutor` initializer.

    :func:`_bind_globals` is not reused because :func:`_process_multimer` also
    needs the per-reference atom mass arrays (for center-of-mass computation),
    which the bucket-dedup path does not.

    :param crystal_: The crystal bound to the module global ``crystal``
    :param monomers_: The reference monomers bound to the module global ``monomers``
    :returns: ``None``
    """
    global crystal, monomers, masses_by_ref
    crystal = crystal_
    monomers = list(monomers_)
    masses_by_ref = [
        np.array([qcel.periodictable.to_mass(s) for s in m.symbols]) for m in monomers
    ]


def _process_multimer(g_N: SymOpList) -> Multimer | None:
    """Construct a :class:`~crystalatte.core.multimer.Multimer` from a deduped
    candidate cluster.

    :param g_N: The deduplicated cluster to materialize
    :returns: The constructed :class:`~crystalatte.core.multimer.Multimer`, or
        ``None`` if ``g_N`` contains two members that are the same
        ``(base reference, SymOp)`` pair (which can happen when multiple
        translations land on a degenerate Wyckoff orbit and would produce
        overlapping monomers)
    """
    refs = g_N.ref_indices
    ops = list(g_N)
    # degeneracy: reject only if two members are the SAME (ref, op) pair
    seen = set(zip(refs.tolist(), ops))
    if len(seen) != len(ops):
        return None
    mons: list[Monomer] = []
    for k, g in enumerate(ops):
        base = monomers[refs[k]]
        masses = masses_by_ref[refs[k]]
        mon_frac = g.apply(base.frac_coords)
        mon_cart = crystal.to_cartesian(mon_frac)
        com_frac = np.average(mon_frac, axis=0, weights=masses)
        com_cart = crystal.to_cartesian(com_frac)
        mons.append(Monomer(base.symbols, mon_frac, mon_cart, com_frac, com_cart, g))
    return Multimer(mons, g_N, g_N.multiplicity)


def generate(crystal: Crystal, N: int, R: float, **kwargs) -> list[Multimer]:
    """Generate unique multimers of every reference molecule in the crystal.

    :param crystal: The crystal structure to generate multimers from; all
        symmetry-independent reference molecules (:meth:`Crystal.get_references`)
        are enumerated
    :param N: The number of monomers in the multimer (e.g. ``N=2`` for dimers)
    :param R: The maximum center-to-center distance in Angstroms for monomers to
        be considered part of the same multimer
    :param kwargs: Additional options forwarded to :func:`_generate_neighbors`
    :returns: A list of unique :class:`~crystalatte.core.multimer.Multimer`
        objects representing the multimers found in the crystal
    """
    monomers = crystal.get_references()
    d = np.linalg.norm(crystal.lattice_vectors, axis=0)
    R_eff = R + max(pdist(m.cart_coords).max() for m in monomers)

    lows, highs = [], []
    for m in monomers:
        t = m.centroid_frac
        lows.append(-np.ceil((R_eff - t * d) / d))
        highs.append(np.ceil((R_eff - (1 - t) * d) / d))
    bounds = np.vstack((np.min(lows, axis=0), np.max(highs, axis=0))).T.astype(int)

    unique_neighbors, _ = _generate_neighbors(crystal, monomers, bounds, R, N, **kwargs)
    _init_multimers(crystal, monomers)

    multimers: list[Multimer] = []
    for g_N in tqdm(
        unique_neighbors,
        total=len(unique_neighbors),
        desc="Processing unique multimers",
        leave=False,
    ):
        m = _process_multimer(g_N)
        if m is not None:
            multimers.append(m)
    return multimers


if __name__ == "__main__":
    pass
