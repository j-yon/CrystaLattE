"""Equivalence tests: does a pair of clusters represent the same multimer?

Two independent strategies live here:

- the anchored relative-transform search (:func:`anchored_match`), used in an
  exact integer basis and a tolerant Cartesian one, which reasons about the
  cluster's symmetry operations;
- labeled point-cloud congruence (:func:`same_multimer_geom`), which ignores
  the operations entirely and superimposes the reconstructed atoms.

Routing between them is :mod:`.equivalence`'s job, not this module's.
"""
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from ...core.sym_ops import SymOpList
from . import context
from .invariants import (
    base_reference,
    cluster_atoms_cart,
    int_inv,
    relative_transforms,
)

def anchored_match(LG, vG, LH, vH, S, D, eq, inv) -> bool:
    """Anchored relative-transform search: the shared core of both matchers.

    Decides whether two clusters satisfy the matching condition of
    ``docs/dedup_correctness.tex`` Thm 1, i.e. whether there is a permutation
    ``pi`` and per-monomer dressings ``sigma_a = (s_a, d_a)`` drawn from
    ``(S, D)`` with ``Delta^G_ab = sigma_a^-1 Delta^H_(pi a)(pi b) sigma_b``.
    Splitting into linear and translation parts, that is

        L^G[a,b] = s_a^-1 @ L^H[pa,pb] @ s_b
        v^G[a,b] = s_a^-1 @ (L^H[pa,pb] @ d_b + v^H[pa,pb] - d_a)

    The dressing is AFFINE because a self-isometry fixes the monomer's CENTROID,
    not the coordinate origin: ``d = c - s @ c``. Dropping ``d`` is valid only
    when every ``s`` fixes ``c``; otherwise it compares frame-origin offsets
    that are not the same geometric quantity, and since every ``s`` is
    orthogonal (norm-preserving) the guard becomes unsatisfiable at ANY
    tolerance whenever the two ``v`` norms differ (the urea case).

    Solved by anchoring node 0: enumerate its image ``h0 = pi(0)`` and dressing
    index ``k0`` (only ``n * |S|`` choices). For each candidate image of ``b``
    the dressing-free linear equation ``s_b = (L^H[h0, pi b])^-1 s0 L^G[0,b]``
    is solved first — that fixes ``d_b``, which the translation guard then needs
    — and finally all pairs are verified. ``O(|S| * n^3)`` instead of the
    ``O(n! * |S|^n)`` brute force, exploring the identical condition, so
    completeness is preserved (Thm 2).

    Note ``(L^H[a,b])^-1`` is just ``L^H[b,a]``: since ``L[a,b] = R_a^-1 R_b``,
    ``L[b,a] = L[a,b]^-1`` exactly, so no inverse is computed in the inner loop.

    :param LG: Cluster G's relative linear parts, shape ``(n, n, 3, 3)``
    :param vG: Cluster G's relative translations, shape ``(n, n, 3)``
    :param LH: Cluster H's relative linear parts
    :param vH: Cluster H's relative translations
    :param S: The dressing group's linear parts
    :param D: The dressing translations, index-aligned with ``S``
    :param eq: Equality predicate (exact for the integer basis, tolerant for the
        Cartesian one)
    :param inv: Inverse of a dressing linear part (transpose for orthogonal
        Cartesian matrices, adjugate for integer lattice-basis ones)
    :returns: True iff the clusters match under some ``(pi, dressings)``
    """
    n = len(LG)
    inv_cache: dict[int, NDArray] = {}

    def s_inv(k: int) -> NDArray:
        cached = inv_cache.get(k)
        if cached is None:
            cached = inv_cache[k] = inv(S[k])
        return cached

    def index_in_S(M: NDArray) -> int | None:
        for k, s in enumerate(S):
            if eq(M, s):
                return k
        return None

    def verify(pi: list[int], ks: list[int]) -> bool:
        for a in range(n):
            sa_inv, da = s_inv(ks[a]), D[ks[a]]
            for b in range(n):
                L_h, v_h = LH[pi[a], pi[b]], vH[pi[a], pi[b]]
                if not eq(vG[a, b], sa_inv @ (L_h @ D[ks[b]] + v_h - da)):
                    return False
                if not eq(LG[a, b], sa_inv @ L_h @ S[ks[b]]):
                    return False
        return True

    for h0 in range(n):
        for k0 in range(len(S)):
            pi = [-1] * n
            ks = [-1] * n
            used = [False] * n
            pi[0], ks[0], used[h0] = h0, k0, True

            def assign(b: int) -> bool:
                if b == n:
                    return verify(pi, ks)
                for h in range(n):
                    if used[h]:
                        continue
                    # Solve the dressing-free linear equation first: it fixes
                    # s_b, hence d_b, which the translation guard then needs.
                    kb = index_in_S(LH[h, h0] @ S[k0] @ LG[0, b])
                    if kb is None:
                        continue
                    if not eq(
                        vG[0, b],
                        s_inv(k0) @ (LH[h0, h] @ D[kb] + vH[h0, h] - D[k0]),
                    ):
                        continue
                    pi[b], ks[b], used[h] = h, kb, True
                    if assign(b + 1):
                        return True
                    pi[b], ks[b], used[h] = -1, -1, False
                return False

            if assign(1):
                return True
    return False


def same_multimer_int(
    g_N: SymOpList,
    h_N: SymOpList,
    W: list[NDArray],
    W_tr: list[NDArray] | None = None,
) -> bool:
    """Return True iff ``g_N`` and ``h_N`` are the same multimer under a
    crystallographic isometry, working entirely in the exact integer lattice
    basis and modding out the site-symmetry dressing ``(W, W_tr)``.

    Exact-integer form of :func:`anchored_match` — roundoff-free and complete
    for every same-orbit (multiplicity-required) merge, including
    centrosymmetric site-inversion cases. Non-crystallographic congruences
    (dressing outside ``W``) are left to the float fallback in
    :func:`are_equivalent`.

    :param g_N: The first cluster
    :param h_N: The second cluster
    :param W: The site-symmetry linear parts of the clusters' shared base
        reference, i.e. ``context.W_lin_by_ref[base]``. Required: there is no safe
        default, since defaulting to reference 0 would silently mod out the
        wrong site symmetry for every other reference under ``Z' > 1``.
    :param W_tr: The matching dressing translations ``context.W_tr_by_ref[base]``, in
        twelfths, index-aligned with ``W``. Defaults to zeros, which is correct
        only when the centroid is a fixed point of every element of ``W``.
    :returns: True iff the clusters are the same multimer via a crystallographic
        isometry
    """
    if len(g_N) != len(h_N):
        return False
    LG, vG = relative_transforms(g_N, integer=True)
    LH, vH = relative_transforms(h_N, integer=True)
    D = [np.zeros(3, dtype=np.int64)] * len(W) if W_tr is None else W_tr
    return anchored_match(LG, vG, LH, vH, W, D, np.array_equal, int_inv)


def same_multimer(g_N: SymOpList, h_N: SymOpList, atol: float = 1e-3) -> bool:
    """Return True iff ``g_N`` and ``h_N`` are the same multimer up to a global
    isometry (rotation, translation, or reflection).

    Cartesian form of :func:`anchored_match`, dressed with the FULL molecular
    self-isometry group (:attr:`~crystalatte.core.multimer.Monomer.self_isometries`)
    rather than just the site symmetry, so it also merges non-crystallographic
    congruences and — being blind to ``det`` — mirror-image enantiomers.

    Callers (:func:`are_equivalent`) gate this behind the cheap
    :func:`fingerprints_match` pre-filter, so it only ever runs on clusters
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
    if len(g_N) != len(h_N):
        return False
    LG, vG = relative_transforms(g_N, integer=False)
    LH, vH = relative_transforms(h_N, integer=False)
    base = base_reference(g_N)
    return anchored_match(
        LG, vG, LH, vH,
        context.self_iso_by_ref[base],
        context.self_iso_tr_by_ref[base],
        lambda a, b: np.allclose(a, b, atol=atol),
        lambda s: s.T,  # orthogonal -> inverse is transpose
    )


def same_multimer_geom(g_N: SymOpList, h_N: SymOpList, atol: float = 1e-3) -> bool:
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
    A = cluster_atoms_cart(g_N)
    B = cluster_atoms_cart(h_N)
    if A.shape != B.shape:
        return False
    ZA = np.concatenate([context.Z_by_ref[g_N.ref_indices[k]] for k in range(len(g_N))])
    ZB = np.concatenate([context.Z_by_ref[h_N.ref_indices[k]] for k in range(len(h_N))])
    if sorted(ZA) != sorted(ZB):
        return False
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)

    n = len(A)

    def _triple(P):
        """Return the BEST-conditioned anchor triple, not merely the first
        acceptable one.

        A triple only has to clear ``|cross| > 1e-3`` to be usable, but a
        barely-acceptable one is a terrible basis to fit a rotation from: the
        rotation about its near-degenerate axis is poorly determined, and that
        error is amplified by the lever arm out to distant atoms. Measured, a
        triple at the acceptance threshold amplifies a 1e-6 A coordinate
        perturbation to ~4e-3 A at an atom 20 A away — larger than the ``atol``
        the fit is then judged against, so a correct correspondence can fail its
        own test. Maximising the triangle area costs the same O(n^2) scan.
        """
        i0, best, best_area = 0, None, 1e-3
        for i1 in range(1, n):
            v1 = P[i1] - P[i0]
            for i2 in range(i1 + 1, n):
                area = np.linalg.norm(np.cross(v1, P[i2] - P[i0]))
                if area > best_area:
                    best, best_area = (i0, i1, i2), area
        return best

    tri = _triple(A)
    if tri is None:          # (near-)linear cluster: fall back to full-pair distance check
        DA = np.sort(cdist(A, A).ravel())
        DB = np.sort(cdist(B, B).ravel())
        return bool(np.allclose(DA, DB, atol=atol))
    ia, ib, ic = tri
    dref = np.array([
        np.linalg.norm(A[ia] - A[ib]),
        np.linalg.norm(A[ia] - A[ic]),
        np.linalg.norm(A[ib] - A[ic]),
    ])
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
                R = kabsch(A[[ia, ib, ic]], B[[jb, jd, jc]])   # allows reflection
                if R is None:
                    continue

                # The triple only BOOTSTRAPS a correspondence; it is too small a
                # basis to judge the fit by. Assign each atom to its nearest
                # unused same-Z partner (most confident first, so a clear match
                # is never displaced by an ambiguous one), then re-fit the
                # rotation on the FULL correspondence. Error then averages down
                # over n atoms instead of being amplified out from three.
                mapped = A @ R.T
                pairing = np.full(n, -1, dtype=int)
                used = np.zeros(n, dtype=bool)
                order = []
                for k in range(n):
                    cand = same[ZA[k]]
                    d = np.linalg.norm(B[cand] - mapped[k], axis=1)
                    order.append((float(d.min()), k))
                for _, k in sorted(order):
                    cand = same[ZA[k]]
                    d = np.linalg.norm(B[cand] - mapped[k], axis=1)
                    free = [(d[t], c) for t, c in enumerate(cand) if not used[c]]
                    if not free:
                        break
                    _, pick = min(free)
                    pairing[k], used[pick] = pick, True
                if (pairing < 0).any():
                    continue

                R = kabsch(A, B[pairing])
                if R is None:
                    continue
                # Accept only on the refined map, at the tight tolerance.
                if np.abs(A @ R.T - B[pairing]).max() < atol:
                    return True
    return False


def kabsch(P: NDArray, Q: NDArray) -> NDArray | None:
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
