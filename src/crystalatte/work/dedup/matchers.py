import numpy as np
from numpy.typing import NDArray

from ...core.sym_ops import SymOpList
from . import context
from .invariants import int_inv, relative_transforms

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
