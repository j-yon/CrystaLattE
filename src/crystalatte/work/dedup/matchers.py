from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ...core.sym_ops import SymOpList
from .invariants import int_inv, relative_transforms


def as_per_reference(
    groups: list[NDArray] | list[list[NDArray]],
    n_refs: int,
) -> list[list[NDArray]]:
    """Normalize a dressing-group argument to one group per base reference.

    Accepts either a single group (a sequence of ``(3, 3)`` linear parts or
    ``(3,)`` translations) shared by every reference -- correct for a
    mono-reference cluster -- or a list of groups already indexed by reference.

    :param groups: A single dressing group, or one per reference
    :param n_refs: How many references the caller needs covered
    :returns: A list of ``n_refs`` groups
    """
    first = groups[0]
    # A lone group's first element is a bare array; a per-reference list's is
    # itself a group. Dimensionality of ``groups`` alone can't tell them apart:
    # a per-reference list of translation groups and a single group of linear
    # parts are both two levels deep.
    if isinstance(first, np.ndarray) and first.ndim <= 2:
        return [groups] * n_refs
    return groups


def anchored_match(
    LG: NDArray,
    vG: NDArray,
    LH: NDArray,
    vH: NDArray,
    S: list[list[NDArray]],
    D: list[list[NDArray]],
    refG: list[int],
    refH: list[int],
    eq: Callable[[NDArray, NDArray], bool],
    inv: Callable[[NDArray], NDArray],
    verify_all: bool = True,
) -> bool:
    """Anchored relative-transform search: the shared core of both matchers.

    Decides the matching condition of ``docs/dedup_correctness.tex``
    (``thm:match``): is there a reference-compatible permutation ``pi`` and
    per-monomer dressings ``sigma_a = (s_a, d_a)``, drawn from the group of
    member ``a``'s own base reference, such that for all ``a, b``

        L^G[a,b] = s_a^-1 @ L^H[pa,pb] @ s_b
        v^G[a,b] = s_a^-1 @ (L^H[pa,pb] @ d_b + v^H[pa,pb] - d_a)

    Runs in ``O(|S| * n^3)`` rather than the ``O(n! * |S|^n)`` brute force, and
    explores exactly the same condition, so it stays complete (``thm:search``).
    Under ``Z' = 1`` every member shares one group and this reduces to the
    single-group form.

    :param LG: Cluster G's relative linear parts, shape ``(n, n, 3, 3)``
    :param vG: Cluster G's relative translations, shape ``(n, n, 3)``
    :param LH: Cluster H's relative linear parts
    :param vH: Cluster H's relative translations
    :param S: The dressing groups' linear parts, indexed by base reference
        (:func:`as_per_reference` also accepts a single shared group)
    :param D: The dressing translations, index-aligned with ``S``
    :param refG: Cluster G's per-member base-reference indices
    :param refH: Cluster H's per-member base-reference indices
    :param eq: Equality predicate (exact for the integer basis, tolerant for the
        Cartesian one)
    :param inv: Inverse of a dressing linear part (transpose for orthogonal
        Cartesian matrices, adjugate for integer lattice-basis ones)
    :param verify_all: Re-check all ``n^2`` pairs before accepting. Redundant in
        exact arithmetic; kept as a backstop where ``eq`` is tolerant.
    :returns: True iff the clusters match under some ``(pi, dressings)``
    """
    n = len(LG)
    # Dressing inverses are reused across every anchor and branch, and the
    # integer-basis adjugate is not free, so compute each one once.
    inv_cache: dict[tuple[int, int], NDArray] = {}

    def s_inv(ref: int, k: int) -> NDArray:
        cached = inv_cache.get((ref, k))
        if cached is None:
            cached = inv_cache[(ref, k)] = inv(S[ref][k])
        return cached

    def index_in_S(ref: int, M: NDArray) -> int | None:
        for k, s in enumerate(S[ref]):
            if eq(M, s):
                return k
        return None

    # The descent only tests the anchor row (0, b). Relative transforms compose
    # (Delta_ab Delta_bc = Delta_ac), so every other pair follows from that row
    # (lem:row) and this sweep cannot reject in exact arithmetic (rem:verify;
    # 0 rejections in 3981 leaves over six crystals). Under a tolerant ``eq``
    # the composition holds only up to accumulated error, which is what this
    # re-check guards against.
    def verify(pi: list[int], ks: list[int]) -> bool:
        for a in range(n):
            ra = refG[a]
            sa_inv, da = s_inv(ra, ks[a]), D[ra][ks[a]]
            for b in range(n):
                rb = refG[b]
                L_h, v_h = LH[pi[a], pi[b]], vH[pi[a], pi[b]]
                if not eq(vG[a, b], sa_inv @ (L_h @ D[rb][ks[b]] + v_h - da)):
                    return False
                if not eq(LG[a, b], sa_inv @ L_h @ S[rb][ks[b]]):
                    return False
        return True

    # Anchor member 0: enumerate its image h0 = pi(0) and dressing k0, only
    # n * |S| choices. Every other member's dressing is then solved for rather
    # than enumerated.
    r0 = refG[0]
    for h0 in range(n):
        # pi must map each member onto one of the same base reference
        # (def:same). Required for correctness; it also prunes the search.
        if refH[h0] != r0:
            continue
        for k0 in range(len(S[r0])):
            pi = [-1] * n
            ks = [-1] * n
            used = [False] * n
            pi[0], ks[0], used[h0] = h0, k0, True

            def assign(b: int) -> bool:
                if b == n:
                    return verify(pi, ks) if verify_all else True
                rb = refG[b]
                for h in range(n):
                    if used[h] or refH[h] != rb:
                        continue
                    # With s_0 fixed, the linear equation for pair (0, b) has
                    # the unique solution s_b = (L^H[h0, h])^-1 @ s_0 @ L^G[0, b];
                    # the candidate survives only if that lies in b's group.
                    # (L^H[h0, h])^-1 is just L^H[h, h0]: L[a,b] = R_a^-1 R_b,
                    # so the swapped pair is the exact inverse, no solve needed.
                    kb = index_in_S(rb, LH[h, h0] @ S[r0][k0] @ LG[0, b])
                    if kb is None:
                        continue
                    # Solving s_b first is what makes this check possible: it
                    # fixes d_b. The dressing is affine, d = c - s @ c, because
                    # a self-isometry fixes the monomer's centroid, not the
                    # frame origin. Dropping d compares origin offsets that are
                    # not the same geometric quantity, and since s preserves
                    # norms the check then fails at ANY tolerance whenever the
                    # two v norms differ (urea; rem:centering).
                    if not eq(
                        vG[0, b],
                        s_inv(r0, k0) @ (LH[h0, h] @ D[rb][kb] + vH[h0, h] - D[r0][k0]),
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
    W: list[list[NDArray]] | list[NDArray],
    W_tr: list[list[NDArray]] | list[NDArray] | None = None,
) -> bool:
    """Return True iff ``g_N`` and ``h_N`` are the same multimer under a
    crystallographic isometry, working entirely in the exact integer lattice
    basis and modding out the site-symmetry dressing ``(W, W_tr)``.

    Exact-integer form of :func:`anchored_match`: roundoff-free and complete
    for every same-orbit merge, including centrosymmetric site-inversion cases.
    Congruences whose dressing lies outside ``W`` (non-crystallographic) are not
    found here; they are left to :mod:`~crystalatte.work.dedup.congruence`.

    :param g_N: The first cluster
    :param h_N: The second cluster
    :param W: Site-symmetry linear parts indexed by base reference, i.e.
        ``context.W_lin_by_ref``. A single group is also accepted and applies to
        every member, which is only right for a mono-reference cluster (see
        :func:`as_per_reference`). Required: defaulting to reference 0 would
        silently mod out the wrong site symmetry for every other reference.
    :param W_tr: The matching dressing translations ``context.W_tr_by_ref``, in
        twelfths, index-aligned with ``W``. Defaults to zeros.
    :returns: True iff the clusters are the same multimer via a crystallographic
        isometry
    """
    if len(g_N) != len(h_N):
        return False
    refG = g_N.ref_indices.tolist()
    refH = h_N.ref_indices.tolist()
    # A reference-compatible pi exists only if both clusters draw on each
    # reference the same number of times; reject before building transforms.
    if sorted(refG) != sorted(refH):
        return False
    LG, vG = relative_transforms(g_N, integer=True)
    LH, vH = relative_transforms(h_N, integer=True)
    n_refs = max(max(refG), max(refH)) + 1
    S = as_per_reference(W, n_refs)
    # Zero translations are correct only when every centroid is a fixed point
    # of every element of its W; otherwise pass W_tr (see rem:centering).
    D = (
        [[np.zeros(3, dtype=np.int64)] * len(grp) for grp in S]
        if W_tr is None
        else as_per_reference(W_tr, n_refs)
    )
    # Exact integers: the closing sweep provably cannot reject (lem:row) and
    # cost ~10% of runtime, so skip it.
    return anchored_match(
        LG, vG, LH, vH, S, D, refG, refH, np.array_equal, int_inv, verify_all=False
    )
