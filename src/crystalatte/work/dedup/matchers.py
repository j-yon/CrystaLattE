from collections.abc import Callable

from numpy.typing import NDArray


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
    """Anchored relative-transform search behind :func:`~crystalatte.work.dedup.congruence.same_multimer`.

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
