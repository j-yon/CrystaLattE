import numpy as np
from numpy.typing import NDArray

from ...core.sym_ops import SymOpList
from . import context
from .invariants import int_inv, relative_transforms

def as_per_reference(groups, n_refs: int):
    """Normalize a dressing-group argument to one group per base reference.

    Accepts either form:

    - a single group (a sequence of ``(3, 3)`` matrices), meaning every
      reference shares it. This is the right thing to pass for a mono-reference
      cluster, and is what the callers did before ``Z' > 1`` was supported.
    - a list of groups, already indexed by base reference.

    The two are told apart by the first element: a lone group's first element is
    a bare array (a ``(3, 3)`` linear part or a ``(3,)`` translation), whereas a
    per-reference list's first element is itself a group, i.e. a sequence. Note
    the test cannot be on dimensionality alone -- a per-reference list of
    translation groups and a single group of linear parts are both "two deep".

    :param groups: A single dressing group, or one per reference
    :param n_refs: How many references the caller needs covered
    :returns: A list of ``n_refs`` groups
    """
    first = groups[0]
    if isinstance(first, np.ndarray) and first.ndim <= 2:
        return [groups] * n_refs
    return groups


def anchored_match(LG, vG, LH, vH, S, D, refG, refH, eq, inv, verify_all=True) -> bool:
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

    The closing ``verify`` sweep is LOGICALLY REDUNDANT: the anchor row is tested
    in full during the descent, and relative transforms compose
    (``Delta_ab Delta_bc = Delta_ac``), so no other pair can fail — see
    ``docs/dedup_correctness.tex`` Lemma ``lem:row``. Measured over six crystals
    it rejected 0 of 3981 leaves. It is kept because in the tolerant Cartesian
    instantiation that composition only holds up to accumulated tolerance, which
    makes the sweep a cheap backstop; in the exact integer instantiation it
    cannot fail at all.

    Note ``(L^H[a,b])^-1`` is just ``L^H[b,a]``: since ``L[a,b] = R_a^-1 R_b``,
    ``L[b,a] = L[a,b]^-1`` exactly, so no inverse is computed in the inner loop.

    Each member is dressed by the group of ITS OWN base reference, and ``pi`` is
    restricted to reference-compatible images (``refH[pi(a)] == refG[a]``). Both
    are Definition~2's requirements rather than additions to it, so the theory
    above is unchanged; the restriction on ``pi`` also prunes the search. Under
    ``Z' = 1`` every member shares one group and the behaviour is identical to
    the single-group form.

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
    :param verify_all: Re-check all ``n^2`` pairs before accepting. Provably
        unnecessary in exact arithmetic (see the note on the closing sweep
        below), so the integer instantiation passes False; the tolerant
        Cartesian one keeps it as a numerical backstop.
    :returns: True iff the clusters match under some ``(pi, dressings)``
    """
    n = len(LG)
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

    r0 = refG[0]
    for h0 in range(n):
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
                    # Solve the dressing-free linear equation first: it fixes
                    # s_b, hence d_b, which the translation guard then needs.
                    kb = index_in_S(rb, LH[h, h0] @ S[r0][k0] @ LG[0, b])
                    if kb is None:
                        continue
                    if not eq(
                        vG[0, b],
                        s_inv(r0, k0)
                        @ (LH[h0, h] @ D[rb][kb] + vH[h0, h] - D[r0][k0]),
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

    Handles cross-reference clusters (``Z' > 1``: co-crystals, hydrates, and
    single-species crystals with several independent molecules). Each member is
    dressed by its own reference's site symmetry and ``pi`` is held
    reference-compatible, which is what Definition 2 asks for.

    :param g_N: The first cluster
    :param h_N: The second cluster
    :param W: Site-symmetry linear parts indexed by base reference, i.e.
        ``context.W_lin_by_ref``. A single group is also accepted and applies to
        every member, which is the right thing only when the cluster is
        mono-reference (see :func:`as_per_reference`). Required either way:
        there is no safe default, since defaulting to reference 0 would silently
        mod out the wrong site symmetry for every other reference.
    :param W_tr: The matching dressing translations ``context.W_tr_by_ref``, in
        twelfths, index-aligned with ``W``. Defaults to zeros, which is correct
        only when the centroid is a fixed point of every element of ``W``.
    :returns: True iff the clusters are the same multimer via a crystallographic
        isometry
    """
    if len(g_N) != len(h_N):
        return False
    refG = g_N.ref_indices.tolist()
    refH = h_N.ref_indices.tolist()
    # Cheap necessary condition: a reference-compatible permutation exists only
    # if the two clusters draw on the same references the same number of times.
    if sorted(refG) != sorted(refH):
        return False
    LG, vG = relative_transforms(g_N, integer=True)
    LH, vH = relative_transforms(h_N, integer=True)
    n_refs = max(max(refG), max(refH)) + 1
    S = as_per_reference(W, n_refs)
    D = ([[np.zeros(3, dtype=np.int64)] * len(grp) for grp in S]
         if W_tr is None else as_per_reference(W_tr, n_refs))
    # verify_all=False: in exact integer arithmetic the cocycle identity
    # Delta_ab Delta_bc = Delta_ac holds exactly, so the anchor row implies every
    # other pair (Lemma lem:row) and the closing n^2 sweep cannot reject. It was
    # measured to reject 0 of 3981 leaves while costing ~10% of runtime.
    return anchored_match(LG, vG, LH, vH, S, D, refG, refH, np.array_equal, int_inv, verify_all=False)


def same_multimer(g_N: SymOpList, h_N: SymOpList, atol: float = 1e-3) -> bool:
    """Return True iff ``g_N`` and ``h_N`` are the same multimer up to a global
    isometry (rotation, translation, or reflection).

    Cartesian form of :func:`anchored_match`, dressed with the FULL molecular
    self-isometry group (:attr:`~crystalatte.core.multimer.Monomer.point_group`)
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
    refG = g_N.ref_indices.tolist()
    refH = h_N.ref_indices.tolist()
    if sorted(refG) != sorted(refH):
        return False
    LG, vG = relative_transforms(g_N, integer=False)
    LH, vH = relative_transforms(h_N, integer=False)
    # Per-member point groups. Reading a single group off `base_reference` was
    # correct only for mono-reference clusters, and silently dressed every
    # member with the FIRST member's point group otherwise.
    return anchored_match(
        LG, vG, LH, vH,
        context.self_iso_by_ref,
        context.self_iso_tr_by_ref,
        refG, refH,
        lambda a, b: np.allclose(a, b, atol=atol),
        lambda s: s.T,  # orthogonal -> inverse is transpose
    )
