"""Generate unique N-mers from a crystal and a reference monomer.

Pipeline:

1. :func:`_generate_neighbors` enumerates symmetry-related neighbor monomers
   within a cutoff (filter convention selected by keyword arguments:
   ``use_origin``, ``use_com``, or the default closest-contact), forms all
   ``C(M, N-1)`` :class:`~crystalatte.core.sym_ops.SymOpList` candidates with
   the reference identity prepended, groups them by a bucket key (the hash of
   the sorted, rounded pairwise member-centroid distances — an isometry
   invariant valid across references), and runs :func:`_process_bucket` per
   bucket to deduplicate via :func:`_is_bijection`.
2. :func:`_is_bijection` tests equivalence cheapest-first:

   a. distance-fingerprint gate (:func:`_fingerprints_match`) — a cached,
      vectorized necessary condition that rejects most non-equivalent pairs;
   b. exact-integer matcher (:func:`_same_multimer_int`) — the complete,
      roundoff-free decision for every crystallographic (same-orbit) merge,
      modding out the site symmetry (twelfths-scale integer arithmetic; no
      floating-point tolerance involved);
   c. float congruence fallback (:func:`_same_multimer`), gated by
      ``_check_congruence`` — runs only when the integer matcher fails and the
      molecular symmetry is richer than the site symmetry. It merges every
      energetically-equivalent congruence: proper molecular-symmetry
      congruences AND mirror-image enantiomers alike, with no chirality gate
      and no chemistry oracle. The dedup target is energetic equivalence, not
      point-group-restricted congruence;
   d. point-cloud congruence (:func:`_same_multimer_geom`) for cross-reference
      clusters, and as a last-resort fallback for LINEAR monomers, whose
      continuous point group no finite dressing set can represent.

   Stages (b) and (c) are the same anchored search (:func:`_anchored_match`)
   over the same relative transforms (:func:`_relative_transforms`), differing
   only in arithmetic — exact integer versus tolerant Cartesian — and in which
   group supplies the dressing.
3. :func:`generate` post-processes each unique
   :class:`~crystalatte.core.sym_ops.SymOpList` into a
   :class:`~crystalatte.core.multimer.Multimer`.
"""

from itertools import combinations, product
from collections import defaultdict

import numpy as np
import qcelemental as qcel
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm
from numpy.typing import NDArray

from ..core.crystal import Crystal
from ..core.multimer import Monomer, Multimer
from ..core.space_group import SITE_ATOL
from ..core.sym_ops import SymOp, SymOpList


# Module-level bindings set by `_bind_globals` (in main process) and by the
# worker initializer (`_init_bucket_worker`, in ProcessPoolExecutor children).
# Per-reference lists indexed by reference index (one entry per independent monomer).
_crystal: Crystal | None = None
_monomers: list[Monomer] = []
_Z_by_ref: list[NDArray] = []
_mass_by_ref: list[NDArray] = []
_self_iso_by_ref: list[list[NDArray]] = [[np.eye(3)]]
_W_lin_by_ref: list[list[NDArray]] = [[np.eye(3, dtype=np.int64)]]
_check_congruence_by_ref: list[bool] = [False]
# Translation parts of the dressings, index-aligned with the groups above.
# A monomer self-isometry is an isometry about the monomer's CENTROID, not
# about the coordinate origin, so as a map on the reference monomer's raw
# coordinates it is the affine map ``x -> s @ x + d`` with ``d = c - s @ c``.
# Dropping ``d`` (using ``s`` alone) is only valid when ``s @ c == c``.
_self_iso_tr_by_ref: list[list[NDArray]] = [[np.zeros(3)]]        # Cartesian, Angstroms
_W_tr_by_ref: list[list[NDArray]] = [[np.zeros(3, dtype=np.int64)]]  # lattice basis, twelfths
# True for references whose atoms are collinear. Such monomers have a
# CONTINUOUS point group (C_inf_v / D_inf_h), which no finite self-isometry
# sample can represent, so the relative-transform matchers cannot find the
# dressing a merge needs and these are routed to point-cloud congruence.
_is_linear_by_ref: list[bool] = [False]
_A: NDArray | None = None      # fractional -> Cartesian matrix (cart = _A @ frac)
_A_inv: NDArray | None = None  # Cartesian -> fractional matrix


def _bind_globals(
    crystal: Crystal, monomers: list[Monomer], site_atol: float = SITE_ATOL
) -> None:
    """Bind the deduper's per-run inputs at module scope.

    Sets the module globals for per-reference state: per-reference Z and mass
    arrays, self-isometries, site-symmetry dressings, the linearity flags and
    the congruence gates. Also sets the lattice vectors ``_A`` / ``_A_inv`` for
    fractional<->Cartesian conversion.

    :param crystal: The crystal supplying the lattice and Cartesian conversion
    :param monomers: The reference monomers (one per crystal asymmetric unit)
    :param site_atol: Fractional tolerance for the site-symmetry test; must
        match the one :func:`_generate_neighbors` passes to
        ``get_unique_sym_ops`` or stabilizer and orbit disagree
    :returns: ``None``
    """
    global _crystal, _monomers, _Z_by_ref, _mass_by_ref, _self_iso_by_ref
    global _W_lin_by_ref, _check_congruence_by_ref, _A, _A_inv
    global _self_iso_tr_by_ref, _W_tr_by_ref, _is_linear_by_ref
    _crystal = crystal
    _monomers = list(monomers)
    _A = np.asarray(crystal.lattice_vectors, dtype=float)
    _A_inv = np.linalg.inv(_A)

    _Z_by_ref, _mass_by_ref, _self_iso_by_ref = [], [], []
    _W_lin_by_ref, _check_congruence_by_ref = [], []
    _self_iso_tr_by_ref, _W_tr_by_ref, _is_linear_by_ref = [], [], []
    for mon in _monomers:
        _Z_by_ref.append(
            np.array([qcel.periodictable.to_Z(s) for s in mon.symbols], dtype=np.int64)
        )
        _mass_by_ref.append(
            np.array([qcel.periodictable.to_mass(s) for s in mon.symbols])
        )
        S = mon.self_isometries
        _self_iso_by_ref.append(S)
        # Cartesian dressing translations: d = c - s @ c (zero iff s fixes c).
        c_cart = np.asarray(mon.centroid_cart, dtype=float)
        _self_iso_tr_by_ref.append([c_cart - s @ c_cart for s in S])

        # Collinear atoms => rank-1 gyration tensor => continuous point group.
        centered = np.asarray(mon.cart_coords, dtype=float)
        centered = centered - centered.mean(axis=0)
        gyration = np.linalg.eigvalsh(centered.T @ centered)  # ascending
        _is_linear_by_ref.append(
            bool(gyration[1] <= 1e-6 * max(float(gyration[-1]), 1.0))
        )

        site_ops = crystal.space_group.get_site_symmetry(
            mon.centroid_frac, atol=site_atol
        )
        c_frac = np.asarray(mon.centroid_frac, dtype=float)
        W, W_tr = [], []
        for op in site_ops:
            r = op.rot.astype(np.int64)
            if any(np.array_equal(r, w) for w in W):
                continue
            # The dressing must fix the centroid EXACTLY, but a site op fixes it
            # only modulo a lattice vector, so strip that lattice part off the
            # op's own translation. All integer, in twelfths.
            tr = op.tr.astype(np.int64)
            lattice = np.rint(r @ c_frac + tr / 12.0 - c_frac).astype(np.int64)
            W.append(r)
            W_tr.append(tr - 12 * lattice)
        if not W:
            W = [np.eye(3, dtype=np.int64)]
            W_tr = [np.zeros(3, dtype=np.int64)]
        _W_lin_by_ref.append(W)
        _W_tr_by_ref.append(W_tr)
        _check_congruence_by_ref.append(len(S) > len(W))


def _init_bucket_worker(
    crystal: Crystal, monomers: list[Monomer], site_atol: float = SITE_ATOL
) -> None:
    """Process-pool initializer: bind the deduper globals in a worker process.

    :param crystal: The crystal passed to :func:`_bind_globals`
    :param monomers: The reference monomers passed to :func:`_bind_globals`
    :param site_atol: The tolerance passed to :func:`_bind_globals`
    :returns: ``None``
    """
    _bind_globals(crystal, monomers, site_atol)


def _int_inv(M: NDArray) -> NDArray:
    """Return the exact integer inverse of a unimodular integer matrix.

    Lattice-basis rotations are integer with ``det = ±1`` but are not
    orthogonal, so the inverse is recovered via the adjugate (``round`` of the
    float inverse), never the transpose. Accepts a single ``(3, 3)`` matrix or a
    stacked ``(n, 3, 3)`` batch.

    :param M: A ``(3, 3)`` or ``(n, 3, 3)`` integer matrix with ``det = ±1``
    :returns: The exact ``int64`` inverse, same shape as ``M``
    """
    inv = np.rint(np.linalg.inv(M)).astype(np.int64)
    identity = np.broadcast_to(np.eye(3, dtype=np.int64), np.shape(M))
    assert np.array_equal(inv @ np.asarray(M, dtype=np.int64), identity), (
        "_int_inv called on a non-unimodular matrix (exact integer inverse does not exist)"
    )
    return inv


def _relative_transforms(g_N: SymOpList, integer: bool) -> tuple[NDArray, NDArray]:
    """Return the cluster's pairwise relative transforms
    ``Delta[a,b] = T_a^-1 T_b = (L[a,b], v[a,b])``.

    These are invariant under any global isometry applied to the whole cluster
    (it cancels in ``T_a^-1 T_b``), including reflection — which is exactly why
    matching them merges identical clusters, lattice translates, and enantiomers
    without ever choosing the global map explicitly.

    Two bases are available, differing only in arithmetic:

    - ``integer=True``: the lattice basis, ``L[a,b] = rot_a^-1 rot_b`` and
      ``v[a,b] = rot_a^-1 (tr_b - tr_a)`` in twelfths. Exact at all distances —
      no Cartesian conversion, no roundoff.
    - ``integer=False``: Cartesian. A crystal sym op ``x' = rot @ x + tr/12``
      becomes ``Rc = A @ rot @ A^-1`` (orthogonal, since sym ops are isometries)
      and ``tc = A @ (tr/12)``, with ``A`` the fractional->Cartesian matrix.

    Cached on the :class:`~crystalatte.core.sym_ops.SymOpList` (one slot per
    basis) since the matchers re-read them across many comparisons.

    :param g_N: The cluster whose relative transforms are requested
    :param integer: Whether to work in the exact integer lattice basis
    :returns: A tuple ``(L, v)`` of shapes ``(n, n, 3, 3)`` and ``(n, n, 3)``
    """
    slot = "_rel_int_cache" if integer else "_rel_cache"
    cached = getattr(g_N, slot, None)
    if cached is not None:
        return cached

    if integer:
        R = g_N.rot_cache.astype(np.int64)                  # (n, 3, 3)
        t = g_N.tr_cache.astype(np.int64)                   # (n, 3), twelfths
        R_inv = _int_inv(R)                                 # batched
    else:
        R = _A @ g_N.rot_cache.astype(float) @ _A_inv       # (n, 3, 3)
        t = (_A @ (g_N.tr_cache.astype(float) / 12.0).T).T  # (n, 3)
        R_inv = np.linalg.inv(R)

    L = np.einsum("aij,bjk->abik", R_inv, R)                # (n, n, 3, 3)
    dt = t[None, :, :] - t[:, None, :]                      # (n, n, 3): t_b - t_a
    v = np.einsum("aij,abj->abi", R_inv, dt)                # (n, n, 3)
    setattr(g_N, slot, (L, v))
    return L, v


def _relative_transforms_int(g_N: SymOpList) -> tuple[NDArray, NDArray]:
    """Exact integer-basis :func:`_relative_transforms`. See that function.

    :param g_N: The cluster whose integer relative transforms are requested
    :returns: A tuple ``(L, v)`` of shapes ``(n, n, 3, 3)`` and ``(n, n, 3)``
    """
    return _relative_transforms(g_N, integer=True)


def _anchored_match(LG, vG, LH, vH, S, D, eq, inv) -> bool:
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


def _same_multimer_int(
    g_N: SymOpList,
    h_N: SymOpList,
    W: list[NDArray],
    W_tr: list[NDArray] | None = None,
) -> bool:
    """Return True iff ``g_N`` and ``h_N`` are the same multimer under a
    crystallographic isometry, working entirely in the exact integer lattice
    basis and modding out the site-symmetry dressing ``(W, W_tr)``.

    Exact-integer form of :func:`_anchored_match` — roundoff-free and complete
    for every same-orbit (multiplicity-required) merge, including
    centrosymmetric site-inversion cases. Non-crystallographic congruences
    (dressing outside ``W``) are left to the float fallback in
    :func:`_is_bijection`.

    :param g_N: The first cluster
    :param h_N: The second cluster
    :param W: The site-symmetry linear parts of the clusters' shared base
        reference, i.e. ``_W_lin_by_ref[base]``. Required: there is no safe
        default, since defaulting to reference 0 would silently mod out the
        wrong site symmetry for every other reference under ``Z' > 1``.
    :param W_tr: The matching dressing translations ``_W_tr_by_ref[base]``, in
        twelfths, index-aligned with ``W``. Defaults to zeros, which is correct
        only when the centroid is a fixed point of every element of ``W``.
    :returns: True iff the clusters are the same multimer via a crystallographic
        isometry
    """
    if len(g_N) != len(h_N):
        return False
    LG, vG = _relative_transforms(g_N, integer=True)
    LH, vH = _relative_transforms(h_N, integer=True)
    D = [np.zeros(3, dtype=np.int64)] * len(W) if W_tr is None else W_tr
    return _anchored_match(LG, vG, LH, vH, W, D, np.array_equal, _int_inv)


def _same_multimer(g_N: SymOpList, h_N: SymOpList, atol: float = 1e-3) -> bool:
    """Return True iff ``g_N`` and ``h_N`` are the same multimer up to a global
    isometry (rotation, translation, or reflection).

    Cartesian form of :func:`_anchored_match`, dressed with the FULL molecular
    self-isometry group (:attr:`~crystalatte.core.multimer.Monomer.self_isometries`)
    rather than just the site symmetry, so it also merges non-crystallographic
    congruences and — being blind to ``det`` — mirror-image enantiomers.

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
    if len(g_N) != len(h_N):
        return False
    LG, vG = _relative_transforms(g_N, integer=False)
    LH, vH = _relative_transforms(h_N, integer=False)
    base = _base(g_N)
    return _anchored_match(
        LG, vG, LH, vH,
        _self_iso_by_ref[base],
        _self_iso_tr_by_ref[base],
        lambda a, b: np.allclose(a, b, atol=atol),
        lambda s: s.T,  # orthogonal -> inverse is transpose
    )


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
    2. Mono-reference with a LINEAR monomer, after both of the above fail:
       point-cloud congruence (:func:`_same_multimer_geom`). A collinear
       monomer's point group is continuous (``C_inf_v`` / ``D_inf_h``), so the
       finite self-isometry sample cannot supply a dressing that is a rotation
       about the molecular axis by an unsampled angle. Point-cloud congruence
       needs no dressing. It is strictly weaker on its own — replacing the
       relative-transform path with it inflates the CO2 dimer count from 38 to
       126 — so it runs only as a last-resort fallback.
    3. Cross-reference (members drawn from different base references, or the
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
        if _same_multimer_int(g_N, h_N, _W_lin_by_ref[i], _W_tr_by_ref[i]):
            return True
        if _check_congruence_by_ref[i] and _same_multimer(g_N, h_N):
            return True
        if _is_linear_by_ref[i]:
            # Backstop for continuous point groups (C_inf_v / D_inf_h), where
            # NO finite self-isometry set is complete: a merge whose dressing is
            # a rotation about the molecular axis by an unsampled angle is
            # invisible to the relative-transform matchers. Point-cloud
            # congruence needs no dressing at all.
            #
            # Since self-isometry closure repair landed this no longer fires for
            # CO2 (its group closes to 48 operations, enough for every merge in
            # the test set — disabling this branch leaves CO2 unchanged). It is
            # kept because sufficiency is a property of that particular sample,
            # not a guarantee for an arbitrary linear species. It is a strictly
            # weaker matcher on its own, so it runs LAST.
            return _same_multimer_geom(g_N, h_N)
        return False
    return _same_multimer_geom(g_N, h_N)


def _process_bucket(
    bucket: list[SymOpList],
) -> tuple[list[SymOpList], list[SymOpList]]:
    """Deduplicate one bucket of clusters that share the same bucket key.

    For each new candidate, compare it to every existing unique representative
    via :func:`_is_bijection`. On a match, the candidate with the smaller summed
    translation norm is kept as the representative (and the other is reclassified
    as a duplicate). This keeps the representative closer to the reference, which
    matters when a downstream radius filter is applied per multimer.

    :param bucket: The clusters sharing a single ``_bucket_key``
    :returns: A tuple ``(unique, duplicate)`` of cluster lists
    """
    local_unique: list[SymOpList] = []
    local_dup: list[SymOpList] = []
    for g_t in bucket:
        for idx, u in enumerate(local_unique):
            if not _is_bijection(g_t, u):
                continue
            if (
                np.linalg.norm(g_t.translations, axis=1).sum()
                < np.linalg.norm(u.translations, axis=1).sum()
            ):
                # Promote g_t in place. Assigning by index rather than
                # list.remove(u) avoids an O(k) sweep of SymOpList.__eq__ (which
                # compares operation sets) and cannot drop a different
                # equal-comparing representative.
                local_dup.append(u)
                local_unique[idx] = g_t
                g_t.multiplicity = u.multiplicity + 1
            else:
                local_dup.append(g_t)
                u.multiplicity += 1
            break
        else:
            local_unique.append(g_t)
    return local_unique, local_dup


def _neighbor_distance(
    cart_mon: NDArray, partner_masses: NDArray, monomer: Monomer, mode: str
) -> float:
    """Return the cutoff distance for a partner monomer.

    :param cart_mon: The partner monomer's Cartesian coordinates
    :param partner_masses: The *partner's* atomic masses, aligned with
        ``cart_mon``. These must come from the partner's own base reference:
        under ``Z' > 1`` the anchor and the partner can be different molecules
        (or the same molecule with a different atom ordering), so weighting the
        partner's coordinates by the anchor's masses gives a wrong center of
        mass, or a shape mismatch for a genuine co-crystal.
    :param monomer: The anchor monomer
    :param mode: The distance convention, as resolved by
        :func:`_distance_mode`: ``"origin"`` (closest partner atom to the anchor
        centroid), ``"com"`` (partner centre of mass to anchor centroid), or
        ``"contact"`` (closest atom-pair contact)
    :returns: The partner-to-anchor distance in Angstroms
    """
    if mode == "origin":
        return float(np.linalg.norm(cart_mon - monomer.centroid_cart, axis=1).min())
    if mode == "com":
        com_mon = np.average(cart_mon, axis=0, weights=partner_masses)
        return float(np.linalg.norm(com_mon - monomer.centroid_cart))
    return float(cdist(monomer.cart_coords, cart_mon).min())


def _distance_mode(kwargs: dict) -> str:
    """Resolve the distance convention once, rather than per candidate.

    :param kwargs: The caller's keyword arguments
    :returns: ``"origin"``, ``"com"`` or ``"contact"``
    """
    if kwargs.get("use_origin"):
        return "origin"
    if kwargs.get("use_com", False):
        return "com"
    return "contact"


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
    verbose = kwargs.get("verbose", 1)
    mode = _distance_mode(kwargs)
    site_atol = kwargs.get("site_atol", SITE_ATOL)

    if kwargs.get("debug", False):
        translations = list(product([-1, 0, 1], repeat=3))
    else:
        translations = list(product(*[range(cm, cp + 1) for cm, cp in cutoff]))

    # Bind per-reference state (workers below get the same via _init_bucket_worker).
    _bind_globals(crystal, monomers, site_atol)

    # Full partner pool across ALL references: (base_index j, SymOp placing monomers[j]).
    pool = []
    for j, mon in enumerate(monomers):
        sym_ops = crystal.space_group.get_unique_sym_ops(
            mon.centroid_frac, atol=site_atol
        )
        for op, tr in product(sym_ops, translations):
            pool.append((j, op.translate(np.array(tr, dtype=np.int16) * 12)))

    # Anchor each reference i; partners = pool members within R of anchor i
    # (excluding anchor i's own zero-translation identity).
    #
    # Note the cutoff is anchor<->partner only: for N > 2 a candidate's partners
    # are each within R of the anchor but may be arbitrarily far from *each
    # other*. This anchor-centric convention is intentional, not an oversight.
    candidates: list[SymOpList] = []
    for i, anchor in enumerate(monomers):
        near = []
        for (j, op) in pool:
            if j == i and op == SymOp.identity():
                continue
            partner_cart = crystal.to_cartesian(op.apply(monomers[j].frac_coords))
            if _neighbor_distance(partner_cart, _mass_by_ref[j], anchor, mode) <= R:
                near.append((j, op))
        if verbose:
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
            # (an isometry invariant of the cluster, and valid across
            # references — a key built only from SymOp translations would not be).
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

        n_jobs = max(1, min(kwargs["n_jobs"], len(buckets), cpu_count() - 1))
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_bucket_worker,
            initargs=(crystal, monomers, site_atol),
        ) as ex:
            results = list(
                tqdm(
                    ex.map(_process_bucket, buckets.values()),
                    total=len(buckets),
                    desc="Processing buckets",
                    leave=False,
                    disable=not verbose,
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
            disable=not verbose,
        ):
            local_unique, local_dup = _process_bucket(bucket)
            unique.extend(local_unique)
            duplicate.extend(local_dup)

    if verbose:
        print(
            f"Found {len(duplicate)} equivalent multimers from neighboring unit cell search."
        )

    return unique, duplicate


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
        base = _monomers[refs[k]]
        masses = _mass_by_ref[refs[k]]
        mon_frac = g.apply(base.frac_coords)
        mon_cart = _crystal.to_cartesian(mon_frac)
        com_frac = np.average(mon_frac, axis=0, weights=masses)
        com_cart = _crystal.to_cartesian(com_frac)
        mons.append(Monomer(base.symbols, mon_frac, mon_cart, com_frac, com_cart, g))
    return Multimer(mons, g_N, g_N.multiplicity)


def generate(crystal: Crystal, N: int, R: float, *, detect_symmetry=True, symprec=1e-3, **kwargs) -> list[Multimer]:
    """Generate unique multimers of every reference molecule in the crystal.

    :param crystal: The crystal structure to generate multimers from; all
        symmetry-independent reference molecules (:meth:`Crystal.get_references`)
        are enumerated
    :param N: The number of monomers in the multimer (e.g. ``N=2`` for dimers)
    :param R: The maximum center-to-center distance in Angstroms for monomers to
        be considered part of the same multimer
    :param detect_symmetry: If True, automatically detect and apply symmetry via
        :meth:`Crystal.detect_symmetry` before enumeration (default: True)
    :param symprec: Symmetry detection tolerance in Angstroms (default: 1e-3)
    :param kwargs: Additional options forwarded to :func:`_generate_neighbors`
    :returns: A list of unique :class:`~crystalatte.core.multimer.Multimer`
        objects representing the multimers found in the crystal
    """
    if detect_symmetry:
        crystal = crystal.detect_symmetry(symprec)
    monomers = crystal.get_references()
    d = np.linalg.norm(crystal.lattice_vectors, axis=0)
    R_eff = R + max(pdist(m.cart_coords).max() for m in monomers)

    lows, highs = [], []
    for m in monomers:
        t = m.centroid_frac
        lows.append(-np.ceil((R_eff - t * d) / d))
        highs.append(np.ceil((R_eff - (1 - t) * d) / d))
    bounds = np.vstack((np.min(lows, axis=0), np.max(highs, axis=0))).T.astype(int)

    # Site-symmetry tests compare FRACTIONAL vectors, but symprec is in
    # Angstroms, so convert. Using the shortest cell edge keeps the fractional
    # tolerance conservative along every axis. Floored at the module default so
    # a tight symprec cannot reintroduce the CIF-rounding failure (a special
    # position written as 0.3333 instead of 1/3 sits ~1e-4 off the element).
    kwargs.setdefault(
        "site_atol", max(SITE_ATOL, float(symprec) / float(d.min()))
    )

    unique_neighbors, _ = _generate_neighbors(crystal, monomers, bounds, R, N, **kwargs)

    multimers: list[Multimer] = []
    for g_N in tqdm(
        unique_neighbors,
        total=len(unique_neighbors),
        desc="Processing unique multimers",
        leave=False,
        disable=not kwargs.get("verbose", 1),
    ):
        m = _process_multimer(g_N)
        if m is not None:
            multimers.append(m)
    return multimers
