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
   b. proper-rotation branch — exact ``int16`` bijection check on the augmented
      sym-op matrices (catches every case whose relating isometry is a crystal
      sym op already in the list);
   c. geometric branch (:func:`_same_multimer`) — matches the two clusters in
      relative-transform space, modding out the monomer's internal symmetry
      (:attr:`~crystalatte.core.multimer.Monomer.self_isometries`). This is a
      complete test that merges identical clusters, lattice translates, and
      enantiomers alike, with no chirality gate and no chemistry oracle.
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
# Defaults are safe: an identity-only self-isometry group and no lattice until
# bindings exist (the geometric branch degenerates to plain congruence).
_crystal: Crystal | None = None
_monomer: Monomer | None = None
_self_iso: list[NDArray] = [np.eye(3)]
_Z_per_monomer: NDArray = np.empty(0, dtype=np.int64)
_A: NDArray | None = None      # fractional -> Cartesian matrix (cart = _A @ frac)
_A_inv: NDArray | None = None  # Cartesian -> fractional matrix


def _bind_globals(crystal: Crystal, monomer: Monomer) -> None:
    """Bind the deduper's per-run inputs at module scope.

    Sets the module globals used by the geometric branch and the
    distance-fingerprint pre-filter: ``_self_iso`` (the monomer's molecular
    point group as Cartesian 3x3 operations), ``_Z_per_monomer`` (the per-atom
    atomic numbers), and ``_A`` / ``_A_inv`` (the fractional<->Cartesian lattice
    matrices used to push the integer sym-op transforms into Cartesian space).

    :param crystal: The crystal supplying the lattice and Cartesian conversion
    :param monomer: The reference monomer supplying symbols and point group
    :returns: ``None``
    """
    global _crystal, _monomer, _self_iso, _Z_per_monomer, _A, _A_inv
    _crystal = crystal
    _monomer = monomer
    _self_iso = monomer.self_isometries
    _Z_per_monomer = np.array(
        [qcel.periodictable.to_Z(s) for s in monomer.symbols], dtype=np.int64
    )
    _A = np.asarray(crystal.lattice_vectors, dtype=float)
    _A_inv = np.linalg.inv(_A)


def _init_bucket_worker(crystal: Crystal, monomer: Monomer) -> None:
    """Process-pool initializer: bind the deduper globals in a worker process.

    :param crystal: The crystal passed to :func:`_bind_globals`
    :param monomer: The reference monomer passed to :func:`_bind_globals`
    :returns: ``None``
    """
    _bind_globals(crystal, monomer)


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


def _cluster_atoms_cart(g_N: SymOpList) -> NDArray:
    """Reconstruct the cluster's Cartesian atom positions from the module-level
    ``_monomer`` / ``_crystal`` bindings.

    The result is not cached: this is called only by
    :func:`_distance_fingerprint`, which caches the derived fingerprint, so a
    per-object cache here would never be re-read.

    :param g_N: The cluster whose atom positions are requested
    :returns: An ``(n_atoms, 3)`` array of Cartesian coordinates
    """
    atoms_frac = np.vstack([op.apply(_monomer.frac_coords) for op in g_N])
    return _crystal.to_cartesian(atoms_frac)


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
    Z = np.tile(_Z_per_monomer, len(g_N))
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
    S = _self_iso

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


def _is_bijection(g_N: SymOpList, h_N: SymOpList) -> bool:
    """Return True iff ``g_N`` and ``h_N`` represent the same multimer.

    Three stages, cheapest first:

    0. Distance-fingerprint gate (:func:`_fingerprints_match`) — a cached,
       vectorized necessary condition. The vast majority of within-bucket pairs
       are not equivalent and are rejected here in ~microseconds, so the more
       expensive branches below run only on fingerprint-matching pairs.
    a. Proper-rotation branch — for each non-identity ``h_i``, check whether the
       composed list ``{g_j @ h_i : j}`` equals ``g_N`` as a set of 4x4
       augmented sym-op matrices (exact ``int16`` equality). Fast exact path for
       equivalences whose relating isometry already appears in the sym-op list.
    b. Geometric branch (:func:`_same_multimer`) — matches the two clusters in
       relative-transform space, modding out the monomer's internal symmetry.
       Complete: catches lattice translates and enantiomers the proper branch
       misses (including rare homometric fingerprint collisions), with no
       chirality gate.

    :param g_N: The first cluster
    :param h_N: The second cluster
    :returns: True iff the two clusters represent the same multimer
    """
    if not _fingerprints_match(g_N, h_N):
        return False

    n = len(g_N)
    g_aug = g_N.aug_cache
    g_fp = g_aug.reshape(n, -1)

    # composed[i, j] = g_j @ h_i, shape (n, n, 4, 4)
    composed = g_aug[:, None, :, :] @ h_N.aug_cache[None, :, :, :]
    for i in range(1, n):
        composed_fp = composed[i].reshape(n, -1)
        in_g = np.all(composed_fp[:, None, :] == g_fp[None, :, :], axis=-1).any(axis=-1)
        if not in_g.all():
            continue
        if len(np.unique(composed_fp, axis=0)) == n:
            return True

    return _same_multimer(g_N, h_N)


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

    - ``use_origin=True``: closest atom of the partner to the origin (which is
      the reference monomer's centroid in this code's convention).
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
        return float(np.linalg.norm(cart_mon, axis=1).min())
    if kwargs.get("use_com", False):
        masses = np.array([qcel.periodictable.to_mass(s) for s in monomer.symbols])
        com_mon = np.average(cart_mon, axis=0, weights=masses)
        return float(np.linalg.norm(com_mon - monomer.centroid_cart))
    return float(cdist(monomer.cart_coords, cart_mon).min())


def _generate_neighbors(
    crystal: Crystal,
    monomer: Monomer,
    cutoff: NDArray,
    R: float,
    N: int,
    **kwargs,
) -> tuple[list[SymOpList], list[SymOpList]]:
    """Generate all unique N-mers in the supercell defined by ``cutoff``, then
    deduplicate via translation-fingerprint bucketing and :func:`_is_bijection`.

    :param crystal: The crystal containing the monomer
    :param monomer: The reference monomer; its centroid is placed at the origin
    :param cutoff: Per-axis ``(low, high)`` integer lattice translation bounds
    :param R: Distance cutoff in Angstroms. The distance convention is selected
        by ``kwargs`` (see :func:`_neighbor_distance`).
    :param N: N-mer order (``N=2`` for dimers, ``N=3`` for trimers, ...)
    :param kwargs: Additional options (``debug``, ``n_jobs``, and the distance
        convention flags consumed by :func:`_neighbor_distance`)
    :returns: A tuple ``(unique, duplicate)`` of cluster lists
    """
    # `get_unique_sym_ops` filters out crystal sym ops whose action coincides
    # with the identity at the monomer's Wyckoff position (so we don't generate
    # the same monomer twice).
    sym_ops = crystal.space_group.get_unique_sym_ops(monomer.centroid_frac)

    if kwargs.get("debug", False):
        translations = list(product([-1, 0, 1], repeat=3))
    else:
        translations = list(product(*[range(cm, cp + 1) for cm, cp in cutoff]))

    # Build (sym op, lattice translation) candidate operations, dropping the
    # (identity, 0) pair (= the reference monomer itself).
    full_ops: list[SymOp] = []
    for op, tr in product(sym_ops, translations):
        if op == SymOp.identity() and all(t == 0 for t in tr):
            continue
        full_ops.append(op.translate(np.array(tr, dtype=np.int16) * 12))

    # Prune by partner-monomer distance to the reference.
    full_ops = [
        op
        for op in full_ops
        if _neighbor_distance(
            crystal.to_cartesian(op.apply(monomer.frac_coords)), monomer, kwargs
        )
        <= R
    ]
    print(f"Found {len(full_ops)} monomers within {R} Angstroms.")

    centroid_frac_by_op = {op: op.apply(monomer.centroid_frac) for op in full_ops}
    centroid_frac_by_op[SymOp.identity()] = monomer.centroid_frac

    # Bind module-level state (`_crystal`, `_monomer`, point group, lattice)
    # for the geometric branch in `_is_bijection`. Workers spawned below get the
    # same bindings via `_init_bucket_worker`.
    _bind_globals(crystal, monomer)

    # Build all (identity + N-1)-monomer SymOpList candidates.
    candidates: list[SymOpList] = []
    for partner_ops in tqdm(
        combinations(full_ops, N - 1),
        total=len(full_ops) ** (N - 1) // math.factorial(N - 1),
        desc="Generating candidates",
        leave=False,
    ):
        ops = [SymOp.identity(), *partner_ops]
        centroid_frac = np.mean([centroid_frac_by_op[op] for op in ops], axis=0)
        sop = SymOpList(ops, 1)
        sop.centroid_frac = centroid_frac
        sop.centroid_cart = crystal.to_cartesian(centroid_frac)
        candidates.append(sop)

    # Bucket by structural translation fingerprint — `_is_bijection` is only
    # ever called within a bucket, since differing fingerprints rule out
    # equivalence.
    buckets: dict[int, list[SymOpList]] = defaultdict(list)
    for g_t in candidates:
        buckets[g_t.translation_fp].append(g_t)

    unique: list[SymOpList] = []
    duplicate: list[SymOpList] = []

    if "n_jobs" in kwargs and kwargs["n_jobs"] > 1:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import cpu_count

        n_jobs = min(kwargs["n_jobs"], len(buckets), cpu_count() - 1)
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_bucket_worker,
            initargs=(crystal, monomer),
        ) as ex:
            results = list(
                tqdm(
                    ex.map(_process_bucket, buckets.values()),
                    total=len(buckets),
                    desc="Processing across unique translation buckets",
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
            desc="Processing across unique translations",
            leave=False,
        ):
            local_unique, local_dup = _process_bucket(bucket)
            unique.extend(local_unique)
            duplicate.extend(local_dup)

    print(
        f"Found {len(duplicate)} equivalent multimers from neighboring unit cell search."
    )

    return unique, duplicate


def _init_multimers(crystal_: Crystal, monomer_: Monomer, masses_: NDArray) -> None:
    """Bind globals used by :func:`_process_multimer` so it can run from a
    :class:`~concurrent.futures.ProcessPoolExecutor` initializer.

    :func:`_bind_globals` is not reused because :func:`_process_multimer` also
    needs the per-atom mass array (for center-of-mass computation), which the
    bucket-dedup path does not.

    :param crystal_: The crystal bound to the module global ``crystal``
    :param monomer_: The reference monomer bound to the module global ``monomer``
    :param masses_: The per-atom masses bound to the module global ``masses``
    :returns: ``None``
    """
    global crystal, monomer, masses
    crystal = crystal_
    monomer = monomer_
    masses = masses_


def _process_multimer(g_N: SymOpList) -> Multimer | None:
    """Construct a :class:`~crystalatte.core.multimer.Multimer` from a deduped
    candidate cluster.

    :param g_N: The deduplicated cluster to materialize
    :returns: The constructed :class:`~crystalatte.core.multimer.Multimer`, or
        ``None`` if ``g_N`` contains duplicate sym ops (which can happen when
        multiple translations land on a degenerate Wyckoff orbit and would
        produce overlapping monomers)
    """
    if len(set(g_N)) != len(g_N):
        return None
    mons: list[Monomer] = []
    for g in g_N:
        mon_frac = g.apply(monomer.frac_coords)
        mon_cart = crystal.to_cartesian(mon_frac)
        com_frac = np.average(mon_frac, axis=0, weights=masses)
        com_cart = crystal.to_cartesian(com_frac)
        mons.append(Monomer(monomer.symbols, mon_frac, mon_cart, com_frac, com_cart, g))
    return Multimer(mons, g_N, g_N.multiplicity)


def generate(
    crystal: Crystal, monomer: Monomer, N: int, R: float, **kwargs
) -> list[Multimer]:
    """Generate unique multimers of the given monomer in the crystal.

    :param crystal: The crystal structure containing the monomer
    :param monomer: The reference monomer to generate multimers from
    :param N: The number of monomers in the multimer (e.g. ``N=2`` for dimers)
    :param R: The maximum center-to-center distance in Angstroms for monomers to
        be considered part of the same multimer
    :param kwargs: Additional options forwarded to :func:`_generate_neighbors`
    :returns: A list of unique :class:`~crystalatte.core.multimer.Multimer`
        objects representing the multimers found in the crystal
    """
    d = np.linalg.norm(crystal.lattice_vectors, axis=0)
    t = monomer.centroid_frac

    # effective bounds to search (accounting for monomer extent)
    R_eff = R + pdist(monomer.cart_coords).max()
    low_bounds = -np.ceil((R_eff - t * d) / d)
    high_bounds = np.ceil((R_eff - (1 - t) * d) / d)
    bounds = np.vstack((low_bounds, high_bounds)).T
    bounds = bounds.astype(int)

    unique_neighbors, _ = _generate_neighbors(crystal, monomer, bounds, R, N, **kwargs)

    masses = np.array([qcel.periodictable.to_mass(s) for s in monomer.symbols])
    _init_multimers(crystal, monomer, masses)

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
