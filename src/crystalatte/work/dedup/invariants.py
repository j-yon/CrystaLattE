"""Isometry invariants of a cluster.

Everything here is a property of a cluster that any global isometry preserves,
so two clusters that are the same multimer necessarily agree on all of them.
They are what the matchers compare, and what the cheap pre-filter gates on.
"""
import numpy as np
from numpy.typing import NDArray

from ...core.sym_ops import SymOpList
from . import context

def int_inv(M: NDArray) -> NDArray:
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
        "int_inv called on a non-unimodular matrix (exact integer inverse does not exist)"
    )
    return inv


def relative_transforms(g_N: SymOpList, integer: bool) -> tuple[NDArray, NDArray]:
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
        R_inv = int_inv(R)                                 # batched
    else:
        R = context.frac_to_cart @ g_N.rot_cache.astype(float) @ context.cart_to_frac       # (n, 3, 3)
        t = (context.frac_to_cart @ (g_N.tr_cache.astype(float) / 12.0).T).T  # (n, 3)
        R_inv = np.linalg.inv(R)

    L = np.einsum("aij,bjk->abik", R_inv, R)                # (n, n, 3, 3)
    dt = t[None, :, :] - t[:, None, :]                      # (n, n, 3): t_b - t_a
    v = np.einsum("aij,abj->abi", R_inv, dt)                # (n, n, 3)
    setattr(g_N, slot, (L, v))
    return L, v


def relative_transforms_int(g_N: SymOpList) -> tuple[NDArray, NDArray]:
    """Exact integer-basis :func:`relative_transforms`. See that function.

    :param g_N: The cluster whose integer relative transforms are requested
    :returns: A tuple ``(L, v)`` of shapes ``(n, n, 3, 3)`` and ``(n, n, 3)``
    """
    return relative_transforms(g_N, integer=True)


def base_reference(g_N: SymOpList) -> int:
    """Return the (single) base reference index of a mono-reference cluster.

    :param g_N: The cluster whose base reference index is requested
    :returns: The base reference index (``g_N.ref_indices[0]``)
    """
    return int(g_N.ref_indices[0])


def is_mono_reference(g_N: SymOpList) -> bool:
    """Return True iff every member of ``g_N`` shares the same base reference.

    :param g_N: The cluster to test
    :returns: True iff ``g_N`` is mono-reference
    """
    return len(set(g_N.ref_indices.tolist())) == 1


def cluster_atoms_cart(g_N: SymOpList) -> NDArray:
    """Reconstruct the cluster's Cartesian atom positions from the module-level
    ``context.references`` / ``context.active_crystal`` bindings, using each member's base reference.

    The result is not cached: this is called only by
    :func:`distance_fingerprint`, which caches the derived fingerprint, so a
    per-object cache here would never be re-read.

    :param g_N: The cluster whose atom positions are requested
    :returns: An ``(n_atoms, 3)`` array of Cartesian coordinates
    """
    ref = g_N.ref_indices
    parts = [op.apply(context.references[ref[k]].frac_coords) for k, op in enumerate(g_N)]
    return context.active_crystal.to_cartesian(np.vstack(parts))


def distance_fingerprint(g_N: SymOpList) -> tuple[NDArray, NDArray]:
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
    atoms = cluster_atoms_cart(g_N)
    ref = g_N.ref_indices
    Z = np.concatenate([context.Z_by_ref[ref[k]] for k in range(len(g_N))])
    i_idx, j_idx = np.triu_indices(len(atoms), k=1)
    dists = np.linalg.norm(atoms[i_idx] - atoms[j_idx], axis=1)
    zz = Z[i_idx] * Z[j_idx]
    order = np.lexsort((dists, zz))
    fp = (zz[order], dists[order])
    g_N._dist_fp_cache = fp
    return fp


def fingerprints_match(g_N: SymOpList, h_N: SymOpList, atol: float = 1e-6) -> bool:
    """Return True iff the two clusters' Z-tagged distance fingerprints coincide.

    This is the cheap necessary gate run before the exact and anchored branches.
    The tight ``atol`` is safe because equivalent clusters share
    exact-arithmetic distances; homometric false positives (rare) are caught
    downstream by :func:`same_multimer`.

    :param g_N: The first cluster
    :param h_N: The second cluster
    :param atol: Absolute tolerance in Angstroms for the distance comparison
    :returns: True iff both fingerprints match
    """
    if len(g_N) != len(h_N):
        return False
    zz_g, d_g = distance_fingerprint(g_N)
    zz_h, d_h = distance_fingerprint(h_N)
    if zz_g.shape != zz_h.shape or not np.array_equal(zz_g, zz_h):
        return False
    return np.allclose(d_g, d_h, atol=atol, rtol=0.0)
