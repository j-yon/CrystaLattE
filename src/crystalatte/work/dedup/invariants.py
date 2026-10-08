from functools import cache

import numpy as np
from numpy.typing import NDArray

from ...core.sym_ops import SymOpList
from . import context

"""
    Forming the adjugate directly in integer arithmetic (nine 2x2 cofactors) is
    exact by construction and looks like the obvious optimisation, since this is
    one of the hottest functions in the deduper. It was tried and MEASURED
    SLOWER at the sizes that actually occur: 22.5 us against 8.7 us for a single
    3x3 and 22.9 vs 10.2 for a batch of three, because a dozen small numpy
    operations cost more in Python overhead than one LAPACK call. It only wins
    past a batch of ~100, and clusters hold 2-3 operations. Do not "fix" this.
"""

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


def relative_transforms(g_N: SymOpList, integer: bool) -> tuple[NDArray, NDArray]:
    """Return the cluster's pairwise relative transforms ``Delta[a,b] = T_a^-1 T_b = (L[a,b], v[a,b])``

    Available in exact integer arithmetic or Cartesian coordinates, cached on the cluster for efficiency

    :param g_N: cluster whose relative transforms are requested
    :param integer: whether to work in the exact integer lattice basis
    :returns: A tuple ``(L, v)`` of shapes ``(n, n, 3, 3)`` and ``(n, n, 3)``
    """
    slot = "_rel_int_cache" if integer else "_rel_cache"

    # check for existing cache
    cached = getattr(g_N, slot, None)
    if cached is not None:
        return cached

    if integer:
        R = g_N.rot_cache.astype(np.int64) # (n, 3, 3)
        t = g_N.tr_cache.astype(np.int64) # (n, 3), twelfths
        R_inv = int_inv(R)
    else:
        R = context.frac_to_cart @ g_N.rot_cache.astype(float) @ context.cart_to_frac       # (n, 3, 3)
        t = (context.frac_to_cart @ (g_N.tr_cache.astype(float) / 12.0).T).T # (n, 3)
        R_inv = np.linalg.inv(R)

    # contract rotations
    L = np.einsum("aij,bjk->abik", R_inv, R) # (n, n, 3, 3)

    # contract translations
    dt = t[None, :, :] - t[:, None, :] # (n, n, 3): t_b - t_a
    v = np.einsum("aij,abj->abi", R_inv, dt) # (n, n, 3)

    # cache and return transform
    setattr(g_N, slot, (L, v))
    return L, v


# TODO: remove, just use relative_transforms(g_N, integer=True) directly
def relative_transforms_int(g_N: SymOpList) -> tuple[NDArray, NDArray]:
    """Exact integer-basis :func:`relative_transforms`

    :param g_N: cluster whose integer relative transforms are requested
    :returns: tuple ``(L, v)`` of shapes ``(n, n, 3, 3)`` and ``(n, n, 3)``
    """
    return relative_transforms(g_N, integer=True)


def base_reference(g_N: SymOpList) -> int:
    """Return the (single) base reference index of a mono-reference cluster

    :param g_N: The cluster whose base reference index is requested
    :returns: The base reference index (``g_N.ref_indices[0]``)
    """
    return int(g_N.ref_indices[0])


def is_mono_reference(g_N: SymOpList) -> bool:
    """Return True iff every member of ``g_N`` shares the same base reference

    :param g_N: cluster to test
    :returns: True iff ``g_N`` is mono-reference
    """
    return len(set(g_N.ref_indices.tolist())) == 1


def cluster_atoms_cart(g_N: SymOpList) -> NDArray:
    """Reconstruct the cluster's Cartesian atom positions from the module-level context bindings

    :param g_N: The cluster whose atom positions are requested
    :returns: An ``(n_atoms, 3)`` array of Cartesian coordinates
    """
    ref = g_N.ref_indices
    parts = [op.apply(context.references[ref[k]].frac_coords) for k, op in enumerate(g_N)]
    return context.active_crystal.to_cartesian(np.vstack(parts))


@cache
def _triu_indices(n: int) -> tuple[NDArray, NDArray]:
    """Cache np.triu_indices for efficiency

    :param n: Number of atoms in the cluster
    :returns: The ``(i, j)`` index arrays for pairs with ``i < j``
    """
    return np.triu_indices(n, k=1)


def distance_fingerprint(g_N: SymOpList) -> tuple[NDArray, NDArray]:
    """Return the cluster's cached Z-tagged sorted pairwise atom-distance fingerprint

    :param g_N: cluster to fingerprint
    :returns: tuple ``(zz, distances)`` of equal-length 1-D arrays, sorted jointly by ``(zz, distance)``
    """
    # if cached, return
    cached = getattr(g_N, "_dist_fp_cache", None)
    if cached is not None:
        return cached

    # not cached, compute here
    atoms = cluster_atoms_cart(g_N) # (n_atoms, 3)
    ref = g_N.ref_indices # (n_ops,)
    Z = np.concatenate([context.Z_by_ref[ref[k]] for k in range(len(g_N))]) # (n_atoms,)
    i_idx, j_idx = _triu_indices(len(atoms)) # (n_atom_pairs,)
    dists = np.linalg.norm(atoms[i_idx] - atoms[j_idx], axis=1) 
    zz = Z[i_idx] * Z[j_idx]
    order = np.lexsort((dists, zz)) # lexsort by (zz, dists)
    fp = (zz[order], dists[order])

    # cache and return
    g_N._dist_fp_cache = fp
    return fp


def fingerprints_match(g_N: SymOpList, h_N: SymOpList, atol: float = 1e-6) -> bool:
    """Return True iff the two clusters' Z-tagged distance fingerprints coincide

    :param g_N: first cluster
    :param h_N: second cluster
    :param atol: absolute tolerance in Angstroms for the distance comparison
    :returns: True iff both fingerprints match
    """
    if len(g_N) != len(h_N): 
        return False # different number of operations (should be impossible but cheap to check)

    zz_g, d_g = distance_fingerprint(g_N)
    zz_h, d_h = distance_fingerprint(h_N)
    if zz_g.shape != zz_h.shape or not np.array_equal(zz_g, zz_h): 
        return False # different number of atom pairs or different Z products

    # compare distances with tolerance
    return np.allclose(d_g, d_h, atol=atol, rtol=0.0)
