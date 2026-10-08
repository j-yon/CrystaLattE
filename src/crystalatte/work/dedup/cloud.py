import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from ...core.sym_ops import SymOpList
from . import context
from .invariants import cluster_atoms_cart


def kabsch(P: NDArray, Q: NDArray) -> NDArray | None:
    """Return the orthogonal R (det +/-1) minimizing ||P R^T - Q|| for centered
    corresponding point sets, or None if degenerate

    :param P: an ``(k, 3)`` array of centered source points
    :param Q: an ``(k, 3)`` array of centered target points
    :returns: orthogonal map ``R``, or None on an SVD failure
    """
    H = P.T @ Q

    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None # SVD failed

    R = (Vt.T @ U.T)
    return R


def same_multimer_geom(g_N: SymOpList, h_N: SymOpList, atol: float = 1e-3) -> bool:
    """Return True iff the two clusters' reconstructed, Z-labeled atom clouds are
    congruent under some isometry (rotation, translation, or reflection)

    Returns False for a (near-)collinear cluster, which admits no anchor triple.

    :param g_N: first cluster (may span multiple base references)
    :param h_N: second cluster (may span multiple base references)
    :param atol: absolute tolerance in Angstroms
    :returns: True iff the two clusters' atom clouds are congruent
    """
    # clear cheap necessary conditions first
    A = cluster_atoms_cart(g_N)
    B = cluster_atoms_cart(h_N)
    if A.shape != B.shape: # different number of atoms
        return False

    ZA = np.concatenate([context.Z_by_ref[g_N.ref_indices[k]] for k in range(len(g_N))])
    ZB = np.concatenate([context.Z_by_ref[h_N.ref_indices[k]] for k in range(len(h_N))])
    if sorted(ZA) != sorted(ZB): # different Z composition
        return False

    # center clouds at the origin
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)

    n = len(A)

    def _triple(P):
        """Return the largest-area anchor triple, or None if all are collinear"""
        # Any triple with |cross| > 1e-3 is usable, but a thin one fits the
        # rotation poorly and the error grows with distance (1e-6 A noise ->
        # ~4e-3 A at 20 A, above ``atol``). Maximising area costs the same scan.
        i0, best, best_area = 0, None, 1e-3
        for i1 in range(1, n):
            v1 = P[i1] - P[i0]
            for i2 in range(i1 + 1, n):
                area = np.linalg.norm(np.cross(v1, P[i2] - P[i0]))
                if area > best_area:
                    best, best_area = (i0, i1, i2), area
        return best

    tri = _triple(A)
    if tri is None:
        # (near-)linear cluster: no correspondence can be bootstrapped, and
        # linear monomers are out of scope (see `generate`), so decline;
        # not equivalent is the safe direction
        return False
    ia, ib, ic = tri
    dref = np.array([
        np.linalg.norm(A[ia] - A[ib]),
        np.linalg.norm(A[ia] - A[ic]),
        np.linalg.norm(A[ib] - A[ic]),
    ])
    DB_full = cdist(B, B)
    same = {z: np.where(ZB == z)[0] for z in np.unique(ZB)}
    # match the anchor to every Z- and distance-compatible triple in H
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

                # The triple only bootstraps a correspondence. Pair each atom
                # with its nearest unused same-Z partner, most confident first,
                # then re-fit on all n atoms so error averages down.
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
