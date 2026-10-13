"""Non-crystallographic congruence merges, run on the representatives left by the exact orbit-key pass.

The orbit key already merges every pair related by a crystal symmetry. What remains are congruences
outside the space group from molecular symmetry richer than the site symmetry, matched within one
base reference. Cross-reference congruences are not searched: they need two independent molecules
identical to the fingerprint tolerance, which in practice signals missed crystal symmetry.
"""

from collections import defaultdict, deque

import numpy as np
from scipy.spatial.distance import pdist

from ...core.sym_ops import SymOpList
from . import context
from .equivalence import merge
from .invariants import (
    base_reference,
    distance_fingerprint,
    fingerprints_match_many,
    is_mono_reference,
    relative_transforms,
)
from .matchers import anchored_match

#: sweep window (Angstroms) on summed member-centroid separations; congruent clusters agree to ~1e-3 per pair
WINDOW = 0.05


def is_needed() -> bool:
    """Return True iff some reference's congruence gate is open

    :returns: whether :func:`merge_congruent` can find anything
    """
    return any(context.check_congruence_by_ref)


def same_multimer(g_N: SymOpList, h_N: SymOpList, atol: float = 1e-3) -> bool:
    """Return True iff ``g_N`` and ``h_N`` are the same multimer up to a global
    isometry (rotation, translation, or reflection).

    Cartesian form of :func:`anchored_match`, dressed with the FULL molecular
    self-isometry group (:attr:`~crystalatte.core.multimer.Monomer.point_group`)
    rather than just the site symmetry, so it also merges non-crystallographic
    congruences and, being blind to ``det``, mirror-image enantiomers.

    Expensive; only called on clusters that already share a distance fingerprint.

    :param g_N: The first cluster
    :param h_N: The second cluster
    :param atol: Absolute tolerance in Angstroms. Can be tight when the
        relevant self-isometries are exact (identity / crystallographic); must
        be looser when a merge relies on an approximate molecular operation
        recovered from distorted CIF coordinates.
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
    # Pass groups for every reference, not one read off the cluster's
    # ``base_reference``: each member must be dressed by its OWN reference's
    # point group, or Z' > 1 clusters get the first member's group throughout.
    return anchored_match(
        LG,
        vG,
        LH,
        vH,
        context.self_iso_by_ref,
        context.self_iso_tr_by_ref,
        refG,
        refH,
        lambda a, b: np.abs(a - b).max() <= atol,
        lambda s: s.T,  # orthogonal -> inverse is transpose
    )


def _congruence_ref(g_N: SymOpList) -> int | None:
    """Return the base reference whose congruence gate ``g_N`` passes through, if any

    Only mono-reference clusters of one base reference with an open gate can be congruent outside the crystal
    symmetry, so this is the key a cluster is swept under.

    :param g_N: cluster to classify
    :returns: the base reference index, or ``None`` if ``g_N`` cannot be congruent to anything
    """
    if not is_mono_reference(g_N):
        return None
    i = base_reference(g_N)
    return i if context.check_congruence_by_ref[i] else None


def _spread(g_N: SymOpList) -> float:
    """Return the summed pairwise member-centroid distance, an isometry invariant used to sort clusters

    :param g_N: cluster to measure
    :returns: summed centroid separations in Angstroms
    """
    cens = [
        context.active_crystal.to_cartesian(
            op.apply(context.references[r].centroid_frac)
        )
        for r, op in zip(g_N.ref_indices, g_N)
    ]
    return float(pdist(np.array(cens)).sum())


def merge_congruent(
    unique: list[SymOpList],
) -> tuple[list[SymOpList], list[SymOpList]]:
    """Merge congruent representatives with a tolerance-window sweep (no rounding boundaries)

    Clusters are visited in order of spread and compared against the group roots of the same base reference
    whose spread lies within :data:`WINDOW`. Distance fingerprints are screened against the whole window in one
    vectorized call; :func:`same_multimer` then runs on the hits in window order, and the first match wins.

    :param unique: representatives from the exact pass
    :returns: a tuple ``(unique, duplicate)`` of cluster lists
    """
    spread = [_spread(g) for g in unique]
    groups: list[list[SymOpList]] = []
    # per base reference: (spread, group index, zz, distances) of each group's first member, ascending spread
    windows: dict[int, deque[tuple[float, int, np.ndarray, np.ndarray]]] = defaultdict(
        deque
    )
    for i in sorted(range(len(unique)), key=spread.__getitem__):
        g = unique[i]
        ref = _congruence_ref(g)
        if ref is None:
            groups.append([g])
            continue

        window = windows[ref]
        while window and spread[i] - window[0][0] > WINDOW:
            window.popleft()
        if window:
            hits = fingerprints_match_many(
                g, np.stack([w[2] for w in window]), np.stack([w[3] for w in window])
            )
            match = None
            for j in np.flatnonzero(hits):
                if same_multimer(groups[window[j][1]][0], g):
                    match = window[j][1]
                    break
            if match is not None:
                groups[match].append(g)
                continue
        window.append((spread[i], len(groups), *distance_fingerprint(g)))
        groups.append([g])

    reps, duplicates = [], []
    for group in groups:
        rep, dups = merge(group)
        reps.append(rep)
        duplicates.extend(dups)
    return reps, duplicates
