"""Non-crystallographic congruence merges, run on the representatives left by the exact orbit-key pass.

The orbit key already merges every pair related by a crystal symmetry. What remains are congruences
outside the space group: molecular symmetry richer than the site symmetry (the Sohncke enantiomer
case) and cross-reference matches under Z' > 1, which in practice signal missed crystal symmetry.
"""
from collections import deque

import numpy as np
from scipy.spatial.distance import pdist

from ....core.sym_ops import SymOpList
from .. import context
from ..equivalence import merge
from ..invariants import base_reference, fingerprints_match, is_mono_reference
from .cloud import same_multimer_geom
from .congruence import same_multimer

#: sweep window (Angstroms) on summed member-centroid separations; congruent clusters agree to ~1e-3 per pair
WINDOW = 0.05


def is_needed() -> bool:
    """Return True iff a congruence gate is open or two references share an atomic composition

    :returns: whether :func:`merge_congruent` can find anything
    """
    if any(context.check_congruence_by_ref):
        return True
    compositions = [tuple(sorted(z.tolist())) for z in context.Z_by_ref]
    return len(set(compositions)) < len(compositions)


def are_congruent(g_N: SymOpList, h_N: SymOpList) -> bool:
    """Return True iff two clusters are congruent outside the crystal symmetry

    :param g_N: first cluster
    :param h_N: second cluster
    :returns: True iff the clusters represent the same multimer
    """
    if not fingerprints_match(g_N, h_N):
        return False
    if is_mono_reference(g_N) and is_mono_reference(h_N) and base_reference(g_N) == base_reference(h_N):
        i = base_reference(g_N)
        return context.check_congruence_by_ref[i] and same_multimer(g_N, h_N)
    return same_multimer_geom(g_N, h_N)


def _spread(g_N: SymOpList) -> float:
    """Return the summed pairwise member-centroid distance, an isometry invariant used to sort clusters

    :param g_N: cluster to measure
    :returns: summed centroid separations in Angstroms
    """
    cens = [
        context.active_crystal.to_cartesian(op.apply(context.references[r].centroid_frac))
        for r, op in zip(g_N.ref_indices, g_N)
    ]
    return float(pdist(np.array(cens)).sum())


def merge_congruent(
    unique: list[SymOpList],
) -> tuple[list[SymOpList], list[SymOpList]]:
    """Merge congruent representatives with a tolerance-window sweep (no rounding boundaries)

    :param unique: representatives from the exact pass
    :returns: a tuple ``(unique, duplicate)`` of cluster lists
    """
    spread = [_spread(g) for g in unique]
    groups: list[list[SymOpList]] = []
    roots: deque[tuple[float, int]] = deque()  # (spread, group index) of each group's first member, ascending
    for i in sorted(range(len(unique)), key=spread.__getitem__):
        # compare against group roots inside the window only; first match wins
        while roots and spread[i] - roots[0][0] > WINDOW:
            roots.popleft()
        for _, k in roots:
            if are_congruent(groups[k][0], unique[i]):
                groups[k].append(unique[i])
                break
        else:
            roots.append((spread[i], len(groups)))
            groups.append([unique[i]])

    reps, duplicates = [], []
    for group in groups:
        rep, dups = merge(group)
        reps.append(rep)
        duplicates.extend(dups)
    return reps, duplicates
