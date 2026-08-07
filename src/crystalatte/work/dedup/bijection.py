"""Cheapest-first routing between the equivalence tests, and bucket dedup.

This module owns the decision of WHICH matcher applies to a given pair; the
matchers themselves make no routing decisions.
"""
import numpy as np

from ...core.sym_ops import SymOpList
from . import context
from . import matchers
from .invariants import _base, _fingerprints_match, _mono

def _is_bijection(g_N: SymOpList, h_N: SymOpList) -> bool:
    """Return True iff ``g_N`` and ``h_N`` represent the same multimer.

    Stages, cheapest first:

    0. Distance-fingerprint gate (:func:`_fingerprints_match`) — a cached,
       vectorized necessary condition rejecting most within-bucket pairs.
    1. Mono-reference (both clusters drawn from the same single base
       reference): exact-integer matcher (:func:`~crystalatte.work.dedup.matchers._same_multimer_int`) — the
       complete, roundoff-free decision for every crystallographic
       (same-orbit) merge, modding out that reference's site symmetry
       (``context._W_lin_by_ref[i]``). Subsumes the former exact matrix branch.
       Falls back to the float congruence matcher (:func:`~crystalatte.work.dedup.matchers._same_multimer`)
       only when the integer matcher fails and that reference's molecular
       symmetry is richer than its site symmetry
       (``context._check_congruence_by_ref[i]``). It uses the FULL molecular
       self-isometry group, so it merges every energetically-equivalent
       congruence: proper molecular-symmetry congruences AND mirror-image
       enantiomers. This is intentional and goes beyond an enantiomer-only
       check; the dedup target is energetic equivalence.
    2. Cross-reference (members drawn from different base references, or the
       two clusters' base references differ): labeled point-cloud congruence
       (:func:`~crystalatte.work.dedup.matchers._same_multimer_geom`) — reflection-allowed Kabsch matching on
       the reconstructed, Z-labeled Cartesian atom clouds.

    :param g_N: The first cluster
    :param h_N: The second cluster
    :returns: True iff the clusters represent the same multimer
    """
    if not _fingerprints_match(g_N, h_N):
        return False
    if _mono(g_N) and _mono(h_N) and _base(g_N) == _base(h_N):
        i = _base(g_N)
        if matchers._same_multimer_int(g_N, h_N, context._W_lin_by_ref[i], context._W_tr_by_ref[i]):
            return True
        if context._check_congruence_by_ref[i] and matchers._same_multimer(g_N, h_N):
            return True
        return False
    return matchers._same_multimer_geom(g_N, h_N)


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
