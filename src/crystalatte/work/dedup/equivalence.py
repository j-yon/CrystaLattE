import numpy as np

from ...core.sym_ops import SymOpList
from . import cloud
from . import context
from . import matchers
from .invariants import base_reference, fingerprints_match, is_mono_reference

def are_equivalent(g_N: SymOpList, h_N: SymOpList) -> bool:
    """Return True iff ``g_N`` and ``h_N`` represent the same multimer

    Stages: distance-fingerprint gate, exact-integer matcher, then either the
    mono-reference Cartesian matcher or the cross-reference atom-cloud matcher.

    :param g_N: first cluster
    :param h_N: second cluster
    :returns: True iff the clusters represent the same multimer
    """
    # reject if the distance fingerprints differ
    if not fingerprints_match(g_N, h_N):
        return False

    # exact integer matcher: complete and roundoff-free for every
    # crystallographic (same-orbit) merge
    if matchers.same_multimer_int(
        g_N, h_N, context.W_lin_by_ref, context.W_tr_by_ref
    ):
        return True

    # mono-reference congruence matcher, only where the molecular symmetry is
    # richer than the site symmetry (see the gate in `context.bind`)
    if is_mono_reference(g_N) and is_mono_reference(h_N) and base_reference(g_N) == base_reference(h_N):
        i = base_reference(g_N)
        return context.check_congruence_by_ref[i] and matchers.same_multimer(g_N, h_N)

    # cross-reference fallback: reflection-allowed Kabsch on Z-labeled clouds
    return cloud.same_multimer_geom(g_N, h_N)


def deduplicate_bucket(
    bucket: list[SymOpList],
) -> tuple[list[SymOpList], list[SymOpList]]:
    """Deduplicate one bucket of clusters that share the same bucket key

    :param bucket: clusters sharing a single ``_bucket_key``
    :returns: a tuple ``(unique, duplicate)`` of cluster lists
    """
    local_unique: list[SymOpList] = []
    local_dup: list[SymOpList] = []
    for g_t in bucket:
        # search for equivalence in the existing unique representatives
        for idx, u in enumerate(local_unique):
            if not are_equivalent(g_t, u):
                continue 

            # promote the duplicate candidate with the smaller translation norm
            if (
                np.linalg.norm(g_t.translations, axis=1).sum()
                < np.linalg.norm(u.translations, axis=1).sum()
            ):
                # promote g_t in place
                local_dup.append(u)
                local_unique[idx] = g_t
                g_t.multiplicity = u.multiplicity + 1
            else:
                local_dup.append(g_t)
                u.multiplicity += 1
            break

        # found a new unique representative
        else:
            local_unique.append(g_t)

    return local_unique, local_dup
