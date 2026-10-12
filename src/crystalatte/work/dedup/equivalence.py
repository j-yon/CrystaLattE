import numpy as np

from ...core.sym_ops import SymOpList


def translation_norm(g_N: SymOpList) -> float:
    """Return the summed translation norm of a cluster's operations, used to pick representatives

    :param g_N: cluster to measure
    :returns: sum of the members' translation norms
    """
    return float(np.linalg.norm(g_N.translations, axis=1).sum())


def merge(group: list[SymOpList]) -> tuple[SymOpList, list[SymOpList]]:
    """Fold a group of equivalent clusters into one representative

    :param group: clusters known to represent the same multimer
    :returns: ``(representative, duplicates)``; the representative carries the group's total multiplicity
    """
    rep = min(group, key=translation_norm)
    rep.multiplicity = sum(g.multiplicity for g in group)
    return rep, [g for g in group if g is not rep]


def deduplicate_bucket(
    bucket: list[SymOpList],
) -> tuple[list[SymOpList], list[SymOpList]]:
    """Deduplicate one bucket of clusters sharing an :func:`~crystalatte.work.dedup.invariants.orbit_key`

    The key is exact, so every cluster in the bucket is the same multimer.

    :param bucket: clusters sharing a single orbit key
    :returns: a tuple ``(unique, duplicate)`` of cluster lists
    """
    rep, duplicates = merge(bucket)
    return [rep], duplicates
