import numpy as np

from ....core.sym_ops import SymOpList
from .. import context
from ..invariants import relative_transforms
from ..matchers import anchored_match


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
        LG, vG, LH, vH,
        context.self_iso_by_ref,
        context.self_iso_tr_by_ref,
        refG, refH,
        lambda a, b: np.allclose(a, b, atol=atol),
        lambda s: s.T,  # orthogonal -> inverse is transpose
    )
