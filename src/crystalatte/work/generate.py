import warnings

import numpy as np
from tqdm import tqdm

from ..core.crystal import Crystal
from ..core.multimer import Monomer, Multimer
from ..core.space_group import SITE_ATOL
from ..core.sym_ops import SymOpList
from .dedup import context
from .neighbors import generate_neighbors


def _process_multimer(g_N: SymOpList) -> Multimer:
    """Construct a :class:`Multimer` from a deduped candidate cluster

    :param g_N: Deduplicated cluster to materialize
    :returns: Constructed :class:`~crystalatte.core.multimer.Multimer`
    """
    refs = g_N.ref_indices
    ops = list(g_N)

    mons: list[Monomer] = []
    for k, g in enumerate(ops):
        base = context.references[refs[k]]
        mon_frac = g.apply(base.frac_coords)
        mon_cart = context.active_crystal.to_cartesian(mon_frac)
        com_frac = g.apply(base.centroid_frac)
        com_cart = context.active_crystal.to_cartesian(com_frac)
        mons.append(Monomer(base.symbols, mon_frac, mon_cart, com_frac, com_cart, g))
    return Multimer(mons, g_N, g_N.multiplicity)


def generate(
    crystal: Crystal, N: int, R: float, *, detect_symmetry=True, symprec=1e-3, **kwargs
) -> list[Multimer]:
    """Generate unique multimers of every reference molecule in the crystal

    :param crystal: :class:`Crystal` structure to generate multimers from
    :param N: Number of monomers in the multimer
    :param R: Cutoff in Angstroms that every pair of monomers must satisfy
    :param detect_symmetry: If True, automatically detect and apply full symmetry
    :param symprec: ``spglib`` symmetry detection tolerance in Angstroms
    :param kwargs: Additional options forwarded to :func:`generate_neighbors`.
    :returns: List of unique multimers found in the crystal
    :raises ValueError: If ``N`` is not an integer ``>= 2``

    .. warning::
       Linear monomers are not currently supported. Linear molecules have only
       two rotational degrees of freedom, but a placement is stored as a full 3x3 
       rotation. The extra parameter is physically meaningless yet is still compared, 
       so two placements of the identical molecule can disagree and their multimers fail
       to merge
    """
    # check N validity
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 2:
        raise ValueError(
            f"N must be an integer >= 2; got {N!r}. For the individual reference molecules, "
            f"use Crystal.get_references() directly."
        )

    if detect_symmetry:
        crystal = crystal.detect_symmetry(symprec)
    else:
        warnings.warn(
            "symmetry detection is off; multimers related by symmetry the declared "
            "space group lacks are possibly not merged, so the result may contain duplicates",
            RuntimeWarning,
            stacklevel=2,
        )
    monomers = crystal.get_references()

    # check for linear monomers and warn
    for i, m in enumerate(monomers):
        pos = np.asarray(m.cart_coords, dtype=float)
        if np.linalg.matrix_rank(pos - pos.mean(axis=0), tol=1e-6) <= 1:
            warnings.warn(
                f"reference {i} is a linear molecule; multimer deduplication is "
                "known to be incomplete for linear monomers and the result will "
                "contain duplicates (see crystalatte.work.dedup)",
                RuntimeWarning,
                stacklevel=2,
            )

    # determine bounding box to fully contain all partners of all anchors
    radius = max(
        np.linalg.norm(m.cart_coords - m.centroid_cart, axis=1).max() for m in monomers
    )
    spacing = 1.0 / np.linalg.norm(np.linalg.inv(crystal.lattice_vectors), axis=1)
    reach = (R + 2 * radius) / spacing # maximum fractional distance to reach any partner of any anchor

    # generate full unit cell for comparison
    anchors = np.array([m.centroid_frac for m in monomers])
    images = np.array(
        [
            op.apply(m.centroid_frac)
            for m in monomers
            for op in crystal.space_group.sym_ops
        ]
    )

    # add max distance between monomers in unit cell to reach to ensure all partners are included
    # in BOTH directions
    lows = np.floor(anchors.min(axis=0) - images.max(axis=0) - reach)
    highs = np.ceil(anchors.max(axis=0) - images.min(axis=0) + reach)
    bounds = np.vstack((lows, highs)).T.astype(int)

    # convert symprec to fractional tolerance for site-symmetry tests
    d = np.linalg.norm(crystal.lattice_vectors, axis=0)
    kwargs.setdefault("site_atol", max(SITE_ATOL, float(symprec) / float(d.min())))

    # generate and deduplicate all unique N-mers in the supercell defined by ``bounds``
    unique_neighbors = generate_neighbors(crystal, monomers, bounds, R, N, **kwargs)

    # process each unique multimer into a Multimer object
    return [
        _process_multimer(g_N)
        for g_N in tqdm(
            unique_neighbors,
            total=len(unique_neighbors),
            desc="Processing unique multimers",
            leave=False,
            disable=not kwargs.get("verbose", 1),
        )
    ]
