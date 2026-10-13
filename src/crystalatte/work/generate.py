import warnings

import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm

from ..core.crystal import Crystal
from ..core.multimer import Monomer, Multimer
from ..core.space_group import SITE_ATOL
from ..core.sym_ops import SymOpList
from .dedup import context
from .neighbors import generate_neighbors


def _process_multimer(g_N: SymOpList) -> Multimer | None:
    """Construct a Multimer from a deduped candidate cluster

    :param g_N: deduplicated cluster to materialize
    :returns: constructed :class:`~crystalatte.core.multimer.Multimer`, or ``None`` if ``g_N`` contains identical members 
    """
    refs = g_N.ref_indices
    ops = list(g_N)

    # degeneracy: reject only if two members are the SAME (ref, op) pair
    seen = set(zip(refs.tolist(), ops))
    if len(seen) != len(ops):
        return None

    mons: list[Monomer] = []
    for k, g in enumerate(ops):
        base = context.references[refs[k]]
        masses = context.mass_by_ref[refs[k]]
        mon_frac = g.apply(base.frac_coords)
        mon_cart = context.active_crystal.to_cartesian(mon_frac)
        com_frac = np.average(mon_frac, axis=0, weights=masses)
        com_cart = context.active_crystal.to_cartesian(com_frac)
        mons.append(Monomer(base.symbols, mon_frac, mon_cart, com_frac, com_cart, g))
    return Multimer(mons, g_N, g_N.multiplicity)


def generate(
    crystal: Crystal, N: int, R: float, *, detect_symmetry=True, symprec=1e-3, **kwargs
) -> list[Multimer]:
    """Generate unique multimers of every reference molecule in the crystal

    :param crystal: crystal structure to generate multimers from
    :param N: number of monomers in the multimer 
    :param R: maximum center-to-center monomer distance in Angstroms 
    :param detect_symmetry: if True, automatically detect and apply symmetry before enumeration
    :param symprec: symmetry detection tolerance in Angstroms 
    :param kwargs: additional options forwarded to :func:`generate_neighbors`.
    :returns: list of unique multimers found in the crystal
    :raises ValueError: if ``N`` is not an integer ``>= 2``

    .. warning::
       **LINEAR MONOMERS ARE NOT SUPPORTED.** A linear molecule (CO2, N2,
       acetylene) has only two rotational degrees of freedom, but a placement is
       stored as a full 3x3 rotation. The extra parameter — the twist about the
       molecular axis — is physically meaningless yet is still compared, so two
       placements of the identical molecule can disagree and their multimers fail
       to merge. Deduplication is therefore INCOMPLETE for such crystals and the
       returned list will contain duplicates. A ``RuntimeWarning`` is emitted when this is
       detected. See the note in :mod:`crystalatte.work.dedup`.
    """
    # check N validity
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 2:
        raise ValueError(
            f"N must be an integer >= 2 (a multimer needs at least two "
            f"monomers); got {N!r}. For the individual reference molecules, "
            f"use Crystal.get_references() directly."
        )

    if detect_symmetry:
        crystal = crystal.detect_symmetry(symprec)
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

    # without detection, independent references can be symmetry copies the declared space group misses
    compositions = [tuple(sorted(m.symbols)) for m in monomers]
    if not detect_symmetry and len(set(compositions)) < len(compositions):
        warnings.warn(
            "symmetry detection is off and two references share an atomic composition; multimers related "
            "by symmetry the declared space group lacks are not merged, so the result can contain duplicates "
            "(their multiplicities are split, so lattice-energy sums are unaffected)",
            RuntimeWarning,
            stacklevel=2,
        )

    # effective radius for bounding box, needed for monoclinic/triclinic cells
    d = np.linalg.norm(crystal.lattice_vectors, axis=0)
    R_eff = R + max(pdist(m.cart_coords).max() for m in monomers)

    # create oversized bounding box 
    lows, highs = [], []
    for m in monomers:
        t = m.centroid_frac
        lows.append(-np.ceil((R_eff - t * d) / d))
        highs.append(np.ceil((R_eff - (1 - t) * d) / d))
    bounds = np.vstack((np.min(lows, axis=0), np.max(highs, axis=0))).T.astype(int)

    # convert symprec to fractional tolerance for site-symmetry tests
    kwargs.setdefault("site_atol", max(SITE_ATOL, float(symprec) / float(d.min())))

    # generate and deduplicate all unique N-mers in the supercell defined by ``bounds``
    unique_neighbors = generate_neighbors(crystal, monomers, bounds, R, N, **kwargs)

    # process each unique multimer into a Multimer object, parallelized if requested
    multimers: list[Multimer] = []
    for g_N in tqdm(
        unique_neighbors,
        total=len(unique_neighbors),
        desc="Processing unique multimers",
        leave=False,
        disable=not kwargs.get("verbose", 1),
    ):
        m = _process_multimer(g_N)
        if m is not None:
            multimers.append(m)

    return multimers
