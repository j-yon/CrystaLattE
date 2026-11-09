import warnings

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ..core.crystal import Crystal
from ..core.multimer import Monomer, Multimer
from ..core.space_group import SITE_ATOL
from ..core.sym_ops import SymOpList
from .dedup import context
from .neighbors import generate_neighbors


def _process_multimers(clusters: list[SymOpList], verbose: int = 1) -> list[Multimer]:
    """Construct a :class:`Multimer` from each deduped candidate cluster

    Members are transformed in batches of one member slot and base reference, each product keeping the shape it
    has for a single monomer (``SymOp.apply`` and ``Crystal.to_cartesian``), so the coordinates and geometric
    means are bit-for-bit those of transforming each monomer on its own.

    :param clusters: Deduplicated clusters of one size to materialize
    :param verbose: Show a progress bar if nonzero
    :returns: Constructed :class:`~crystalatte.core.multimer.Multimer` objects, in the order of ``clusters``
    """
    if not clusters:
        return []
    rot = np.stack([g.rot_cache for g in clusters])
    tr = np.stack([g.tr_cache for g in clusters]) / 12
    refs = np.stack([g.ref_indices for g in clusters])
    U, n = refs.shape
    F2C = context.active_crystal.lattice_vectors  # the matrix to_cartesian applies

    frac: list[list[NDArray]] = [[None] * n for _ in range(U)]
    cart: list[list[NDArray]] = [[None] * n for _ in range(U)]
    com_frac = np.empty((U, n, 3))
    com_cart = np.empty((U, n, 3))
    for k in range(n):
        for r in np.unique(refs[:, k]).tolist():
            at = np.flatnonzero(refs[:, k] == r)
            base = context.references[r]
            f = np.swapaxes(rot[at, k] @ base.frac_coords.T, 1, 2) + tr[at, k, None]
            c = np.swapaxes(F2C @ np.swapaxes(f, 1, 2), 1, 2)
            com_frac[at, k] = rot[at, k] @ base.centroid_frac + tr[at, k]
            com_cart[at, k] = (F2C @ com_frac[at, k, :, None])[:, :, 0]
            for a, u in enumerate(at.tolist()):
                frac[u][k], cart[u][k] = f[a], c[a]

    # geometric mean of the centroid distances, in pdist's pair order and arithmetic
    i, j = np.triu_indices(n, 1)
    d = com_cart[:, i] - com_cart[:, j]
    prods = np.prod(np.sqrt(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]), axis=1)

    out = []
    for u, g_N in enumerate(
        tqdm(clusters, desc="Processing unique multimers", leave=False, disable=not verbose)
    ):
        mons = [
            Monomer(context.references[refs[u, k]].symbols, frac[u][k], cart[u][k], com_frac[u, k], com_cart[u, k], g)
            for k, g in enumerate(g_N.sym_ops)
        ]
        out.append(Multimer.from_parts(mons, g_N, g_N.multiplicity, float(prods[u] ** (1 / len(i)))))
    return out


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
    return _process_multimers(unique_neighbors, kwargs.get("verbose", 1))
