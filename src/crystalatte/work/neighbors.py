from collections import defaultdict
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from tqdm import tqdm

from ..core.crystal import Crystal
from ..core.multimer import Monomer
from ..core.space_group import SITE_ATOL
from ..core.sym_ops import SymOp, SymOpList
from .dedup import congruence, context
from .dedup.equivalence import deduplicate_bucket
from .dedup.invariants import orbit_labels


def _distance_mode(kwargs: dict) -> str:
    """Resolve the desired distance convention

    :param kwargs: user keyword arguments
    :returns: ``"origin"``, ``"com"`` or ``"contact"``
    """
    # closest partner atom to anchor centroid
    if kwargs.get("use_origin"):
        return "origin"
    # closest atom-pair contact
    if kwargs.get("use_com", False):
        return "com"
    # default: closest atom-pair contact
    return "contact"


def _neighbor_distance(
    cart_mon: NDArray, partner_masses: NDArray, monomer: Monomer, mode: str
) -> float:
    """Return the cutoff distance for a partner monomer

    :param cart_mon: partner monomer's Cartesian coordinates
    :param partner_masses: partner's atomic masses, aligned with ``cart_mon``.
        (these must come from the partner's own base reference for ``Z' > 1``)
    :param monomer: anchor monomer
    :param mode: desired distance convention
    :returns: partner-to-anchor distance in Angstroms
    """
    if mode == "origin":
        return float(np.linalg.norm(cart_mon - monomer.centroid_cart, axis=1).min())
    if mode == "com":
        com_mon = np.average(cart_mon, axis=0, weights=partner_masses)
        return float(np.linalg.norm(com_mon - monomer.centroid_cart))
    return float(cdist(monomer.cart_coords, cart_mon).min())


def _partner_adjacency(
    carts: list[NDArray], masses: list[NDArray], R: float, mode: str
) -> NDArray:
    """Return which pairs of partners lie within ``R`` of each other under the distance convention

    The origin convention is anchor-centred, so for a partner pair it takes the smaller of the two
    closest-atom-to-centroid distances.

    :param carts: partner Cartesian coordinates
    :param masses: partner atomic masses, aligned with ``carts``
    :param R: distance cutoff in Angstroms
    :param mode: desired distance convention
    :returns: a symmetric ``(k, k)`` boolean matrix
    """
    coms = np.array([np.average(c, axis=0, weights=m) for c, m in zip(carts, masses)])
    D = cdist(coms, coms)
    if mode == "com":
        return D <= R

    # any pair beyond R + both radii is out of range; only the rest need an exact distance
    radii = np.array(
        [np.linalg.norm(c - com, axis=1).max() for c, com in zip(carts, coms)]
    )
    adj = np.zeros_like(D, dtype=bool)
    for p, q in zip(
        *np.nonzero(np.triu(D <= R + radii[:, None] + radii[None, :], k=1))
    ):
        if mode == "origin":
            d = min(
                np.linalg.norm(carts[q] - coms[p], axis=1).min(),
                np.linalg.norm(carts[p] - coms[q], axis=1).min(),
            )
        else:
            d = cdist(carts[p], carts[q]).min()
        adj[p, q] = adj[q, p] = d <= R
    return adj


def _cliques(adj: NDArray | None, n: int, size: int):
    """Yield sorted index tuples of ``size`` partners that are pairwise adjacent

    :param adj: partner adjacency matrix, or ``None`` when ``size == 1``
    :param n: number of partners
    :param size: number of partners per combination
    :returns: generator of index tuples
    """

    def extend(chosen: tuple, cands: NDArray):
        if len(chosen) == size:
            yield chosen
            return
        for x, k in enumerate(cands):
            rest = cands[x + 1 :]
            yield from extend(
                (*chosen, int(k)), rest[adj[k, rest]] if adj is not None else rest
            )

    yield from extend((), np.arange(n))


# TODO: split this into two functions, one for generating candidates and one for deduplicating them
def generate_neighbors(
    crystal: Crystal,
    monomers: list[Monomer],
    cutoff: NDArray,
    R: float,
    N: int,
    **kwargs,
) -> list[SymOpList]:
    """Generate and deduplicate all unique multimers in the supercell defined by ``cutoff``.
    Multimers are not actually constructed, only their symmetry operations are considered

    :param crystal: :class:`Crystal` containing the monomers
    :param monomers: Reference :class:`Monomer` instance(s) used to anchor the N-mer search
    :param cutoff: Per-axis ``(low, high)`` integer lattice translation bounds
    :param R: Distance cutoff in Angstroms, applied to every pair of members. The distance convention is selected by ``kwargs``
    :param N: N-mer order
    :param kwargs: Additional options (``verbose``, ``site_atol``, etc.)
    :returns: Unique clusters as a list of :class:`SymOpList` instances
    """
    verbose = kwargs.get("verbose", 1)
    mode = _distance_mode(kwargs)
    site_atol = kwargs.get("site_atol", SITE_ATOL)

    # generate all integer translations within the cutoff box
    translations = list(product(*[range(cm, cp + 1) for cm, cp in cutoff]))

    # bind per-reference state
    context.bind(crystal, monomers, site_atol)

    # full partner pool across ALL references
    pool = []
    for j, mon in enumerate(monomers):
        sym_ops = crystal.space_group.get_unique_sym_ops(
            mon.centroid_frac, atol=site_atol
        )
        for op, tr in product(sym_ops, translations):
            pool.append((j, op.translate(np.array(tr, dtype=np.int16) * 12)))

    # anchor each reference i
    candidates: list[SymOpList] = []
    identity = SymOp.identity()  # parse once; SymOps are never mutated, so members can share it
    with tqdm(
        total=len(monomers) * len(pool),
        desc="Generating candidate operations",
        leave=False,
        disable=not verbose,
    ) as pbar:
        for i, anchor in enumerate(monomers):
            near, carts = [], []
            for j, op in pool:
                pbar.update()
                if j == i and op == identity:
                    continue  # skip self-identity

                # filter partners by distance to anchor i
                partner_cart = crystal.to_cartesian(op.apply(monomers[j].frac_coords))
                if (_neighbor_distance(partner_cart, context.mass_by_ref[j], anchor, mode) <= R):
                    near.append((j, op))
                    carts.append(partner_cart)

            if verbose:
                tqdm.write(f"Reference {i}: {len(near)} monomers within {R} Angstroms.")

            # every pair of members must lie within R, so partners must also be within R of each other
            adj = None
            if N > 2:
                adj = _partner_adjacency(
                    carts, [context.mass_by_ref[j] for j, _ in near], R, mode
                )

            cliques = tqdm(
                _cliques(adj, len(near), N - 1),
                desc=f"Enumerating clusters for reference {i}",
                unit=" clusters",
                position=1,
                leave=False,
                disable=not verbose,
            )
            idx = np.array(list(cliques), dtype=np.int64)
            if len(idx) == 0:
                continue

            # gather every clique's members at once, the anchor's identity first; each candidate is a view
            C = len(idx)
            rot = np.concatenate(
                (np.broadcast_to(identity.rot, (C, 1, 3, 3)), np.stack([op.rot for _, op in near])[idx]),
                axis=1,
            )
            tr = np.concatenate(
                (np.broadcast_to(identity.tr, (C, 1, 3)), np.stack([op.tr for _, op in near])[idx]),
                axis=1,
            )
            refs = np.concatenate(
                (np.full((C, 1), i, dtype=np.int64), np.array([j for j, _ in near], dtype=np.int64)[idx]),
                axis=1,
            )
            candidates.extend(SymOpList.from_components(rot[c], tr[c], 1, refs[c]) for c in range(C))

    # exact pass: clusters sharing an orbit key are the same multimer
    buckets: dict[int, list[SymOpList]] = defaultdict(list)
    for label, g_t in zip(orbit_labels(candidates).tolist(), candidates):
        buckets[label].append(g_t)

    unique: list[SymOpList] = []
    duplicate: list[SymOpList] = []
    for bucket in tqdm(
        buckets.values(), 
        desc="Deduplicating candidates with exact integer checks", 
        leave=False,
        disable=not verbose
    ):
        local_unique, local_dup = deduplicate_bucket(bucket)
        unique.extend(local_unique)
        duplicate.extend(local_dup)

    # congruence pass: congruences outside the crystal symmetry
    if congruence.is_needed():
        unique, local_dup = congruence.merge_congruent(unique)
        duplicate.extend(local_dup)

    if verbose:
        print(
            f"Found {len(duplicate)} equivalent multimers from neighboring unit cell search."
        )

    return unique
