from collections import defaultdict
from itertools import product
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm

from ..core.crystal import Crystal
from ..core.multimer import Monomer
from ..core.space_group import SITE_ATOL
from ..core.sym_ops import SymOp, SymOpList
from .dedup import context
from .dedup.context import init_worker
from .dedup.equivalence import deduplicate_bucket

# below this much pairwise work, deduplicate serially even when ``n_jobs > 1`` to avoid overhead
PARALLEL_MIN_PAIR_WORK: Final[int] = 50_000


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


def _partner_adjacency(carts: list[NDArray], masses: list[NDArray], R: float, mode: str) -> NDArray:
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
    radii = np.array([np.linalg.norm(c - com, axis=1).max() for c, com in zip(carts, coms)])
    adj = np.zeros_like(D, dtype=bool)
    for p, q in zip(*np.nonzero(np.triu(D <= R + radii[:, None] + radii[None, :], k=1))):
        if mode == "origin":
            d = min(np.linalg.norm(carts[q] - coms[p], axis=1).min(), np.linalg.norm(carts[p] - coms[q], axis=1).min())
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
            rest = cands[x + 1:]
            yield from extend((*chosen, int(k)), rest[adj[k, rest]] if adj is not None else rest)

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
    """Generate and deduplicate all unique N-mers in the supercell defined by ``cutoff`` 

    :param crystal: crystal containing the monomers
    :param monomers: reference monomer(s) used to anchor the N-mer search
    :param cutoff: per-axis ``(low, high)`` integer lattice translation bounds
    :param R: distance cutoff in Angstroms, applied to every pair of members. The distance convention is selected by ``kwargs``
    :param N: N-mer order
    :param kwargs: additional options (``n_jobs``, ``verbose``, etc.)
    :returns: unique cluster lists
    """
    verbose = kwargs.get("verbose", 1)
    mode = _distance_mode(kwargs)
    site_atol = kwargs.get("site_atol", SITE_ATOL)
    full_congruence = kwargs.get("full_congruence_check", False)

    # generate all integer translations within the cutoff box
    translations = list(product(*[range(cm, cp + 1) for cm, cp in cutoff]))

    # bind per-reference state
    context.bind(crystal, monomers, site_atol, full_congruence)

    # full partner pool across ALL references
    pool = []
    for j, mon in enumerate(monomers):
        sym_ops = crystal.space_group.get_unique_sym_ops(
            mon.centroid_frac, atol=site_atol
        )
        for op, tr in product(sym_ops, translations):
            pool.append((j, op.translate(np.array(tr, dtype=np.int16) * 12)))

    # anchor each reference i; partners = pool members within R of anchor i
    # excludes anchor i's own zero-translation identity
    candidates: list[SymOpList] = []
    for i, anchor in enumerate(monomers):
        near, carts = [], []
        for j, op in pool:
            if j == i and op == SymOp.identity():
                continue # skip self-identity

            # filter partners by distance to anchor i
            partner_cart = crystal.to_cartesian(op.apply(monomers[j].frac_coords))
            if (_neighbor_distance(partner_cart, context.mass_by_ref[j], anchor, mode) <= R):
                near.append((j, op))
                carts.append(partner_cart)

        if verbose:
            print(f"Reference {i}: {len(near)} monomers within {R} Angstroms.")

        # every pair of members must lie within R, so partners must also be within R of each other
        adj = None
        if N > 2:
            adj = _partner_adjacency(carts, [context.mass_by_ref[j] for j, _ in near], R, mode)

        for idx in _cliques(adj, len(near), N - 1):
            members = [(i, SymOp.identity()), *(near[k] for k in idx)]
            ops = [op for (_, op) in members]
            refs = np.array([b for (b, _) in members], dtype=np.int64)
            member_cens = np.array(
                [
                    crystal.to_cartesian(op.apply(monomers[b].centroid_frac))
                    for (b, op) in members
                ]
            )
            sop = SymOpList(ops, 1)
            sop.ref_indices = refs

            # bucket key = coarse pairwise member-centroid distances
            if len(member_cens) > 1:
                dfp = np.sort(np.round(pdist(member_cens), 2))
            else:
                dfp = np.array([0.0])
            sop._bucket_key = hash(dfp.tobytes())
            candidates.append(sop)

    # group candidates by bucket key for parallel deduplication
    buckets: dict[int, list[SymOpList]] = defaultdict(list)
    for g_t in candidates:
        buckets[g_t._bucket_key].append(g_t)

    unique: list[SymOpList] = []
    duplicate: list[SymOpList] = []

    # predict pair work by summing number of pairwise comparisons in each bucket
    pair_work = sum(len(b) * len(b) for b in buckets.values())

    # parallelize if the user requested it and the work is large enough to amortize the overhead
    if kwargs.get("n_jobs", 1) > 1 and pair_work >= PARALLEL_MIN_PAIR_WORK:
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        from multiprocessing import cpu_count

        # limit n_jobs to the number of buckets and available CPUs
        n_jobs = max(1, min(kwargs["n_jobs"], len(buckets), cpu_count() - 1))

        # chunk buckets to amortize pickling overhead
        chunksize = max(1, len(buckets) // (n_jobs * 8))
        try:
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=init_worker,
                initargs=(crystal, monomers, site_atol, full_congruence),
            ) as ex:
                # wrap the parallel map in tqdm for progress reporting
                results = list(
                    tqdm(
                        ex.map(
                            deduplicate_bucket, buckets.values(), chunksize=chunksize
                        ),
                        total=len(buckets),
                        desc="Processing buckets",
                        leave=False,
                        disable=not verbose,
                    )
                )
        except BrokenProcessPool as exc:
            raise RuntimeError(
                "the worker pool died during deduplication. The usual cause on macOS and Windows is calling "
                "generate(..., n_jobs > 1) from a script whose top level is not protected: wrap the call in "
                '`if __name__ == "__main__":`'
            ) from exc

        # combine results from all workers
        for local_unique, local_dup in results:
            unique.extend(local_unique)
            duplicate.extend(local_dup)

    else:
        # serial deduplication wrapped in tqdm for progress reporting
        for bucket in tqdm(
            buckets.values(),
            total=len(buckets),
            desc="Processing buckets",
            leave=False,
            disable=not verbose,
        ):
            local_unique, local_dup = deduplicate_bucket(bucket)
            unique.extend(local_unique)
            duplicate.extend(local_dup)

    if verbose:
        print(
            f"Found {len(duplicate)} equivalent multimers from neighboring unit cell search."
        )

    return unique
