"""Neighbour enumeration: build the candidate N-mers to be deduplicated.

Anchors each reference monomer in turn, collects the symmetry- and
translation-generated partners within the cutoff, and forms every
``C(M, N-1)`` combination. Deduplication of the result is :mod:`.dedup`'s job.
"""
from itertools import combinations, product
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm
from numpy.typing import NDArray

from ..core.crystal import Crystal
from ..core.multimer import Monomer
from ..core.space_group import SITE_ATOL
from ..core.sym_ops import SymOp, SymOpList
from .dedup import context
from .dedup.equivalence import deduplicate_bucket
from .dedup.context import init_worker
# below this much pairwise work, deduplicate serially even when ``n_jobs > 1`` to avoid overhead
PARALLEL_MIN_PAIR_WORK: Final[int] = 50_000

def _distance_mode(kwargs: dict) -> str:
    """Resolve the distance convention once, rather than per candidate.

    :param kwargs: The caller's keyword arguments
    :returns: ``"origin"``, ``"com"`` or ``"contact"``
    """
    if kwargs.get("use_origin"):
        return "origin"
    if kwargs.get("use_com", False):
        return "com"
    return "contact"


def _neighbor_distance(
    cart_mon: NDArray, partner_masses: NDArray, monomer: Monomer, mode: str
) -> float:
    """Return the cutoff distance for a partner monomer.

    :param cart_mon: The partner monomer's Cartesian coordinates
    :param partner_masses: The *partner's* atomic masses, aligned with
        ``cart_mon``. These must come from the partner's own base reference:
        under ``Z' > 1`` the anchor and the partner can be different molecules
        (or the same molecule with a different atom ordering), so weighting the
        partner's coordinates by the anchor's masses gives a wrong center of
        mass, or a shape mismatch for a genuine co-crystal.
    :param monomer: The anchor monomer
    :param mode: The distance convention, as resolved by
        :func:`_distance_mode`: ``"origin"`` (closest partner atom to the anchor
        centroid), ``"com"`` (partner centre of mass to anchor centroid), or
        ``"contact"`` (closest atom-pair contact)
    :returns: The partner-to-anchor distance in Angstroms
    """
    if mode == "origin":
        return float(np.linalg.norm(cart_mon - monomer.centroid_cart, axis=1).min())
    if mode == "com":
        com_mon = np.average(cart_mon, axis=0, weights=partner_masses)
        return float(np.linalg.norm(com_mon - monomer.centroid_cart))
    return float(cdist(monomer.cart_coords, cart_mon).min())


def generate_neighbors(
    crystal: Crystal,
    monomers: list[Monomer],
    cutoff: NDArray,
    R: float,
    N: int,
    **kwargs,
) -> tuple[list[SymOpList], list[SymOpList]]:
    """Generate all unique N-mers in the supercell defined by ``cutoff`` by
    anchoring each reference monomer in turn, then deduplicate via a
    centroid-distance bucketing and :func:`are_equivalent`.

    :param crystal: The crystal containing the monomers
    :param monomers: The reference monomers (one per crystal asymmetric unit);
        each is anchored in turn as the multimer's first member
    :param cutoff: Per-axis ``(low, high)`` integer lattice translation bounds
    :param R: Distance cutoff in Angstroms. The distance convention is selected
        by ``kwargs`` (see :func:`_neighbor_distance`).
    :param N: N-mer order (``N=2`` for dimers, ``N=3`` for trimers, ...)
    :param kwargs: Additional options (``debug``, ``n_jobs``, and the distance
        convention flags consumed by :func:`_neighbor_distance`)
    :returns: A tuple ``(unique, duplicate)`` of cluster lists
    """
    verbose = kwargs.get("verbose", 1)
    mode = _distance_mode(kwargs)
    site_atol = kwargs.get("site_atol", SITE_ATOL)

    translations = list(product(*[range(cm, cp + 1) for cm, cp in cutoff]))

    # Bind per-reference state (workers below get the same via init_worker).
    context.bind(crystal, monomers, site_atol)

    # Full partner pool across ALL references: (base_index j, SymOp placing monomers[j]).
    pool = []
    for j, mon in enumerate(monomers):
        sym_ops = crystal.space_group.get_unique_sym_ops(
            mon.centroid_frac, atol=site_atol
        )
        for op, tr in product(sym_ops, translations):
            pool.append((j, op.translate(np.array(tr, dtype=np.int16) * 12)))

    # Anchor each reference i; partners = pool members within R of anchor i
    # (excluding anchor i's own zero-translation identity).
    #
    # Note the cutoff is anchor<->partner only: for N > 2 a candidate's partners
    # are each within R of the anchor but may be arbitrarily far from *each
    # other*. This anchor-centric convention is intentional, not an oversight.
    candidates: list[SymOpList] = []
    for i, anchor in enumerate(monomers):
        near = []
        for (j, op) in pool:
            if j == i and op == SymOp.identity():
                continue
            partner_cart = crystal.to_cartesian(op.apply(monomers[j].frac_coords))
            if _neighbor_distance(partner_cart, context.mass_by_ref[j], anchor, mode) <= R:
                near.append((j, op))
        if verbose:
            print(f"Reference {i}: {len(near)} monomers within {R} Angstroms.")
        for partner_combo in combinations(near, N - 1):
            members = [(i, SymOp.identity()), *partner_combo]
            ops = [op for (_, op) in members]
            refs = np.array([b for (b, _) in members], dtype=np.int64)
            member_cens = np.array(
                [crystal.to_cartesian(op.apply(monomers[b].centroid_frac))
                 for (b, op) in members]
            )
            centroid_frac = np.mean(
                [op.apply(monomers[b].centroid_frac) for (b, op) in members], axis=0
            )
            sop = SymOpList(ops, 1)
            sop.ref_indices = refs
            sop.centroid_frac = centroid_frac
            sop.centroid_cart = crystal.to_cartesian(centroid_frac)
            # bucket key = sorted rounded pairwise member-centroid distances
            # (an isometry invariant of the cluster, and valid across
            # references — a key built only from SymOp translations would not be).
            # Deliberately coarse (Euclidean Angstroms, 2 dp) and over-inclusive:
            # equivalent clusters' centroid distances agree to ~1e-9, so a 0.01 A
            # bin never straddles a rounding boundary in practice, while any
            # extra within-bucket collisions are harmless — correctness is
            # decided by are_equivalent, not by the bucket key itself.
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

    return unique, duplicate
