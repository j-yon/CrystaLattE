from collections import defaultdict
from itertools import combinations, product
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from tqdm import tqdm

from ..core.crystal import Crystal
from ..core.multimer import Monomer
from ..core.space_group import SITE_ATOL
from ..core.sym_ops import SymOp, SymOpList
from .dedup import context, fallback
from .dedup.context import init_worker
from .dedup.equivalence import deduplicate_bucket
from .dedup.invariants import orbit_key

# below this much key work, compute orbit keys serially even when ``n_jobs > 1``: the pool costs ~1 s to start.
# One unit is a fast-path key (trivial site symmetry); a general-path key costs ~10 units.
PARALLEL_MIN_KEY_WORK: Final[int] = 75_000


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


def _orbit_keys(chunk: list[SymOpList]) -> list[tuple]:
    """Worker task: orbit keys for a chunk of candidates

    :param chunk: candidates to key
    :returns: their orbit keys, in order
    """
    return [orbit_key(g) for g in chunk]


def _compute_keys(
    candidates: list[SymOpList], n_jobs: int, initargs: tuple, verbose: int
) -> list[tuple]:
    """Compute orbit keys, in a process pool when requested and the workload amortizes it

    :param candidates: candidate clusters
    :param n_jobs: requested worker count
    :param initargs: arguments for :func:`~crystalatte.work.dedup.context.init_worker`
    :param verbose: show a progress bar if truthy
    :returns: one orbit key per candidate, in order
    """
    weight = 1 if all(len(W) == 1 for W, _, _ in context.W_stacked_by_ref) else 10
    if n_jobs <= 1 or weight * len(candidates) < PARALLEL_MIN_KEY_WORK:
        return [
            orbit_key(g)
            for g in tqdm(candidates, desc="Keying candidates", leave=False, disable=not verbose)
        ]

    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from multiprocessing import cpu_count

    n_jobs = max(1, min(n_jobs, cpu_count() - 1))
    size = -(-len(candidates) // (n_jobs * 8))  # ceil: ~8 chunks per worker
    chunks = [candidates[i:i + size] for i in range(0, len(candidates), size)]
    try:
        with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=init_worker, initargs=initargs
        ) as ex:
            results = list(
                tqdm(
                    ex.map(_orbit_keys, chunks),
                    total=len(chunks),
                    desc="Keying candidates",
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
    return [key for chunk in results for key in chunk]


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
    :param R: distance cutoff in Angstroms. The distance convention is selected by ``kwargs``
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
        near = []
        for j, op in pool:
            if j == i and op == SymOp.identity():
                continue # skip self-identity

            # filter partners by distance to anchor i
            partner_cart = crystal.to_cartesian(op.apply(monomers[j].frac_coords))
            if (_neighbor_distance(partner_cart, context.mass_by_ref[j], anchor, mode) <= R):
                near.append((j, op))

        if verbose:
            print(f"Reference {i}: {len(near)} monomers within {R} Angstroms.")

        # generate all combinations of N-1 partners from the nearby pool
        for partner_combo in combinations(near, N - 1):
            members = [(i, SymOp.identity()), *partner_combo]
            ops = [op for (_, op) in members]
            refs = np.array([b for (b, _) in members], dtype=np.int64)
            sop = SymOpList(ops, 1)
            sop.ref_indices = refs
            candidates.append(sop)

    # exact pass: clusters sharing an orbit key are the same multimer
    keys = _compute_keys(
        candidates, kwargs.get("n_jobs", 1), (crystal, monomers, site_atol, full_congruence), verbose
    )
    buckets: dict[tuple, list[SymOpList]] = defaultdict(list)
    for key, g_t in zip(keys, candidates):
        buckets[key].append(g_t)

    unique: list[SymOpList] = []
    duplicate: list[SymOpList] = []
    for bucket in buckets.values():
        local_unique, local_dup = deduplicate_bucket(bucket)
        unique.extend(local_unique)
        duplicate.extend(local_dup)

    # fallback pass: congruences outside the crystal symmetry
    if fallback.is_needed():
        unique, local_dup = fallback.merge_congruent(unique)
        duplicate.extend(local_dup)

    if verbose:
        print(
            f"Found {len(duplicate)} equivalent multimers from neighboring unit cell search."
        )

    return unique
