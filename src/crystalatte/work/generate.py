import math
from itertools import combinations, product
from collections import defaultdict

import numpy as np
import qcelemental as qcel
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm
from numpy.typing import NDArray

from ..core.crystal import Crystal
from ..core.multimer import Monomer, Multimer
from ..core.sym_ops import SymOp, SymOpList


def _is_bijection(g_N, h_N):
    n = len(g_N)

    # Precompute flattened cache of target matrices for quick matching
    g_fp = g_N.aug_cache.reshape(n, -1)

    # We want to test every operation in h_N composed with every operation in g_N.
    composed = g_N.aug_cache[:, None, :, :] @ h_N.aug_cache[None, :, :, :]

    # Loop over each 'candidate' transformation h_i (outer dimension), skipping the identity (i=0) since it will just reproduce g_N and we want to find a non-trivial bijection
    for i in range(1, n):
        # Shape: (n, 4, 4) -> represents (h_i * g_j) for all j
        h_i_composed = composed[i]
        composed_fp = h_i_composed.reshape(n, -1)

        # Check if each composed result exists inside the original g_N set
        in_g = np.all(composed_fp[:, None, :] == g_fp[None, :, :], axis=-1).any(axis=-1)

        if not in_g.all():
            continue

        # Injectivity check
        unique_rows = np.unique(composed_fp, axis=0)
        if len(unique_rows) == n:
            return True

    return False


def _process_bucket(bucket: list[SymOpList]) -> tuple[list[SymOpList], list[SymOpList]]:
    """Given a bucket of SymOpLists that share the same translation fingerprint, identify which are unique multimers and which are duplicates based on bijections between their symmetry operations.

    :param bucket: A list of SymOpList objects that share the same translation fingerprint.
    :returns: A tuple of (unique_multimers, duplicate_multimers) where each is a list of SymOpList representing the unique and duplicate multimers found in the bucket.
    """
    local_unique: list[SymOpList] = []
    local_dup: list[SymOpList] = []
    for g_t in bucket:
        for u in local_unique:
            if _is_bijection(g_t, u):
                # sometimes g_t can be closer (even though the distance matrix is the same) and this can lead to a missed unique multimer when filtering by radius
                if (
                    np.linalg.norm(g_t.translations, axis=1).sum()
                    < np.linalg.norm(u.translations, axis=1).sum()
                ):
                    local_dup.append(u)
                    local_unique.remove(u)
                    local_unique.append(g_t)
                    g_t.multiplicity = u.multiplicity + 1
                else:
                    local_dup.append(g_t)
                    u.multiplicity += 1

                break
        else:
            local_unique.append(g_t)

    return local_unique, local_dup


def _generate_neighbors(
    crystal: Crystal,
    monomer: Monomer,
    cutoff: NDArray,
    R: float,
    N: int,
    **kwargs,
) -> tuple[list[SymOpList], list[SymOpList]]:
    """Generate all unique multimers in the central unit cell and neighboring cells, and identify duplicates among them.

    :param crystal: The crystal structure containing the monomer.
    :type crystal: Crystal

    :param N: The number of monomers in the multimer (e.g. N=2 for dimers).
    :type N: int

    :returns: A tuple of (unique_multimers, duplicate_multimers) where each is a list of SymOpList representing the unique and duplicate multimers found in the central and neighboring unit cells.
    """
    # Get the symmetry operations that are unique for the monomer's position in the unit cell. This accounts for special Wyckoff positions where some symmetry operations may map the monomer onto itself and therefore not generate a new multimer.
    sym_ops = crystal.space_group.get_unique_sym_ops(monomer.centroid_frac)

    if kwargs.get("debug", False):
        translations = list(product([-1, 0, 1], repeat=3))
    else:
        translations = list(product(*[range(cm, cp + 1) for cm, cp in cutoff]))
    op_tr_pairs = list(product(sym_ops, translations))

    # drop the identity operation with zero translation since it will be the reference multimer we compare against and doesn't need to be generated
    op_tr_pairs = [
        (pair[0], np.array(pair[1], dtype=np.int16) * 12)
        for pair in op_tr_pairs
        if not (pair[0] == SymOp.identity() and all(t == 0 for t in pair[1]))
    ]
    # actually translate the symops
    full_ops: list[SymOp] = [pair[0].translate(pair[1]) for pair in op_tr_pairs]
    candidates: list[SymOpList] = []

    # further prune based on distance to reference monomer
    frac_mons = [op.apply(monomer.frac_coords) for op in full_ops]
    cart_mons = [crystal.to_cartesian(m) for m in frac_mons]

    delete_indices = []
    for i, cart_mon in enumerate(cart_mons):
        if kwargs.get("use_com", False):
            masses = np.array(
                [qcel.periodictable.to_mass(sym) for sym in monomer.symbols]
            )
            com_mon = np.average(cart_mon, axis=0, weights=masses)
            dist = np.linalg.norm(com_mon - monomer.centroid_cart)
        else:
            dist = cdist(monomer.cart_coords, cart_mon).min()

        if dist > R:
            delete_indices.append(i)

    full_ops = [op for i, op in enumerate(full_ops) if i not in delete_indices]

    # Expensive for large N, but it's the price we pay
    for g_N in tqdm(
        combinations(full_ops, N - 1),
        total=len(full_ops) ** (N - 1) // math.factorial(N - 1),
        desc="Generating candidates",
        leave=False,
    ):
        candidates.append(SymOpList([SymOp.identity()] + list(g_N), 1))

    # Group candidates by a fast structural key (sorted tuple of translation norms), then only run _is_bijection within each bucket.
    buckets: dict[int, list[SymOpList]] = defaultdict(list)
    for g_t in candidates:
        key = g_t.translation_fp
        buckets[key].append(g_t)

    # Buckets are independent, process is parallelized if desired
    unique: list[SymOpList] = []
    duplicate: list[SymOpList] = []

    if "n_jobs" in kwargs and kwargs["n_jobs"] > 1:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import cpu_count

        # make sure n_jobs is a reasonable number
        if kwargs["n_jobs"] > len(buckets):
            kwargs["n_jobs"] = len(buckets)
        elif kwargs["n_jobs"] > cpu_count():
            kwargs["n_jobs"] = cpu_count() - 1

        with ProcessPoolExecutor(max_workers=kwargs["n_jobs"]) as ex:
            results = list(
                tqdm(
                    ex.map(_process_bucket, buckets.values()),
                    total=len(buckets),
                    desc="Processing across unique translation buckets",
                    leave=False,
                )
            )

        for local_unique, local_dup in results:
            unique.extend(local_unique)
            duplicate.extend(local_dup)

    else:
        for bucket in tqdm(
            buckets.values(),
            total=len(buckets),
            desc="Processing across unique translations",
            leave=False,
        ):
            local_unique, local_dup = _process_bucket(bucket)
            unique.extend(local_unique)
            duplicate.extend(local_dup)

    print(
        f"Found {len(duplicate)} equivalent multimers from neighboring unit cell search."
    )

    return unique, duplicate


def _init_multimers(crystal_, monomer_, R_, masses_, use_com_):
    """Initialize global variables for use in parallel processing of multimers."""
    global crystal, monomer, R, masses, use_com
    crystal = crystal_
    monomer = monomer_
    R = R_
    masses = masses_
    use_com = use_com_


def _process_multimer(
    g_N: SymOpList,
) -> Multimer | None:
    """Given a SymOpList representing the symmetry operations that generate a candidate multimer, construct the corresponding Multimer object if it is valid

    :param g_N: A SymOpList representing the symmetry operations that generate a candidate multimer.
    :type g_N: SymOpList

    :returns: A Multimer object representing the multimer generated by g_N if it is valid, None otherwise.
    :rtype: Multimer | None
    """
    global crystal, monomer, R, masses, use_com

    # if there are any refs with identical symops, skip multimer (multiple identities)
    if len(set(g_N)) != len(g_N):
        return None

    mons = []

    for g in g_N:
        mon_g_frac = g.apply(monomer.frac_coords)
        mon_g_cart = crystal.to_cartesian(mon_g_frac)
        com_g_frac = np.average(mon_g_frac, axis=0, weights=masses)
        com_g_cart = crystal.to_cartesian(com_g_frac)

        mon_g = Monomer(
            monomer.symbols,
            mon_g_frac,
            mon_g_cart,
            com_g_frac,
            com_g_cart,
            g,
        )

        mons.append(mon_g)

    return Multimer(mons, g_N, g_N.multiplicity)


def generate(
    crystal: Crystal, monomer: Monomer, N: int, R: float, **kwargs
) -> list[Multimer]:
    """Generate unique multimers of the given monomer in the crystal.

    :param crystal: The crystal structure containing the monomer.
    :type crystal: Crystal

    :param monomer: The reference monomer to generate multimers from.
    :type monomer: Monomer

    :param N: The number of monomers in the multimer (e.g. N=2 for dimers).
    :type N: int

    :param R: The maximum center-to-center distance in Angstroms for monomers to be considered part of the same multimer.
    :type R: float

    :returns: A list of unique Multimer objects representing the multimers found in the crystal.
    :rtype: list[Multimer]
    """
    d = np.linalg.norm(crystal.lattice_vectors, axis=0)
    t = monomer.centroid_frac

    # effective bounds to search (accounting for monomer extent)
    R_eff = R + pdist(monomer.cart_coords).max()
    low_bounds = -np.ceil((R_eff - t * d) / d)
    high_bounds = np.ceil((R_eff - (1 - t) * d) / d)
    bounds = np.vstack((low_bounds, high_bounds)).T
    bounds = bounds.astype(int)

    unique_neighbors, duplicate_neighbors = _generate_neighbors(
        crystal, monomer, bounds, R, N, **kwargs
    )

    masses = np.array([qcel.periodictable.to_mass(sym) for sym in monomer.symbols])
    _init_multimers(crystal, monomer, R, masses, kwargs.get("use_com", False))

    # if requested, process multimers in parallel
    # only do in parallel if there are a large number of unique neighbors to process, otherwise the overhead of parallelization may outweigh the benefits
    if "n_jobs" in kwargs and kwargs["n_jobs"] > 1 and False:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import cpu_count

        # make sure n_jobs is a reasonable number
        if kwargs["n_jobs"] > len(unique_neighbors):
            kwargs["n_jobs"] = len(unique_neighbors)
        elif kwargs["n_jobs"] > cpu_count():
            kwargs["n_jobs"] = cpu_count() - 1

        with ProcessPoolExecutor(
            initializer=_init_multimers,
            initargs=(crystal, monomer, R, masses, kwargs.get("use_com", False)),
            max_workers=kwargs["n_jobs"],
        ) as ex:
            multimers = list(
                tqdm(
                    ex.map(_process_multimer, unique_neighbors),
                    total=len(unique_neighbors),
                    desc="Processing unique multimers",
                    leave=False,
                )
            )
            multimers = [m for m in multimers if m is not None]

    else:
        multimers = []
        for g_N in tqdm(
            unique_neighbors,
            total=len(unique_neighbors),
            desc="Processing unique multimers",
            leave=False,
        ):
            multimer = _process_multimer(g_N)
            if multimer is not None:
                multimers.append(multimer)

    return multimers


if __name__ == "__main__":
    pass

"""
things to add still:
    multiple molecules in P1 unit cell 
"""
