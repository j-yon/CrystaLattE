import math
from itertools import combinations, product
from collections import defaultdict

import numpy as np
import qcelemental as qcel
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm

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
    cutoff: list[tuple[int, int]],
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
    :rtype: tuple[list[SymOpList], list[SymOpList]]
    """
    sym_ops = crystal.space_group.sym_ops  # TODO: get power/inverse too
    if "debug" in kwargs and kwargs["debug"]:
        translations = list(product([-1, 0, 1], repeat=3))
        # cutoff = [(-3, 3), (-6, 6), (-2, 2)]
        # translations = list(product(*[range(cm - 1, cp + 2) for cm, cp in cutoff]))
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

    # Expensive for large N, but its the price we pay
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


# def _generate_distant(
#     unique_neighbors: list[SymOpList],
#     duplicate_neighbors: list[SymOpList],
#     cutoff: list[int],
#     N: int,
# ) -> list[SymOpList]:
#     # finally, generate all translations of the unique central unit cell multimers within the cutoff using previous neighbor translations to determine further equivalencies
#     final_unique = []
#     distances = product(*[range(-c, c) for c in cutoff])

#     for i, g_N in enumerate(unique_neighbors):
#         duplicate_d = []
#         for d in product(distances, repeat=N - 1):
#             d = np.array(d, dtype=np.float64)
#             d = np.insert(d, 0, [0, 0, 0], axis=0)

#             # now check if translations are a multiple of a duplicate neighbor translation and if rotations are equal, if so skip
#             dup = False
#             for neighbor in duplicate_neighbors:
#                 if g_N.rotation_equality(neighbor) and any(
#                     np.array_equal(d, dup_d) for dup_d in neighbor_translations
#                 ):
#                     dup = True
#                     break
#             if dup:
#                 continue

#             # check for duplicates in opposite directions by pure translation
#             if any(np.array_equal(d, duplicate_d_i) for duplicate_d_i in duplicate_d):
#                 continue
#             else:
#                 duplicate_d.append(-d)

#             g_T = g_N.translate_list(d)
#             for neighbor in unique_neighbors:
#                 if _is_multiple_translation(g_T.rotations, d, neighbor):
#                     g_T.multiplicity = max(g_T.multiplicity, neighbor.multiplicity)
#                     break

#             final_unique.append(g_T)

#         return final_unique


# def _is_multiple_translation(
#     rot: list[NDArray], tr: NDArray, neighbor: SymOpList
# ) -> bool:
#     for rot_i, rot_neighbor in zip(rot, neighbor.rotations):
#         if not all(
#             np.array_equal(rot_i, rot_neighbor_i)
#             for rot_i, rot_neighbor_i in zip(rot, neighbor.rotations)
#         ):
#             return False

#     for tr_i, tr_neighbor in zip(tr, neighbor.translations):
#         # only check nonzero translation components to avoid division by zero
#         nonzero_i = np.abs(tr_i) > 1e-6
#         nonzero_n = np.abs(tr_neighbor) > 1e-6

#         # if the nonzero components are different, they can't be multiples
#         if not np.array_equal(nonzero_i, nonzero_n):
#             return False

#         # if there are no nonzero components, then the translations are effectively the same and we can skip the ratio check
#         if not np.any(nonzero_i):
#             continue

#         # check to see if all ratios are equivalent
#         if not np.allclose(
#             tr_i[nonzero_i] / tr_neighbor[nonzero_n],
#             tr_i[nonzero_i][0] / tr_neighbor[nonzero_n][0],
#             1e-6,
#         ):
#             return False

#         # ratios are equivalent, but if they are negative then the translations are in opposite directions and can't be multiples
#         if tr_i[nonzero_i][0] / tr_neighbor[nonzero_n][0] < 0:
#             return False

#     return True


def _init_multimers(crystal_, monomer_, R_, masses_, use_com_):
    # This is needed to avoid issues with pickling large objects like the crystal and monomer when using ProcessPoolExecutor
    global crystal, monomer, R, masses, use_com
    crystal = crystal_
    monomer = monomer_
    R = R_
    masses = masses_
    use_com = use_com_


def _process_multimer(
    g_N: SymOpList,
) -> Multimer | None:
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
    """
    Generate unique multimers of the given monomer in the crystal.

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
    lat_vecs = crystal.lattice_vectors

    V = crystal.volume
    b1 = np.cross(lat_vecs[:, 1], lat_vecs[:, 2]) / V
    b2 = np.cross(lat_vecs[:, 2], lat_vecs[:, 0]) / V
    b3 = np.cross(lat_vecs[:, 0], lat_vecs[:, 1]) / V

    # Perpendicular face-to-face distances
    d = np.array(
        [
            1.0 / np.linalg.norm(b1),
            1.0 / np.linalg.norm(b2),
            1.0 / np.linalg.norm(b3),
        ]
    )

    # Distance from reference point to each face (+ and - sides)
    # t[i] * d[i]         = distance to the "lower" face in direction i
    # (1 - t[i]) * d[i]   = distance to the "upper" face in direction i
    t = monomer.centroid_frac
    dist_lower = t * d  # gap between ref and the face behind it
    dist_upper = (1 - t) * d  # gap between ref and the face ahead of it

    # effective radius to account for monomer extent
    R_eff = R + pdist(monomer.cart_coords).max()

    bounds = []
    for i in range(3):
        n_minus = (
            int(np.ceil((R_eff - dist_lower[i]) / d[i])) if R_eff > dist_lower[i] else 0
        )
        n_plus = (
            int(np.ceil((R_eff - dist_upper[i]) / d[i])) if R_eff > dist_upper[i] else 0
        )
        bounds.append((-n_minus, n_plus))

    unique_neighbors, duplicate_neighbors = _generate_neighbors(
        crystal, monomer, bounds, R, N, **kwargs
    )
    if "v1" in kwargs and kwargs["v1"]:
        return []

    # neighbor_translations = []
    # for neighbor in unique_neighbors:
    #     # get integer translations for each neighbor relative to the central unit cell
    #     tr = []
    #     for g in neighbor:
    #         tr.append([np.floor(x) if x >= 0 else np.ceil(x) for x in g.tr])

    #     neighbor_translations.append(tr)

    masses = np.array([qcel.periodictable.to_mass(sym) for sym in monomer.symbols])
    _init_multimers(crystal, monomer, R, masses, kwargs.get("use_com", False))

    # if requested, process multimers in parallel
    # only do in parallel if there are a large number of unique neighbors to process, otherwise the overhead of parallelization may outweigh the benefits
    # TODO: determine when this is actually faster
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
    computation of powers of symops
    inverses (would come with powers)
    molecular symmetry
"""
