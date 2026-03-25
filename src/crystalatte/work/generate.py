import math
from itertools import combinations, product
from collections import defaultdict

import numpy as np
import qcelemental as qcel
from scipy.spatial.distance import cdist
from tqdm import tqdm

from ..core.crystal import Crystal
from ..core.multimer import Monomer, Multimer
from ..core.sym_ops import SymOp, SymOpList


def _is_bijection(g_N: SymOpList, h_N: SymOpList) -> bool:
    """Check if there exists a bijection between the sets of symmetry operations g_N and h_N.

    :param g_N: A SymOpList representing the first set of symmetry operations.
    :type g_N: SymOpList

    :param h_N: A SymOpList representing the second set of symmetry operations.
    :type h_N: SymOpList

    :returns: True if there exists a bijection between g_N and h_N, False otherwise.
    :rtype: bool
    """
    n = len(g_N)

    # Precompute augment matrix caches for g_N for fast comparison
    g_aug = g_N.aug_cache.copy()
    g_fp = g_aug.reshape(n, -1)

    for i in range(n):
        # composed = h_N.aug_cache @ g_N.aug_cache[i]  # (n, 4, 4) # doesnt work

        # Need to multiply h_rot by g_tr to get correct composed translation
        g_tr_rotated = np.tile(np.eye(4), (n, 1, 1))  # (n, 4, 4)
        g_tr_rotated[:, :3, :3] = g_N.rot_cache[i]
        g_tr_rotated[:, :3, 3] = h_N.rot_cache @ g_N.tr_cache[i]
        composed = h_N.aug_cache @ g_tr_rotated  # (n, 4, 4)
        composed_fp = composed.reshape(n, -1)

        # Check each composed result is in g_N:
        in_g = np.all(composed_fp[:, None, :] == g_fp[None, :, :], axis=-1).any(axis=-1)

        if not in_g.all():
            continue

        # Injectivity check
        unique_rows = np.unique(composed_fp, axis=0)
        if len(unique_rows) == n:
            return True  # Surjectivity is implied since |composed| == |g_N| == n

    return False


def _process_bucket(bucket: list[SymOpList]) -> tuple[list[SymOpList], list[SymOpList]]:
    local_unique: list[SymOpList] = []
    local_dup: list[SymOpList] = []
    for g_t in bucket:
        for u in local_unique:
            if _is_bijection(g_t, u):
                local_dup.append(g_t)
                u.multiplicity += 1
                break
        else:
            local_unique.append(g_t)
    return local_unique, local_dup


def _generate_neighbors(
    crystal: Crystal,
    cutoff: list[int],
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
    translations = list(product(*[range(-c, c + 1) for c in cutoff]))
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

    # Expensive for large N, but its the price we pay
    for g_N in tqdm(
        combinations(full_ops, N - 1),
        total=len(op_tr_pairs) ** (N - 1) // math.factorial(N - 1),
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
    # first identify equivalencies in the central unit cell and neighboring cells
    cutoff = [0] * 3
    a, b, c, _, _, _ = crystal.lattice_parameters
    for i, v in enumerate((a, b, c)):
        if v > 0:
            cutoff[i] = int(np.ceil(R / v)) + 1
        # with CoM, don't need to search as far
        if "use_com" in kwargs and kwargs["use_com"]:
            cutoff[i] = max(0, cutoff[i] - 1)

    unique_neighbors, duplicate_neighbors = _generate_neighbors(
        crystal, cutoff, N, **kwargs
    )

    # neighbor_translations = []
    # for neighbor in unique_neighbors:
    #     # get integer translations for each neighbor relative to the central unit cell
    #     tr = []
    #     for g in neighbor:
    #         tr.append([np.floor(x) if x >= 0 else np.ceil(x) for x in g.tr])

    #     neighbor_translations.append(tr)

    # actually create multimers to return
    multimers = []
    masses = np.array([qcel.periodictable.to_mass(sym) for sym in monomer.symbols])
    for i, g_N in enumerate(unique_neighbors):
        # if there are any monomers with identical symops, skip multimer (multiple identities)
        if len(set(g_N)) != len(g_N):
            continue

        mons = []

        bounded = True
        for g in g_N:
            mon_g_frac = g.apply(monomer.frac_coords)
            mon_g_cart = crystal.to_cartesian(mon_g_frac)
            com_g_frac = np.average(mon_g_frac, axis=0, weights=masses)
            com_g_cart = crystal.to_cartesian(com_g_frac)

            # check if mon_g is within R of the reference monomer based on center of mass distance or minimum atomic distance, depending on user preference
            if "use_com" in kwargs and kwargs["use_com"]:
                dist = np.linalg.norm(com_g_cart - monomer.centroid_cart)
            else:
                dist = cdist(monomer.cart_coords, mon_g_cart).min()

            if dist > R:
                bounded = False
                break

            mon_g = Monomer(
                monomer.symbols,
                mon_g_frac,
                mon_g_cart,
                com_g_frac,
                com_g_cart,
                g,
            )

            mons.append(mon_g)

        if not bounded:
            continue

        # if we made it here, we have a valid multimer
        multimers.append(Multimer(mons, g_N.multiplicity))

    return multimers


if __name__ == "__main__":
    pass

"""
things to add still:
    computation of powers of symops
    inverses (would come with powers)
    molecular symmetry
"""
