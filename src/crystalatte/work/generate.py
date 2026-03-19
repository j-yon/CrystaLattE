from itertools import combinations, product
import numpy as np
from numpy.typing import NDArray
import qcelemental as qcel

from ..core.crystal import Crystal
from ..core.multimer import Monomer, Multimer
from ..core.sym_ops import SymOp, SymOpList


def _is_bijection(g_N: SymOpList, h_N: SymOpList) -> bool:
    # test for bijection with new reference monomer
    bijection = False
    for g in g_N:
        map = [h.compose_augment(g) for h in h_N]

        if (
            all(m in g_N for m in map)
            and len(set(map)) == len(h_N)
            and set(map) == set(g_N)
        ):
            bijection = True
            break

    return bijection


def _generate_central(
    crystal: Crystal, N: int
) -> tuple[list[SymOpList], list[SymOpList]]:
    # generate all combinations of N-1 symmetry operations from the space group
    symop_lists = combinations(crystal._space_group.sym_ops, N - 1)
    symop_lists = [
        SymOpList([SymOp.identity()] + list(symop_list), 1)
        for symop_list in symop_lists
    ]

    combos = combinations(symop_lists, 2)
    duplicates = []
    for g_N, h_N in combos:
        if _is_bijection(g_N, h_N):
            symop_lists.remove(h_N)
            duplicates.append(h_N)
            g_N.multiplicity += 1

    print(
        f"Found {len(duplicates)} equivalent multimers from central unit cell search."
    )
    return symop_lists, duplicates


def _generate_neighbors(
    crystal: Crystal,
    symop_lists: list[SymOpList],
    N: int,
) -> tuple[list[SymOpList], list[SymOpList]]:
    # add translation to all symop lists and check for bijections with other symop lists
    translated = []
    for g_N in symop_lists:
        for d in product(product([-1, 0, 1], repeat=3), repeat=N - 1):
            g_t = g_N.translate_list(
                np.array(d, dtype=np.float64), exclude_reference=True
            )

            translated.append(g_t)

    duplicates = []
    for g_t, h_t in combinations(translated, 2):
        if _is_bijection(g_t, h_t):
            translated.remove(h_t)
            duplicates.append(h_t)
            g_t.multiplicity += 1

    print(
        f"Found {len(duplicates)} equivalent multimers from neighboring unit cell search."
    )
    return translated, duplicates


def _is_multiple_translation(
    rot: list[NDArray], tr: NDArray, neighbor: SymOpList
) -> bool:
    for rot_i, rot_neighbor in zip(rot, neighbor.rotations):
        if not all(
            np.array_equal(rot_i, rot_neighbor_i)
            for rot_i, rot_neighbor_i in zip(rot, neighbor.rotations)
        ):
            return False

    for tr_i, tr_neighbor in zip(tr, neighbor.translations):
        # only check nonzero translation components to avoid division by zero
        nonzero_i = np.abs(tr_i) > 1e-6
        nonzero_n = np.abs(tr_neighbor) > 1e-6

        # if the nonzero components are different, they can't be multiples
        if not np.array_equal(nonzero_i, nonzero_n):
            return False

        # if there are no nonzero components, then the translations are effectively the same and we can skip the ratio check
        if not np.any(nonzero_i):
            continue

        # check to see if all ratios are equivalent
        if not np.allclose(
            tr_i[nonzero_i] / tr_neighbor[nonzero_n],
            tr_i[nonzero_i][0] / tr_neighbor[nonzero_n][0],
            1e-6,
        ):
            return False

        # ratios are equivalent, but if they are negative then the translations are in opposite directions and can't be multiples
        if tr_i[nonzero_i][0] / tr_neighbor[nonzero_n][0] < 0:
            return False

    return True


def generate(crystal: Crystal, monomer: Monomer, N: int, R: float) -> list[Multimer]:
    """
    Generate unique multimers of the given monomer in the crystal.

    Parameters
    ----------
    crystal : Crystal
        The crystal structure containing the monomer.
    monomer : Monomer
        The reference monomer to generate multimers from.
    N : int
        The number of monomers in the multimer (e.g. N=2 for dimers).
    R : float
        The maximum center-to-center distance in Angstroms for monomers to be considered part of the same multimer.

    Returns
    -------
    List[Multimer]
        A list of unique multimers generated from the reference monomer.
    """

    # first identify equivalencies in the central unit cell and neighboring cells
    unique_central, duplicate_central = _generate_central(crystal, N)
    unique_neighbors, duplicate_neighbors = _generate_neighbors(
        crystal, unique_central, N
    )

    # given cell dimensions and R, calculate how far you have to go in each direction to find all monomers within R of the reference monomer
    # this is a loose overestimate, but is further pruned once multimers are formed
    cutoff = [1] * 3
    a, b, c, _, _, _ = crystal.lattice_parameters
    for i, v in enumerate((a, b, c)):
        if v > 0:
            cutoff[i] = max(cutoff[i], int(np.ceil(R / v)))

    # finally, generate all translations of the unique central unit cell multimers within the cutoff
    # using previous neighbor translations to determine further equivalencies
    final_unique = []
    for g_N in unique_central:
        for d in product(
            product(range(-max(cutoff), max(cutoff) + 1), repeat=3), repeat=N - 1
        ):
            # TODO: avoid this check by only generating translations within cutoff in the first place
            outside = False
            for d_i in d:
                if any(abs(d_i_j) > cutoff_j for d_i_j, cutoff_j in zip(d_i, cutoff)):
                    outside = True
                    break

            if outside:
                continue
            d = np.array(d, dtype=np.float64)
            d = np.insert(d, 0, [0, 0, 0], axis=0)

            # now check if translations are a multiple of a duplicate neighbor translation and if rotations are equal, if so skip
            if any(
                _is_multiple_translation(
                    g_N.rotations,
                    d,
                    duplicate_neighbor,
                )
                for duplicate_neighbor in duplicate_neighbors
            ):
                continue

            g_T = g_N.translate_list(d)
            for neighbor in unique_neighbors:
                if _is_multiple_translation(g_T.rotations, d, neighbor):
                    g_T.multiplicity = max(g_T.multiplicity, neighbor.multiplicity)
                    break

            final_unique.append(g_T)

    # actually create multimers to return
    multimers = []
    masses = np.array([qcel.periodictable.to_mass(sym) for sym in monomer.symbols])
    for i, g_N in enumerate(final_unique):
        # if there are any monomers with identical symops, skip multimer (multiple identities)
        if len(set(g_N)) != len(g_N):
            continue

        # create multimers in current SymOpList
        mons = []

        bounded = True
        for g in g_N:
            mon_g_frac = g.apply(monomer.frac_coords)
            mon_g_cart = crystal.to_cartesian(mon_g_frac)
            com_g_frac = np.average(mon_g_frac, axis=0, weights=masses)
            com_g_cart = crystal.to_cartesian(com_g_frac)

            # check if mon_g is within R of the reference monomer based on center of mass distance
            # TODO: make checking available for closest contact as well
            dist = np.linalg.norm(com_g_cart - monomer.centroid_cart)
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

        if len(multimers) == 1:
            print(multimers[-1].monomers[-1].symop)
            print(multimers[-1].monomers[-1].cart_coords)

    return multimers
