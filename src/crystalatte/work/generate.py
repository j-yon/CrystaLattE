from itertools import combinations, combinations_with_replacement, product
import numpy as np
from copy import deepcopy

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


def _generate_central(crystal: Crystal, N: int) -> list[SymOpList]:
    # generate all combinations of N-1 symmetry operations from the space group
    symop_lists = combinations(crystal._space_group.sym_ops, N - 1)
    symop_lists = [
        SymOpList([SymOp.identity()] + list(symop_list), 1)
        for symop_list in symop_lists
    ]

    # check for duplicates in each SymOpList
    # for symop_list in symop_lists:
    #     if len(set(symop_list)) != len(symop_list):
    #         symop_lists.remove(symop_list)

    unique = []
    for g_N, h_N in combinations(symop_lists, 2):
        # determine ordering of g_N and h_N for comparison, so that we always compare the new multimer to the representative
        if g_N in unique:
            # if g_N is already in the unique list, it should be the representative
            rep = g_N
            new = h_N
        else:
            # if h_N is already in the unique list, it should be the representative
            # if neither, representative does not matter, so just pick one
            rep = h_N
            new = g_N

        if rep not in unique:
            unique.append(rep)

        if not _is_bijection(rep, new):
            if new not in unique:
                unique.append(new)
        else:
            # print(f"Found bijection between:\n {rep} \nand\n {new}")
            rep.multiplicity += 1

    return unique


def _generate_neighbors(
    crystal: Crystal,
    symop_list: list[SymOpList],
) -> list[SymOpList]:
    directions = product([-1, 0, 1], repeat=3)

    unique = []
    for g_N, h_N in combinations(symop_list, 2):
        for d in combinations(directions, 2):
            rep = max(g_N, h_N, key=lambda x: x.multiplicity)
            new = min(g_N, h_N, key=lambda x: x.multiplicity)

            rep_t = rep.translate_list(
                np.array(d[0], dtype=np.float64), exclude_reference=True
            )
            new_t = new.translate_list(
                np.array(d[1], dtype=np.float64), exclude_reference=True
            )

            if rep_t not in unique:
                unique.append(rep_t)

            if not _is_bijection(rep_t, new_t):
                if new_t not in unique:
                    unique.append(new_t)
            else:
                # print(f"Found bijection between:\n {rep_t} \nand\n {new_t}")
                rep_t.multiplicity += 1

    return unique


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
    unique = _generate_central(crystal, N)
    unique = _generate_neighbors(crystal, unique)

    # propagate outwards, actually creating multimers
    multimers = []
    for g_N in unique:
        # create multimers in current SymOpList
        mons = []

        bounded = True
        for g in g_N:
            mon_g_frac = g.apply(monomer.frac_coords)
            mon_g_cart = crystal.to_cartesian(mon_g_frac)
            com_g_frac = np.mean(mon_g_frac, axis=0)
            com_g_cart = np.mean(mon_g_cart, axis=0)

            # check if mon_g is within R of the reference monomer
            dist = np.linalg.norm(com_g_cart - monomer.centroid_cart)
            if dist > R:
                bounded = False
                break

            mon_g = Monomer(
                symbols=monomer.symbols,
                frac_coords=mon_g_frac,
                cart_coords=mon_g_cart,
                centroid_frac=com_g_frac,
                centroid_cart=com_g_cart,
            )

            mons.append(mon_g)

        if not bounded:
            continue

        # sanity check, make sure no overlapping monomers in any of [mons]
        if any(
            np.linalg.norm(m1.centroid_cart - m2.centroid_cart) < 1.0
            for m1, m2 in combinations(mons, 2)
        ):
            continue

        # if we made it here, we have a valid multimer
        multimers.append(Multimer(mons, g_N.multiplicity))

    return multimers
