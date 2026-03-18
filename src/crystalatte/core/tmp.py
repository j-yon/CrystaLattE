"""
from cif grab latatice parameters, space group, table number, sym ops, and asu atoms
create Crystal object that has these attributes and methods to convert between frac and cart
lattice params will be floats, space group will be strin/enum, table number will be in
symop will be list of SymOp objects, asu atoms will be list of symbols and frac coords (direct from cif, not symmetry expanded)

create method to generate molecules with centroids in the central unit cell by applying symops to asu atoms and checking which ones are in the central cell

determine a reference monomer (e.g. the one closest to the center of the unit cell)

determine molecular symmetry by applying symops to the reference monomer and checking which symops leave it invariant (within some tolerance)

for most basic lattice: P1 with cubic unit cell (ammonia)
only possible operations are lattice translations
starting from reference monomer, apply lattice translations and discount opposite translations
"""

import numpy as np
from itertools import product

vals = range(-1, 2)
origin = (0, 0)
lattice = 3 * np.eye(3)


# def minimum_image_distance(fi: np.ndarray, fj: np.ndarray, lattice):
#     fi = np.linalg.inv(lattice) @ fi
#     fj = np.linalg.inv(lattice) @ fj

#     d = fj - fi
#     print(d)
#     d -= np.round(d)
#     return lattice @ d


def canonicalize(points, lattice):
    diffs = []

    for anchor in points:
        pass

    return tuple(sorted(diffs))


def gen_unique_configs(N):
    seen = set()
    unique = []

    for others in product(product(vals, repeat=2), repeat=N - 1):
        points = [origin] + list(others)
        if len(set(points)) != N:
            continue

        fingerprint = canonicalize(points, lattice)

        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(points)

    return unique


for N in range(2, 4):
    configs = gen_unique_configs(N)
    print(
        f"\n{len(configs)} unique configurations of {N} points in a 3x3x3 grid (modulo translations):"
    )
    for config in configs:
        print(tuple(config))
