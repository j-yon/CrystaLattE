"""Reference detection on hand-built crystals.

Each crystal is small enough that its correct references can be read off by hand.
"""

import numpy as np

from crystalatte.core.crystal import Crystal
from crystalatte.core.multimer import ASU
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp

ORTHO = (90.0, 90.0, 90.0)


def _crystal(vectors, symbols, positions, ops=()) -> Crystal:
    group = SpaceGroup.from_sym_ops([SymOp.identity(), *ops], number=1)
    return Crystal(vectors, ORTHO, group, ASU(symbols, np.array(positions, dtype=float)))


def test_edge_fragment_is_dropped_when_rounding_would_split_its_site():
    # 0.0015 rounds differently from its image at -0.9985, where the C is cut off from its O
    crystal = _crystal(
        (10.0, 10.0, 10.0), ["C", "O"], [[0.0015, 0.5, 0.5], [0.8815, 0.5, 0.5]]
    )
    assert [sorted(r.symbols) for r in crystal.get_references()] == [["C", "O"]]


def test_one_orbit_gives_one_reference_when_rounding_would_split_its_centroid():
    # every image of this lone atom is the same molecule, whatever its coordinate rounds to
    crystal = _crystal((10.0, 10.0, 10.0), ["Ar"], [[0.0015, 0.5, 0.5]])
    assert [r.symbols for r in crystal.get_references()] == [["Ar"]]

