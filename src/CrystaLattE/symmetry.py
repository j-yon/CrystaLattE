from crystalatte.sym_ops import SymOp
from crystalatte.tmp import Lattice, CrystalSystem


class SpaceGroup:
    number: int
    name: str
    point_group: str
    crystal_system: CrystalSystem
    lattice: Lattice
    sym_ops: list[SymOp]

    def __init__(self, name):
        self.name = name
        pass

    def __str__(self):
        return f"Space Group: {self.name}, Operations: {len(self.operations)}"
