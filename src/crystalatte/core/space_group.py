from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Final
from pathlib import Path
from enum import Enum
import json

from .sym_ops import SymOp

conventions: Final = ["Hermann-Mauguin", "Schoenflies", "PDB", "Hall", "short"]


class CrystalSystem(Enum):
    """The 7 crystal systems in 3D."""

    TRICLINIC = "triclinic"
    MONOCLINIC = "monoclinic"
    ORTHORHOMBIC = "orthorhombic"
    TETRAGONAL = "tetragonal"
    TRIGONAL = "trigonal"
    HEXAGONAL = "hexagonal"
    CUBIC = "cubic"


class LatticeSystem(Enum):
    """The 7 lattice systems in 3D."""

    TRICLINIC = "triclinic"
    MONOCLINIC = "monoclinic"
    ORTHORHOMBIC = "orthorhombic"
    TETRAGONAL = "tetragonal"
    HEXAGONAL = "hexagonal"
    RHOMBOHEDRAL = "rhombohedral"
    CUBIC = "cubic"


@dataclass
class SpaceGroup:
    _registry: ClassVar[dict[int, SpaceGroup]] = {}
    _data: ClassVar[list[dict]] = []

    def __new__(cls, number: int) -> SpaceGroup:
        """Create a SpaceGroup instance from a space group number."""
        if number in cls._registry:
            return cls._registry[number]

        instance = super().__new__(cls)
        cls._registry[number] = instance
        return instance

    def __init__(self, number: int):
        if hasattr(self, "_initialized"):
            return  # Avoid re-initialization

        if SpaceGroup._data == []:
            SpaceGroup._load_data()

        # TODO: turn this back to dict, will have multiple entries for some space groups
        if number < 1 or number > len(SpaceGroup._data):
            raise ValueError(f"Space group number {number} not found in data.")

        data = SpaceGroup._data[number - 1]

        self._number = data["number"]
        self._symbol = data["symbol"]
        self._point_group = data["point_group"]
        self._crystal_system = CrystalSystem(data["crystal_system"])
        self._lattice_system = data["lattice_system"]
        self._sym_ops = [SymOp(op) for op in data["sym_ops"]]
        self._lattice_constraints = data["lattice_constraints"]

    @classmethod
    def _load_data(cls):
        """Load space group data from a JSON file."""
        # TODO: make this more robust and efficient (e.g. use a binary format, or a database)
        path = Path(__file__).parent / "space_groups.json"
        with open(path, "r") as f:
            cls._data = json.load(f)

    @property
    def number(self) -> int:
        """Return the space group number."""
        return self._number

    @property
    def symbol(self) -> dict[str, str]:
        """Return the symbol of the space group for all available conventions."""
        return self._symbol

    @property
    def point_group(self) -> str:
        """Return the point group of the space group."""
        return self._point_group

    @property
    def crystal_system(self) -> CrystalSystem:
        """Return the crystal system of the space group."""
        return self._crystal_system

    @property
    def lattice_system(self) -> str:
        """Return the lattice system of the space group."""
        return self._lattice_system

    @property
    def sym_ops(self) -> list[SymOp]:
        """Return the list of symmetry operations in the space group."""
        return self._sym_ops

    @property
    def lattice_constraints(self) -> dict:
        """Return the lattice constraints for the space group."""
        return self._lattice_constraints

    def __repr__(self) -> str:
        return f"SpaceGroup(number={self._number}, symbol='{self._symbol}')"

    def __str__(self) -> str:
        return f"Space Group {self._number} ({self._symbol}) - {self._crystal_system.name.capitalize()}"


if __name__ == "__main__":
    pass
