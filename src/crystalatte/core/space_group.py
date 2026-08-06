from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Final
from pathlib import Path
from enum import Enum
import json

import numpy as np
from numpy.typing import NDArray

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
    """Represents a space group, which is a mathematical description of the symmetry of a crystal structure. Contains information about the space group number, symbol, point group, crystal system, lattice system, symmetry operations, lattice constraints, and special Wyckoff positions.

    :param _number: The space group number (1-230)
    :param _hall: The Hall symbol identifying the specific setting (e.g. "-P 2yab" for P2₁/a). When None, the standard setting for the given number is used.
    :ivar _registry: A class variable that stores instances of SpaceGroup keyed by their lookup key (Hall symbol or str(number)) to ensure one instance per setting
    :ivar _data: A class variable that stores the space group data loaded from a JSON
    """

    _number: int
    _registry: ClassVar[dict[str, SpaceGroup]] = {}
    _data: ClassVar[dict[str, dict]] = {}

    def __new__(cls, number: int, hall: str | None = None) -> SpaceGroup:
        """Create a SpaceGroup instance from a space group number and optional Hall symbol.

        :param number: The space group number (1-230)
        :param hall: Optional Hall symbol (e.g. "-P 2yab") to select a specific setting. When omitted, the standard setting for the number is used.
        :returns: A SpaceGroup instance for the requested setting. Instances are cached per setting so that the same setting is never constructed twice.
        """
        key = hall.strip() if hall else str(number)
        if key in cls._registry:
            return cls._registry[key]

        instance = super().__new__(cls)
        cls._registry[key] = instance
        return instance

    def __init__(self, number: int, hall: str | None = None):
        if hasattr(self, "_initialized"):
            return  # Avoid re-initialization

        if SpaceGroup._data == {}:
            SpaceGroup._load_data()

        # Try Hall symbol first (setting-specific), then fall back to number (standard setting)
        key = hall.strip() if hall else str(number)
        if key not in SpaceGroup._data:
            key = str(number)
        if key not in SpaceGroup._data:
            raise ValueError(f"Space group number {number} (hall={hall!r}) not found in data.")

        data = SpaceGroup._data[key]

        self._number = data["number"]
        self._hall = hall.strip() if hall else data["symbol"].get("Hall")
        self._symbol = data["symbol"]
        self._point_group = data["point_group"]
        self._crystal_system = CrystalSystem(data["crystal_system"])
        self._lattice_system = data["lattice_system"]
        self._sym_ops = [SymOp(op) for op in data["sym_ops"]]
        self._lattice_constraints = data["lattice_constraints"]
        self._special_positions = data["special_wyckoff_positions"]

    def __reduce__(self) -> tuple[type[SpaceGroup], tuple[int, str | None]]:
        return (self.__class__, (self._number, self._hall))

    @classmethod
    def _load_data(cls):
        """Load space group data from a JSON file."""
        path = Path(__file__).parent / "space_groups_all.json"
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

    def get_unique_sym_ops(self, coords: NDArray) -> list[SymOp]:
        """Return a list of unique symmetry operations in the space group given a center of mass. Used to account for special Wyckoff positions.

        :param coords: The fractional coordinates of the center of mass of the monomer. Should be a 3-element array of floats between 0 and 1
        :returns: A list of unique symmetry operations that should be applied to the monomer to generate full multimers, accounting for any special Wyckoff positions
        :raises ValueError: If the number of unique symmetry operations found does not match the expected multiplicity for the special position
        """
        for letter, pos in self._special_positions.items():
            for site in pos["coordinates"]:
                if np.allclose(coords, site):
                    unique = {}
                    for op in self._sym_ops:
                        res = op.apply(coords)
                        res = tuple(np.round(res, decimals=8) % 1)

                        if res not in unique:
                            unique[res] = op

                    if len(unique) != pos["multiplicity"]:
                        raise ValueError(
                            f"Expected {pos['multiplicity']} unique symmetry operations for position {letter}, but found {len(unique)}."
                        )
                    return list(unique.values())

        return self._sym_ops

    def get_site_symmetry(self, coords: NDArray) -> list[SymOp]:
        """Return the site-symmetry stabilizer of ``coords``: the space-group
        operations that map ``coords`` onto itself modulo a lattice translation.

        :param coords: Fractional coordinates of the monomer centroid
        :returns: The subset of symmetry operations fixing ``coords`` (always
            includes the identity)
        """
        coords = np.asarray(coords, dtype=float)
        result: list[SymOp] = []
        for op in self._sym_ops:
            delta = op.apply(coords) - coords
            if np.allclose(delta - np.round(delta), 0.0, atol=1e-6):
                result.append(op)
        return result

    def __repr__(self) -> str:
        return f"SpaceGroup(number={self._number}, symbol='{self._symbol}')"

    def __str__(self) -> str:
        return f"Space Group {self._number} ({self._symbol}) - {self._crystal_system.name.capitalize()}"


if __name__ == "__main__":
    pass
