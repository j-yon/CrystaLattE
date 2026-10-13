from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, Final, Self

import numpy as np
from numpy.typing import NDArray

from .sym_ops import SymOp

conventions: Final = ["Hermann-Mauguin", "Schoenflies", "PDB", "Hall", "short"]

SITE_ATOL: Final[float] = 1e-4


class CrystalSystem(Enum):
    """The 7 crystal systems in 3D"""

    TRICLINIC = "triclinic"
    MONOCLINIC = "monoclinic"
    ORTHORHOMBIC = "orthorhombic"
    TETRAGONAL = "tetragonal"
    TRIGONAL = "trigonal"
    HEXAGONAL = "hexagonal"
    CUBIC = "cubic"


@dataclass
class SpaceGroup:
    """Represents a space group and contains information about the space group number, symbol,
    point group, crystal system, lattice system, symmetry operations, lattice constraints,
    and special Wyckoff positions

    :param _number: space group number (1-230)
    :param _hall: Hall symbol identifying the specific setting
    :param _symbol: dictionary of symbols for different conventions
    :param _point_group: point group symbol
    :param _crystal_system: crystal system enum
    :param _lattice_system: lattice system enum
    :param _sym_ops: list of SymOp objects representing the symmetry operations
    :param _lattice_constraints: dictionary of lattice constraints
    :param _special_positions: dictionary of special Wyckoff positions
    :ivar _registry: stores instances of SpaceGroup keyed by their lookup key to ensure one instance per setting
    :ivar _data: stores the space group data loaded from a JSON
    """

    _number: int
    _hall: str | None
    _symbol: dict[str, str]
    _point_group: str
    _crystal_system: CrystalSystem
    _lattice_system: str
    _sym_ops: list[SymOp]
    _lattice_constraints: dict
    _special_positions: dict[str, dict]

    _registry: ClassVar[dict[str, SpaceGroup]] = {}
    _data: ClassVar[dict[str, dict]] = {}

    def __new__(cls, number: int, hall: str | None = None) -> Self:
        """Create a SpaceGroup instance from a space group number and optional Hall symbol.

        :param number: The space group number (1-230)
        :param hall: Optional Hall symbol (e.g. "-P 2yab") to select a specific setting
        :returns: A cached SpaceGroup instance for the requested setting
        """
        key = hall.strip() if hall else str(number)
        if key in cls._registry:
            return cls._registry[key]

        instance = super().__new__(cls)
        cls._registry[key] = instance
        return instance

    def __init__(self, number: int, hall: str | None = None) -> None:
        if hasattr(self, "_initialized"):
            return  # avoid re-initialization

        # load data if not already loaded
        if SpaceGroup._data == {}:
            SpaceGroup.load_data()

        # try Hall symbol first (setting-specific), then fall back to number (standard setting)
        key = hall.strip() if hall else str(number)
        if key not in SpaceGroup._data:
            key = str(number)
        if key not in SpaceGroup._data:
            raise ValueError(
                f"Space group number {number} (hall={hall!r}) not found in data."
            )

        # grab data from determined key
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
        self._initialized = True

    def __repr__(self) -> str:
        return f"SpaceGroup(number={self._number}, symbol='{self._symbol}')"

    def __str__(self) -> str:
        return f"Space Group {self._number} ({self._symbol}) - {self._crystal_system.name.capitalize()}"

    def __reduce__(self):
        if getattr(self, "_detected", False):
            return (
                SpaceGroup._reconstruct_detected,
                (self._sym_ops, self._number, self._symbol),
            )
        return (self.__class__, (self._number, self._hall))

    @property
    def number(self) -> int:
        return self._number

    @property
    def symbol(self) -> dict[str, str]:
        return self._symbol

    @property
    def point_group(self) -> str:
        return self._point_group

    @property
    def crystal_system(self) -> CrystalSystem:
        return self._crystal_system

    @property
    def lattice_system(self) -> str:
        return self._lattice_system

    @property
    def sym_ops(self) -> list[SymOp]:
        return self._sym_ops

    @property
    def lattice_constraints(self) -> dict:
        return self._lattice_constraints

    @classmethod
    def _reconstruct_detected(cls, sym_ops, number, symbol) -> Self:
        """Reconstruct a detected SpaceGroup from pickled data"""
        return cls.from_sym_ops(sym_ops, number=number, symbol=symbol)

    @classmethod
    def load_data(cls) -> None:
        """Load space group data from a JSON file"""
        path = Path(__file__).parent / "space_groups_all.json"
        with open(path, "r") as f:
            cls._data = json.load(f)

    @classmethod
    def from_sym_ops(cls, sym_ops, number=None, symbol=None) -> Self:
        """Build a SpaceGroup from an explicit operation list and bypass the DB/registry

        Has no special-position table, ``get_unique_sym_ops`` dedups by centroid image

        :param sym_ops: symmetry operations
        :param number: optional international number
        :param symbol: optional symbol dict
        :returns: fresh, uncached SpaceGroup carrying exactly these ops
        """
        instance = object.__new__(cls)
        instance._number = number
        instance._hall = None
        instance._symbol = symbol if symbol is not None else {}
        instance._point_group = None
        instance._crystal_system = None
        instance._lattice_system = None
        instance._sym_ops = list(sym_ops)
        instance._lattice_constraints = {}
        instance._special_positions = {}
        instance._detected = True
        instance._initialized = True  # guard against __init__ re-running
        return instance

    def get_unique_sym_ops(
        self, coords: NDArray, atol: float = SITE_ATOL
    ) -> list[SymOp]:
        """Return one representative operation per distinct image of ``coords`` (modulo the lattice)

        :param coords: fractional coordinates of the monomer centroid
        :param atol: fractional tolerance for deciding two images coincide
        :returns: one SymOp per distinct centroid image
        :raises ValueError: if an unambiguously matched special position's multiplicity disagrees with the computed orbit size
        """
        coords = np.asarray(coords, dtype=float)

        # deduplicate by image wrapped to central unit cell
        images: list[NDArray] = []
        result: list[SymOp] = []
        for op in self._sym_ops:
            image = np.asarray(op.apply(coords), dtype=float) % 1.0

            # use loose-ish tolerance since special positions (1/3) are written as floats
            if any(
                np.allclose(((image - seen + 0.5) % 1.0) - 0.5, 0.0, atol=atol)
                for seen in images
            ):
                continue

            images.append(image)
            result.append(op)

        # validation: compare coordinates to special positions and check multiplicity
        matches = {
            letter: pos["multiplicity"]
            for letter, pos in self._special_positions.items()
            if any(np.allclose(coords, site) for site in pos["coordinates"])
        }

        # only an unambiguous match is checked: parameterised positions (e.g. (x, x, z)) are
        # tabulated with free parameters at zero, so at the origin SG 217 matches g, e, c and a
        # at once and the true multiplicity is unknown; orbit-stabilizer fixes the count anyway
        if len(matches) == 1:
            ((letter, multiplicity),) = matches.items()
            if len(result) != multiplicity:
                raise ValueError(
                    f"Expected {multiplicity} unique symmetry "
                    f"operations for position {letter}, but found {len(result)}."
                )

        return result

    def get_site_symmetry(
        self, coords: NDArray, atol: float = SITE_ATOL
    ) -> list[SymOp]:
        """Get the site-symmetry stabilizer of ``coords``

        :param coords: fractional coordinates of the monomer centroid
        :param atol: fractional tolerance for deciding an operation fixes the site
        :returns: subset of symmetry operations fixing ``coords`` including identity
        """
        coords = np.asarray(coords, dtype=float)

        result: list[SymOp] = []
        for op in self._sym_ops:
            # simply check if this opp leaves the site invariant (mod lattice translations)
            # site symmetries will always have coincident atom types, so no permutation check
            delta = op.apply(coords) - coords
            if np.allclose(delta - np.round(delta), 0.0, atol=atol):
                result.append(op)

        return result


if __name__ == "__main__":
    pass
