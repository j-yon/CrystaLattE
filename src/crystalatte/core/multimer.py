from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray

from .sym_ops import SymOp


@dataclass
class ASU:
    """
    Represents an asymmetric unit (ASU) of a crystal structure, which is the smallest portion of the crystal that can generate the entire unit cell through the application of the space group's symmetry operations. Contains the atoms and their positions within the ASU

    :param atoms: A list of atomic symbols corresponding to the atoms in the ASU
    :type atoms: list[str]

    :param positions: A list of fractional coordinates corresponding to the positions of the atoms in the ASU
    :type positions: list[NDArray[np.float64]]
    """

    _atoms: list[str]
    _positions: np.ndarray

    def __post_init__(self):
        """Constructor method"""
        if len(self._atoms) != len(self._positions):
            raise ValueError("Number of atoms must match number of positions")

    @property
    def atoms(self) -> list[str]:
        """Return the list of atomic symbols in the ASU."""
        return self._atoms

    @property
    def positions(self) -> np.ndarray:
        """Return the list of fractional coordinates for the atoms in the ASU."""
        return self._positions


@dataclass
class Monomer:
    """
    Represents a molecular monomer in the crystal.

    Attributes
    ----------
    symbols : List[str]
        Atom symbols in the monomer.
    frac_coords : NDArray[np.float64]
        Fractional coordinates of atoms, shape (N, 3).
    cart_coords : NDArray[np.float64]
        Cartesian coordinates of atoms, shape (N, 3).
    centroid_frac : NDArray[np.float64]
        Centroid in fractional coordinates.
    centroid_cart : NDArray[np.float64]
        Centroid in Cartesian coordinates (Angstroms).
    symop: Symmetry operation that generates this monomer from the reference monomer (optional).
    """

    _symbols: list[str]
    _frac_coords: NDArray[np.float64]
    _cart_coords: NDArray[np.float64]
    _centroid_frac: NDArray[np.float64]
    _centroid_cart: NDArray[np.float64]
    _symop: SymOp | None = None

    @property
    def symbols(self) -> list[str]:
        """Return the list of atomic symbols in the monomer."""
        return self._symbols

    @property
    def frac_coords(self) -> NDArray[np.float64]:
        """Return the fractional coordinates of the atoms in the monomer."""
        return self._frac_coords

    @property
    def cart_coords(self) -> NDArray[np.float64]:
        """Return the Cartesian coordinates of the atoms in the monomer."""
        return self._cart_coords

    @property
    def centroid_frac(self) -> NDArray[np.float64]:
        """Return the centroid of the monomer in fractional coordinates."""
        return self._centroid_frac

    @property
    def centroid_cart(self) -> NDArray[np.float64]:
        """Return the centroid of the monomer in Cartesian coordinates."""
        return self._centroid_cart

    @property
    def symop(self) -> SymOp | None:
        """Return the symmetry operation that generates this monomer from the reference monomer, if applicable."""
        return self._symop

    def to_molecule(self) -> qcel.models.Molecule:
        """Convert to QCElemental Molecule object."""
        cart_bohr = self._cart_coords / qcel.constants.bohr2angstroms
        return qcel.models.Molecule(
            symbols=self._symbols,
            geometry=cart_bohr.flatten(),
        )


@dataclass
class Multimer:
    """Represents a unique multimer pairing of N monomers.

    Attributes:
    monomers: List[Monomer]
        List of monomers in the multimer, with the reference monomer first.
    multiplicity : int
        Number of symmetry-equivalent copies of this multimer type.
    """

    _monomers: list[Monomer]
    _multiplicity: int = 1
    _geometric_mean: float = field(init=False)

    def __post_init__(self):
        """Calculate the geometric mean of the inter-monomer distances."""
        if len(self._monomers) < 2:
            self._geometric_mean = 0.0
            return

        dists = []
        for i in range(len(self._monomers)):
            for j in range(i + 1, len(self._monomers)):
                dist = np.linalg.norm(
                    self._monomers[i]._centroid_cart - self._monomers[j]._centroid_cart
                )
                dists.append(dist)

        self._geometric_mean = np.prod(dists) ** (1 / len(dists))

    @property
    def monomers(self) -> list[Monomer]:
        """Return the list of monomers in the multimer."""
        return self._monomers

    @property
    def multiplicity(self) -> int:
        """Return the multiplicity of the multimer."""
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, value: int):
        """Set the multiplicity of the multimer."""
        self._multiplicity = value

    @property
    def geometric_mean(self) -> float:
        """Return the geometric mean of the inter-monomer distances."""
        return self._geometric_mean

    def to_molecule(self) -> qcel.models.Molecule:
        """Convert dimer to QCElemental Molecule object."""
        symbols = [monomer._symbols for monomer in self._monomers]
        coords = [monomer._cart_coords for monomer in self._monomers]
        coords = np.vstack(coords)
        cart_bohr = coords / qcel.constants.bohr2angstroms

        # TODO: does this return a fragmented molecule?
        return qcel.models.Molecule(
            symbols=symbols,
            geometry=cart_bohr.flatten(),
        )

    def get_molecules(self) -> list[qcel.models.Molecule]:
        """Return the monomers as separate QCElemental Molecules."""
        return [monomer.to_molecule() for monomer in self._monomers]

    def to_qcel_molecule(self) -> qcel.models.Molecule:
        """Convert the multimer to a single QCElemental Molecule object."""
        symbols = []
        coords = []

        for monomer in self._monomers:
            symbols.extend(monomer._symbols)
            coords.append(monomer._cart_coords)
        coords = np.vstack(coords)
        print(coords)
        cart_bohr = coords / qcel.constants.bohr2angstroms

        return qcel.models.Molecule(
            symbols=symbols,
            geometry=cart_bohr.flatten(),
        )
