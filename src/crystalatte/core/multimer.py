from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray


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
    """

    symbols: list[str]
    frac_coords: NDArray[np.float64]
    cart_coords: NDArray[np.float64]
    centroid_frac: NDArray[np.float64]
    centroid_cart: NDArray[np.float64]

    def to_molecule(self) -> qcel.models.Molecule:
        """Convert to QCElemental Molecule object."""
        cart_bohr = self.cart_coords / qcel.constants.bohr2angstroms
        return qcel.models.Molecule(
            symbols=self.symbols,
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

    monomers: list[Monomer]
    multiplicity: int = 1

    geometric_mean: float = field(init=False)

    def __post_init__(self):
        """Calculate the geometric mean of the inter-monomer distances."""
        if len(self.monomers) < 2:
            self.geometric_mean = 0.0
            return

        dists = []
        for i in range(len(self.monomers)):
            for j in range(i + 1, len(self.monomers)):
                dist = np.linalg.norm(
                    self.monomers[i].centroid_cart - self.monomers[j].centroid_cart
                )
                dists.append(dist)

        self.geometric_mean = np.prod(dists) ** (1 / len(dists))

    def to_molecule(self) -> qcel.models.Molecule:
        """Convert dimer to QCElemental Molecule object."""
        symbols = [monomer.symbols for monomer in self.monomers]
        coords = [monomer.cart_coords for monomer in self.monomers]
        coords = np.vstack(coords)
        cart_bohr = coords / qcel.constants.bohr2angstroms

        # TODO: does this return a fragmented molecule?
        return qcel.models.Molecule(
            symbols=symbols,
            geometry=cart_bohr.flatten(),
        )

    def get_molecules(self) -> list[qcel.models.Molecule]:
        """Return the monomers as separate QCElemental Molecules."""
        return [monomer.to_molecule() for monomer in self.monomers]
