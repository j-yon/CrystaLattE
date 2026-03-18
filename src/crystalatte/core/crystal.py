from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray

from .sym_ops import SymOp
from .space_group import SpaceGroup
from .multimer import ASU


@dataclass
class Crystal:
    """
    Represents a crystal unit cell with lattice parameters and space group symmetry.

    :param a: Lattice parameter a in Angstroms
    :type a: np.float64

    :param b: Lattice parameter b in Angstroms
    :type b: np.float64

    :param c: Lattice parameter c in Angstroms
    :type c: np.float64

    :param alpha: Lattice angle alpha in degrees
    :type alpha: np.float64

    :param beta: Lattice angle beta in degrees
    :type beta: np.float64

    :param gamma: Lattice angle gamma in degrees
    :type gamma: np.float64

    :param space_group: The crystal's space group
    :type space_group: SpaceGroup

    :param symops: List of symmetry operations for the space group
    :type symops: List[SymOp]
    """

    _a: float
    _b: float
    _c: float
    _alpha: float
    _beta: float
    _gamma: float
    _space_group: SpaceGroup
    _asu: ASU

    # Computed matrices for coordinate transformations
    _frac_to_cart: NDArray[np.float64] = field(init=False, repr=False)
    _cart_to_frac: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize transformation matrices."""
        self._validate_crystal()
        self._update_matrices()

    def _update_matrices(self):
        """Compute fractional <-> Cartesian transformation matrices."""
        # Convert angles to radians
        alpha, beta, gamma = np.radians([self._alpha, self._beta, self._gamma])

        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)

        # Volume factor
        omega = np.sqrt(
            1
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
            + 2 * cos_alpha * cos_beta * cos_gamma
        )

        # Fractional to Cartesian matrix (column vectors are lattice vectors)
        # Using the standard crystallographic convention:
        # a along x, b in xy plane, c general
        c_y = self._c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        self._frac_to_cart = np.array(
            [
                [self._a, self._b * cos_gamma, self._c * cos_beta],
                [0.0, self._b * sin_gamma, c_y],
                [0.0, 0.0, self._c * omega / sin_gamma],
            ]
        )

        # Cartesian to fractional matrix
        self._cart_to_frac = np.linalg.inv(self._frac_to_cart)

    def _validate_crystal(self):
        """Things to validate
        - Lattice parameters must be positive
        - Lattice angles must be between 0 and 180 degrees
        - Space group must be consistent with lattice parameters
        - ASU atoms must be within the ASU region of space group

        """
        if self._a <= 0 or self._b <= 0 or self._c <= 0:
            raise ValueError("Lattice parameters must be positive.")

        if (
            not (0 < self._alpha < 180)
            or not (0 < self._beta < 180)
            or not (0 < self._gamma < 180)
        ):
            raise ValueError("Lattice angles must be between 0 and 180 degrees.")

        sg = self._space_group
        asu = self._asu

        # TODO: check lattice parameters and other stuff

    @property
    def volume(self) -> float:
        """Calculate unit cell volume in cubic Angstroms."""
        return abs(np.linalg.det(self._frac_to_cart))

    def to_cartesian(self, frac_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert fractional coordinates to Cartesian coordinates.

        Parameters
        ----------
        frac_coords : NDArray[np.float64]
            Fractional coordinates, shape (N, 3).

        Returns
        -------
        NDArray[np.float64]
            Cartesian coordinates in Angstroms.
        """
        return (self._frac_to_cart @ frac_coords.T).T

    def to_fractional(self, cart_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert Cartesian coordinates to fractional coordinates.

        Parameters
        ----------
        cart_coords : NDArray[np.float64]
            Cartesian coordinates in Angstroms, shape (N, 3).

        Returns
        -------
        NDArray[np.float64]
            Fractional coordinates.
        """
        return (self._cart_to_frac @ cart_coords.T).T

    def __repr__(self) -> str:
        return (
            f"Crystal(a={self._a}, b={self._b}, c={self._c}, "
            f"alpha={self._alpha}, beta={self._beta}, gamma={self._gamma}, "
            f"space_group={self._space_group})"
        )


"""
from cif grab latatice parameters, space group, table number, sym ops, and asu atoms
create Crystal object that has these attributes and methods to convert between frac and cart
lattice params will be floats, space group will be strin/enum, table number will be in
symop will be list of SymOp objects, asu atoms will be list of symbols and frac coords (direct from cif, not symmetry expanded)

create method to generate molecules with centroids in the central unit cell by applying symops to asu atoms and checking which ones are in the central cell

determine a reference monomer (e.g. the one closest to the center of the unit cell)

determine molecular symmetry by applying symops to the reference monomer and checking which symops leave it invariant (within some tolerance)


"""
