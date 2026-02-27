from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray

from CrystaLattE.sym_ops import SymOp


@dataclass
class Crystal:
    """
    Represents a crystal unit cell with lattice parameters and space group symmetry.

    Attributes
    ----------
    a : float
        Length of the a-axis in Angstroms.
    b : float
        Length of the b-axis in Angstroms.
    c : float
        Length of the c-axis in Angstroms.
    alpha : float
        Angle between b and c axes in degrees.
    beta : float
        Angle between a and c axes in degrees.
    gamma : float
        Angle between a and b axes in degrees.
    space_group_name : str
        Space group name (Hermann-Mauguin notation).
    space_group_number : int
        International Tables space group number.
    symops : List[SymOp]
        List of symmetry operations.
    """

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    space_group_name: str = "P1"
    space_group_number: int = 1
    symops: list[SymOp] = field(default_factory=list)

    # Computed matrices (set in __post_init__)
    _frac_to_cart: NDArray[np.float64] = field(init=False, repr=False)
    _cart_to_frac: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize transformation matrices."""
        self._update_matrices()
        if not self.symops:
            self.symops = [SymOp.identity()]

    def _update_matrices(self):
        """Compute fractional <-> Cartesian transformation matrices."""
        # Convert angles to radians
        alpha_rad = np.radians(self.alpha)
        beta_rad = np.radians(self.beta)
        gamma_rad = np.radians(self.gamma)

        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad)
        cos_gamma = np.cos(gamma_rad)
        sin_gamma = np.sin(gamma_rad)

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
        c_y = self.c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        self._frac_to_cart = np.array(
            [
                [self.a, self.b * cos_gamma, self.c * cos_beta],
                [0.0, self.b * sin_gamma, c_y],
                [0.0, 0.0, self.c * omega / sin_gamma],
            ]
        )

        # Cartesian to fractional matrix
        self._cart_to_frac = np.linalg.inv(self._frac_to_cart)

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

    def apply_symop(
        self, frac_coords: NDArray[np.float64], symop: SymOp
    ) -> NDArray[np.float64]:
        """
        Apply a symmetry operation to fractional coordinates.

        Parameters
        ----------
        frac_coords : NDArray[np.float64]
            Fractional coordinates.
        symop : SymOp
            Symmetry operation to apply.

        Returns
        -------
        NDArray[np.float64]
            Transformed fractional coordinates.
        """
        return symop.apply(frac_coords)

    def generate_symmetry_equivalents(
        self,
        frac_coords: NDArray[np.float64],
        wrap_to_unit_cell: bool = True,
    ) -> NDArray[np.float64]:
        """
        Generate all symmetry-equivalent positions for given fractional coordinates.

        Parameters
        ----------
        frac_coords : NDArray[np.float64]
            Fractional coordinates, shape (N, 3).
        wrap_to_unit_cell : bool
            If True, wrap coordinates to [0, 1).

        Returns
        -------
        NDArray[np.float64]
            All symmetry-equivalent positions, shape (N * M, 3)
            where M is the number of symmetry operations.
        """
        all_positions = []
        for symop in self.symops:
            transformed = symop.apply(frac_coords)
            if wrap_to_unit_cell:
                transformed = transformed % 1.0
            all_positions.append(transformed)

        result = np.vstack(all_positions)
        return result


def get_crystal_info(filepath: str | Path) -> dict:
    """
    Extract crystal information from a CIF file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.

    Returns
    -------
    dict
        Dictionary containing crystal information:
        - 'a', 'b', 'c': lattice parameters (Angstroms)
        - 'alpha', 'beta', 'gamma': lattice angles (degrees)
        - 'volume': unit cell volume (cubic Angstroms)
        - 'space_group_name': space group name
        - 'space_group_number': IT space group number
        - 'n_symops': number of symmetry operations
        - 'n_atoms_asu': number of atoms in asymmetric unit
    """
    crystal, symbols, frac_coords = parse_cif_file(filepath)

    return {
        "a": crystal.a,
        "b": crystal.b,
        "c": crystal.c,
        "alpha": crystal.alpha,
        "beta": crystal.beta,
        "gamma": crystal.gamma,
        "volume": crystal.volume,
        "space_group_name": crystal.space_group_name,
        "space_group_number": crystal.space_group_number,
        "n_symops": len(crystal.symops),
        "n_atoms_asu": len(symbols),
    }


def generate_spherical_cluster(
    filepath: str | Path,
    radius: float,
    center: NDArray[np.float64] | None = None,
    remove_duplicates: bool = True,
    duplicate_tolerance: float = 0.01,
) -> tuple[qcel.models.Molecule, Crystal, list[tuple[int, int, int]]]:
    """
    Generate an approximately spherical cluster of molecules from a CIF file.

    This function creates a supercell that extends far enough in all directions
    to include all unit cells whose centers fall within the specified radius
    from a central reference point. The result is an approximately spherical
    cluster of molecules.

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.
    radius : float
        Radius in Angstroms from the center point. Unit cells with any atom
        within this radius from the center will be included.
    center : NDArray[np.float64], optional
        Center point in fractional coordinates. Defaults to (0.5, 0.5, 0.5),
        the center of the reference unit cell.
    remove_duplicates : bool, default=True
        If True, remove duplicate atoms at unit cell boundaries.
    duplicate_tolerance : float, default=0.01
        Distance tolerance (in Angstroms) for detecting duplicates.

    Returns
    -------
    Tuple[qcel.models.Molecule, Crystal, List[Tuple[int, int, int]]]
        - QCElemental Molecule object with all atoms in the cluster
        - Crystal object with unit cell information
        - List of (i, j, k) unit cell indices included in the cluster

    Examples
    --------
    >>> mol, crystal, cells = generate_spherical_cluster("structure.cif", 15.0)
    >>> print(f"Cluster contains {len(mol.symbols)} atoms")
    >>> print(f"From {len(cells)} unit cells")
    """
    crystal, asu_symbols, asu_frac_coords = parse_cif_file(filepath)

    if center is None:
        center = np.array([0.5, 0.5, 0.5])
    else:
        center = np.asarray(center)

    # First, generate the full unit cell (symmetry-expanded)
    unit_cell_symbols = []
    unit_cell_frac_coords = []

    for symbol, frac_coord in zip(asu_symbols, asu_frac_coords):
        equiv_coords = crystal.generate_symmetry_equivalents(
            frac_coord, wrap_to_unit_cell=True
        )
        for coord in equiv_coords:
            unit_cell_symbols.append(symbol)
            unit_cell_frac_coords.append(coord)

    unit_cell_frac_coords = np.array(unit_cell_frac_coords)

    # Remove duplicates within the unit cell
    if remove_duplicates:
        unit_cell_symbols, unit_cell_frac_coords = remove_duplicate_atoms(
            unit_cell_symbols, unit_cell_frac_coords, tolerance=0.01
        )

    # Calculate how many unit cells we need in each direction
    # Use the lattice vectors to estimate the range
    center_cart = crystal.to_cartesian(center)

    # Calculate the maximum number of cells needed in each direction
    # by considering the lattice parameters
    n_a = int(np.ceil(radius / crystal.a)) + 1
    n_b = int(np.ceil(radius / crystal.b)) + 1
    n_c = int(np.ceil(radius / crystal.c)) + 1

    # Collect all atoms within the spherical region
    all_symbols = []
    all_cart_coords = []
    included_cells = []

    # Iterate over all potentially relevant unit cells
    for i in range(-n_a, n_a + 1):
        for j in range(-n_b, n_b + 1):
            for k in range(-n_c, n_c + 1):
                # Translation vector for this unit cell
                translation = np.array([float(i), float(j), float(k)])

                # Check if any atom from this cell is within radius
                cell_atoms_in_range = False
                cell_symbols = []
                cell_coords = []

                for sym, frac in zip(unit_cell_symbols, unit_cell_frac_coords):
                    # Translate fractional coordinates
                    translated_frac = frac + translation

                    # Convert to Cartesian
                    cart = crystal.to_cartesian(translated_frac)

                    # Check distance from center
                    dist = np.linalg.norm(cart - center_cart)

                    if dist <= radius:
                        cell_atoms_in_range = True
                        cell_symbols.append(sym)
                        cell_coords.append(cart)

                # If any atom is in range, include all atoms from this cell
                # that are within the radius
                if cell_atoms_in_range:
                    all_symbols.extend(cell_symbols)
                    all_cart_coords.extend(cell_coords)
                    if (i, j, k) not in included_cells:
                        included_cells.append((i, j, k))

    all_cart_coords = np.array(all_cart_coords)

    # Remove duplicate atoms at boundaries (in Cartesian space)
    if remove_duplicates and len(all_symbols) > 0:
        all_symbols, all_cart_coords = _remove_duplicate_atoms_cartesian(
            all_symbols, all_cart_coords, tolerance=duplicate_tolerance
        )

    # Convert Angstroms to Bohr for QCElemental
    cart_coords_bohr = all_cart_coords / qcel.constants.bohr2angstroms

    # Create QCElemental Molecule
    mol = qcel.models.Molecule(
        symbols=all_symbols,
        geometry=cart_coords_bohr.flatten(),
    )

    return mol, crystal, included_cells


def _remove_duplicate_atoms_cartesian(
    symbols: list[str],
    coords: NDArray[np.float64],
    tolerance: float = 0.01,
) -> tuple[list[str], NDArray[np.float64]]:
    """
    Remove duplicate atoms based on Cartesian coordinate proximity.

    Parameters
    ----------
    symbols : List[str]
        List of atom symbols.
    coords : NDArray[np.float64]
        Cartesian coordinates array of shape (N, 3).
    tolerance : float
        Distance tolerance in Angstroms for considering atoms as duplicates.

    Returns
    -------
    Tuple[List[str], NDArray[np.float64]]
        Unique atom symbols and coordinates.
    """
    if len(symbols) == 0:
        return symbols, coords

    unique_symbols = [symbols[0]]
    unique_coords = [coords[0]]

    for i in range(1, len(symbols)):
        is_duplicate = False
        for j in range(len(unique_coords)):
            dist = np.linalg.norm(coords[i] - unique_coords[j])
            if dist < tolerance:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_symbols.append(symbols[i])
            unique_coords.append(coords[i])

    return unique_symbols, np.array(unique_coords)


def generate_supercell(
    filepath: str | Path,
    na: int = 1,
    nb: int = 1,
    nc: int = 1,
    remove_duplicates: bool = True,
    duplicate_tolerance: float = 0.01,
) -> qcel.models.Molecule:
    """
    Generate a supercell by replicating the unit cell.

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.
    na, nb, nc : int
        Number of unit cell replications along a, b, c axes.
        Use negative values to extend in both directions (e.g., na=-2
        creates cells from -2 to +2 along a).
    remove_duplicates : bool, default=True
        If True, remove duplicate atoms at unit cell boundaries.
    duplicate_tolerance : float, default=0.01
        Distance tolerance (in Angstroms) for detecting duplicates.

    Returns
    -------
    qcel.models.Molecule
        QCElemental Molecule object with the supercell.

    Examples
    --------
    >>> mol = generate_supercell("structure.cif", na=2, nb=2, nc=2)
    >>> # Creates a 2x2x2 supercell (8 unit cells)
    """
    crystal, asu_symbols, asu_frac_coords = parse_cif_file(filepath)

    # Generate full unit cell
    unit_cell_symbols = []
    unit_cell_frac_coords = []

    for symbol, frac_coord in zip(asu_symbols, asu_frac_coords):
        equiv_coords = crystal.generate_symmetry_equivalents(
            frac_coord, wrap_to_unit_cell=True
        )
        for coord in equiv_coords:
            unit_cell_symbols.append(symbol)
            unit_cell_frac_coords.append(coord)

    unit_cell_frac_coords = np.array(unit_cell_frac_coords)

    if remove_duplicates:
        unit_cell_symbols, unit_cell_frac_coords = remove_duplicate_atoms(
            unit_cell_symbols, unit_cell_frac_coords, tolerance=0.01
        )

    # Determine range for each axis
    def get_range(n):
        if n < 0:
            return range(n, -n + 1)
        return range(n)

    range_a = get_range(na)
    range_b = get_range(nb)
    range_c = get_range(nc)

    # Generate supercell
    all_symbols = []
    all_cart_coords = []

    for i in range_a:
        for j in range_b:
            for k in range_c:
                translation = np.array([float(i), float(j), float(k)])

                for sym, frac in zip(unit_cell_symbols, unit_cell_frac_coords):
                    translated_frac = frac + translation
                    cart = crystal.to_cartesian(translated_frac)
                    all_symbols.append(sym)
                    all_cart_coords.append(cart)

    all_cart_coords = np.array(all_cart_coords)

    if remove_duplicates and len(all_symbols) > 0:
        all_symbols, all_cart_coords = _remove_duplicate_atoms_cartesian(
            all_symbols, all_cart_coords, tolerance=duplicate_tolerance
        )

    # Convert to Bohr
    cart_coords_bohr = all_cart_coords / qcel.constants.bohr2angstroms

    mol = qcel.models.Molecule(
        symbols=all_symbols,
        geometry=cart_coords_bohr.flatten(),
    )

    return mol
