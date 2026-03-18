from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray


def _get_covalent_radius(symbol: str) -> float:
    """Get covalent radius for an element in Angstroms.

    Parameters:
    symbol : str
        Element symbol.

    Returns:
    float
        Covalent radius in Angstroms.
    """
    # qcelemental returns covalent radii in Bohr
    radius_bohr = qcel.covalentradii.get(symbol)
    return radius_bohr * qcel.constants.bohr2angstroms


def _build_bond_graph(
    symbols: list[str],
    cart_coords: NDArray[np.float64],
    bond_tolerance: float = 0.4,
) -> list[list[int]]:
    """Build adjacency list for atoms based on covalent bonding.

    Two atoms are considered bonded if their distance is less than
    the sum of their covalent radii plus a tolerance.

    Parameters:
    symbols : List[str]
        Atom symbols.
    cart_coords : NDArray[np.float64]
        Cartesian coordinates, shape (N, 3).
    bond_tolerance : float, default=0.4
        Tolerance in Angstroms added to sum of covalent radii.

    Returns:
    List[List[int]]
        Adjacency list where adj[i] contains indices of atoms bonded to atom i.
    """
    n_atoms = len(symbols)
    adj = [[] for _ in range(n_atoms)]

    # Get covalent radii
    radii = [_get_covalent_radius(s) for s in symbols]

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(cart_coords[i] - cart_coords[j])
            max_bond = radii[i] + radii[j] + bond_tolerance
            if dist < max_bond:
                adj[i].append(j)
                adj[j].append(i)

    return adj


def _find_connected_components(adj: list[list[int]]) -> list[list[int]]:
    """Find connected components in a graph using BFS.

    Parameters:
    adj : List[List[int]]
        Adjacency list representation of the graph.

    Returns:
    List of components, where each component is a list of atom indices.
    """
    n = len(adj)
    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue

        # BFS from this node
        component = []
        queue = [start]
        visited[start] = True

        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        components.append(sorted(component))

    return components


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
    cell_index : Tuple[int, int, int]
        Unit cell index (i, j, k) where this monomer is located.
    symop_index : int
        Index of the symmetry operation that generated this monomer.
    centroid_frac : NDArray[np.float64]
        Centroid in fractional coordinates.
    centroid_cart : NDArray[np.float64]
        Centroid in Cartesian coordinates (Angstroms).
    """

    symbols: list[str]
    frac_coords: NDArray[np.float64]
    cart_coords: NDArray[np.float64]
    cell_index: tuple[int, int, int]
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
    distance : float
        Geometric mean of center-to-center distances between monomers in Angstroms.
    symop_indices : List[int]
        Symmetry operation index for each monomer in the multimer.
    cell_indices : List[Tuple[int, int, int]]
        Unit cell index (i, j, k) for each monomer in the multimer.
    multiplicity : int
        Number of symmetry-equivalent copies of this multimer type.
    """

    monomers: list[Monomer]
    distance: float
    symop_indices: list[int]
    cell_indices: list[tuple[int, int, int]]
    multiplicity: int = 1

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


def _identify_molecules_in_cell(
    crystal: Crystal,
    asu_symbols: list[str],
    asu_frac_coords: list[NDArray[np.float64]],
    bond_tolerance: float = 0.4,
) -> list[Monomer]:
    """
    Identify complete molecules in the unit cell using connectivity.

    This function:
    1. Generates all symmetry-equivalent atoms in a 3x3x3 supercell
    2. Builds a bond graph based on covalent radii
    3. Finds connected components (molecules)
    4. Returns molecules whose centroids are in the central unit cell

    Parameters
    ----------
    crystal : Crystal
        Crystal object with symmetry operations.
    asu_symbols : List[str]
        Atom symbols in the asymmetric unit.
    asu_frac_coords : List[NDArray[np.float64]]
        Fractional coordinates of ASU atoms.
    bond_tolerance : float, default=0.4
        Tolerance for bond detection in Angstroms.

    Returns
    -------
    List[MolecularCluster]
        Complete molecules with centroids in the central unit cell.
    """
    # Generate atoms in a 3x3x3 supercell to capture molecules
    # that cross unit cell boundaries
    all_symbols = []
    all_frac = []
    all_cart = []

    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                translation = np.array([float(di), float(dj), float(dk)])
                for symop in crystal.symops:
                    for sym, frac in zip(asu_symbols, asu_frac_coords):
                        new_frac = symop.apply(frac) + translation
                        new_cart = crystal.to_cartesian(new_frac)
                        all_symbols.append(sym)
                        all_frac.append(new_frac)
                        all_cart.append(new_cart)

    all_frac = np.array(all_frac)
    all_cart = np.array(all_cart)

    # Remove duplicate atoms (same position)
    unique_indices = []
    for i in range(len(all_symbols)):
        is_dup = False
        for j in unique_indices:
            if np.linalg.norm(all_cart[i] - all_cart[j]) < 0.01:
                is_dup = True
                break
        if not is_dup:
            unique_indices.append(i)

    symbols = [all_symbols[i] for i in unique_indices]
    frac_coords = all_frac[unique_indices]
    cart_coords = all_cart[unique_indices]

    # Build bond graph and find molecules
    adj = _build_bond_graph(symbols, cart_coords, bond_tolerance)
    components = _find_connected_components(adj)

    # Create Monomer objects for molecules in the central cell
    molecules = []
    mol_idx = 0

    for component in components:
        mol_symbols = [symbols[i] for i in component]
        mol_frac = frac_coords[component]
        mol_cart = cart_coords[component]
        centroid_frac = np.mean(mol_frac, axis=0)
        centroid_cart = np.mean(mol_cart, axis=0)

        # Check if centroid is in the central unit cell [0, 1)
        if (
            0 <= centroid_frac[0] < 1
            and 0 <= centroid_frac[1] < 1
            and 0 <= centroid_frac[2] < 1
        ):
            mol = Monomer(
                symbols=mol_symbols,
                frac_coords=mol_frac,
                cart_coords=mol_cart,
                cell_index=(0, 0, 0),
                centroid_frac=centroid_frac,
                centroid_cart=centroid_cart,
            )
            molecules.append(mol)
            mol_idx += 1

    return molecules
