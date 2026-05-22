from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import qcelemental as qcel
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .space_group import SpaceGroup
from .multimer import ASU, Monomer


@dataclass
class Crystal:
    """
    Represents a crystal unit cell with lattice parameters and space group symmetry.

    :param _vectors: Lattice vectors (a, b, c) in Angstroms
    :param _angles: Lattice angles (alpha, beta, gamma) in degrees
    :param _space_group: The crystal's space group
    :param _asu: The asymmetric unit of the crystal, containing the unique atoms and their positions
    :ivar _frac_to_cart: A 3x3 matrix for converting fractional coordinates to Cartesian coordinates
    :ivar _cart_to_frac: A 3x3 matrix for converting Cartesian coordinates to fractional coordinates
    """

    _vectors: tuple[float, float, float]
    _angles: tuple[float, float, float]
    _space_group: SpaceGroup
    _asu: ASU

    # Computed matrices for coordinate transformations
    _frac_to_cart: NDArray[np.float64] = field(init=False, repr=False)
    _cart_to_frac: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        """Validate crystal attributes and initialize transformation matrices."""
        self._validate_crystal()
        self._update_matrices()

    def _update_matrices(self):
        """Compute fractional <-> Cartesian transformation matrices."""
        a, b, c = self._vectors
        alpha, beta, gamma = np.radians(self._angles)

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
        c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        self._frac_to_cart = np.array(
            [
                [a, b * cos_gamma, c * cos_beta],
                [0.0, b * sin_gamma, c_y],
                [0.0, 0.0, c * omega / sin_gamma],
            ]
        )

        # Cartesian to fractional matrix
        self._cart_to_frac = np.linalg.inv(self._frac_to_cart)  # type: ignore

    def _validate_crystal(self):
        """Validate the provided crystal attributes.

        :raises ValueError: If lattice parameters are non-positive or if angles are not between 0 and 180 degrees
        """
        a, b, c = self._vectors
        alpha, beta, gamma = self._angles
        if a <= 0 or b <= 0 or c <= 0:
            raise ValueError("Lattice parameters must be positive.")

        if not (0 < alpha < 180) or not (0 < beta < 180) or not (0 < gamma < 180):
            raise ValueError("Lattice angles must be between 0 and 180 degrees.")

        # sg = self._space_group
        # asu = self._asu

        # TODO: check lattice parameters and other stuff

    @property
    def space_group(self) -> SpaceGroup:
        """Return the space group of the crystal."""
        return self._space_group

    @property
    def lattice_parameters(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return lattice parameters (a, b, c), (alpha, beta, gamma)."""
        return self._vectors, self._angles

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        """Return lattice vectors as a 3x3 array (columns are a, b, c)."""
        return self._frac_to_cart

    @property
    def volume(self) -> float:
        """Calculate unit cell volume in cubic Angstroms."""
        return abs(np.linalg.det(self._frac_to_cart))

    def to_cartesian(self, frac_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert fractional coordinates to Cartesian coordinates.

        :param frac_coords: Fractional coordinates, shape (N, 3)
        :returns: Cartesian coordinates in Angstroms, shape (N, 3)
        """
        return (self._frac_to_cart @ frac_coords.T).T

    def to_fractional(self, cart_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert Cartesian coordinates to fractional coordinates.

        :param cart_coords: Cartesian coordinates in Angstroms, shape (N, 3)
        :returns: Fractional coordinates, shape (N, 3)
        """
        return (self._cart_to_frac @ cart_coords.T).T

    def _get_covalent_radius(self, symbol: str) -> float:
        """Get covalent radius for an element in Angstroms.

        :param symbol: The atomic symbol of the element (e.g., 'C', 'O', 'N').
        :returns: The covalent radius of the element in Angstroms.
        """
        # qcelemental returns covalent radii in Bohr
        radius_bohr = qcel.covalentradii.get(symbol)
        return radius_bohr * qcel.constants.bohr2angstroms  # type: ignore

    def _bfs(self, coords: NDArray, elements: NDArray, tol=1.2) -> list[list[int]]:
        """Perform a breadth-first search to find connected components of atoms based on covalent bonding.

        :param coords: Cartesian coordinates of atoms, shape (N, 3)
        :param elements: Atomic symbols corresponding to the coordinates, shape (N,)
        :param tol: Tolerance factor for determining bonding (default 1.2)
        :returns: A list of lists, where each inner list contains the indices of atoms in a connected component (monomer)
        """
        coords = np.asarray(coords)

        N = len(coords)
        radii = np.array([self._get_covalent_radius(elem) for elem in elements])

        # --- KD-tree ---
        tree = cKDTree(coords)
        max_radius = np.max(radii)
        global_cutoff = tol * (2 * max_radius)

        pairs = tree.query_pairs(r=global_cutoff)

        adjacency = [[] for _ in range(N)]

        # --- Build graph ---
        for i, j in pairs:
            cutoff = tol * (radii[i] + radii[j])
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                adjacency[i].append(j)
                adjacency[j].append(i)

        # --- BFS with shift propagation ---
        fragments = []
        visited = np.zeros(N, dtype=bool)

        for start in range(N):
            if visited[start]:
                continue

            # BFS
            stack = [start]
            visited[start] = True
            fragment = [start]

            while stack:
                node = stack.pop()
                for neighbor in adjacency[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
                        fragment.append(neighbor)

            fragments.append(fragment)

        return fragments

    def get_reference(self, tol: float = 1.2) -> Monomer:
        """Identify a reference monomer in the crystal by finding the connected component of atoms closest to the origin.

        :param tol: Tolerance factor for determining bonding (default 1.2)
        :returns: A Monomer object representing the reference monomer
        """
        all_symbols = []
        all_frac = []
        for d in product(range(-1, 2), repeat=3):
            translation = np.array(d)
            for i, symop in enumerate(self._space_group.sym_ops):
                for atom, pos in zip(self._asu.atoms, self._asu.positions):
                    new_pos = symop.apply(pos) + translation
                    all_symbols.append(atom)
                    all_frac.append(new_pos)

        all_frac = np.array(all_frac)
        all_symbols = np.array(all_symbols)

        mask = ~np.any(np.abs(all_frac) > 2.0, axis=1)
        all_frac = all_frac[mask]
        all_symbols = all_symbols[mask]

        unique_indices = []
        for i in range(len(all_frac)):
            if not unique_indices:
                unique_indices.append(i)
                continue
            accepted = all_frac[unique_indices]
            diffs = accepted - all_frac[i]
            dists = np.linalg.norm(diffs, axis=1)
            if np.all(dists >= 1e-3):
                unique_indices.append(i)

        unique_frac = all_frac[unique_indices]
        unique_symbols = all_symbols[unique_indices]

        unique_cart = self.to_cartesian(unique_frac)
        components = self._bfs(unique_cart, unique_symbols, tol)

        # NOTE: always assumes a full molecule is formed (which should be true)
        mol_len = max(len(c) for c in components)

        min_d = np.inf
        ref_idx = 0
        for i, component in enumerate(components):
            if len(component) < mol_len:
                continue

            mol_frac = unique_frac[component]

            # needed for mass-accurate CoM
            masses = np.array(
                [
                    qcel.periodictable.to_mass(str(sym))
                    for sym in [unique_symbols[j] for j in component]
                ]
            )
            centroid_frac = np.average(mol_frac, axis=0, weights=masses)

            # Check if centroid is new closest to origin (NOT center of cell, can mess with symops)
            curr_d = np.linalg.norm(centroid_frac)
            if curr_d < min_d:
                min_d = curr_d
                ref_idx = i

        ref_component = components[ref_idx]
        masses = np.array(
            [
                qcel.periodictable.to_mass(str(sym))
                for sym in [unique_symbols[j] for j in ref_component]
            ]
        )

        # create Monomer from closest molecule
        monomer = Monomer(
            [str(unique_symbols[j]) for j in ref_component],
            unique_frac[ref_component],
            unique_cart[ref_component],
            np.average(unique_frac[ref_component], axis=0, weights=masses),
            np.average(unique_cart[ref_component], axis=0, weights=masses),
        )

        return monomer

    def __repr__(self) -> str:
        return (
            f"Crystal(vectors={self._vectors}, angles={self._angles}, "
            f"space_group={self._space_group})"
        )


if __name__ == "__main__":
    pass
