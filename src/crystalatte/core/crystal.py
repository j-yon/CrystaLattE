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
    def lattice_parameters(self) -> tuple[float, float, float, float, float, float]:
        """Return lattice parameters (a, b, c, alpha, beta, gamma)."""
        return self._a, self._b, self._c, self._alpha, self._beta, self._gamma

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        """Return lattice vectors as a 3x3 array (columns are a, b, c)."""
        return self._frac_to_cart

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

    def _get_covalent_radius(self, symbol: str) -> float:
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
        self,
        symbols: list[str],
        cart_coords: NDArray[np.float64],
        bond_tolerance: float = 1.2,
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
        radii = [self._get_covalent_radius(s) for s in symbols]

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(cart_coords[i] - cart_coords[j])
                max_bond = (radii[i] + radii[j]) * bond_tolerance
                if dist < max_bond:
                    adj[i].append(j)
                    adj[j].append(i)

        return adj

    def _find_connected_components(self, adj: list[list[int]]) -> list[list[int]]:
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

    def _distance_matrix(self, a, b):
        """Euclidean distance matrix between rows of arrays `a` and `b`.
        Equivalent to `scipy.spatial.distance.cdist(a, b, 'euclidean')`.
        Returns a.shape[0] x b.shape[0] array. It also returns the shortest
        distance between the atoms of `a` and of `b`.
        """

        assert a.shape[1] == b.shape[1]

        distm = np.sqrt(
            np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :]) ** 2, axis=2)
        )

        r_min = np.min(distm)

        return distm, r_min

    def _bfs(self, coords: NDArray, elements: list[str], cell: NDArray, tol=1.2):
        coords = np.asarray(coords)
        # cell = np.asarray(cell)
        # inv_cell = np.linalg.inv(cell)

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
        """Identify a reference monomer in the crystal by finding the connected component of atoms closest to the origin"""
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

        unique_symbols = []
        unique_frac = []
        for symbol, frac in zip(all_symbols, all_frac):
            if any(np.abs(frac) > 2.0):
                continue

            if not any(np.linalg.norm(frac - uf) < 1e-3 for uf in unique_frac):
                unique_frac.append(frac)
                unique_symbols.append(symbol)

        unique_frac = np.array(unique_frac)
        unique_cart = self.to_cartesian(unique_frac)

        components = self._bfs(unique_cart, unique_symbols, self.lattice_vectors, tol)

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
                    qcel.periodictable.to_mass(sym)
                    for sym in [unique_symbols[j] for j in component]
                ]
            )
            centroid_frac = np.average(mol_frac, axis=0, weights=masses)

            # Check if centroid is new closest to origin
            curr_d = np.linalg.norm(centroid_frac)
            if curr_d < min_d:
                min_d = curr_d
                ref_idx = i

        ref_component = components[ref_idx]
        masses = np.array(
            [
                qcel.periodictable.to_mass(sym)
                for sym in [unique_symbols[j] for j in ref_component]
            ]
        )

        # create Monomer from closest molecule
        monomer = Monomer(
            [unique_symbols[j] for j in ref_component],
            unique_frac[ref_component],
            unique_cart[ref_component],
            np.average(unique_frac, axis=0, weights=masses),
            np.average(unique_cart, axis=0, weights=masses),
        )

        return monomer

    def __repr__(self) -> str:
        return (
            f"Crystal(a={self._a}, b={self._b}, c={self._c}, "
            f"alpha={self._alpha}, beta={self._beta}, gamma={self._gamma}, "
            f"space_group={self._space_group})"
        )
