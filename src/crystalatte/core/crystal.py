from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import time
import itertools

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray

from .sym_ops import SymOp
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
        radii = [self._get_covalent_radius(s) for s in symbols]

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(cart_coords[i] - cart_coords[j])
                max_bond = radii[i] + radii[j] + bond_tolerance
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

    def _bfs(self, geom, elems, bfs_thresh):
        """Linear scaling breadth first search for partitioning a set of atoms into covalently bound molecules"""

        natom = geom.shape[0]

        # van der waals radii of each atom type
        radii = np.array([qcel.covalentradii.get(elem) for elem in elems])
        max_radius = np.max(radii)

        # this is the max distance between any two covalently bound atoms
        blocksize = int(np.ceil(2.0 * bfs_thresh * max_radius))

        # map each atom to a "cube" with dimension blocksize ** 3
        geom_floor = np.floor(geom).astype(int)
        atomkeys = [
            (pos[0], pos[1], pos[2])
            for pos in geom_floor - (np.mod(geom_floor, blocksize))
        ]
        atomkey_to_atoms = dict.fromkeys(atomkeys)

        for k, _ in atomkey_to_atoms.items():
            atomkey_to_atoms[k] = []

        for atomind, atomkey in enumerate(atomkeys):
            atomkey_to_atoms[atomkey].append(atomind)

        # list of covalently bonded neighbors for each atom
        neighborlist = [[] for atomind in range(natom)]

        for atomkey, atoms in atomkey_to_atoms.items():
            # within the loop, we'll find neighbors of the atoms in "cube" atomkey
            # neighbors have to be in the same cube, or one cube over
            otherkeys = []
            for keyx in [atomkey[0] - blocksize, atomkey[0], atomkey[0] + blocksize]:
                for keyy in [
                    atomkey[1] - blocksize,
                    atomkey[1],
                    atomkey[1] + blocksize,
                ]:
                    for keyz in [
                        atomkey[2] - blocksize,
                        atomkey[2],
                        atomkey[2] + blocksize,
                    ]:
                        otherkey = (keyx, keyy, keyz)
                        if otherkey in atomkey_to_atoms:
                            otherkeys.append(otherkey)

            # the atoms in either our cube or the neighboring cube
            otheratoms = list(
                itertools.chain(*[atomkey_to_atoms[otherkey] for otherkey in otherkeys])
            )

            geom_atoms = geom[atoms]
            geom_others = geom[otheratoms]

            rad_atoms = radii[atoms]
            rad_others = radii[otheratoms]

            dist, _ = self._distance_matrix(geom_atoms, geom_others)
            bound = 1.2 * (rad_atoms.reshape(-1, 1) + rad_others.reshape(1, -1))

            # list of all bonds involving atoms from our cube
            bond_sources, bond_targets = np.where(dist < bound)

            # update the neighbor list with bonds
            for bond_ind, bond_source in enumerate(bond_sources):
                bond_target = bond_targets[bond_ind]
                atomind = atoms[bond_source]
                otheratomind = otheratoms[bond_target]
                atomkey = atomkeys[atomind]
                if otheratomind != atomind:
                    neighborlist[atomind].append(otheratomind)

        # now that we have the neighborlist, we need to perform BFS to get fragments

        # has the atom already been assigned to a fragment?
        in_fragment = [False] * natom

        # list of complete fragments
        fragments = []

        # this index traverses all atoms, looking for atoms not in a fragment
        outerind = 0
        while outerind < natom:
            # atom outerind already in a fragment
            if in_fragment[outerind]:
                outerind += 1
                continue

            # make a new fragment, containing outerind
            fragment = [outerind]

            # this indexes traverses neighbors of outerind, adding them to this fragment
            innerind = 0

            while innerind < len(fragment):
                # we've already added the neighbor to this fragment
                if in_fragment[fragment[innerind]]:
                    innerind += 1
                    continue

                in_fragment[fragment[innerind]] = True
                fragment.extend(neighborlist[fragment[innerind]])
                innerind += 1

            fragment = sorted(set(fragment))
            fragments.append(fragment)
            outerind += 1

        return fragments

    def get_reference(self, tol: float = 0.1) -> Monomer:
        """Identify a reference monomer in the crystal by finding the connected component of atoms closest to the center of the unit cell (0.5, 0.5, 0.5) in fractional coordinates."""
        all_symbols = []
        all_frac = []
        all_cart = []

        # Generate atoms in a 3x3x3 supercell to capture molecules that cross unit cell boundaries
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    translation = np.array([float(di), float(dj), float(dk)])
                    for symop in self._space_group.sym_ops:
                        for sym, frac in zip(self._asu.atoms, self._asu.positions):
                            new_frac = symop.apply(frac) + translation
                            new_cart = self.to_cartesian(new_frac)
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
        adj = self._build_bond_graph(symbols, cart_coords, tol)
        components = self._find_connected_components(adj)

        # TODO: improve bfs algorithm
        # components = self._bfs(frac_coords, symbols, tol)

        min_d = np.inf
        close_idx = 0
        for i, component in enumerate(components):
            mol_symbols = [symbols[j] for j in component]
            mol_frac = frac_coords[component]
            mol_cart = cart_coords[component]
            centroid_frac = np.mean(mol_frac, axis=0)
            centroid_cart = np.mean(mol_cart, axis=0)

            # Check if centroid is new closest to center (0.5, 0.5, 0.5)
            curr_d = np.linalg.norm(centroid_frac - np.array([0.5, 0.5, 0.5]))
            if curr_d < min_d:
                min_d = curr_d
                close_idx = i

        # create Monomer from closest molecule
        ref_component = components[close_idx]
        monomer = Monomer(
            symbols=[symbols[j] for j in ref_component],
            frac_coords=frac_coords[ref_component],
            cart_coords=cart_coords[ref_component],
            centroid_frac=np.mean(frac_coords[ref_component], axis=0),
            centroid_cart=np.mean(cart_coords[ref_component], axis=0),
        )

        return monomer

    def __repr__(self) -> str:
        return (
            f"Crystal(a={self._a}, b={self._b}, c={self._c}, "
            f"alpha={self._alpha}, beta={self._beta}, gamma={self._gamma}, "
            f"space_group={self._space_group})"
        )
