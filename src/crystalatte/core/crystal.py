from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .multimer import ASU, Monomer
from .space_group import SpaceGroup
from .sym_ops import SymOp


@dataclass
class Crystal:
    """Represents a crystal unit cell with lattice parameters and space group symmetry

    :param _vectors: lattice vectors (a, b, c)
    :param _angles: lattice angles (alpha, beta, gamma)
    :param _space_group: SpaceGroup object representing this crystal's symmetry
    :param _asu: asymmetric unit of this crystal
    :ivar _frac_to_cart: 3x3 matrix to convert fractional to Cartesian
    """

    _vectors: tuple[float, float, float]
    _angles: tuple[float, float, float]
    _space_group: SpaceGroup
    _asu: ASU

    # computed matrices for coordinate transformations
    _frac_to_cart: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        """Validate crystal attributes and initialize transformation matrices"""
        self._validate_crystal()
        self._update_matrices()

    def _update_matrices(self):
        """Compute the fractional -> Cartesian transformation matrix"""
        a, b, c = self._vectors
        alpha, beta, gamma = np.radians(self._angles)

        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)

        # volume factor
        omega = np.sqrt(
            1
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
            + 2 * cos_alpha * cos_beta * cos_gamma
        )

        # standard crystallographic convention:
        # a along x, b in xy plane, c general
        c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        self._frac_to_cart = np.array(
            [
                [a, b * cos_gamma, c * cos_beta],
                [0.0, b * sin_gamma, c_y],
                [0.0, 0.0, c * omega / sin_gamma],
            ]
        )

    def _validate_crystal(self):
        """Validate the provided crystal attributes

        :raises ValueError: if lattice parameters are non-positive or if angles are not between 0 and 180 degrees
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

    def __repr__(self) -> str:
        return (
            f"Crystal(vectors={self._vectors}, angles={self._angles}, "
            f"space_group={self._space_group})"
        )

    @property
    def space_group(self) -> SpaceGroup:
        return self._space_group

    @property
    def lattice_parameters(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return self._vectors, self._angles

    @property
    def lattice_vectors(self) -> NDArray[np.float64]:
        return self._frac_to_cart

    @property
    def volume(self) -> float:
        return abs(np.linalg.det(self._frac_to_cart))

    def _get_covalent_radius(self, symbol: str) -> float:
        """Get covalent radius for an element in Angstroms

        :param symbol: atomic symbol of the element
        :returns: covalent radius of the element in Angstroms
        """
        # TODO: dont use qcel
        radius_bohr = qcel.covalentradii.get(symbol)
        return radius_bohr * qcel.constants.bohr2angstroms

    def _bfs(self, coords: NDArray, elements: NDArray, tol=1.2) -> list[list[int]]:
        """Perform BFS to find connected components of atoms based on covalent bonding

        :param coords: Cartesian coordinates of atoms, shape (N, 3)
        :param elements: atomic symbols corresponding to the coordinates, shape (N,)
        :param tol: tolerance factor for determining bonding
        :returns: list of indices of atoms in a connected component (monomer)
        """
        coords = np.asarray(coords)

        N = len(coords)
        radii = np.array([self._get_covalent_radius(elem) for elem in elements])

        # KD-tree
        tree = cKDTree(coords)
        max_radius = np.max(radii)
        global_cutoff = tol * (2 * max_radius)

        pairs = tree.query_pairs(r=global_cutoff)

        adjacency = [[] for _ in range(N)]

        # build adjacency matrix
        for i, j in pairs:
            cutoff = tol * (radii[i] + radii[j])
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                adjacency[i].append(j)
                adjacency[j].append(i)

        # BFS with shift propagation
        fragments = []
        visited = np.zeros(N, dtype=bool)
        for start in range(N):
            if visited[start]:
                continue

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

    def _find_molecules(self, tol: float = 1.2):
        """Expand the ASU and BFS into complete molecules

        :param tol: tolerance factor for determining bonding
        :returns: unique fractional/Cartesian coordinates, symbols, and list of complete molecule indices
        """
        all_symbols, all_frac = [], []

        # build 3x3x3 supercell of ASU images to ensure complete molecules are captured
        for d in product(range(-1, 2), repeat=3):
            translation = np.array(d)
            for symop in self._space_group.sym_ops:
                for atom, pos in zip(self._asu.atoms, self._asu.positions):
                    all_symbols.append(atom)
                    all_frac.append(symop.apply(pos) + translation)

        all_frac = np.array(all_frac)
        all_symbols = np.array(all_symbols)

        # filter out any atoms that are more than 2 fractional units away from the origin
        mask = ~np.any(np.abs(all_frac) > 2.0, axis=1)
        all_frac, all_symbols = all_frac[mask], all_symbols[mask]

        # determine superimposed atoms: an atom is dropped iff an earlier KEPT atom lies within 1e-3.
        # The tree only proposes pairs; the strict ``< 1e-3`` test is the original norm, so results are exact
        pairs = cKDTree(all_frac).query_pairs(1e-3, output_type="ndarray")  # i < j
        close = np.linalg.norm(all_frac[pairs[:, 0]] - all_frac[pairs[:, 1]], axis=1) < 1e-3
        pairs = pairs[close]
        keep = np.ones(len(all_frac), dtype=bool)
        # visiting pairs by ascending j settles every keep[i] (i < j) before it is read
        for i, j in pairs[np.argsort(pairs[:, 1], kind="stable")].tolist():
            if keep[i]:
                keep[j] = False
        unique_indices = np.flatnonzero(keep)

        # deduplicate the arrays based on the unique indices
        unique_frac = all_frac[unique_indices]
        unique_symbols = all_symbols[unique_indices]
        unique_cart = self.to_cartesian(unique_frac)

        # determine connected components using BFS
        components = self._bfs(unique_cart, unique_symbols, tol)
        mol_len = max(len(c) for c in components)
        full = [c for c in components if len(c) == mol_len]
        return unique_frac, unique_cart, unique_symbols, full

    def _build_monomer(self, comp, unique_frac, unique_cart, unique_symbols) -> Monomer:
        """Construct a Monomer object from a component of unique atoms

        :param comp: indices of the atoms in the component
        :param unique_frac: array of unique fractional coordinates
        :param unique_cart: array of unique Cartesian coordinates
        :param unique_symbols: array of unique atomic symbols
        :returns: Monomer object representing the component
        """
        syms = [str(unique_symbols[j]) for j in comp]
        masses = np.array([qcel.periodictable.to_mass(s) for s in syms])
        return Monomer(
            syms,
            unique_frac[comp],
            unique_cart[comp],
            np.average(unique_frac[comp], axis=0, weights=masses),
            np.average(unique_cart[comp], axis=0, weights=masses),
        )

    def to_cartesian(self, frac_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert fractional coordinates to Cartesian coordinates

        :param frac_coords: fractional coordinates, shape (N, 3)
        :returns: Cartesian coordinates in Angstroms, shape (N, 3)
        """
        return (self._frac_to_cart @ frac_coords.T).T

    def get_references(self, tol: float = 1.2) -> list[Monomer]:
        """Return one Monomer per molecular symmetry-orbit in the ASU (total Z')

        Two complete molecules are the same reference iff a SymOp maps one onto the other

        :param tol: tolerance factor for determining bonding
        :returns: Monomer list representing the reference monomers
        """
        unique_frac, unique_cart, unique_symbols, full = self._find_molecules(tol)
        ops = self._space_group.sym_ops

        reps: dict[tuple, tuple[float, list]] = {}

        # iterate over all complete molecules and find the representative of each symmetry orbit
        for comp in full:
            syms = [str(unique_symbols[j]) for j in comp]
            masses = np.array([qcel.periodictable.to_mass(s) for s in syms])
            cen = np.average(unique_frac[comp], axis=0, weights=masses)
            imgs = []

            # apply all symmetry operations to the centroid and find images
            for op in ops:
                r = op.apply(cen) % 1.0
                r[r > 1.0 - 1e-3] -= 1.0  # wrap 0.999... -> ~0
                imgs.append(tuple(np.round(r, 3)))

            # assign minimum image as representative key
            key = min(imgs)
            d = float(np.linalg.norm(cen))
            if key not in reps or d < reps[key][0]:
                reps[key] = (d, comp)

        # only keep the closest representative of each symmetry orbit
        ordered = [comp for _, comp in sorted(reps.values(), key=lambda t: t[0])]
        return [
            self._build_monomer(c, unique_frac, unique_cart, unique_symbols)
            for c in ordered
        ]

    def get_reference(self, tol: float = 1.2) -> Monomer:
        """Return the single reference molecule closest to the origin

        :param tol: tolerance factor for determining bonding
        :returns: Monomer object representing the reference monomer
        """
        return self.get_references(tol)[0]

    def detect_symmetry(self, symprec: float = 1e-3) -> Crystal:
        """Attempt to detect a higher true space group with spglib

        :param symprec: spglib symmetry tolerance
        :returns: higher-symmetry Crystal, or ``self``
        """
        import warnings

        import spglib

        lattice = self._frac_to_cart.T

        # expand ASU to full cell using declared symmetry ops
        frac, nums = [], []
        for op in self._space_group.sym_ops:
            for pos, sym in zip(self._asu.positions, self._asu.atoms):
                frac.append(op.apply(np.asarray(pos, dtype=float)) % 1.0)
                nums.append(qcel.periodictable.to_Z(sym))
        frac = np.array(frac)
        nums = np.array(nums)

        # dedup coincident atoms (mod 1)
        keep = []
        for i in range(len(frac)):
            if not keep:
                keep.append(i)
                continue
            d = (frac[keep] - frac[i] + 0.5) % 1.0 - 0.5  # periodic distance
            if np.all(np.linalg.norm(d, axis=1) >= 1e-3):
                keep.append(i)
        positions = frac[keep]
        numbers = nums[keep].tolist()

        # call spglib to detect symmetry, ignoring warnings (spglib can be noisy)
        # spglib signals failure either by raising or by returning None; both fall back
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sym = spglib.get_symmetry(
                    (lattice, positions, numbers), symprec=symprec
                )
        except Exception as exc:
            warnings.warn(
                f"symmetry detection failed, using declared symmetry: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self

        # if no symmetry is detected, return with declared symmetry
        if sym is None:
            warnings.warn(
                "symmetry detection failed, using declared symmetry",
                RuntimeWarning,
                stacklevel=2,
            )
            return self

        # if detected symmetry is lower than declared, return with declared symmetry
        detected = [
            SymOp.from_components(
                np.asarray(r, dtype=np.int16),
                np.round(np.asarray(t) * 12).astype(np.int16),
            )
            for r, t in zip(sym["rotations"], sym["translations"])
        ]
        declared = self._space_group.sym_ops
        if len(detected) <= len(declared):
            if len(detected) < len(declared):
                warnings.warn(
                    "symmetry detection found lower symmetry than declared, "
                    "still using declared symmetry",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return self

        # check that each declared op appears in new detected symmetry
        def _same_op(a, b):
            return np.array_equal(a.rot, b.rot) and np.array_equal(
                (a.tr - b.tr) % 12, np.zeros(3, dtype=a.tr.dtype)
            )

        if not all(any(_same_op(op, d) for d in detected) for op in declared):
            return self

        # isolate metadata and create new Crystal with detected symmetry
        # number is not important if it can't be determined
        number = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = spglib.get_symmetry_dataset(
                    (lattice, positions, numbers), symprec=symprec
                )
            number = getattr(ds, "number", None) if ds is not None else None
        except Exception:
            number = None
        new_sg = SpaceGroup.from_sym_ops(detected, number=number)
        return Crystal(self._vectors, self._angles, new_sg, self._asu)


if __name__ == "__main__":
    pass
