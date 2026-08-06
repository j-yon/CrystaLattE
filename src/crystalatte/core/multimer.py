from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from itertools import combinations

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray

from .sym_ops import SymOp, SymOpList


def _z_labeled_match(
    transformed: NDArray, target: NDArray, Z: NDArray, atol: float
) -> bool:
    """Return True iff ``transformed`` and ``target`` are the same labeled point
    set: every target atom is matched to exactly one transformed atom of the
    same atomic number within Cartesian tolerance ``atol``.

    Used by the self-isometry detection below and by any caller that wants to
    check whether two equally-Z-labeled atom clouds coincide as point sets.

    :param transformed: The transformed atom positions, shape ``(n, 3)``
    :param target: The reference atom positions, shape ``(n, 3)``
    :param Z: The per-atom atomic numbers, shape ``(n,)``
    :param atol: Cartesian matching tolerance in Angstroms
    :returns: True iff the two labeled point sets coincide
    """
    n = len(transformed)
    if n != len(target):
        return False
    matched = np.zeros(n, dtype=bool)
    for i in range(n):
        diff = target - transformed[i]
        d2 = np.einsum("ij,ij->i", diff, diff)
        candidates = np.where((Z == Z[i]) & ~matched & (d2 < atol * atol))[0]
        if len(candidates) == 0:
            return False
        matched[candidates[0]] = True
    return True


def _candidate_axes(pos: NDArray) -> list[NDArray]:
    """Enumerate candidate symmetry axes / mirror normals for a
    centroid-centered atom cloud.

    The candidates are atom directions, atom-pair midpoints and perpendiculars,
    and the PCA eigenvectors (which catch planar molecules), deduplicated by
    direction (an axis and its negative are merged). This is intentionally
    minimal — a candidate library, not a rigorous point-group solver — but it
    covers the axes that real molecular symmetry elements tend to lie along.

    :param pos: The centroid-centered atom positions, shape ``(n, 3)``
    :returns: A list of unit-vector candidate axes
    """
    axes: list[NDArray] = []
    n_atoms = len(pos)
    for p in pos:
        nrm = np.linalg.norm(p)
        if nrm > 1e-3:
            axes.append(p / nrm)
    for i, j in combinations(range(n_atoms), 2):
        for v in ((pos[i] + pos[j]) / 2, np.cross(pos[i], pos[j])):
            nrm = np.linalg.norm(v)
            if nrm > 1e-3:
                axes.append(v / nrm)
    eigvecs = np.linalg.eigh(pos.T @ pos)[1]
    axes.extend(eigvecs[:, k] for k in range(3))

    unique_axes: list[NDArray] = []
    for v in axes:
        if not any(abs(abs(np.dot(u, v)) - 1) < 1e-5 for u in unique_axes):
            unique_axes.append(v)
    return unique_axes


def _rotation(axis: NDArray, angle: float) -> NDArray:
    """Return the Rodrigues rotation matrix about a unit axis.

    :param axis: The unit rotation axis, shape ``(3,)``
    :param angle: The rotation angle in radians
    :returns: The ``(3, 3)`` rotation matrix
    """
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _symmetry_candidates(pos: NDArray) -> list[NDArray]:
    """Generate candidate orthogonal operations to test against a
    centroid-centered atom cloud, pruned by the cloud's gyration tensor.

    Any symmetry operation ``O`` of the cloud commutes with the gyration tensor
    ``G = sum_i outer(r_i, r_i)`` (it just permutes the points), so ``O`` must
    map each of ``G``'s eigenspaces to itself:

    - Asymmetric top (three distinct eigenvalues — the generic / chiral case):
      every eigenaxis is unique, so ``O`` is forced to be
      ``diag(+/-1, +/-1, +/-1)`` in the principal-axis frame. That is exactly
      the 8 elements of D2h, so we return just those — no atom-pair axis sweep.
      This collapses the common case from ``~O(n_atoms^2)`` candidates to 8.
    - Degenerate top (a repeated eigenvalue — symmetric/spherical, e.g. planar
      or high-symmetry molecules): a rotation axis can lie anywhere in the
      degenerate eigenplane, so fall back to the broad :func:`_candidate_axes`
      library (identity + inversion, a mirror per axis, rotations of order
      2/3/4/6).

    :param pos: The centroid-centered atom positions, shape ``(n, 3)``
    :returns: A list of candidate ``(3, 3)`` orthogonal matrices
    """
    w, V = np.linalg.eigh(pos.T @ pos)  # ascending eigenvalues, eigenvectors in columns
    scale = max(float(w[-1]), 1.0)
    asymmetric = (w[1] - w[0] > 1e-3 * scale) and (w[2] - w[1] > 1e-3 * scale)

    if asymmetric:
        signs = np.array([[sx, sy, sz] for sx in (1.0, -1.0)
                          for sy in (1.0, -1.0) for sz in (1.0, -1.0)])
        return [V @ np.diag(s) @ V.T for s in signs]

    candidates: list[NDArray] = [np.eye(3), -np.eye(3)]
    for axis in _candidate_axes(pos):
        candidates.append(np.eye(3) - 2.0 * np.outer(axis, axis))  # mirror
        for order in (2, 3, 4, 6):
            for k in range(1, order):
                candidates.append(_rotation(axis, 2.0 * np.pi * k / order))
    return candidates


def _self_isometries(
    symbols: list[str], cart_coords: NDArray, atol: float = 0.05
) -> list[NDArray]:
    """Return the monomer's molecular point group as a list of Cartesian 3x3
    orthogonal matrices (acting about the centroid) that map the labeled atom
    cloud onto itself. The identity is always first; improper operations
    (``det < 0``) are present iff the monomer is achiral.

    Candidates come from :func:`_symmetry_candidates` (gyration-tensor pruned)
    and are filtered to those that preserve the labeled cloud. This is not a
    rigorous point-group solver, but it recovers the operations the deduper must
    mod out when matching clusters in relative-transform space.

    :param symbols: The atomic symbols of the monomer
    :param cart_coords: The monomer's Cartesian coordinates, shape ``(n, 3)``
    :param atol: Cartesian matching tolerance in Angstroms. The 0.05 default
        absorbs the small thermal / measurement distortion in typical CIF
        coordinates while still rejecting genuine asymmetry.
    :returns: A list of ``(3, 3)`` orthogonal matrices, identity first
    """
    Z = np.array([qcel.periodictable.to_Z(s) for s in symbols])
    pos = cart_coords - cart_coords.mean(axis=0)

    found: list[NDArray] = []
    for u in _symmetry_candidates(pos):
        if any(np.allclose(u, v, atol=1e-4) for v in found):
            continue
        if _z_labeled_match(pos @ u.T, pos, Z, atol):
            found.append(u)
    return found


def _has_improper_symmetry(
    symbols: list[str], cart_coords: NDArray, atol: float = 0.05
) -> bool:
    """Return True iff the monomer is achiral — i.e. its self-isometry group
    contains an improper operation (``det < 0``).

    :param symbols: The atomic symbols of the monomer
    :param cart_coords: The monomer's Cartesian coordinates, shape ``(n, 3)``
    :param atol: Cartesian matching tolerance in Angstroms
    :returns: True iff some improper isometry preserves the labeled atom cloud
    """
    return any(
        float(np.linalg.det(u)) < 0
        for u in _self_isometries(symbols, cart_coords, atol)
    )


@dataclass
class ASU:
    """Represents an asymmetric unit (ASU) of a crystal structure, which is the smallest portion of the crystal that can generate the entire unit cell through the application of the space group's symmetry operations. Contains the atoms and their positions within the ASU.

    :param _atoms: A list of atomic symbols corresponding to the atoms in the ASU
    :param _positions: A list of fractional coordinates corresponding to the positions of the atoms in the ASU
    """

    _atoms: list[str]
    _positions: NDArray[np.float64]

    def __post_init__(self):
        """ASU validation.

        :raises ValueError: If the number of atoms does not match the number of positions or if the positions are not in the correct format.
        """
        if len(self._atoms) != len(self._positions):
            raise ValueError("Number of atoms must match number of positions")

    @property
    def atoms(self) -> list[str]:
        """Return the list of atomic symbols in the ASU."""
        return self._atoms

    @property
    def positions(self) -> NDArray:
        """Return the list of fractional coordinates for the atoms in the ASU."""
        return self._positions


@dataclass
class Monomer:
    """Represents a molecular monomer in the crystal.

    :param _symbols: A list of atomic symbols corresponding to the atoms in the monomer
    :param _frac_coords: A list of fractional coordinates corresponding to the positions of the atoms in the monomer
    :param _cart_coords: A list of Cartesian coordinates corresponding to the positions of the atoms in the monomer
    :param _centroid_frac: The centroid of the monomer in fractional coordinates
    :param _centroid_cart: The centroid of the monomer in Cartesian coordinates
    :param _symop: The symmetry operation that generates this monomer from the reference monomer, if applicable
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

    @cached_property
    def self_isometries(self) -> list[NDArray]:
        """The monomer's molecular point group: Cartesian 3x3 orthogonal
        operations (about the centroid) that map the labeled atom cloud onto
        itself. Always includes the identity; includes improper operations
        (``det < 0``) iff the monomer is achiral. Cached after first access.

        The multimer deduper uses these to mod out internal symmetry when
        testing two clusters for geometric equivalence (identity *or*
        enantiomer) in relative-transform space: two clusters are the same
        multimer iff their pairwise relative transforms match under some
        monomer permutation after dressing each monomer by an element of this
        group.

        :returns: A list of ``(3, 3)`` orthogonal matrices, identity first
        """
        return _self_isometries(self._symbols, self._cart_coords)

    def has_improper_internal_symmetry(self) -> bool:
        """Return True iff the monomer is achiral — its self-isometry group
        contains an improper operation (``det < 0``).

        :returns: True iff some improper isometry preserves the atom cloud
        """
        return any(float(np.linalg.det(u)) < 0 for u in self.self_isometries)

    def __repr__(self) -> str:
        """Return a string representation of the monomer."""
        return f"Monomer(symbols={self._symbols}, centroid_cart={self._centroid_cart}, SymOp={self._symop})"


@dataclass
class Multimer:
    """Represents a unique multimer pairing of N monomers.

    :param _monomers: A list of Monomer objects that make up the multimer
    :param _sym_ops: A list of symmetry operations that generate the monomers in the multimer
    :param _multiplicity: The multiplicity of the multimer, which is the number of times this multimer appears in the crystal due to symmetry
    :ivar _geometric_mean: The geometric mean of the inter-monomer distances in the multimer, which can be used as a measure of the overall size of the multimer and to filter out multimers that are too large or too small based on a specified cutoff
    """

    _monomers: list[Monomer]
    _sym_ops: SymOpList
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
    def sym_ops(self) -> SymOpList:
        """Return the list of symmetry operations that generate the monomers in the multimer."""
        return self._sym_ops

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
        """Convert multimer to QCElemental Molecule object.

        :returns: A QCElemental Molecule object representing the multimer, with atomic symbols and Cartesian coordinates for all atoms in the multimer
        """
        symbols = [monomer._symbols for monomer in self._monomers]
        coords = [monomer._cart_coords for monomer in self._monomers]
        symbols = np.array(symbols)
        coords = np.vstack(coords)

        fragments = []
        cnt = 0
        for monomer in self._monomers:
            fragments.append(list(range(cnt, cnt + len(monomer._symbols))))
            cnt += len(monomer._symbols)

        cart_bohr = coords / qcel.constants.bohr2angstroms
        return qcel.models.Molecule(
            symbols=symbols.flatten(),
            geometry=cart_bohr.flatten(),
            fragments=fragments,
        )

    def to_monomer_molecules(self) -> list[qcel.models.Molecule]:
        """Return the monomers as separate QCElemental Molecules.

        :returns: A list of QCElemental Molecule objects, each representing a monomer in the multimer, with atomic symbols and Cartesian coordinates for the atoms in each monomer
        """
        return [monomer.to_molecule() for monomer in self._monomers]

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, Multimer):
            return NotImplemented

        if len(self._monomers) != len(value._monomers):
            return False

        for m1, m2 in zip(self._monomers, value._monomers):
            if m1._symbols != m2._symbols:
                return False
            if not np.allclose(m1._centroid_cart, m2._centroid_cart, atol=1e-3):
                return False
            if m1._symop != m2._symop:
                return False

        return True

    def __repr__(self) -> str:
        return f"Multimer(Monomers={self._monomers}, multiplicity={self._multiplicity}, geometric_mean={self._geometric_mean:.2f} Å)"


if __name__ == "__main__":
    pass
