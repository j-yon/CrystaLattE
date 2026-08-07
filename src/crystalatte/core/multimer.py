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


#: Orders of the finite subgroups of O(3) a molecule can realise. Used only as
#: a coherence check on a repaired set — an order outside this set proves the
#: result is not a group (e.g. the raw filter returns 42 for hexamine).
VALID_POINT_GROUP_ORDERS: frozenset[int] = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 48, 60,
     72, 96, 120}
)

#: Tolerance ladder, as multiples of the caller's ``atol``, tried loosest-first.
_TOLERANCE_LADDER: tuple[float, ...] = (1.0, 0.4, 0.2, 0.1, 0.04, 0.02)

#: Largest group we will build; the icosahedral groups cap out at 120.
_MAX_GROUP_ORDER = 120


def _close_isometries(
    seed: list[NDArray], pos: NDArray, Z: NDArray, atol: float
) -> list[NDArray] | None:
    """Close ``seed`` under composition, keeping only genuine symmetries.

    Returns ``None`` if the closure escapes the symmetry set (a product that is
    not itself a symmetry proves some seed element was spurious, since a
    composition of real symmetries is always real) or exceeds
    :data:`_MAX_GROUP_ORDER`.

    :param seed: Detected operations to close over
    :param pos: Centroid-centered atom positions
    :param Z: Per-atom atomic numbers
    :param atol: Cartesian matching tolerance in Angstroms
    :returns: The closed list (identity first), or ``None`` if incoherent
    """
    # ``stack`` mirrors ``group`` as one (k, 3, 3) array so membership is a
    # single vectorized reduction rather than k Python-level allclose calls —
    # the closure is a triple loop, so this is the hot path.
    group: list[NDArray] = [np.eye(3)]
    stack = np.empty((_MAX_GROUP_ORDER + 2, 3, 3))
    stack[0] = group[0]

    def known(X: NDArray) -> bool:
        return bool(
            (np.abs(stack[: len(group)] - X).max(axis=(1, 2)) < 1e-3).any()
        )

    def add(X: NDArray) -> None:
        stack[len(group)] = X
        group.append(X)

    for s in seed:
        if not known(s):
            add(s)

    # Breadth-first: only products involving a newly added operation can be
    # new, so each element pairs against the group once instead of every sweep.
    frontier = list(group)
    while frontier:
        fresh: list[NDArray] = []
        for a in frontier:
            for b in list(group):
                for product in (a @ b, b @ a):
                    if known(product):
                        continue
                    if not _z_labeled_match(pos @ product.T, pos, Z, atol):
                        return None
                    if len(group) > _MAX_GROUP_ORDER:
                        return None
                    add(product)
                    fresh.append(product)
        frontier = fresh
    return group


def _is_coherent_group(group: list[NDArray]) -> bool:
    """Return True iff ``group`` looks like a genuine finite point group:
    a valid order, and every element's inverse present.

    :param group: The candidate operation list
    :returns: True iff the list passes the coherence checks
    """
    if len(group) not in VALID_POINT_GROUP_ORDERS:
        return False
    return all(
        any(np.allclose(g.T, h, atol=1e-3) for h in group) for g in group
    )


def _self_isometries(
    symbols: list[str], cart_coords: NDArray, atol: float = 0.05
) -> list[NDArray]:
    """Return the monomer's molecular point group as a list of Cartesian 3x3
    orthogonal matrices (acting about the centroid) that map the labeled atom
    cloud onto itself. The identity is always first; improper operations
    (``det < 0``) are present iff the monomer is achiral.

    Candidates come from :func:`_symmetry_candidates` (gyration-tensor pruned)
    and are filtered to those that preserve the labeled cloud. That library is
    NOT closed under composition — it holds ``±I``, one mirror per axis, and
    proper rotations of order 2/3/4/6, but no rotoreflections, so ``S4 = sigma
    . C4`` and ``S6 = sigma . C6`` can never appear. A T_d molecule therefore
    filters to ``E + 8 C3 + 3 C2 + 6 sigma_d = 18`` operations, and 18 does not
    divide 24, proving the raw filter does not return a group. Since
    ``docs/dedup_correctness.tex`` Thm 1/2 assume it does, the filtered set is
    repaired here by closing it under composition.

    Closure alone is not enough: it amplifies over-detection, because composing
    two operations each off by ``eps`` gives one off by ``~2 eps`` that a loose
    tolerance still accepts (closing hexamine's 42-element set at 0.05 A runs
    past 200 operations). So the detection tolerance is stepped down through
    :data:`_TOLERANCE_LADDER` and the FIRST rung whose closure is a coherent
    group is taken — the most permissive tolerance that still yields something
    mathematically well-formed. If no rung does, the raw filter is returned
    unchanged rather than fabricating a group.

    :param symbols: The atomic symbols of the monomer
    :param cart_coords: The monomer's Cartesian coordinates, shape ``(n, 3)``
    :param atol: Cartesian matching tolerance in Angstroms, and the loosest rung
        of the ladder. The 0.05 default absorbs the small thermal / measurement
        distortion in typical CIF coordinates while still rejecting genuine
        asymmetry.
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

    # Tighter rungs re-test the operations already accepted at ``atol`` rather
    # than re-scanning the (large) candidate library.
    for factor in _TOLERANCE_LADDER:
        rung = atol * factor
        seed = (
            found
            if factor == 1.0
            else [u for u in found if _z_labeled_match(pos @ u.T, pos, Z, rung)]
        )
        group = _close_isometries(seed, pos, Z, rung)
        if group is not None and _is_coherent_group(group):
            return group
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
