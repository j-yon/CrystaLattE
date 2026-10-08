from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Final, Self

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

# ADD lazy

# orders of the finite subgroups of O(3) a molecule can realise
VALID_POINT_GROUP_ORDERS: frozenset[int] = frozenset(
    { 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 48, 60, 72, 96, 120 }
)

# tolerance ladder
_TOLERANCE_LADDER: Final[tuple[float, ...]] = (1.0, 0.4, 0.2, 0.1, 0.04, 0.02)

# largest possible group
_MAX_GROUP_ORDER: Final[int] = 120


# tolerance handed to the point group solver, deliberately loose
# measured across the X23 references, 0.3 names the correct group in every case where one is known
_SOLVER_TOLERANCE: Final[float] = 0.3


def _rotation(axis: NDArray, angle: float) -> NDArray:
    """Return the Rodrigues rotation matrix about a unit axis

    :param axis: unit rotation axis, shape ``(3,)``
    :param angle: rotation angle in radians
    :returns: The ``(3, 3)`` rotation matrix
    """
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _solver_analyzer(symbols: Sequence[str], cart_coords: NDArray, tolerance: float) -> PointGroupAnalyzer:
    """Return pymatgen's point-group analyzer for the centred molecule

    :param symbols: atomic symbols
    :param cart_coords: Cartesian coordinates, shape ``(n, 3)``
    :param tolerance: tolerance passed to the solver
    :returns: :class:`pymatgen.symmetry.analyzer.PointGroupAnalyzer`
    """
    pos = np.asarray(cart_coords, dtype=float)
    return PointGroupAnalyzer(Molecule(list(symbols), pos - pos.mean(axis=0)), tolerance=tolerance)


@dataclass(frozen=True)
class _Frame:
    """A centroid-centred, Z-labelled molecule that symmetry operations act on.
    Used internally only for point-group detection

    :param pos: centroid-centred atom positions, shape ``(n, 3)``
    :param Z: per-atom atomic numbers, shape ``(n,)``
    """

    _pos: NDArray
    _Z: NDArray

    @classmethod
    def from_coords(cls, symbols: Sequence[str], cart_coords: NDArray) -> Self:
        """Construct a centroid-centred frame from atomic symbols and Cartesian coordinates

        :param symbols: atomic symbols
        :param cart_coords: Cartesian coordinates, shape ``(n, 3)``
        :returns: :class:`_Frame` with centroid-centred positions and atomic numbers
        """
        Z = np.array([qcel.periodictable.to_Z(s) for s in symbols])
        return cls(cart_coords - cart_coords.mean(axis=0), Z)

    def _axes(self) -> list[NDArray]:
        """Enumerate candidate symmetry axes and mirror normals"""
        pos = self._pos

        # axes aligned with atoms
        axes: list[NDArray] = []
        for p in pos:
            nrm = np.linalg.norm(p)
            if nrm > 1e-3:
                axes.append(p / nrm)

        # axes along the midpoints (v[0]) and cross products (v[1]) of all pairs of atoms
        for i, j in combinations(range(len(pos)), 2):
            for v in ((pos[i] + pos[j]) / 2, np.cross(pos[i], pos[j])):
                nrm = np.linalg.norm(v)
                if nrm > 1e-3:
                    axes.append(v / nrm)

        # axes along the eigenvectors of the gyration tensor
        eigvecs = np.linalg.eigh(pos.T @ pos)[1]
        axes.extend(eigvecs[:, k] for k in range(3))

        # prune parallel or antiparallel axes
        unique_axes: list[NDArray] = []
        for v in axes:
            if not any(abs(abs(np.dot(u, v)) - 1) < 1e-5 for u in unique_axes):
                unique_axes.append(v)

        return unique_axes

    def _verified(self, ops: Sequence[NDArray], atol: float) -> list[NDArray]:
        """Return those of ``ops`` that are symmetries at ``atol``"""
        ops = [np.asarray(u, dtype=float) for u in ops]
        return [u for u in ops if self.is_symmetry(u, atol)]

    def is_symmetry(self, op: NDArray, atol: float) -> bool:
        """Return True iff ``op`` maps the labelled atoms onto themselves within ``atol``

        :param op: ``(3, 3)`` Cartesian operation
        :param atol: cartesian matching tolerance in Angstroms
        :returns: True iff ``op`` is a symmetry at ``atol``
        """
        # positions with the operation applied
        transformed = self._pos @ op.T

        # maintain a boolean mask of which atoms have already been matched
        matched = np.zeros(len(self._pos), dtype=bool)
        for i in range(len(transformed)):
            diff = self._pos - transformed[i]
            d2 = np.einsum("ij,ij->i", diff, diff)  # compute squared distances to avoid square root

            # find unmatched atoms with the same atomic number and within the distance tolerance
            candidates = np.where((self._Z == self._Z[i]) & ~matched & (d2 < atol * atol))[0]
            if len(candidates) == 0:
                return False
            
            # mark candidates as matched
            matched[candidates[0]] = True
        return True

    def candidates(self) -> list[NDArray]:
        """Generate candidate orthogonal operations, pruned by the gyration tensor

        :returns: list of candidate ``(3, 3)`` orthogonal matrices
        """
        # ascending eigenvalues, eigenvectors in columns
        w, V = np.linalg.eigh(self._pos.T @ self._pos)
        scale = max(float(w[-1]), 1.0)
        asymmetric = (w[1] - w[0] > 1e-3 * scale) and (w[2] - w[1] > 1e-3 * scale)

        # generic case, three distinct eigenvalues, so the rotation axes are the eigenvectors
        if asymmetric:
            signs = [
                (sx, sy, sz)
                for sx in (1.0, -1.0)
                for sy in (1.0, -1.0)
                for sz in (1.0, -1.0)
            ]
            return [V @ np.diag(s) @ V.T for s in signs]

        # degenerate case, so the rotation axis can lie anywhere in the degenerate plane
        candidates: list[NDArray] = [np.eye(3), -np.eye(3)]
        for axis in self._axes():
            candidates.append(np.eye(3) - 2.0 * np.outer(axis, axis))  # mirror

            # for all possible rotation orders, generate nontrivial rotations about the axis
            for order in (2, 3, 4, 6):
                candidates.extend(_rotation(axis, 2.0 * np.pi * k / order) for k in range(1, order))
        return candidates

    def close(self, seed: Sequence[NDArray], atol: float) -> PointGroup | None:
        """Close ``seed`` under composition with genuine symmetries

        :param seed: Detected operations to close over
        :param atol: Cartesian matching tolerance in Angstroms
        :returns: The closed group or ``None`` if incoherent
        """
        # stack mirrors group as one (k, 3, 3) array so membership is a single vectorized reduction
        group: list[NDArray] = [np.eye(3)]
        stack = np.empty((_MAX_GROUP_ORDER + 2, 3, 3))
        stack[0] = group[0]

        def known(X: NDArray) -> bool:
            """Determine if ``X`` is already in the group by matching max elementwise difference"""
            return bool((np.abs(stack[: len(group)] - X).max(axis=(1, 2)) < 1e-3).any())

        def add(X: NDArray) -> None:
            """Add ``X`` to the group and stack"""
            stack[len(group)] = X
            group.append(X)

        # add the seed first
        for s in seed:
            if known(s):
                continue
            # if tolerance is too loose, the seed may overrun the maximum group order
            if len(group) > _MAX_GROUP_ORDER:
                return None
            add(s)

        # add operation products until closure, or return None if incoherent or too large
        frontier = list(group)
        while frontier:
            # only products involving a newly added operation can be new, so each element pairs 
            # against the group once instead of every sweep
            fresh: list[NDArray] = []
            for a in frontier:
                for b in group:
                    for product in (a @ b, b @ a):
                        if known(product):
                            continue
                        if not self.is_symmetry(product, atol):
                            return None
                        if len(group) > _MAX_GROUP_ORDER:
                            return None
                        add(product)
                        fresh.append(product)
            frontier = fresh

        return PointGroup(group, tolerance=atol)


class PointGroup(Sequence[NDArray]):
    """A molecule's point group as Cartesian ``(3, 3)`` operations

    :param operations: ``(3, 3)`` orthogonal matrices
    :param tolerance: tolerance the operations closed into a group at, or ``None`` if they never did
    """

    def __init__(self, operations: Sequence[NDArray], tolerance: float | None = None) -> None:
        self._operations = list(operations)
        self._tolerance = tolerance

    def __getitem__(self, index):
        return self._operations[index]

    def __len__(self) -> int:
        return len(self._operations)

    def __repr__(self) -> str:
        return f"PointGroup(order={self.order}, tolerance={self._tolerance})"

    @property
    def operations(self) -> list[NDArray]:
        return self._operations

    @property
    def order(self) -> int:
        return len(self._operations)

    @property
    def tolerance(self) -> float | None:
        return self._tolerance

    @property
    def is_coherent(self) -> bool:
        """True iff the operations look like a genuine finite point group"""
        if self.order not in VALID_POINT_GROUP_ORDERS:
            return False
        ops = self._operations
        return all(any(np.allclose(g.T, h, atol=1e-3) for h in ops) for g in ops)

    @classmethod
    def from_solver(
        cls,
        symbols: Sequence[str],
        cart_coords: NDArray,
        tolerance: float = _SOLVER_TOLERANCE,
    ) -> Self:
        """The idealised point group named by pymatgen's solver

        :param symbols: The atomic symbols
        :param cart_coords: Cartesian coordinates, shape ``(n, 3)``
        :param tolerance: Tolerance passed to the solver
        :returns: The solver's operations, unverified against the coordinates
        """
        analyzer = _solver_analyzer(symbols, cart_coords, tolerance)
        return cls(
            [
                np.asarray(op.rotation_matrix, dtype=float)
                for op in analyzer.get_symmetry_operations()
            ]
        )

    @classmethod
    def detect(
        cls,
        symbols: Sequence[str],
        cart_coords: NDArray,
        atol: float = 0.05,
        seed: Sequence[NDArray] = (),
        use_solver: bool = True,
    ) -> Self:
        """Detect the largest coherent point group the coordinates support at ``atol``

        :param symbols: atomic symbols
        :param cart_coords: Cartesian coordinates, shape ``(n, 3)``
        :param atol: Cartesian matching tolerance in Angstroms
        :param seed: operations known to be exact symmetries, typically the crystallographic site symmetry
        :param use_solver: consult pymatgen point-group solver and add its operations to ``seed``
        :returns: The detected point group
        """
        # create a frame to test candidate operations against with imperfect coordinates
        frame = _Frame.from_coords(symbols, cart_coords)

        # if desired, attempt to identify the idealised point group and add its operations to the seed
        extra = list(cls.from_solver(symbols, cart_coords)) if use_solver else []

        found: list[NDArray] = []
        for u in frame.candidates():
            # if the operation is already in the found list, skip it
            if any(np.allclose(u, v, atol=1e-4) for v in found):
                continue

            # if the operation is a symmetry at the given tolerance, add it to the found list
            if frame.is_symmetry(u, atol):
                found.append(u)

        # iterate over the symmetry tolerance ladder, tightening the rung each time
        for factor in _TOLERANCE_LADDER:
            rung = atol * factor

            # the operations accepted at the previous rung are re-verified at the current rung
            accepted = found if factor == 1.0 else frame._verified(found, rung)

            # create a base set of operations based on the current rung
            base = frame._verified(seed, rung)

            # if using the solver, try closing the group with the solver-augmented seed first
            attempts = [base + frame._verified(extra, rung), base] if extra else [base]
            for attempt in attempts:
                group = frame.close(attempt + accepted, rung)
                # found a coherent group, return it immediately
                if group is not None and group.is_coherent:
                    return group

        # no rung closed coherently
        # currently returns the largest set of verified operations
        # in the future, fallback to NRE/ChSEV check
        result = list(found)
        for s in frame._verified(seed, atol):
            if not any(np.allclose(s, v, atol=1e-4) for v in result):
                result.append(s)
        return cls(result)

def point_group_report(
    symbols: Sequence[str],
    cart_coords: NDArray,
    atol: float = 0.05,
    seed: Sequence[NDArray] = (),
    site_symmetry: Sequence[NDArray] | None = None,
) -> dict:
    """Describe how complete the detected point group is

    :param symbols: atomic symbols
    :param cart_coords: Cartesian coordinates, shape ``(n, 3)``
    :param atol: tolerance for symmetry detection
    :param seed: operations to seed detection with
    :param site_symmetry: site symmetry as integer matrices in the lattice basis, for labelling
    :returns: A dict with ``detected``, ``sweep``, ``site_symmetry``, ``idealised``, ``schoenflies`` and ``complete``
    """
    # run the detection at a few tolerances to see if the group is stable
    sweep = {
        tol: PointGroup.detect(symbols, cart_coords, atol=tol, seed=seed).order
        for tol in (atol, atol * 2, atol * 4, atol * 10)
    }
    detected = sweep[atol]

    # if site_symmetry is provided, use spglib to get the Schoenflies label
    label = None
    if site_symmetry:
        import spglib

        label = spglib.get_pointgroup(
            [np.asarray(w, dtype=np.intc) for w in site_symmetry]
        )[0].strip()

    # use pymatgen's solver to get the idealised point group and its Schoenflies label
    analyzer = _solver_analyzer(symbols, cart_coords, _SOLVER_TOLERANCE)
    idealised = len(analyzer.get_symmetry_operations())

    return {
        "detected": detected,
        "sweep": sweep,
        "site_symmetry": label,
        "idealised": idealised,
        "schoenflies": analyzer.sch_symbol,
        "complete": detected >= idealised,
    }

