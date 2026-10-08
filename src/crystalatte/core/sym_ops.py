from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray


@dataclass()
class SymOp:
    """
    A symmetry operation ``x' = rot @ x + tr`` on fractional coordinates

    :param _op: operation string like 'x,y,z' or '-x+1/2,y,-z+1/2'
    :ivar _rot: 3x3 rotational part, fractional basis
    :ivar _tr: translational part, fractional basis
    :ivar _augment: 4x4 augmented matrix ``[[rot, tr], [0, 0, 0, 1]]``
    """

    _op: str | None
    _rot: NDArray[np.int16] = field(init=False)
    _tr: NDArray[np.int16] = field(init=False)
    _augment: NDArray[np.int16] = field(init=False)

    def __post_init__(self):
        """Constructor method"""
        if self._op is None:
            raise ValueError(
                "SymOp must be initialized with a symmetry operation string"
            )
        self._rot, self._tr = self._parse_symop(self._op)

        augment = np.eye(4, dtype=np.int16)
        augment[:3, :3] = self._rot
        augment[:3, 3] = self._tr
        self._augment = augment

    def _parse_symop(self, op: str) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
        """Parse a symmetry operation from an xyz string

        :param op: A symmetry operation string like 'x,y,z' or '-x+1/2,y,-z+1/2'
        :returns: A tuple of (rot, tr) where rot is a 3x3 matrix corresponding to the rotational component of the symmetry operation in fractional coordinates, and tr is a 3D vector corresponding to the translational component of the symmetry operation in fractional coordinates (integers representing multiples of 1/12 for efficiency)
        """
        rot = np.zeros((3, 3), dtype=np.int16)
        tr = np.zeros(3, dtype=np.int16)

        _op = op.replace(" ", "").split(",")

        # supposedly fast parsing of symmetry operations
        for i, comp in enumerate(_op):
            sign = 1
            j = 0

            while j < len(comp):
                # rotation
                if comp[j] in "+-":
                    sign = 1 if comp[j] == "+" else -1
                    j += 1
                elif comp[j] in "xyz":
                    axis = "xyz".index(comp[j])
                    rot[i, axis] = sign
                    j += 1

                # translation
                else:
                    num_str = ""
                    while j < len(comp) and (comp[j].isdigit() or comp[j] in "/."):
                        num_str += comp[j]
                        j += 1
                    if num_str:
                        tr[i] += int(Fraction(num_str) * 12) * sign
                        sign = 1

        return rot, tr

    @property
    def op(self) -> str | None:
        """Return the original symmetry operation string."""
        return self._op

    @property
    def rot(self) -> NDArray[np.int16]:
        """Return the rotational component of the symmetry operation."""
        return self._rot

    @property
    def tr(self) -> NDArray[np.int16]:
        """Return the translational component of the symmetry operation."""
        return self._tr

    @property
    def augment(self) -> NDArray[np.int16]:
        """Return the augmented matrix representation of the symmetry operation."""
        return self._augment

    @classmethod
    def identity(cls) -> SymOp:
        """Return the identity symmetry operation."""
        return cls("x,y,z")

    @classmethod
    def from_components(
        cls, rot: NDArray[np.int16], tr: NDArray[np.int16], op: str | None = None
    ) -> SymOp:
        """Create a SymOp from given rotation and translation components

        :param rot: 3x3 rotational part, fractional basis
        :param tr: translational part, fractional basis
        :param op: optional operation string, kept for display only
        :returns: the SymOp
        """
        symop = cls.__new__(cls)
        symop._rot = rot
        symop._tr = tr

        augment = np.eye(4, dtype=np.int16)
        augment[:3, :3] = rot
        augment[:3, 3] = tr
        symop._augment = augment
        symop._op = op

        return symop

    def translate(self, translation: NDArray[np.int16]) -> SymOp:
        """Return a new SymOp with the same rotation but translated by the given vector. The original vector in fractional coordinates must be given multiplied by 12, since translations are stored as integers for efficiency.

        :param translation: A 3D vector in fractional coordinates to translate the symmetry operation by
        :returns: A new SymOp with the same rotation but translated by the given vector
        """
        new_tr = self._tr + translation
        return SymOp.from_components(self._rot, new_tr, self._op)

    def apply(self, frac_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply this SymOp to the given fractional coordinates

        :param frac_coords: An Nx3 array of fractional coordinates to transform
        :returns: The transformed fractional coordinates after applying the symmetry operation, calculated as x' = rot @ x + tr, where rot is the rotational component and tr is the translational component of the symmetry operation
        """
        return (self._rot @ frac_coords.T).T + (self._tr / 12)

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, SymOp):
            return NotImplemented
        return np.array_equal(self._augment, value._augment)

    def __hash__(self) -> int:
        return hash(self._augment.tobytes())

    def __iter__(self):
        yield self._rot
        yield self._tr

    def __repr__(self) -> str:
        return f"SymOp(op='{self._op}', rot='{self._rot}', tr='{self._tr}', augment=\n{self._augment})"


@dataclass
class SymOpList:
    """
    A list of symmetry operations, typically a subset of a space group's

    :param _sym_ops: the :class:`SymOp` members
    :param _multiplicity: number of symmetry-equivalent multimers this list generates
    :param _ref_indices: per-member base-reference (ASU molecule) index; defaults to zeros
    :ivar _rot_cache: stacked rotational parts
    :ivar _tr_cache: stacked translational parts
    :ivar _rel_cache: lazy ``(L, v)`` relative Cartesian transforms, set by the deduper
    :ivar _dist_fp_cache: lazy ``(zz, distances)`` fingerprint, set by the deduper
    """

    _sym_ops: list[SymOp] | None
    _multiplicity: int
    _ref_indices: NDArray | None = field(default=None, repr=False)
    _rot_cache: NDArray = field(init=False, repr=False)
    _tr_cache: NDArray = field(init=False, repr=False)

    # lazily populated geometry caches used by the multimer deduper
    _rel_cache: tuple[NDArray, NDArray] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _rel_int_cache: tuple[NDArray, NDArray] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _dist_fp_cache: tuple[NDArray, NDArray] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # lazily-derived geometry caches
    _PICKLE_EXCLUDED = ("_rel_cache", "_rel_int_cache", "_dist_fp_cache")

    def __getstate__(self):
        """Drop the derived caches when pickling, rebuild on demand.

        :returns: The instance dict with the derived cache slots cleared
        """
        state = self.__dict__.copy()
        for name in self._PICKLE_EXCLUDED:
            state[name] = None
        return state

    def __post_init__(self):
        """Constructor method to validate inputs and compute caches for the symmetry operations.

        :raises ValueError: If the list of symmetry operations is empty or if the multiplicity is not a positive integer
        """
        if not self._sym_ops:
            raise ValueError("SymOpList must contain at least one SymOp")
        if self._multiplicity < 1:
            raise ValueError("Multiplicity must be a positive integer")

        # compute matrices for all sym_ops and store in cache
        self._rot_cache = np.stack([op._rot for op in self._sym_ops])
        self._tr_cache = np.stack([op._tr for op in self._sym_ops])

        if self._ref_indices is None:
            self._ref_indices = np.zeros(len(self._sym_ops), dtype=np.int64)

    @property
    def sym_ops(self) -> list[SymOp]:
        """Return the list of symmetry operations."""
        if self._sym_ops is None:
            self._sym_ops = [
                SymOp.from_components(self._rot_cache[i], self._tr_cache[i])
                for i in range(len(self._rot_cache))
            ]
        return self._sym_ops

    @property
    def multiplicity(self) -> int:
        """Return the multiplicity of multimers generated by this list of symmetry operations."""
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, value: int):
        self._multiplicity = value

    @property
    def translations(self) -> list[NDArray]:
        """Return a list of the translational components of the symmetry operations."""
        return [sym_op.tr for sym_op in self.sym_ops]

    @property
    def rot_cache(self) -> NDArray:
        """Return the cached array of rotational components of the symmetry operations."""
        return self._rot_cache

    @property
    def tr_cache(self) -> NDArray:
        """Return the cached array of translational components of the symmetry operations."""
        return self._tr_cache

    @property
    def ref_indices(self) -> NDArray:
        """Per-member base-reference index. Defaults to all-zeros for single-reference clusters."""
        return self._ref_indices

    @ref_indices.setter
    def ref_indices(self, value: NDArray):
        self._ref_indices = np.asarray(value, dtype=np.int64)

    @classmethod
    def from_components(
        cls,
        rot: NDArray,
        tr: NDArray,
        multiplicity: int,
        ref_indices: NDArray | None = None,
    ) -> SymOpList:
        """Create a SymOpList from given rotation and translation components.

        :param rot: ``(N, 3, 3)`` rotational parts, fractional basis
        :param tr: ``(N, 3)`` translational parts as integer twelfths
        :param multiplicity: number of symmetry-equivalent multimers this list generates
        :param ref_indices: per-member base-reference indices; defaults to zeros
        :returns: the SymOpList
        """
        instance = cls.__new__(cls)
        instance._sym_ops = None
        instance._multiplicity = multiplicity
        instance._ref_indices = (
            np.zeros(len(rot), dtype=np.int64)
            if ref_indices is None
            else np.asarray(ref_indices, dtype=np.int64)
        )

        # set caches
        instance._rot_cache = rot
        instance._tr_cache = tr

        # __new__ bypasses the dataclass __init__, so the lazily-populated geometry caches must be initialized explicitly here
        instance._rel_cache = None
        instance._rel_int_cache = None
        instance._dist_fp_cache = None

        return instance

    def __iter__(self):
        return iter(self.sym_ops)

    def __len__(self):
        return len(self.sym_ops)

    def __getitem__(self, index: int) -> SymOp:
        """Return the symmetry operation at the specified index."""
        return self.sym_ops[index]

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, SymOpList):
            return NotImplemented

        return (
            isinstance(value, SymOpList)
            and set(self.sym_ops) == set(value.sym_ops)
            and self._multiplicity == value._multiplicity
        )

    def __repr__(self) -> str:
        return f"SymOpList(sym_ops={self._sym_ops}, multiplicity={self._multiplicity})"
