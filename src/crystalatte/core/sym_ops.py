from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray


@dataclass()
class SymOp:
    """
    A symmetry operation defined by a rotation matrix and translation vector. 
    The symmetry operation transforms fractional coordinates as: 

    .. math::
        x' = rot @ x + tr

    :param _op: Symmetry operation string like ``'x,y,z'`` or ``'-x+1/2,y,-z+1/2'``
    :ivar _rot: 3x3 matrix corresponding to the rotational component of the operation in fractional coordinates
    :ivar _tr: 3D vector corresponding to the translational component of the operation in fractional coordinates
    :ivar _augment: 4x4 matrix representing the symmetry operation in augmented matrix form
    """

    _op: str | None
    _rot: NDArray[np.int16] = field(init=False)
    _tr: NDArray[np.int16] = field(init=False)
    _augment: NDArray[np.int16] = field(init=False)

    def __post_init__(self):
        if self._op is None:
            raise ValueError(
                "SymOp must be initialized with a symmetry operation string"
            )
        self._rot, self._tr = self._parse_symop(self._op)

        augment = np.eye(4, dtype=np.int16)
        augment[:3, :3] = self._rot
        augment[:3, 3] = self._tr
        self._augment = augment

    @staticmethod
    def _parse_symop(op: str) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
        """Parse a symmetry operation from an xyz string

        :param op: Symmetry operation string like ``'x,y,z'`` or ``'-x+1/2,y,-z+1/2'``
        :returns: Rotations and translations in matrix/vector form parsed from the string
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

    @staticmethod
    def _format_symop(rot: NDArray[np.int16], tr: NDArray[np.int16]) -> str:
        """Format rotation and translation components as an xyz string (inverse of :meth:`_parse_symop`)

        :param rot: 3x3 matrix corresponding to the rotational component of the operation in fractional coordinates
        :param tr: 3D vector corresponding to the translational component in twelfths, lattice part included
        :returns: Canonical symmetry operation string like ``'-x+y+1,-x,z+2/3'``
        :raises ValueError: If a rotation coefficient lies outside ±1, which the xyz syntax cannot express
        """
        rows = []
        for coeffs, t in zip(rot.tolist(), tr.tolist()):
            if any(abs(c) > 1 for c in coeffs):
                raise ValueError(
                    f"cannot format rotation row {coeffs} as xyz: coefficients must be in ±1"
                )
            terms = "".join(
                ("+" if c > 0 else "-") + v for c, v in zip(coeffs, "xyz") if c
            )
            if t:
                f = Fraction(int(t), 12)
                terms += f"{'+' if f > 0 else '-'}{abs(f)}"
            rows.append(terms.lstrip("+") or "0")
        return ",".join(rows)

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

    @property
    def op(self) -> str:
        if self._op is None:
            self._op = self._format_symop(self._rot, self._tr)
        return self._op

    @property
    def rot(self) -> NDArray[np.int16]:
        return self._rot

    @property
    def tr(self) -> NDArray[np.int16]:
        return self._tr

    @property
    def augment(self) -> NDArray[np.int16]:
        return self._augment

    @classmethod
    def identity(cls) -> SymOp:
        """Return the identity symmetry operation"""
        return cls("x,y,z")

    @classmethod
    def from_components(
        cls, rot: NDArray[np.int16], tr: NDArray[np.int16], op: str | None = None
    ) -> SymOp:
        """Create a :class:`SymOp` from given rotation and translation components

        :param rot: 3x3 matrix corresponding to the rotational component of the operation in fractional coordinates
        :param tr: 3D vector corresponding to the translational component of the operation in fractional coordinates
        :param op: Optional symmetry operation string to associate with the operation for reference
        :returns: :class:`SymOp` object with the specified rotation and translation components
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

    def translate(self, translation: NDArray[np.int16 | np.float64]) -> SymOp:
        """Translate a class:`SymOp` by a vector in fractional space

        .. note::
            If the translation vector has dtype ``np.int16``, it is assumed to be scaled by 12 
            to match the internal representation of the translation vector

        :param translation: 3D vector in fractional coordinates to translate the symmetry operation by
        :returns: New :class:`SymOp` with the same rotation but translated by the given vector
        """
        # if the translation is given in float, convert it to integer representation in twelfths
        if translation.dtype == np.float64:
            translation = np.round(translation * 12).astype(np.int16)
        new_tr = self._tr + translation

        # a shifted op no longer matches the original string, so let it be formatted on demand
        op = self._op if not np.any(translation) else None
        return SymOp.from_components(self._rot, new_tr, op)

    def apply(self, frac_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply this :class:`SymOp` to the given fractional coordinates

        :param frac_coords: Nx3 array of fractional coordinates to transform
        :returns: Transformed fractional coordinates after applying the symmetry operation
        """
        return (self._rot @ frac_coords.T).T + (self._tr / 12)


@dataclass
class SymOpList:
    """
    A list of symmetry operations, typically representing a subset of symmetry operations of a :class:`SpaceGroup`

    :param _sym_ops: List of :class:`SymOp` objects representing the symmetry operations
    :param _multiplicity: Number of equivalent multimers that correspond to these symmetry operations
    :param _ref_indices: Per-member base-reference index, defaults to all-zeros for single-reference clusters
    :ivar _rot_cache: Cached array of the rotational components of the symmetry operations
    :ivar _tr_cache: Cached array of the translational components of the symmetry operations
    :ivar _rel_cache: Lazily cached ``(L, v)`` pairwise relative Cartesian transforms
    :ivar _dist_fp_cache: Lazily cached ``(zz, distances)`` Z-tagged distance fingerprint
    """

    _sym_ops: list[SymOp] | None
    _multiplicity: int
    _ref_indices: NDArray | None = field(default=None, repr=False)
    _rot_cache: NDArray = field(init=False, repr=False)
    _tr_cache: NDArray = field(init=False, repr=False)

    _rel_cache: tuple[NDArray, NDArray] | None = field(default=None, init=False, repr=False, compare=False)
    _rel_int_cache: tuple[NDArray, NDArray] | None = field(default=None, init=False, repr=False, compare=False)
    _dist_fp_cache: tuple[NDArray, NDArray] | None = field(default=None, init=False, repr=False, compare=False)

    # lazily-derived geometry caches
    _PICKLE_EXCLUDED = ("_rel_cache", "_rel_int_cache", "_dist_fp_cache")

    def __getstate__(self):
        """Drop the derived caches when pickling, rebuild on demand

        :returns: The instance dict with the derived cache slots cleared
        """
        state = self.__dict__.copy()
        for name in self._PICKLE_EXCLUDED:
            state[name] = None
        return state

    def __post_init__(self):
        """Constructor method to validate inputs and compute caches for the symmetry operations

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

    def __iter__(self):
        return iter(self.sym_ops)

    def __len__(self):
        return len(self._rot_cache)

    def __getitem__(self, index: int) -> SymOp:
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

    @property
    def sym_ops(self) -> list[SymOp]:
        # lazily populate the list of SymOp objects from the cached rotation and translation components
        if self._sym_ops is None:
            self._sym_ops = [
                SymOp.from_components(self._rot_cache[i], self._tr_cache[i])
                for i in range(len(self._rot_cache))
            ]
        return self._sym_ops

    @property
    def multiplicity(self) -> int:
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, value: int):
        self._multiplicity = value

    @property
    def translations(self) -> list[NDArray]:
        return [sym_op.tr for sym_op in self.sym_ops]

    @property
    def rot_cache(self) -> NDArray:
        return self._rot_cache

    @property
    def tr_cache(self) -> NDArray:
        return self._tr_cache

    @property
    def ref_indices(self) -> NDArray:
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
        """Create a SymOpList from given rotation and translation components

        :param rot: Nx3x3 array of the rotational components of the operations in fractional coordinates
        :param tr: Nx3 array of the translational components of the operations in fractional coordinates
        :param multiplicity: Number of equivalent multimers that can correspond to this operation list
        :param ref_indices: Optional array of per-member base-reference indices (defaults to all-zeros)
        :returns: A :class:`SymOpList` object with the specified rotation and translation components and multiplicity
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

        # __new__ bypasses the dataclass __init__, so initialize lazily-populated geometry caches
        instance._rel_cache = None
        instance._rel_int_cache = None
        instance._dist_fp_cache = None

        return instance
