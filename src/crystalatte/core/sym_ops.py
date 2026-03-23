from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import overload

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist


@dataclass()
class SymOp:
    """
    A symmetry operation defined by a rotation matrix and translation vector.
    The symmetry operation transforms fractional coordinates as: x' = rot @ x + tr

    :param rot: A 3x3 matrix corresponding to the rotational component of the symmetry operation in fractional coordinates
    :type rot: NDArray[np.float64]

    :param tr: A 3D vector corresponding to the translational component of the symmetry operation in fractional coordinates
    :type tr: NDArray[np.float64]
    """

    _op: str | None
    _rot: NDArray[np.int16] = field(init=False)
    _tr: NDArray[np.int16] = field(init=False)
    _augment: NDArray[np.int16] = field(init=False)

    def __post_init__(self):
        """Constructor method

        :param op: A symmetry operation string like 'x,y,z' or '-x+1/2,y,-z+1/2'
        :type op: str
        """
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
        :type op: str
        """
        rot = np.zeros((3, 3), dtype=np.int16)
        tr = np.zeros(3, dtype=np.int16)

        _op = op.replace(" ", "").split(",")

        # Supposedly fast parsing of symmetry operations
        for i, comp in enumerate(_op):
            sign = 1
            j = 0

            while j < len(comp):
                # Rotation
                if comp[j] in "+-":
                    sign = 1 if comp[j] == "+" else -1
                    j += 1
                elif comp[j] in "xyz":
                    axis = "xyz".index(comp[j])
                    rot[i, axis] = sign
                    j += 1

                # Translation
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

    @op.setter
    def op(self, value: str):
        self._op = value

    @property
    def rot(self) -> NDArray[np.int16]:
        """Return the rotational component of the symmetry operation."""
        return self._rot

    @rot.setter
    def rot(self, value: NDArray[np.int16]):
        self._rot = value

    @property
    def tr(self) -> NDArray[np.int16]:
        """Return the translational component of the symmetry operation."""
        return self._tr

    @tr.setter
    def tr(self, value: NDArray[np.int16]):
        self._tr = value

    @property
    def augment(self) -> NDArray[np.int16]:
        """Return the augmented matrix representation of the symmetry operation."""
        return self._augment

    @augment.setter
    def augment(self, value: NDArray[np.int16]):
        self._augment = value

    @classmethod
    def identity(cls) -> SymOp:
        """Return the identity symmetry operation."""
        return cls("x,y,z")

    @classmethod
    def from_components(
        cls, rot: NDArray[np.int16], tr: NDArray[np.int16], op: str | None = None
    ) -> SymOp:
        """Create a SymOp from given rotation and translation components

        :param rot: A 3x3 matrix corresponding to the rotational component of the symmetry operation in fractional coordinates
        :type rot: NDArray[np.int8]

        :param tr: A 3D vector corresponding to the translational component of the symmetry operation in fractional coordinates
        :type tr: NDArray
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
        """Return a new SymOp with the same rotation but translated by the given vector. The vector must be given as a multiple of 1/12 in fractional coordinates, since translations are stored as integers multiplied by 12.

        :param translation: A 3D vector in fractional coordinates to translate the symmetry operation by
        :type translation: NDArray
        """
        new_tr = self._tr + translation
        return SymOp.from_components(self._rot, new_tr, self._op)

    def compose(self, other: SymOp) -> SymOp:
        """Return the composition of this symmetry operation with another: self * other

        :param other: Another symmetry operation to compose with
        :type other: SymOp
        """
        new_rot = self._rot @ other._rot
        new_tr = self._rot @ other._tr + self._tr
        return SymOp.from_components(new_rot, new_tr)

    def compose_augment(self, other: SymOp) -> SymOp:
        """Return the composition of this symmetry operation with another using augmented matrices: self * other

        :param other: Another symmetry operation to compose with
        :type other: SymOp
        """
        new_augment = self._augment @ other._augment
        new_rot = new_augment[:3, :3]
        new_tr = new_augment[:3, 3]
        return SymOp.from_components(new_rot, new_tr)

    def apply(self, frac_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply this SymOp to the given fractional coordinates

        :param frac_coords: An Nx3 array of fractional coordinates to transform
        :type frac_coords: NDArray[np.float64]
        """
        return (self._rot @ frac_coords.T).T + (self._tr / 12)

    def __eq__(self, value: object, /) -> bool:
        """Check if two SymOps are equal by comparing their augmented matrices."""
        if not isinstance(value, SymOp):
            return NotImplemented
        return np.array_equal(self._augment, value._augment)

    def __hash__(self) -> int:
        """Hash the SymOp based on its augmented matrix."""
        return hash(self._augment.tobytes())

    def __iter__(self):
        """Return an iterator over the components of the symmetry operation (rotation and translation)."""
        yield self._rot
        yield self._tr

    def __repr__(self) -> str:
        return f"SymOp(op='{self._op}', rot='{self._rot}', tr='{self._tr}', augment=\n{self._augment})"


@dataclass
class SymOpList:
    """
    A list of symmetry operations, typically representing a subset of symmetry operations of a space group.

    :param sym_ops: A list of SymOp objects representing the symmetry operations
    :type sym_ops: list[SymOp]

    :param multiplicity: The number of equivalent multimers that can be generated by this list of symmetry operations
    :type multiplicity: int
    """

    _sym_ops: list[SymOp] | None
    _multiplicity: int

    _ops: list[str] | None = field(default=None, repr=False)
    _rot_cache: NDArray = field(init=False, repr=False)
    _tr_cache: NDArray = field(init=False, repr=False)
    _aug_cache: NDArray = field(init=False, repr=False)

    def __post_init__(self):
        """Constructor method"""
        if not self._sym_ops:
            raise ValueError("SymOpList must contain at least one SymOp")
        if self._multiplicity < 1:
            raise ValueError("Multiplicity must be a positive integer")

        # compute matrices for all sym_ops and store in cache
        self._rot_cache = np.stack([op._rot for op in self._sym_ops])
        self._tr_cache = np.stack([op._tr for op in self._sym_ops])
        self._aug_cache = np.stack([op._augment for op in self._sym_ops])

    @property
    def sym_ops(self) -> list[SymOp]:
        """Return the list of symmetry operations."""
        if self._sym_ops is None:
            self._sym_ops = [
                SymOp.from_components(
                    self._rot_cache[i],
                    self._tr_cache[i],
                    self._ops[i] if self._ops else None,
                )
                for i in range(len(self._rot_cache))
            ]
        return self._sym_ops

    @sym_ops.setter
    def sym_ops(self, value: list[SymOp], ops: list[str] | None = None):
        self._sym_ops = value
        self._ops = ops

    @property
    def multiplicity(self) -> int:
        """Return the multiplicity of multimers generated by this list of symmetry operations."""
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, value: int):
        self._multiplicity = value

    @property
    def rotations(self) -> list[NDArray[np.int16]]:
        """Return a list of the rotational components of the symmetry operations."""
        return [sym_op.rot for sym_op in self.sym_ops]

    @property
    def translations(self) -> list[NDArray]:
        """Return a list of the translational components of the symmetry operations."""
        return [sym_op.tr for sym_op in self.sym_ops]

    @property
    def augments(self) -> list[NDArray]:
        """Return a list of the augmented matrix representations of the symmetry operations."""
        return [sym_op.augment for sym_op in self.sym_ops]

    @property
    def rot_cache(self) -> NDArray:
        """Return the cached array of rotational components of the symmetry operations."""
        return self._rot_cache

    @property
    def tr_cache(self) -> NDArray:
        """Return the cached array of translational components of the symmetry operations."""
        return self._tr_cache

    @property
    def aug_cache(self) -> NDArray:
        """Return the cached array of augmented matrix representations of the symmetry operations."""
        return self._aug_cache

    @property
    def rotation_fp(self) -> int:
        """Return a fingerprint of the symmetry operations based on their rotational components."""
        rots = self._rot_cache[:].reshape(len(self._rot_cache), -1)
        fp = np.sort(rots.flatten()).tobytes()
        return hash(fp)

    @property
    def translation_fp(self) -> int:
        """Return a fingerprint of the symmetry operations based on the pairwise squared distances between their translation components."""
        translations = self._tr_cache[:] / 12
        sq_dists = pdist(translations, metric="sqeuclidean").astype(np.int32)
        fp = np.sort(sq_dists).tobytes()
        return hash(fp)

    @property
    def augment_fp(self) -> int:
        """Return a fingerprint of the symmetry operations based on their augmented matrix representations."""
        augments = self._aug_cache[:].reshape(len(self._aug_cache), -1)
        fp = np.sort(augments.flatten()).tobytes()
        return hash(fp)

    # return a symoplist of N identity symops
    @classmethod
    def identities(cls, N: int) -> SymOpList:
        """Return a SymOpList containing N identity symmetry operations."""
        return cls([SymOp.identity()] * N, 1)

    @classmethod
    def from_components(
        cls, rot: NDArray, tr: NDArray, multiplicity: int, ops: list[str] | None = None
    ) -> SymOpList:
        instance = cls.__new__(cls)
        instance._sym_ops = None
        instance._multiplicity = multiplicity

        # set caches
        instance._rot_cache = rot
        instance._tr_cache = tr
        aug_cache = np.eye(4, dtype=np.int16).reshape(1, 4, 4).repeat(len(rot), axis=0)
        aug_cache[:, :3, :3] = rot
        aug_cache[:, :3, 3] = tr
        instance._aug_cache = aug_cache

        return instance

    def rotate_list(
        self, rotation: NDArray, exclude_reference: bool = False
    ) -> SymOpList:
        """Return a new SymOpList with all symmetry operations rotated by the given matrix in fractional coordinates.

        :param translation: A tensor of rotations in fractional coordinates to translate all symmetry operations by
        :type translation: NDArray
        """
        new_rot = self._rot_cache.copy()
        new_rot[:] = rotation @ new_rot.transpose(0, 2, 1)
        new_rot = new_rot.transpose(0, 2, 1)
        return SymOpList.from_components(new_rot, self._tr_cache, self._multiplicity)

    def translate_list(
        self, translation: NDArray, exclude_reference: bool = False
    ) -> SymOpList:
        """Return a new SymOpList with all symmetry operations translated by the given vector in fractional coordinates. The vector must be given as a multiple of 1/12 in fractional coordinates, since translations are stored as integers multiplied by 12.

        :param translation: A 3D matrix of translations in fractional coordinates to translate all symmetry operations by
        :type translation: NDArray
        """
        n = len(self._tr_cache)

        if exclude_reference and translation.shape[0] == n:
            translation[0] = 0
        elif exclude_reference and translation.shape[0] == n - 1:
            translation = np.vstack((np.zeros(3, dtype=np.int32), translation))
        elif translation.shape[0] != n:
            raise ValueError(
                "Translation matrix must have the same number of rows as the "
                "number of symmetry operations, or one row for all operations."
            )

        new_tr = self._tr_cache.copy()
        new_tr[:] += translation
        return SymOpList.from_components(self._rot_cache, new_tr, self._multiplicity)

    def __iter__(self):
        """Return an iterator over the symmetry operations."""
        return iter(self.sym_ops)

    def __len__(self):
        """Return the number of symmetry operations in the list."""
        return len(self.sym_ops)

    @overload
    def __getitem__(self, index: int) -> SymOp: ...
    @overload
    def __getitem__(self, index: slice) -> SymOpList: ...

    def __getitem__(self, index: int | slice) -> SymOp | SymOpList:
        """Return the symmetry operation at the specified index."""
        result = self.sym_ops[index]
        if isinstance(result, list):
            return SymOpList(result, self._multiplicity)
        return result

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


if __name__ == "__main__":
    pass
