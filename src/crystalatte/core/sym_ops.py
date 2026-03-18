from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray


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

    _op: str
    _rot: NDArray[np.int8] = field(init=False)
    _tr: NDArray = field(init=False)
    _augment: NDArray = field(init=False)

    def __post_init__(self):
        """Constructor method

        :param op: A symmetry operation string like 'x,y,z' or '-x+1/2,y,-z+1/2'
        :type op: str
        """
        self._rot, self._tr = self._parse_symop(self._op)

        augment = np.zeros((4, 4))
        augment[:3, :3] = self._rot
        augment[:3, 3] = self._tr
        augment[3, :3] = 0
        augment[3, 3] = 1
        self._augment = augment

    def _parse_symop(self, op: str) -> tuple[NDArray[np.int8], NDArray]:
        """Parse a symmetry operation from an xyz string

        :param op: A symmetry operation string like 'x,y,z' or '-x+1/2,y,-z+1/2'
        :type op: str
        """
        rot = np.zeros((3, 3), dtype=np.int8)
        tr = np.zeros(3)

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
                        tr[i] += float(Fraction(num_str)) * sign
                        sign = 1

        return rot, tr

    @property
    def op(self) -> str:
        """Return the original symmetry operation string."""
        return self._op

    @op.setter
    def op(self, value: str):
        self._op = value

    @property
    def rot(self) -> NDArray[np.int8]:
        """Return the rotational component of the symmetry operation."""
        return self._rot

    @rot.setter
    def rot(self, value: NDArray[np.int8]):
        self._rot = value

    @property
    def tr(self) -> NDArray:
        """Return the translational component of the symmetry operation."""
        return self._tr

    @tr.setter
    def tr(self, value: NDArray):
        self._tr = value

    @property
    def augment(self) -> NDArray:
        """Return the augmented matrix representation of the symmetry operation."""
        return self._augment

    @augment.setter
    def augment(self, value: NDArray):
        self._augment = value

    @classmethod
    def identity(cls) -> SymOp:
        """Return the identity symmetry operation."""
        return cls("x,y,z")

    @classmethod
    def from_components(
        cls, rot: NDArray[np.int8], tr: NDArray, op: str | None = None
    ) -> SymOp:
        """Create a SymOp from given rotation and translation components

        :param rot: A 3x3 matrix corresponding to the rotational component of the symmetry operation in fractional coordinates
        :type rot: NDArray[np.int8]

        :param tr: A 3D vector corresponding to the translational component of the symmetry operation in fractional coordinates
        :type tr: NDArray
        """
        symop = cls.identity()
        symop._rot = rot
        symop._tr = tr

        augment = np.zeros((4, 4))
        augment[:3, :3] = rot
        augment[:3, 3] = tr
        augment[3, :3] = 0
        augment[3, 3] = 1
        symop._augment = augment

        if op:
            symop._op = op

        return symop

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

    # def inverse(self) -> SymOp:
    #     """Return the inverse of this symmetry operation."""
    #     inv_rot = self._rot.T
    #     inv_tr = -inv_rot @ self._tr
    #     return SymOp.from_components(inv_rot, inv_tr)

    def apply(self, frac_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply this SymOp to the given fractional coordinates

        :param frac_coords: An Nx3 array of fractional coordinates to transform
        :type frac_coords: NDArray[np.float64]
        """
        return (self._rot @ frac_coords.T).T + self._tr

    def __eq__(self, value: object, /) -> bool:
        """Check if two SymOps are equal by comparing their augmented matrices."""
        if not isinstance(value, SymOp):
            return NotImplemented
        return np.array_equal(self._augment, value._augment)

    def __hash__(self) -> int:
        """Hash the SymOp based on its augmented matrix."""
        return hash(self._augment.tobytes())

    def __repr__(self) -> str:
        return f"SymOp(op='{self._op}', augment=\n{self._augment})"


@dataclass
class SymOpList:
    """
    A list of symmetry operations, typically representing a subset of symmetry operations of a space group.

    :param sym_ops: A list of SymOp objects representing the symmetry operations
    :type sym_ops: list[SymOp]

    :param multiplicity: The number of equivalent multimers that can be generated by this list of symmetry operations
    :type multiplicity: int
    """

    _sym_ops: list[SymOp]
    _multiplicity: int

    @property
    def sym_ops(self) -> list[SymOp]:
        """Return the list of symmetry operations."""
        return self._sym_ops

    @sym_ops.setter
    def sym_ops(self, value: list[SymOp]):
        self._sym_ops = value

    @property
    def multiplicity(self) -> int:
        """Return the multiplicity of multimers generated by this list of symmetry operations."""
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, value: int):
        self._multiplicity = value

    def translate_list(
        self, translation: NDArray, exclude_reference: bool = False
    ) -> SymOpList:
        """Return a new SymOpList with all symmetry operations translated by the given vector

        :param translation: A 3D vector in fractional coordinates to translate all symmetry operations by
        :type translation: NDArray
        """
        new_sym_ops = []
        for i, sym_op in enumerate(self._sym_ops):
            if exclude_reference and sym_op == SymOp.identity() and i == 0:
                new_sym_ops.append(sym_op)
                continue

            new_tr = sym_op.tr + translation
            new_sym_op = SymOp.from_components(sym_op.rot, new_tr, sym_op.op)
            new_sym_ops.append(new_sym_op)

        return SymOpList(new_sym_ops, self._multiplicity)

    def __iter__(self):
        """Return an iterator over the symmetry operations."""
        return iter(self._sym_ops)

    def __len__(self):
        """Return the number of symmetry operations in the list."""
        return len(self._sym_ops)

    def __getitem__(self, index: int) -> SymOp:
        """Return the symmetry operation at the specified index."""
        return self._sym_ops[index]

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, SymOpList) and set(self._sym_ops) == set(
            value._sym_ops
        )

    def __repr__(self) -> str:
        return f"SymOpList(sym_ops={self._sym_ops}, multiplicity={self._multiplicity})"
