from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
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

    def __post_init__(self):
        """Constructor method

        :param op: A symmetry operation string like 'x,y,z' or '-x+1/2,y,-z+1/2'
        :type op: str
        """
        _rot, _tr = self._parse_symop(self._op)
        object.__setattr__(self, "_rot", _rot)
        object.__setattr__(self, "_tr", _tr)

    def _parse_symop(self, op: str) -> tuple[NDArray[np.int8], NDArray]:
        """Parse a symmetry operation from an xyz string

        :param op: A symmetry operation string like 'x,y,z' or '-x+1/2,y,-z+1/2'
        :type op: str
        """
        rot = np.zeros((3, 3), dtype=np.int8)
        tr = np.zeros(3)

        op = op.replace(" ", "").split(",")

        # Supposedly fast parsing of symmetry operations
        for i, comp in enumerate(op):
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

    @property
    def rot(self) -> NDArray[np.int8]:
        """Return the rotational component of the symmetry operation."""
        return self._rot

    @property
    def tr(self) -> NDArray:
        """Return the translational component of the symmetry operation."""
        return self._tr

    @classmethod
    def identity(cls) -> SymOp:
        """Return the identity symmetry operation."""
        return cls("x,y,z")

    @classmethod
    def from_components(cls, rot: NDArray[np.int8], tr: NDArray) -> SymOp:
        """Create a SymOp from given rotation and translation components

        :param rot: A 3x3 matrix corresponding to the rotational component of the symmetry operation in fractional coordinates
        :type rot: NDArray[np.int8]

        :param tr: A 3D vector corresponding to the translational component of the symmetry operation in fractional coordinates
        :type tr: NDArray
        """
        # This is a bit hacky but it allows us to create a SymOp without parsing a string
        op = "x,y,z"  # dummy string, won't be used
        symop = cls(op)
        object.__setattr__(symop, "_rot", rot)
        object.__setattr__(symop, "_tr", tr)
        return symop

    def compose(self, other: SymOp) -> SymOp:
        """Return the composition of this symmetry operation with another: self * other

        :param other: Another symmetry operation to compose with
        :type other: SymOp
        """
        new_rot = self._rot @ other._rot
        new_tr = self._rot @ other._tr + self._tr
        return SymOp.from_components(new_rot, new_tr)

    def inverse(self) -> SymOp:
        """Return the inverse of this symmetry operation."""
        inv_rot = self._rot.T
        inv_tr = -inv_rot @ self._tr
        return SymOp.from_components(inv_rot, inv_tr)

    def apply(self, frac_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply this SymOp to the given fractional coordinates

        :param frac_coords: An Nx3 array of fractional coordinates to transform
        :type frac_coords: NDArray[np.float64]
        """
        return (self._rot @ frac_coords.T).T + self._tr

    def __eq__(self, value: object, /) -> bool:
        """Check if two SymOps are equal by comparing their rotation and translation components."""
        if not isinstance(value, SymOp):
            return NotImplemented
        return np.array_equal(self._rot, value._rot) and np.allclose(
            self._tr, value._tr
        )
