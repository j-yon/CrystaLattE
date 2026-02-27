from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray


@dataclass
class SymOp:
    """
    A symmetry operation defined by a rotation matrix and translation vector.

    The symmetry operation transforms fractional coordinates as:
        x' = rot @ x + tr

    Attributes
    ----------
    rot : NDArray[np.float64]
        The 3x3 rotation matrix in fractional coordinates.
    tr : NDArray[np.float64]
        The translation vector in fractional coordinates.
    """

    rot: NDArray[np.float64]
    tr: NDArray[np.float64]

    def __post_init__(self):
        self.rot = np.asarray(self.rot, dtype=np.float64)
        self.tr = np.asarray(self.tr, dtype=np.float64)

    def apply(self, frac_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply symmetry operation to fractional coordinates.

        Parameters:
        frac_coords : NDArray[np.float64]
            Fractional coordinates (N, 3).

        Returns
        Transformed fractional coordinates.
        """
        return (self.rot @ frac_coords.T).T + self.tr

    def as_matrix(self) -> NDArray[np.float64]:
        """
        Return the SymOp as a 4x4 augmented matrix.

        Returns
        -------
        NDArray[np.float64]
            4x4 matrix representation.
        """
        mat = np.eye(4)
        mat[:3, :3] = self.rot
        mat[:3, 3] = self.tr
        return mat

    def to_xyz_string(self) -> str:
        """
        Convert symmetry operation to xyz string format.

        Returns
        -------
        str
            String like 'x,y,z' or '-x+1/2,y,-z+1/2'.
        """
        axes = ["x", "y", "z"]
        parts = []
        for i in range(3):
            terms = []
            for j, axis in enumerate(axes):
                coef = self.rot[i, j]
                if abs(coef) > 1e-10:
                    if abs(coef - 1.0) < 1e-10:
                        terms.append(f"+{axis}")
                    elif abs(coef + 1.0) < 1e-10:
                        terms.append(f"-{axis}")
                    else:
                        terms.append(f"{coef:+g}*{axis}")
            # Add translation
            tr_val = self.tr[i]
            if abs(tr_val) > 1e-10:
                # Convert to fraction if possible
                tr_str = _float_to_fraction_str(tr_val)
                terms.append(tr_str)
            if not terms:
                terms.append("0")
            part = "".join(terms)
            if part.startswith("+"):
                part = part[1:]
            parts.append(part)
        return ",".join(parts)

    @classmethod
    def identity(cls) -> "SymOp":
        """Return the identity symmetry operation."""
        return cls(rot=np.eye(3), tr=np.zeros(3))


def _float_to_fraction_str(val: float) -> str:
    """Convert float to fraction string for common crystallographic fractions."""
    fractions = {
        1 / 2: "+1/2",
        1 / 3: "+1/3",
        2 / 3: "+2/3",
        1 / 4: "+1/4",
        3 / 4: "+3/4",
        1 / 6: "+1/6",
        5 / 6: "+5/6",
        -1 / 2: "-1/2",
        -1 / 3: "-1/3",
        -2 / 3: "-2/3",
        -1 / 4: "-1/4",
        -3 / 4: "-3/4",
        -1 / 6: "-1/6",
        -5 / 6: "-5/6",
    }
    for frac_val, frac_str in fractions.items():
        if abs(val - frac_val) < 1e-10:
            return frac_str
    return f"{val:+g}"


def parse_symop_xyz(xyz_str: str) -> SymOp:
    """
    Parse a symmetry operation from xyz string format.

    Parses strings like:
        - "x,y,z"
        - "-x,y,-z"
        - "1/2+x,1/2-y,z"
        - "x+1/2,-y+1/2,z+1/2"

    Parameters
    ----------
    xyz_str : str
        The symmetry operation in xyz string format.

    Returns
    -------
    SymOp
        The parsed symmetry operation.

    Examples
    --------
    >>> symop = parse_symop_xyz("x,y,z")
    >>> symop = parse_symop_xyz("-x+1/2,y,-z+1/2")
    """
    rot = np.zeros((3, 3))
    tr = np.zeros(3)

    # Clean up the string
    xyz_str = xyz_str.lower().replace(" ", "")
    parts = xyz_str.split(",")

    if len(parts) != 3:
        raise ValueError(f"Invalid symmetry operation string: {xyz_str}")

    for i, part in enumerate(parts):
        # Parse each component (x, y, z translations)
        rot[i], tr[i] = _parse_symop_component(part)

    return SymOp(rot=rot, tr=tr)


def _parse_symop_component(component: str) -> tuple[NDArray[np.float64], float]:
    """
    Parse a single component of a symmetry operation string.

    Parameters
    ----------
    component : str
        A single component like "x", "-y", "1/2+z", "x-1/2".

    Returns
    -------
    Tuple[NDArray[np.float64], float]
        The rotation row and translation value.
    """
    rot_row = np.zeros(3)
    translation = 0.0

    # Map axis letters to indices
    axis_map = {"x": 0, "y": 1, "z": 2}

    # Normalize the component: ensure it starts with + or -
    if component and component[0] not in "+-":
        component = "+" + component

    pos = 0
    while pos < len(component):
        # Check for fraction followed by axis (e.g., "1/2x" or "+1/2x")
        frac_axis_match = re.match(r"([+-]?)(\d+/\d+)([xyz])", component[pos:])
        if frac_axis_match:
            sign_str, frac, axis = frac_axis_match.groups()
            sign = -1.0 if sign_str == "-" else 1.0
            # This is coefficient * axis (unusual but possible)
            coef = _parse_fraction(frac) * sign
            rot_row[axis_map[axis]] = coef
            pos += frac_axis_match.end()
            continue

        # Check for axis with optional sign (e.g., "x", "-x", "+x")
        axis_match = re.match(r"([+-]?)([xyz])", component[pos:])
        if axis_match:
            sign_str, axis = axis_match.groups()
            sign = -1.0 if sign_str == "-" else 1.0
            rot_row[axis_map[axis]] = sign
            pos += axis_match.end()
            continue

        # Check for standalone fraction (e.g., "+1/2", "-1/3")
        frac_match = re.match(r"([+-]?)(\d+/\d+)", component[pos:])
        if frac_match:
            sign_str, frac = frac_match.groups()
            sign = -1.0 if sign_str == "-" else 1.0
            translation += _parse_fraction(frac) * sign
            pos += frac_match.end()
            continue

        # Check for decimal number
        num_match = re.match(r"([+-]?\d*\.?\d+)", component[pos:])
        if num_match:
            translation += float(num_match.group(1))
            pos += num_match.end()
            continue

        # Skip unrecognized characters (shouldn't happen with valid input)
        pos += 1

    return rot_row, translation


def _parse_fraction(frac_str: str) -> float:
    """
    Parse a fraction string like '1/2' to a float.

    Parameters
    ----------
    frac_str : str
        Fraction string (e.g., "1/2", "2/3").

    Returns
    -------
    float
        The float value.
    """
    if "/" in frac_str:
        num, denom = frac_str.split("/")
        return float(num) / float(denom)
    return float(frac_str)
