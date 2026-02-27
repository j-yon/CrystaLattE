"""
Space group symmetry operations for crystallographic systems.

This module provides functionality to:
1. Parse CIF files to extract unit cell and symmetry information
2. Parse symmetry operation strings (e.g., "x,y,z", "1/2+x,-y,z+1/2")
3. Apply symmetry operations to generate symmetry-equivalent positions
4. Convert between fractional and Cartesian coordinates
5. Create qcelemental Molecule objects from CIF files
"""

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
        """
        Apply symmetry operation to fractional coordinates.

        Parameters
        ----------
        frac_coords : NDArray[np.float64]
            Fractional coordinates (3,) or (N, 3).

        Returns
        -------
        NDArray[np.float64]
            Transformed fractional coordinates.
        """
        frac_coords = np.asarray(frac_coords)
        if frac_coords.ndim == 1:
            return self.rot @ frac_coords + self.tr
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


@dataclass
class Crystal:
    """
    Represents a crystal unit cell with lattice parameters and space group symmetry.

    Attributes
    ----------
    a : float
        Length of the a-axis in Angstroms.
    b : float
        Length of the b-axis in Angstroms.
    c : float
        Length of the c-axis in Angstroms.
    alpha : float
        Angle between b and c axes in degrees.
    beta : float
        Angle between a and c axes in degrees.
    gamma : float
        Angle between a and b axes in degrees.
    space_group_name : str
        Space group name (Hermann-Mauguin notation).
    space_group_number : int
        International Tables space group number.
    symops : List[SymOp]
        List of symmetry operations.
    """

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    space_group_name: str = "P1"
    space_group_number: int = 1
    symops: list[SymOp] = field(default_factory=list)

    # Computed matrices (set in __post_init__)
    _frac_to_cart: NDArray[np.float64] = field(init=False, repr=False)
    _cart_to_frac: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize transformation matrices."""
        self._update_matrices()
        if not self.symops:
            self.symops = [SymOp.identity()]

    def _update_matrices(self):
        """Compute fractional <-> Cartesian transformation matrices."""
        # Convert angles to radians
        alpha_rad = np.radians(self.alpha)
        beta_rad = np.radians(self.beta)
        gamma_rad = np.radians(self.gamma)

        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad)
        cos_gamma = np.cos(gamma_rad)
        sin_gamma = np.sin(gamma_rad)

        # Volume factor
        omega = np.sqrt(
            1
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
            + 2 * cos_alpha * cos_beta * cos_gamma
        )

        # Fractional to Cartesian matrix (column vectors are lattice vectors)
        # Using the standard crystallographic convention:
        # a along x, b in xy plane, c general
        c_y = self.c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        self._frac_to_cart = np.array(
            [
                [self.a, self.b * cos_gamma, self.c * cos_beta],
                [0.0, self.b * sin_gamma, c_y],
                [0.0, 0.0, self.c * omega / sin_gamma],
            ]
        )

        # Cartesian to fractional matrix
        self._cart_to_frac = np.linalg.inv(self._frac_to_cart)

    @property
    def volume(self) -> float:
        """Calculate unit cell volume in cubic Angstroms."""
        return abs(np.linalg.det(self._frac_to_cart))

    def to_cartesian(self, frac_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert fractional coordinates to Cartesian coordinates.

        Parameters
        ----------
        frac_coords : NDArray[np.float64]
            Fractional coordinates, shape (3,) or (N, 3).

        Returns
        -------
        NDArray[np.float64]
            Cartesian coordinates in Angstroms.
        """
        frac_coords = np.asarray(frac_coords)
        if frac_coords.ndim == 1:
            return self._frac_to_cart @ frac_coords
        return (self._frac_to_cart @ frac_coords.T).T

    def to_fractional(self, cart_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert Cartesian coordinates to fractional coordinates.

        Parameters
        ----------
        cart_coords : NDArray[np.float64]
            Cartesian coordinates in Angstroms, shape (3,) or (N, 3).

        Returns
        -------
        NDArray[np.float64]
            Fractional coordinates.
        """
        cart_coords = np.asarray(cart_coords)
        if cart_coords.ndim == 1:
            return self._cart_to_frac @ cart_coords
        return (self._cart_to_frac @ cart_coords.T).T

    def apply_symop(
        self, frac_coords: NDArray[np.float64], symop: SymOp
    ) -> NDArray[np.float64]:
        """
        Apply a symmetry operation to fractional coordinates.

        Parameters
        ----------
        frac_coords : NDArray[np.float64]
            Fractional coordinates.
        symop : SymOp
            Symmetry operation to apply.

        Returns
        -------
        NDArray[np.float64]
            Transformed fractional coordinates.
        """
        return symop.apply(frac_coords)

    def generate_symmetry_equivalents(
        self,
        frac_coords: NDArray[np.float64],
        wrap_to_unit_cell: bool = True,
    ) -> NDArray[np.float64]:
        """
        Generate all symmetry-equivalent positions for given fractional coordinates.

        Parameters
        ----------
        frac_coords : NDArray[np.float64]
            Fractional coordinates, shape (3,) or (N, 3).
        wrap_to_unit_cell : bool
            If True, wrap coordinates to [0, 1).

        Returns
        -------
        NDArray[np.float64]
            All symmetry-equivalent positions, shape (M, 3) or (N*M, 3)
            where M is the number of symmetry operations.
        """
        frac_coords = np.asarray(frac_coords)
        single_atom = frac_coords.ndim == 1

        if single_atom:
            frac_coords = frac_coords.reshape(1, 3)

        all_positions = []
        for symop in self.symops:
            transformed = symop.apply(frac_coords)
            if wrap_to_unit_cell:
                transformed = transformed % 1.0
            all_positions.append(transformed)

        result = np.vstack(all_positions)
        return result


def parse_cif_file(
    filepath: str | Path,
) -> tuple[Crystal, list[str], NDArray[np.float64]]:
    """
    Parse a CIF file to extract crystal and atomic information.

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.

    Returns
    -------
    Tuple[Crystal, List[str], NDArray[np.float64]]
        - Crystal object with unit cell and symmetry information
        - List of atom labels/symbols
        - Fractional coordinates array of shape (N, 3)

    Examples
    --------
    >>> crystal, symbols, frac_coords = parse_cif_file("structure.cif")
    """
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        content = f.read()

    # Parse unit cell parameters
    a = _extract_cif_value(content, "_cell_length_a")
    b = _extract_cif_value(content, "_cell_length_b")
    c = _extract_cif_value(content, "_cell_length_c")
    alpha = _extract_cif_value(content, "_cell_angle_alpha")
    beta = _extract_cif_value(content, "_cell_angle_beta")
    gamma = _extract_cif_value(content, "_cell_angle_gamma")

    # Parse space group
    sg_number = _extract_cif_int(content, "_space_group_IT_number")
    sg_name = _extract_cif_string(content, "_symmetry_space_group_name_H-M")
    if sg_name:
        sg_name = sg_name.strip("'\"")

    # Parse symmetry operations
    symops = _parse_cif_symmetry_operations(content)

    # Parse atom sites
    symbols, frac_coords = _parse_cif_atom_sites(content)

    crystal = Crystal(
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        space_group_name=sg_name or "P1",
        space_group_number=sg_number or 1,
        symops=symops if symops else [SymOp.identity()],
    )

    return crystal, symbols, frac_coords


def _extract_cif_value(content: str, key: str) -> float:
    """Extract a numeric value from CIF content, removing uncertainty in parentheses."""
    pattern = rf"{re.escape(key)}\s+(\S+)"
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        val_str = match.group(1)
        # Remove uncertainty in parentheses, e.g., "5.624(1)" -> "5.624"
        val_str = re.sub(r"\([^)]*\)", "", val_str)
        return float(val_str)
    raise ValueError(f"Could not find {key} in CIF file")


def _extract_cif_int(content: str, key: str) -> Optional[int]:
    """Extract an integer value from CIF content."""
    pattern = rf"{re.escape(key)}\s+(\d+)"
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_cif_string(content: str, key: str) -> Optional[str]:
    """Extract a string value from CIF content."""
    pattern = rf"{re.escape(key)}\s+(.+)"
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _parse_cif_symmetry_operations(content: str) -> List[SymOp]:
    """
    Parse symmetry operations from CIF file.

    Looks for the _symmetry_equiv_pos_as_xyz loop.
    """
    symops = []

    # Find the symmetry loop
    # Pattern to match symmetry operations in a loop
    loop_pattern = (
        r"loop_\s*_symmetry_equiv_pos_as_xyz\s*((?:[^\n]+\n)+?)(?=loop_|_\w|$)"
    )
    match = re.search(loop_pattern, content, re.IGNORECASE)

    if match:
        lines = match.group(1).strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("_"):
                symops.append(parse_symop_xyz(line))
        return symops

    # Alternative format with site_id
    loop_pattern2 = r"loop_\s*(?:_symmetry_equiv_pos_site_id\s*)?_symmetry_equiv_pos_as_xyz\s*((?:[^\n]+\n)+?)(?=loop_|_\w|$)"
    match = re.search(loop_pattern2, content, re.IGNORECASE)

    if not match:
        # Try another common format
        pattern = r"_symmetry_equiv_pos_as_xyz\s*\n((?:.*\n)*?)(?=loop_|_\w)"
        match = re.search(pattern, content, re.IGNORECASE)

    if match:
        lines = match.group(1).strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("_") and not line.startswith("loop"):
                # Handle format with id number prefix
                parts = line.split()
                if len(parts) >= 1:
                    xyz_str = parts[-1] if len(parts) > 1 else parts[0]
                    # Remove quotes if present
                    xyz_str = xyz_str.strip("'\"")
                    if "," in xyz_str:
                        symops.append(parse_symop_xyz(xyz_str))

    return symops


def _parse_cif_atom_sites(content: str) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Parse atom site information from CIF file.

    Returns atom symbols and fractional coordinates.
    """
    symbols = []
    frac_coords = []

    lines = content.split("\n")

    # Find the atom_site loop with fractional coordinates
    # We need to find a loop_ that contains _atom_site_fract_x
    i = 0
    found_loop = False
    columns = []
    col_map = {}

    while i < len(lines):
        line = lines[i].strip()

        # Look for loop_ followed by _atom_site headers
        if line.lower() == "loop_":
            # Check if this loop has _atom_site_fract_x
            j = i + 1
            temp_columns = []
            while j < len(lines):
                header_line = lines[j].strip()
                if header_line.lower().startswith("_atom_site_"):
                    # Extract the field name
                    field = header_line.lower().replace("_atom_site_", "")
                    temp_columns.append(field)
                    j += 1
                elif header_line.startswith("_") or header_line.lower() == "loop_":
                    # Different type of header or new loop
                    break
                elif header_line == "" or header_line.startswith("#"):
                    j += 1
                else:
                    # Data line - check if we found the right loop
                    break

            # Check if this loop has fractional coordinates
            if (
                "fract_x" in temp_columns
                and "fract_y" in temp_columns
                and "fract_z" in temp_columns
            ):
                found_loop = True
                columns = temp_columns
                col_map = {col: idx for idx, col in enumerate(columns)}
                i = j  # Start reading data from here
                break

        i += 1

    if not found_loop:
        raise ValueError(
            "Could not find atom_site loop with fractional coordinates in CIF file"
        )

    # Required columns for coordinates
    fract_x_col = col_map.get("fract_x")
    fract_y_col = col_map.get("fract_y")
    fract_z_col = col_map.get("fract_z")

    # Label column (element symbol)
    label_col = col_map.get("label")
    type_symbol_col = col_map.get("type_symbol")

    # Helper function to parse coordinates
    def parse_coord(s: str) -> float:
        s = re.sub(r"\([^)]*\)", "", s)
        return float(s)

    # Parse data lines
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        # Stop at next loop or new data block or header
        if (
            line.lower().startswith("loop_")
            or line.startswith("_")
            or line.lower().startswith("data_")
        ):
            break

        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < len(columns):
            continue

        # Get symbol
        if type_symbol_col is not None:
            symbol = parts[type_symbol_col]
        elif label_col is not None:
            # Extract element from label (e.g., "C1" -> "C", "O2" -> "O")
            label = parts[label_col]
            label_match = re.match(r"([A-Za-z]+)", label)
            if label_match is None:
                continue
            symbol = label_match.group(1)
        else:
            continue

        # Type assertions to satisfy the type checker (we know these are not None from above)
        assert fract_x_col is not None
        assert fract_y_col is not None
        assert fract_z_col is not None

        x = parse_coord(parts[fract_x_col])
        y = parse_coord(parts[fract_y_col])
        z = parse_coord(parts[fract_z_col])

        symbols.append(symbol)
        frac_coords.append([x, y, z])

    return symbols, np.array(frac_coords)


def remove_duplicate_atoms(
    symbols: List[str],
    coords: NDArray[np.float64],
    tolerance: float = 0.01,
) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Remove duplicate atoms based on coordinate proximity.

    Parameters
    ----------
    symbols : List[str]
        List of atom symbols.
    coords : NDArray[np.float64]
        Coordinates array of shape (N, 3).
    tolerance : float
        Distance tolerance for considering atoms as duplicates.

    Returns
    -------
    Tuple[List[str], NDArray[np.float64]]
        Unique atom symbols and coordinates.
    """
    if len(symbols) == 0:
        return symbols, coords

    unique_symbols = [symbols[0]]
    unique_coords = [coords[0]]

    for i in range(1, len(symbols)):
        is_duplicate = False
        for j in range(len(unique_coords)):
            dist = np.linalg.norm(coords[i] - unique_coords[j])
            # Also check for periodic images
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        shifted = coords[i] + np.array([dx, dy, dz])
                        dist = np.linalg.norm(shifted - unique_coords[j])
                        if dist < tolerance:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        break
                if is_duplicate:
                    break
            if is_duplicate:
                break

        if not is_duplicate:
            unique_symbols.append(symbols[i])
            unique_coords.append(coords[i])

    return unique_symbols, np.array(unique_coords)


def cif_to_molecule(
    filepath: str | Path,
    expand_symmetry: bool = True,
    wrap_to_unit_cell: bool = True,
    remove_duplicates: bool = True,
    duplicate_tolerance: float = 0.01,
) -> qcel.models.Molecule:
    """
    Create a QCElemental Molecule object from a CIF file.

    This function parses a CIF file, optionally applies space group symmetry
    operations to expand the asymmetric unit to the full unit cell, and
    returns a QCElemental Molecule object with Cartesian coordinates.

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.
    expand_symmetry : bool, default=True
        If True, apply all symmetry operations to generate the full unit cell.
        If False, only the asymmetric unit atoms are returned.
    wrap_to_unit_cell : bool, default=True
        If True, wrap all coordinates to the unit cell [0, 1) in fractional space.
    remove_duplicates : bool, default=True
        If True, remove duplicate atoms that may arise from special positions.
    duplicate_tolerance : float, default=0.01
        Distance tolerance (in fractional coordinates) for detecting duplicates.

    Returns
    -------
    qcel.models.Molecule
        QCElemental Molecule object with Cartesian coordinates.

    Examples
    --------
    >>> mol = cif_to_molecule("carbon_dioxide.cif")
    >>> print(mol.symbols)
    >>> print(mol.geometry)  # Cartesian coordinates in Bohr
    """
    crystal, asu_symbols, asu_frac_coords = parse_cif_file(filepath)

    if expand_symmetry:
        # Apply all symmetry operations
        all_symbols = []
        all_frac_coords = []

        for i, (symbol, frac_coord) in enumerate(zip(asu_symbols, asu_frac_coords)):
            equiv_coords = crystal.generate_symmetry_equivalents(
                frac_coord, wrap_to_unit_cell=wrap_to_unit_cell
            )
            for coord in equiv_coords:
                all_symbols.append(symbol)
                all_frac_coords.append(coord)

        all_frac_coords = np.array(all_frac_coords)

        if remove_duplicates:
            all_symbols, all_frac_coords = remove_duplicate_atoms(
                all_symbols, all_frac_coords, tolerance=duplicate_tolerance
            )
    else:
        all_symbols = asu_symbols
        all_frac_coords = asu_frac_coords

    # Convert to Cartesian coordinates
    cart_coords = crystal.to_cartesian(all_frac_coords)

    # Convert Angstroms to Bohr for QCElemental
    cart_coords_bohr = cart_coords / qcel.constants.bohr2angstroms

    # Create QCElemental Molecule
    mol = qcel.models.Molecule(
        symbols=all_symbols,
        geometry=cart_coords_bohr.flatten(),
    )

    return mol


def get_crystal_info(filepath: str | Path) -> dict:
    """
    Extract crystal information from a CIF file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.

    Returns
    -------
    dict
        Dictionary containing crystal information:
        - 'a', 'b', 'c': lattice parameters (Angstroms)
        - 'alpha', 'beta', 'gamma': lattice angles (degrees)
        - 'volume': unit cell volume (cubic Angstroms)
        - 'space_group_name': space group name
        - 'space_group_number': IT space group number
        - 'n_symops': number of symmetry operations
        - 'n_atoms_asu': number of atoms in asymmetric unit
    """
    crystal, symbols, frac_coords = parse_cif_file(filepath)

    return {
        "a": crystal.a,
        "b": crystal.b,
        "c": crystal.c,
        "alpha": crystal.alpha,
        "beta": crystal.beta,
        "gamma": crystal.gamma,
        "volume": crystal.volume,
        "space_group_name": crystal.space_group_name,
        "space_group_number": crystal.space_group_number,
        "n_symops": len(crystal.symops),
        "n_atoms_asu": len(symbols),
    }


def generate_spherical_cluster(
    filepath: str | Path,
    radius: float,
    center: Optional[NDArray[np.float64]] = None,
    remove_duplicates: bool = True,
    duplicate_tolerance: float = 0.01,
) -> Tuple[qcel.models.Molecule, Crystal, List[Tuple[int, int, int]]]:
    """
    Generate an approximately spherical cluster of molecules from a CIF file.

    This function creates a supercell that extends far enough in all directions
    to include all unit cells whose centers fall within the specified radius
    from a central reference point. The result is an approximately spherical
    cluster of molecules.

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.
    radius : float
        Radius in Angstroms from the center point. Unit cells with any atom
        within this radius from the center will be included.
    center : NDArray[np.float64], optional
        Center point in fractional coordinates. Defaults to (0.5, 0.5, 0.5),
        the center of the reference unit cell.
    remove_duplicates : bool, default=True
        If True, remove duplicate atoms at unit cell boundaries.
    duplicate_tolerance : float, default=0.01
        Distance tolerance (in Angstroms) for detecting duplicates.

    Returns
    -------
    Tuple[qcel.models.Molecule, Crystal, List[Tuple[int, int, int]]]
        - QCElemental Molecule object with all atoms in the cluster
        - Crystal object with unit cell information
        - List of (i, j, k) unit cell indices included in the cluster

    Examples
    --------
    >>> mol, crystal, cells = generate_spherical_cluster("structure.cif", 15.0)
    >>> print(f"Cluster contains {len(mol.symbols)} atoms")
    >>> print(f"From {len(cells)} unit cells")
    """
    crystal, asu_symbols, asu_frac_coords = parse_cif_file(filepath)

    if center is None:
        center = np.array([0.5, 0.5, 0.5])
    else:
        center = np.asarray(center)

    # First, generate the full unit cell (symmetry-expanded)
    unit_cell_symbols = []
    unit_cell_frac_coords = []

    for symbol, frac_coord in zip(asu_symbols, asu_frac_coords):
        equiv_coords = crystal.generate_symmetry_equivalents(
            frac_coord, wrap_to_unit_cell=True
        )
        for coord in equiv_coords:
            unit_cell_symbols.append(symbol)
            unit_cell_frac_coords.append(coord)

    unit_cell_frac_coords = np.array(unit_cell_frac_coords)

    # Remove duplicates within the unit cell
    if remove_duplicates:
        unit_cell_symbols, unit_cell_frac_coords = remove_duplicate_atoms(
            unit_cell_symbols, unit_cell_frac_coords, tolerance=0.01
        )

    # Calculate how many unit cells we need in each direction
    # Use the lattice vectors to estimate the range
    center_cart = crystal.to_cartesian(center)

    # Calculate the maximum number of cells needed in each direction
    # by considering the lattice parameters
    n_a = int(np.ceil(radius / crystal.a)) + 1
    n_b = int(np.ceil(radius / crystal.b)) + 1
    n_c = int(np.ceil(radius / crystal.c)) + 1

    # Collect all atoms within the spherical region
    all_symbols = []
    all_cart_coords = []
    included_cells = []

    # Iterate over all potentially relevant unit cells
    for i in range(-n_a, n_a + 1):
        for j in range(-n_b, n_b + 1):
            for k in range(-n_c, n_c + 1):
                # Translation vector for this unit cell
                translation = np.array([float(i), float(j), float(k)])

                # Check if any atom from this cell is within radius
                cell_atoms_in_range = False
                cell_symbols = []
                cell_coords = []

                for sym, frac in zip(unit_cell_symbols, unit_cell_frac_coords):
                    # Translate fractional coordinates
                    translated_frac = frac + translation

                    # Convert to Cartesian
                    cart = crystal.to_cartesian(translated_frac)

                    # Check distance from center
                    dist = np.linalg.norm(cart - center_cart)

                    if dist <= radius:
                        cell_atoms_in_range = True
                        cell_symbols.append(sym)
                        cell_coords.append(cart)

                # If any atom is in range, include all atoms from this cell
                # that are within the radius
                if cell_atoms_in_range:
                    all_symbols.extend(cell_symbols)
                    all_cart_coords.extend(cell_coords)
                    if (i, j, k) not in included_cells:
                        included_cells.append((i, j, k))

    all_cart_coords = np.array(all_cart_coords)

    # Remove duplicate atoms at boundaries (in Cartesian space)
    if remove_duplicates and len(all_symbols) > 0:
        all_symbols, all_cart_coords = _remove_duplicate_atoms_cartesian(
            all_symbols, all_cart_coords, tolerance=duplicate_tolerance
        )

    # Convert Angstroms to Bohr for QCElemental
    cart_coords_bohr = all_cart_coords / qcel.constants.bohr2angstroms

    # Create QCElemental Molecule
    mol = qcel.models.Molecule(
        symbols=all_symbols,
        geometry=cart_coords_bohr.flatten(),
    )

    return mol, crystal, included_cells


def _remove_duplicate_atoms_cartesian(
    symbols: List[str],
    coords: NDArray[np.float64],
    tolerance: float = 0.01,
) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Remove duplicate atoms based on Cartesian coordinate proximity.

    Parameters
    ----------
    symbols : List[str]
        List of atom symbols.
    coords : NDArray[np.float64]
        Cartesian coordinates array of shape (N, 3).
    tolerance : float
        Distance tolerance in Angstroms for considering atoms as duplicates.

    Returns
    -------
    Tuple[List[str], NDArray[np.float64]]
        Unique atom symbols and coordinates.
    """
    if len(symbols) == 0:
        return symbols, coords

    unique_symbols = [symbols[0]]
    unique_coords = [coords[0]]

    for i in range(1, len(symbols)):
        is_duplicate = False
        for j in range(len(unique_coords)):
            dist = np.linalg.norm(coords[i] - unique_coords[j])
            if dist < tolerance:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_symbols.append(symbols[i])
            unique_coords.append(coords[i])

    return unique_symbols, np.array(unique_coords)


def generate_supercell(
    filepath: str | Path,
    na: int = 1,
    nb: int = 1,
    nc: int = 1,
    remove_duplicates: bool = True,
    duplicate_tolerance: float = 0.01,
) -> qcel.models.Molecule:
    """
    Generate a supercell by replicating the unit cell.

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.
    na, nb, nc : int
        Number of unit cell replications along a, b, c axes.
        Use negative values to extend in both directions (e.g., na=-2
        creates cells from -2 to +2 along a).
    remove_duplicates : bool, default=True
        If True, remove duplicate atoms at unit cell boundaries.
    duplicate_tolerance : float, default=0.01
        Distance tolerance (in Angstroms) for detecting duplicates.

    Returns
    -------
    qcel.models.Molecule
        QCElemental Molecule object with the supercell.

    Examples
    --------
    >>> mol = generate_supercell("structure.cif", na=2, nb=2, nc=2)
    >>> # Creates a 2x2x2 supercell (8 unit cells)
    """
    crystal, asu_symbols, asu_frac_coords = parse_cif_file(filepath)

    # Generate full unit cell
    unit_cell_symbols = []
    unit_cell_frac_coords = []

    for symbol, frac_coord in zip(asu_symbols, asu_frac_coords):
        equiv_coords = crystal.generate_symmetry_equivalents(
            frac_coord, wrap_to_unit_cell=True
        )
        for coord in equiv_coords:
            unit_cell_symbols.append(symbol)
            unit_cell_frac_coords.append(coord)

    unit_cell_frac_coords = np.array(unit_cell_frac_coords)

    if remove_duplicates:
        unit_cell_symbols, unit_cell_frac_coords = remove_duplicate_atoms(
            unit_cell_symbols, unit_cell_frac_coords, tolerance=0.01
        )

    # Determine range for each axis
    def get_range(n):
        if n < 0:
            return range(n, -n + 1)
        return range(n)

    range_a = get_range(na)
    range_b = get_range(nb)
    range_c = get_range(nc)

    # Generate supercell
    all_symbols = []
    all_cart_coords = []

    for i in range_a:
        for j in range_b:
            for k in range_c:
                translation = np.array([float(i), float(j), float(k)])

                for sym, frac in zip(unit_cell_symbols, unit_cell_frac_coords):
                    translated_frac = frac + translation
                    cart = crystal.to_cartesian(translated_frac)
                    all_symbols.append(sym)
                    all_cart_coords.append(cart)

    all_cart_coords = np.array(all_cart_coords)

    if remove_duplicates and len(all_symbols) > 0:
        all_symbols, all_cart_coords = _remove_duplicate_atoms_cartesian(
            all_symbols, all_cart_coords, tolerance=duplicate_tolerance
        )

    # Convert to Bohr
    cart_coords_bohr = all_cart_coords / qcel.constants.bohr2angstroms

    mol = qcel.models.Molecule(
        symbols=all_symbols,
        geometry=cart_coords_bohr.flatten(),
    )

    return mol


@dataclass
class Monomer:
    """
    Represents a molecular monomer in the crystal.

    Attributes
    ----------
    symbols : List[str]
        Atom symbols in the monomer.
    frac_coords : NDArray[np.float64]
        Fractional coordinates of atoms, shape (N, 3).
    cart_coords : NDArray[np.float64]
        Cartesian coordinates of atoms, shape (N, 3).
    cell_index : Tuple[int, int, int]
        Unit cell index (i, j, k) where this monomer is located.
    symop_index : int
        Index of the symmetry operation that generated this monomer.
    centroid_frac : NDArray[np.float64]
        Centroid in fractional coordinates.
    centroid_cart : NDArray[np.float64]
        Centroid in Cartesian coordinates (Angstroms).
    """

    symbols: List[str]
    frac_coords: NDArray[np.float64]
    cart_coords: NDArray[np.float64]
    cell_index: Tuple[int, int, int]
    symop_index: int
    centroid_frac: NDArray[np.float64]
    centroid_cart: NDArray[np.float64]

    def to_molecule(self) -> qcel.models.Molecule:
        """Convert to QCElemental Molecule object."""
        cart_bohr = self.cart_coords / qcel.constants.bohr2angstroms
        return qcel.models.Molecule(
            symbols=self.symbols,
            geometry=cart_bohr.flatten(),
        )


@dataclass
class DimerPair:
    """
    Represents a unique dimer pairing between two monomers.

    Attributes
    ----------
    monomer_a : Monomer
        The reference monomer (always from the central unit cell).
    monomer_b : Monomer
        The partner monomer.
    distance : float
        Center-to-center distance in Angstroms.
    symop_index_a : int
        Symmetry operation index for monomer A.
    symop_index_b : int
        Symmetry operation index for monomer B.
    cell_index_b : Tuple[int, int, int]
        Unit cell index of monomer B relative to A.
    multiplicity : int
        Number of symmetry-equivalent copies of this dimer type.
    """

    monomer_a: Monomer
    monomer_b: Monomer
    distance: float
    symop_index_a: int
    symop_index_b: int
    cell_index_b: Tuple[int, int, int]
    multiplicity: int = 1

    def to_molecule(self) -> qcel.models.Molecule:
        """Convert dimer to QCElemental Molecule object."""
        symbols = list(self.monomer_a.symbols) + list(self.monomer_b.symbols)
        coords_a = self.monomer_a.cart_coords
        coords_b = self.monomer_b.cart_coords
        coords = np.vstack([coords_a, coords_b])
        cart_bohr = coords / qcel.constants.bohr2angstroms
        return qcel.models.Molecule(
            symbols=symbols,
            geometry=cart_bohr.flatten(),
        )


@dataclass
class MolecularCluster:
    """
    Represents a complete molecule identified by connectivity analysis.

    Attributes
    ----------
    symbols : List[str]
        Atom symbols in the molecule.
    frac_coords : NDArray[np.float64]
        Fractional coordinates of atoms, shape (N, 3).
    cart_coords : NDArray[np.float64]
        Cartesian coordinates of atoms, shape (N, 3).
    centroid_frac : NDArray[np.float64]
        Centroid in fractional coordinates.
    centroid_cart : NDArray[np.float64]
        Centroid in Cartesian coordinates (Angstroms).
    molecule_index : int
        Index identifying this molecule in the unit cell.
    cell_index : Tuple[int, int, int]
        Unit cell translation from reference cell.
    """

    symbols: List[str]
    frac_coords: NDArray[np.float64]
    cart_coords: NDArray[np.float64]
    centroid_frac: NDArray[np.float64]
    centroid_cart: NDArray[np.float64]
    molecule_index: int
    cell_index: Tuple[int, int, int] = (0, 0, 0)

    def to_molecule(self) -> qcel.models.Molecule:
        """Convert to QCElemental Molecule object."""
        cart_bohr = self.cart_coords / qcel.constants.bohr2angstroms
        return qcel.models.Molecule(
            symbols=self.symbols,
            geometry=cart_bohr.flatten(),
        )


@dataclass
class MolecularDimer:
    """
    Represents a dimer of two complete molecules.

    Attributes
    ----------
    molecule_a : MolecularCluster
        The reference molecule.
    molecule_b : MolecularCluster
        The partner molecule.
    distance : float
        Center-to-center distance in Angstroms.
    multiplicity : int
        Number of symmetry-equivalent copies of this dimer type.
    """

    molecule_a: MolecularCluster
    molecule_b: MolecularCluster
    distance: float
    multiplicity: int = 1

    def to_molecule(self) -> qcel.models.Molecule:
        """Convert dimer to QCElemental Molecule object."""
        symbols = list(self.molecule_a.symbols) + list(self.molecule_b.symbols)
        coords = np.vstack([self.molecule_a.cart_coords, self.molecule_b.cart_coords])
        cart_bohr = coords / qcel.constants.bohr2angstroms
        return qcel.models.Molecule(
            symbols=symbols,
            geometry=cart_bohr.flatten(),
        )

    def to_molecules(self) -> Tuple[qcel.models.Molecule, qcel.models.Molecule]:
        """Return the two monomers as separate QCElemental Molecules."""
        return self.molecule_a.to_molecule(), self.molecule_b.to_molecule()


def _get_covalent_radius(symbol: str) -> float:
    """
    Get covalent radius for an element in Angstroms.

    Parameters
    ----------
    symbol : str
        Element symbol.

    Returns
    -------
    float
        Covalent radius in Angstroms.
    """
    # qcelemental returns covalent radii in Bohr
    radius_bohr = qcel.covalentradii.get(symbol)
    return radius_bohr * qcel.constants.bohr2angstroms


def _build_bond_graph(
    symbols: List[str],
    cart_coords: NDArray[np.float64],
    bond_tolerance: float = 0.4,
) -> List[List[int]]:
    """
    Build adjacency list for atoms based on covalent bonding.

    Two atoms are considered bonded if their distance is less than
    the sum of their covalent radii plus a tolerance.

    Parameters
    ----------
    symbols : List[str]
        Atom symbols.
    cart_coords : NDArray[np.float64]
        Cartesian coordinates, shape (N, 3).
    bond_tolerance : float, default=0.4
        Tolerance in Angstroms added to sum of covalent radii.

    Returns
    -------
    List[List[int]]
        Adjacency list where adj[i] contains indices of atoms bonded to atom i.
    """
    n_atoms = len(symbols)
    adj = [[] for _ in range(n_atoms)]

    # Get covalent radii
    radii = [_get_covalent_radius(s) for s in symbols]

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(cart_coords[i] - cart_coords[j])
            max_bond = radii[i] + radii[j] + bond_tolerance
            if dist < max_bond:
                adj[i].append(j)
                adj[j].append(i)

    return adj


def _find_connected_components(adj: List[List[int]]) -> List[List[int]]:
    """
    Find connected components in a graph using BFS.

    Parameters
    ----------
    adj : List[List[int]]
        Adjacency list representation of the graph.

    Returns
    -------
    List[List[int]]
        List of components, where each component is a list of atom indices.
    """
    n = len(adj)
    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue

        # BFS from this node
        component = []
        queue = [start]
        visited[start] = True

        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        components.append(sorted(component))

    return components


def _identify_molecules_in_cell(
    crystal: Crystal,
    asu_symbols: List[str],
    asu_frac_coords: List[NDArray[np.float64]],
    bond_tolerance: float = 0.4,
) -> List[MolecularCluster]:
    """
    Identify complete molecules in the unit cell using connectivity.

    This function:
    1. Generates all symmetry-equivalent atoms in a 3x3x3 supercell
    2. Builds a bond graph based on covalent radii
    3. Finds connected components (molecules)
    4. Returns molecules whose centroids are in the central unit cell

    Parameters
    ----------
    crystal : Crystal
        Crystal object with symmetry operations.
    asu_symbols : List[str]
        Atom symbols in the asymmetric unit.
    asu_frac_coords : List[NDArray[np.float64]]
        Fractional coordinates of ASU atoms.
    bond_tolerance : float, default=0.4
        Tolerance for bond detection in Angstroms.

    Returns
    -------
    List[MolecularCluster]
        Complete molecules with centroids in the central unit cell.
    """
    # Generate atoms in a 3x3x3 supercell to capture molecules
    # that cross unit cell boundaries
    all_symbols = []
    all_frac = []
    all_cart = []

    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                translation = np.array([float(di), float(dj), float(dk)])
                for symop in crystal.symops:
                    for sym, frac in zip(asu_symbols, asu_frac_coords):
                        new_frac = symop.apply(frac) + translation
                        new_cart = crystal.to_cartesian(new_frac)
                        all_symbols.append(sym)
                        all_frac.append(new_frac)
                        all_cart.append(new_cart)

    all_frac = np.array(all_frac)
    all_cart = np.array(all_cart)

    # Remove duplicate atoms (same position)
    unique_indices = []
    for i in range(len(all_symbols)):
        is_dup = False
        for j in unique_indices:
            if np.linalg.norm(all_cart[i] - all_cart[j]) < 0.01:
                is_dup = True
                break
        if not is_dup:
            unique_indices.append(i)

    symbols = [all_symbols[i] for i in unique_indices]
    frac_coords = all_frac[unique_indices]
    cart_coords = all_cart[unique_indices]

    # Build bond graph and find molecules
    adj = _build_bond_graph(symbols, cart_coords, bond_tolerance)
    components = _find_connected_components(adj)

    # Create MolecularCluster objects for molecules in the central cell
    molecules = []
    mol_idx = 0

    for component in components:
        mol_symbols = [symbols[i] for i in component]
        mol_frac = frac_coords[component]
        mol_cart = cart_coords[component]
        centroid_frac = np.mean(mol_frac, axis=0)
        centroid_cart = np.mean(mol_cart, axis=0)

        # Check if centroid is in the central unit cell [0, 1)
        if (
            0 <= centroid_frac[0] < 1
            and 0 <= centroid_frac[1] < 1
            and 0 <= centroid_frac[2] < 1
        ):
            mol = MolecularCluster(
                symbols=mol_symbols,
                frac_coords=mol_frac,
                cart_coords=mol_cart,
                centroid_frac=centroid_frac,
                centroid_cart=centroid_cart,
                molecule_index=mol_idx,
                cell_index=(0, 0, 0),
            )
            molecules.append(mol)
            mol_idx += 1

    return molecules


def generate_molecular_dimers(
    filepath: str | Path,
    radius: float,
    bond_tolerance: float = 0.4,
    distance_tolerance: float = 0.01,
) -> Tuple[List[MolecularDimer], Crystal, List[MolecularCluster]]:
    """
    Generate symmetry-unique molecular dimers within a spherical radius.

    This function identifies complete molecules using connectivity analysis
    (based on covalent radii), then generates all unique dimer pairings.
    This is the recommended function for obtaining valid QCElemental
    dimer molecules.

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.
    radius : float
        Maximum center-to-center distance in Angstroms for dimer pairs.
    bond_tolerance : float, default=0.4
        Tolerance in Angstroms for bond detection (added to sum of
        covalent radii).
    distance_tolerance : float, default=0.01
        Tolerance in Angstroms for considering distances as equivalent
        when identifying unique dimers.

    Returns
    -------
    Tuple[List[MolecularDimer], Crystal, List[MolecularCluster]]
        - List of unique MolecularDimer objects, sorted by distance
        - Crystal object with unit cell information
        - List of reference molecules in the central unit cell

    Examples
    --------
    >>> dimers, crystal, mols = generate_molecular_dimers("co2.cif", 10.0)
    >>> for d in dimers:
    ...     print(f"Distance: {d.distance:.2f} A, mult: {d.multiplicity}")
    ...     qcel_mol = d.to_molecule()  # Valid QCElemental dimer molecule

    Notes
    -----
    Unlike generate_unique_dimers() which uses ASU fragments, this function
    reconstructs complete molecules by analyzing covalent connectivity.
    This ensures that each monomer in the dimer is a chemically valid
    molecule.
    """
    crystal, asu_symbols, asu_frac_coords = parse_cif_file(filepath)

    # Identify complete molecules in the central unit cell
    central_molecules = _identify_molecules_in_cell(
        crystal, asu_symbols, asu_frac_coords, bond_tolerance
    )

    if not central_molecules:
        return [], crystal, []

    # Use the first molecule as the reference
    ref_molecule = central_molecules[0]
    ref_centroid = ref_molecule.centroid_cart

    # Calculate range of unit cells to search
    n_a = int(np.ceil(radius / crystal.a)) + 1
    n_b = int(np.ceil(radius / crystal.b)) + 1
    n_c = int(np.ceil(radius / crystal.c)) + 1

    # Generate candidate dimers by translating molecules to neighboring cells
    candidate_dimers = []

    for i in range(-n_a, n_a + 1):
        for j in range(-n_b, n_b + 1):
            for k in range(-n_c, n_c + 1):
                cell_idx = (i, j, k)
                translation = np.array([float(i), float(j), float(k)])

                for mol in central_molecules:
                    # Skip self-pairing (same molecule in same cell)
                    is_ref = (
                        cell_idx == (0, 0, 0)
                        and mol.molecule_index == ref_molecule.molecule_index
                    )
                    if is_ref:
                        continue

                    # Translate molecule
                    trans_frac = mol.frac_coords + translation
                    trans_cart = crystal.to_cartesian(trans_frac)
                    trans_centroid = np.mean(trans_cart, axis=0)

                    # Check distance
                    dist = np.linalg.norm(trans_centroid - ref_centroid)
                    if dist > radius:
                        continue

                    partner = MolecularCluster(
                        symbols=mol.symbols,
                        frac_coords=trans_frac,
                        cart_coords=trans_cart,
                        centroid_frac=np.mean(trans_frac, axis=0),
                        centroid_cart=trans_centroid,
                        molecule_index=mol.molecule_index,
                        cell_index=cell_idx,
                    )

                    dimer = MolecularDimer(
                        molecule_a=ref_molecule,
                        molecule_b=partner,
                        distance=dist,
                        multiplicity=1,
                    )
                    candidate_dimers.append(dimer)

    # Identify unique dimers based on distance
    unique_dimers = _identify_unique_molecular_dimers(
        candidate_dimers, crystal, distance_tolerance
    )

    # Sort by distance
    unique_dimers.sort(key=lambda d: d.distance)

    return unique_dimers, crystal, central_molecules


def _identify_unique_molecular_dimers(
    dimers: List[MolecularDimer],
    crystal: Crystal,
    tolerance: float = 0.01,
) -> List[MolecularDimer]:
    """
    Identify symmetry-unique molecular dimers from a list of candidates.

    Parameters
    ----------
    dimers : List[MolecularDimer]
        Candidate dimers.
    crystal : Crystal
        Crystal object for symmetry operations.
    tolerance : float, default=0.01
        Distance tolerance in Angstroms.

    Returns
    -------
    List[MolecularDimer]
        Unique dimers with multiplicity set.
    """
    if not dimers:
        return []

    dimers_sorted = sorted(dimers, key=lambda d: d.distance)

    unique = []
    used = [False] * len(dimers_sorted)

    for i, dimer_i in enumerate(dimers_sorted):
        if used[i]:
            continue

        multiplicity = 1
        used[i] = True

        # Find equivalent dimers
        for j in range(i + 1, len(dimers_sorted)):
            if used[j]:
                continue

            dimer_j = dimers_sorted[j]

            # Check if distances match
            if abs(dimer_i.distance - dimer_j.distance) > tolerance:
                if dimer_j.distance > dimer_i.distance + tolerance:
                    break
                continue

            # Check if geometry is equivalent
            if _are_molecular_dimers_equivalent(dimer_i, dimer_j, crystal, tolerance):
                multiplicity += 1
                used[j] = True

        unique_dimer = MolecularDimer(
            molecule_a=dimer_i.molecule_a,
            molecule_b=dimer_i.molecule_b,
            distance=dimer_i.distance,
            multiplicity=multiplicity,
        )
        unique.append(unique_dimer)

    return unique


def _are_molecular_dimers_equivalent(
    dimer1: MolecularDimer,
    dimer2: MolecularDimer,
    crystal: Crystal,
    tolerance: float = 0.01,
) -> bool:
    """
    Check if two molecular dimers are symmetry-equivalent.

    Parameters
    ----------
    dimer1, dimer2 : MolecularDimer
        Dimers to compare.
    crystal : Crystal
        Crystal object with symmetry operations.
    tolerance : float
        Distance tolerance in Angstroms.

    Returns
    -------
    bool
        True if dimers are equivalent.
    """
    if abs(dimer1.distance - dimer2.distance) > tolerance:
        return False

    # Compare relative vectors between centroids
    vec1 = dimer1.molecule_b.centroid_cart - dimer1.molecule_a.centroid_cart
    vec2 = dimer2.molecule_b.centroid_cart - dimer2.molecule_a.centroid_cart

    # Check if vectors are related by a point group operation
    for symop in crystal.symops:
        rot_vec = crystal.to_cartesian(symop.rot @ crystal.to_fractional(vec1))
        if np.allclose(rot_vec, vec2, atol=tolerance):
            return True
        if np.allclose(rot_vec, -vec2, atol=tolerance):
            return True

    return False


def get_dimer_molecules(
    filepath: str | Path,
    radius: float,
    bond_tolerance: float = 0.4,
    distance_tolerance: float = 0.01,
) -> List[qcel.models.Molecule]:
    """
    Convenience function to get a list of QCElemental dimer molecules.

    This is the simplest interface for obtaining valid molecular dimers
    from a CIF file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.
    radius : float
        Maximum center-to-center distance in Angstroms for dimer pairs.
    bond_tolerance : float, default=0.4
        Tolerance for bond detection.
    distance_tolerance : float, default=0.01
        Tolerance for distance equivalence.

    Returns
    -------
    List[qcel.models.Molecule]
        List of QCElemental Molecule objects, one for each unique dimer,
        sorted by center-to-center distance.

    Examples
    --------
    >>> dimers = get_dimer_molecules("crystal.cif", radius=10.0)
    >>> for mol in dimers:
    ...     print(f"Atoms: {len(mol.symbols)}, Formula: {mol.get_molecular_formula()}")
    """
    dimers, _, _ = generate_molecular_dimers(
        filepath, radius, bond_tolerance, distance_tolerance
    )
    return [d.to_molecule() for d in dimers]


def generate_unique_dimers(
    filepath: str | Path,
    radius: float,
    reference_symop: int = 0,
    distance_tolerance: float = 0.01,
) -> Tuple[List[DimerPair], Crystal, Monomer]:
    """
    Generate symmetry-unique dimer pairings within a spherical radius.

    This function identifies all unique molecular dimers by considering
    space group symmetry. For each unique dimer type, only one
    representative is returned along with its multiplicity.

    The reference monomer is always the molecule generated by the
    specified symmetry operation in the central unit cell (0, 0, 0).

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.
    radius : float
        Maximum center-to-center distance in Angstroms for dimer pairs.
    reference_symop : int, default=0
        Index of the symmetry operation to use for the reference monomer.
        Default is 0 (identity operation).
    distance_tolerance : float, default=0.01
        Tolerance in Angstroms for considering distances as equivalent.

    Returns
    -------
    Tuple[List[DimerPair], Crystal, Monomer]
        - List of unique DimerPair objects, sorted by distance
        - Crystal object with unit cell information
        - The reference Monomer object

    Examples
    --------
    >>> dimers, crystal, ref_mol = generate_unique_dimers("co2.cif", 10.0)
    >>> for d in dimers:
    ...     print(f"Distance: {d.distance:.2f} A, mult: {d.multiplicity}")

    Notes
    -----
    Two dimers are considered symmetry-equivalent if they have:
    1. The same center-to-center distance (within tolerance)
    2. The same relative symmetry operation relationship

    The multiplicity counts how many times each unique dimer type
    appears when considering all symmetry operations applied to the
    reference monomer.
    """
    crystal, asu_symbols, asu_frac_coords = parse_cif_file(filepath)

    # Generate all monomers in the central unit cell
    # Each monomer corresponds to one symmetry operation applied to the ASU
    central_monomers = []
    for symop_idx, symop in enumerate(crystal.symops):
        mon_symbols = []
        mon_frac = []
        for sym, frac in zip(asu_symbols, asu_frac_coords):
            new_frac = symop.apply(frac)
            new_frac = new_frac % 1.0  # Wrap to unit cell
            mon_symbols.append(sym)
            mon_frac.append(new_frac)

        mon_frac = np.array(mon_frac)
        mon_cart = crystal.to_cartesian(mon_frac)
        centroid_frac = np.mean(mon_frac, axis=0)
        centroid_cart = np.mean(mon_cart, axis=0)

        monomer = Monomer(
            symbols=mon_symbols,
            frac_coords=mon_frac,
            cart_coords=mon_cart,
            cell_index=(0, 0, 0),
            symop_index=symop_idx,
            centroid_frac=centroid_frac,
            centroid_cart=centroid_cart,
        )
        central_monomers.append(monomer)

    # Remove duplicate monomers (from special positions)
    unique_central = _remove_duplicate_monomers(central_monomers, tolerance=0.01)

    # The reference monomer
    ref_monomer = None
    for mon in unique_central:
        if mon.symop_index == reference_symop:
            ref_monomer = mon
            break
    if ref_monomer is None:
        ref_monomer = unique_central[0]

    ref_centroid = ref_monomer.centroid_cart

    # Calculate range of unit cells to search
    n_a = int(np.ceil(radius / crystal.a)) + 1
    n_b = int(np.ceil(radius / crystal.b)) + 1
    n_c = int(np.ceil(radius / crystal.c)) + 1

    # Generate all candidate dimers
    candidate_dimers = []

    for i in range(-n_a, n_a + 1):
        for j in range(-n_b, n_b + 1):
            for k in range(-n_c, n_c + 1):
                cell_idx = (i, j, k)
                translation = np.array([float(i), float(j), float(k)])

                for base_mon in unique_central:
                    # Skip self-pairing in central cell
                    if cell_idx == (0, 0, 0):
                        if base_mon.symop_index == ref_monomer.symop_index:
                            continue

                    # Translate monomer to this cell
                    trans_frac = base_mon.frac_coords + translation
                    trans_cart = crystal.to_cartesian(trans_frac)
                    trans_centroid = np.mean(trans_cart, axis=0)

                    # Check distance
                    dist = np.linalg.norm(trans_centroid - ref_centroid)
                    if dist > radius:
                        continue

                    partner = Monomer(
                        symbols=base_mon.symbols,
                        frac_coords=trans_frac,
                        cart_coords=trans_cart,
                        cell_index=cell_idx,
                        symop_index=base_mon.symop_index,
                        centroid_frac=np.mean(trans_frac, axis=0),
                        centroid_cart=trans_centroid,
                    )

                    dimer = DimerPair(
                        monomer_a=ref_monomer,
                        monomer_b=partner,
                        distance=dist,
                        symop_index_a=ref_monomer.symop_index,
                        symop_index_b=base_mon.symop_index,
                        cell_index_b=cell_idx,
                        multiplicity=1,
                    )
                    candidate_dimers.append(dimer)

    # Identify unique dimers based on distance and symmetry relationship
    unique_dimers = _identify_unique_dimers(
        candidate_dimers,
        crystal,
        distance_tolerance,
    )

    # Sort by distance
    unique_dimers.sort(key=lambda d: d.distance)

    return unique_dimers, crystal, ref_monomer


def _remove_duplicate_monomers(
    monomers: List[Monomer],
    tolerance: float = 0.01,
) -> List[Monomer]:
    """Remove duplicate monomers based on centroid proximity."""
    if not monomers:
        return monomers

    unique = [monomers[0]]
    for mon in monomers[1:]:
        is_dup = False
        for u in unique:
            dist = np.linalg.norm(mon.centroid_cart - u.centroid_cart)
            if dist < tolerance:
                is_dup = True
                break
        if not is_dup:
            unique.append(mon)
    return unique


def _identify_unique_dimers(
    dimers: List[DimerPair],
    crystal: Crystal,
    tolerance: float = 0.01,
) -> List[DimerPair]:
    """
    Identify symmetry-unique dimers from a list of candidates.

    Two dimers are equivalent if:
    1. They have the same distance (within tolerance)
    2. They represent the same symmetry relationship

    Returns unique dimers with multiplicity set.
    """
    if not dimers:
        return []

    # Group by distance first
    dimers_sorted = sorted(dimers, key=lambda d: d.distance)

    unique = []
    used = [False] * len(dimers_sorted)

    for i, dimer_i in enumerate(dimers_sorted):
        if used[i]:
            continue

        # This is a new unique dimer
        multiplicity = 1
        used[i] = True

        # Find equivalent dimers
        for j in range(i + 1, len(dimers_sorted)):
            if used[j]:
                continue

            dimer_j = dimers_sorted[j]

            # Check if distances match
            if abs(dimer_i.distance - dimer_j.distance) > tolerance:
                # Since sorted, no more matches possible at this distance
                if dimer_j.distance > dimer_i.distance + tolerance:
                    break
                continue

            # Check if symmetry relationship is equivalent
            if _are_dimers_equivalent(dimer_i, dimer_j, crystal, tolerance):
                multiplicity += 1
                used[j] = True

        # Create unique dimer with multiplicity
        unique_dimer = DimerPair(
            monomer_a=dimer_i.monomer_a,
            monomer_b=dimer_i.monomer_b,
            distance=dimer_i.distance,
            symop_index_a=dimer_i.symop_index_a,
            symop_index_b=dimer_i.symop_index_b,
            cell_index_b=dimer_i.cell_index_b,
            multiplicity=multiplicity,
        )
        unique.append(unique_dimer)

    return unique


def _are_dimers_equivalent(
    dimer1: DimerPair,
    dimer2: DimerPair,
    crystal: Crystal,
    tolerance: float = 0.01,
) -> bool:
    """
    Check if two dimers are symmetry-equivalent.

    Two dimers are equivalent if they have the same relative geometry,
    meaning one can be transformed into the other by a space group
    symmetry operation.
    """
    # Same distance is a prerequisite (already checked by caller)
    if abs(dimer1.distance - dimer2.distance) > tolerance:
        return False

    # Check if the symop relationships are the same
    # This is a simplification: same symop_b index and same relative
    # cell translation pattern indicates equivalence
    if dimer1.symop_index_b == dimer2.symop_index_b:
        # Check if cell translations differ only by a lattice vector
        # that would make them equivalent under translation
        cell1 = np.array(dimer1.cell_index_b)
        cell2 = np.array(dimer2.cell_index_b)

        # They're equivalent if the cell index difference is the same
        # (accounting for inversion symmetry)
        if np.allclose(cell1, cell2):
            return True
        if np.allclose(cell1, -cell2):
            return True

    # More sophisticated check: compare actual geometry
    # Get relative vectors between centroids
    vec1 = dimer1.monomer_b.centroid_cart - dimer1.monomer_a.centroid_cart
    vec2 = dimer2.monomer_b.centroid_cart - dimer2.monomer_a.centroid_cart

    # Check if vectors are related by a point group operation
    # (rotation or reflection that preserves the lattice)
    for symop in crystal.symops:
        # Apply rotation part only (not translation)
        rot_vec = crystal.to_cartesian(symop.rot @ crystal.to_fractional(vec1))
        if np.allclose(rot_vec, vec2, atol=tolerance):
            return True
        if np.allclose(rot_vec, -vec2, atol=tolerance):
            return True

    return False


if __name__ == "__main__":
    # Example usage / test
    import sys

    if len(sys.argv) > 1:
        cif_path = sys.argv[1]
    else:
        # Default test file
        cif_path = (
            "/home/awallace43/projects/x23_dmetcalf_2022_si/cifs/carbon_dioxide.cif"
        )

    # Parse optional radius argument
    radius = None
    if len(sys.argv) > 2:
        radius = float(sys.argv[2])

    print(f"Processing CIF file: {cif_path}")
    print("-" * 60)

    # Get crystal info
    info = get_crystal_info(cif_path)
    print("Crystal Information:")
    print(
        f"  Lattice parameters: a={info['a']:.3f}, b={info['b']:.3f}, c={info['c']:.3f}"
    )
    print(
        f"  Lattice angles: alpha={info['alpha']:.1f}, "
        f"beta={info['beta']:.1f}, gamma={info['gamma']:.1f}"
    )
    print(f"  Volume: {info['volume']:.3f} A^3")
    sg_name = info["space_group_name"]
    sg_num = info["space_group_number"]
    print(f"  Space group: {sg_name} (#{sg_num})")
    print(f"  Symmetry operations: {info['n_symops']}")
    print(f"  Atoms in ASU: {info['n_atoms_asu']}")
    print("-" * 60)

    if radius is not None:
        # Generate spherical cluster
        print(f"\nGenerating spherical cluster with radius {radius} A...")
        mol, crystal, cells = generate_spherical_cluster(cif_path, radius)
        print(f"  Number of unit cells: {len(cells)}")
        print(f"  Number of atoms: {len(mol.symbols)}")
        print(f"  Molecular formula: {mol.get_molecular_formula()}")

        # Calculate actual extent
        coords = mol.geometry.reshape(-1, 3) * qcel.constants.bohr2angstroms
        center = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        print(f"  Max distance from center: {np.max(distances):.2f} A")
        print(f"  Unit cells included: {cells[:10]}...")
    else:
        # Create single unit cell molecule
        mol = cif_to_molecule(cif_path)
        print("\nQCElemental Molecule (single unit cell):")
        print(f"  Number of atoms: {len(mol.symbols)}")
        print(f"  Symbols: {list(mol.symbols)}")
        print(f"  Molecular formula: {mol.get_molecular_formula()}")

        # Print coordinates
        coords = mol.geometry.reshape(-1, 3) * qcel.constants.bohr2angstroms
        print("\n  Cartesian coordinates (Angstroms):")
        for sym, coord in zip(mol.symbols, coords):
            print(f"    {sym:2s}  {coord[0]:10.6f}  {coord[1]:10.6f}  {coord[2]:10.6f}")
