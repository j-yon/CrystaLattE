# from __future__ import annotations

# import re
# from dataclasses import dataclass, field
# from pathlib import Path

# import numpy as np
# import qcelemental as qcel
# from numpy.typing import NDArray

# from CrystaLattE.crystal import Crystal
# from CrystaLattE.sym_ops import SymOp, parse_symop_xyz


# def parse_cif_file(
#     filepath: str | Path,
# ) -> tuple[Crystal, list[str], NDArray[np.float64]]:
#     """
#     Parse a CIF file to extract crystal and atomic information.

#     Parameters
#     ----------
#     filepath : str or Path
#         Path to the CIF file.

#     Returns
#     -------
#     Tuple[Crystal, List[str], NDArray[np.float64]]
#         - Crystal object with unit cell and symmetry information
#         - List of atom labels/symbols
#         - Fractional coordinates array of shape (N, 3)

#     Examples
#     --------
#     >>> crystal, symbols, frac_coords = parse_cif_file("structure.cif")
#     """
#     filepath = Path(filepath)
#     with open(filepath, "r") as f:
#         content = f.read()

#     # Parse unit cell parameters
#     a = _extract_cif_value(content, "_cell_length_a")
#     b = _extract_cif_value(content, "_cell_length_b")
#     c = _extract_cif_value(content, "_cell_length_c")
#     alpha = _extract_cif_value(content, "_cell_angle_alpha")
#     beta = _extract_cif_value(content, "_cell_angle_beta")
#     gamma = _extract_cif_value(content, "_cell_angle_gamma")

#     # Parse space group
#     sg_number = _extract_cif_int(content, "_space_group_IT_number")
#     sg_name = _extract_cif_string(content, "_symmetry_space_group_name_H-M")
#     if sg_name:
#         sg_name = sg_name.strip("'\"")

#     # Parse symmetry operations
#     symops = _parse_cif_symmetry_operations(content)

#     # Parse atom sites
#     symbols, frac_coords = _parse_cif_atom_sites(content)

#     crystal = Crystal(
#         a=a,
#         b=b,
#         c=c,
#         alpha=alpha,
#         beta=beta,
#         gamma=gamma,
#         space_group_name=sg_name or "P1",
#         space_group_number=sg_number or 1,
#         symops=symops if symops else [SymOp.identity()],
#     )

#     return crystal, symbols, frac_coords


# def _extract_cif_value(content: str, key: str) -> float:
#     """Extract a numeric value from CIF content, removing uncertainty in parentheses."""
#     pattern = rf"{re.escape(key)}\s+(\S+)"
#     match = re.search(pattern, content, re.IGNORECASE)
#     if match:
#         val_str = match.group(1)
#         # Remove uncertainty in parentheses, e.g., "5.624(1)" -> "5.624"
#         val_str = re.sub(r"\([^)]*\)", "", val_str)
#         return float(val_str)
#     raise ValueError(f"Could not find {key} in CIF file")


# def _extract_cif_int(content: str, key: str) -> int | None:
#     """Extract an integer value from CIF content."""
#     pattern = rf"{re.escape(key)}\s+(\d+)"
#     match = re.search(pattern, content, re.IGNORECASE)
#     if match:
#         return int(match.group(1))
#     return None


# def _extract_cif_string(content: str, key: str) -> str | None:
#     """Extract a string value from CIF content."""
#     pattern = rf"{re.escape(key)}\s+(.+)"
#     match = re.search(pattern, content, re.IGNORECASE)
#     if match:
#         return match.group(1).strip()
#     return None


# def _parse_cif_symmetry_operations(content: str) -> list[SymOp]:
#     """
#     Parse symmetry operations from CIF file.

#     Looks for the _symmetry_equiv_pos_as_xyz loop.
#     """
#     symops = []

#     # Find the symmetry loop
#     # Pattern to match symmetry operations in a loop
#     loop_pattern = (
#         r"loop_\s*_symmetry_equiv_pos_as_xyz\s*((?:[^\n]+\n)+?)(?=loop_|_\w|$)"
#     )
#     match = re.search(loop_pattern, content, re.IGNORECASE)

#     if match:
#         lines = match.group(1).strip().split("\n")
#         for line in lines:
#             line = line.strip()
#             if line and not line.startswith("_"):
#                 symops.append(parse_symop_xyz(line))
#         return symops

#     # Alternative format with site_id
#     loop_pattern2 = r"loop_\s*(?:_symmetry_equiv_pos_site_id\s*)?_symmetry_equiv_pos_as_xyz\s*((?:[^\n]+\n)+?)(?=loop_|_\w|$)"
#     match = re.search(loop_pattern2, content, re.IGNORECASE)

#     if not match:
#         # Try another common format
#         pattern = r"_symmetry_equiv_pos_as_xyz\s*\n((?:.*\n)*?)(?=loop_|_\w)"
#         match = re.search(pattern, content, re.IGNORECASE)

#     if match:
#         lines = match.group(1).strip().split("\n")
#         for line in lines:
#             line = line.strip()
#             if line and not line.startswith("_") and not line.startswith("loop"):
#                 # Handle format with id number prefix
#                 parts = line.split()
#                 if len(parts) >= 1:
#                     xyz_str = parts[-1] if len(parts) > 1 else parts[0]
#                     # Remove quotes if present
#                     xyz_str = xyz_str.strip("'\"")
#                     if "," in xyz_str:
#                         symops.append(parse_symop_xyz(xyz_str))

#     return symops


# def _parse_cif_atom_sites(content: str) -> tuple[list[str], NDArray[np.float64]]:
#     """
#     Parse atom site information from CIF file.

#     Returns atom symbols and fractional coordinates.
#     """
#     symbols = []
#     frac_coords = []

#     lines = content.split("\n")

#     # Find the atom_site loop with fractional coordinates
#     # We need to find a loop_ that contains _atom_site_fract_x
#     i = 0
#     found_loop = False
#     columns = []
#     col_map = {}

#     while i < len(lines):
#         line = lines[i].strip()

#         # Look for loop_ followed by _atom_site headers
#         if line.lower() == "loop_":
#             # Check if this loop has _atom_site_fract_x
#             j = i + 1
#             temp_columns = []
#             while j < len(lines):
#                 header_line = lines[j].strip()
#                 if header_line.lower().startswith("_atom_site_"):
#                     # Extract the field name
#                     field = header_line.lower().replace("_atom_site_", "")
#                     temp_columns.append(field)
#                     j += 1
#                 elif header_line.startswith("_") or header_line.lower() == "loop_":
#                     # Different type of header or new loop
#                     break
#                 elif header_line == "" or header_line.startswith("#"):
#                     j += 1
#                 else:
#                     # Data line - check if we found the right loop
#                     break

#             # Check if this loop has fractional coordinates
#             if (
#                 "fract_x" in temp_columns
#                 and "fract_y" in temp_columns
#                 and "fract_z" in temp_columns
#             ):
#                 found_loop = True
#                 columns = temp_columns
#                 col_map = {col: idx for idx, col in enumerate(columns)}
#                 i = j  # Start reading data from here
#                 break

#         i += 1

#     if not found_loop:
#         raise ValueError(
#             "Could not find atom_site loop with fractional coordinates in CIF file"
#         )

#     # Required columns for coordinates
#     fract_x_col = col_map.get("fract_x")
#     fract_y_col = col_map.get("fract_y")
#     fract_z_col = col_map.get("fract_z")

#     # Label column (element symbol)
#     label_col = col_map.get("label")
#     type_symbol_col = col_map.get("type_symbol")

#     # Helper function to parse coordinates
#     def parse_coord(s: str) -> float:
#         s = re.sub(r"\([^)]*\)", "", s)
#         return float(s)

#     # Parse data lines
#     while i < len(lines):
#         line = lines[i].strip()
#         i += 1

#         # Stop at next loop or new data block or header
#         if (
#             line.lower().startswith("loop_")
#             or line.startswith("_")
#             or line.lower().startswith("data_")
#         ):
#             break

#         if not line or line.startswith("#"):
#             continue

#         parts = line.split()
#         if len(parts) < len(columns):
#             continue

#         # Get symbol
#         if type_symbol_col is not None:
#             symbol = parts[type_symbol_col]
#         elif label_col is not None:
#             # Extract element from label (e.g., "C1" -> "C", "O2" -> "O")
#             label = parts[label_col]
#             label_match = re.match(r"([A-Za-z]+)", label)
#             if label_match is None:
#                 continue
#             symbol = label_match.group(1)
#         else:
#             continue

#         # Type assertions to satisfy the type checker (we know these are not None from above)
#         assert fract_x_col is not None
#         assert fract_y_col is not None
#         assert fract_z_col is not None

#         x = parse_coord(parts[fract_x_col])
#         y = parse_coord(parts[fract_y_col])
#         z = parse_coord(parts[fract_z_col])

#         symbols.append(symbol)
#         frac_coords.append([x, y, z])

#     return symbols, np.array(frac_coords)


# def remove_duplicate_atoms(
#     symbols: list[str],
#     coords: NDArray[np.float64],
#     tolerance: float = 0.01,
# ) -> tuple[list[str], NDArray[np.float64]]:
#     """
#     Remove duplicate atoms based on coordinate proximity.

#     Parameters
#     ----------
#     symbols : List[str]
#         List of atom symbols.
#     coords : NDArray[np.float64]
#         Coordinates array of shape (N, 3).
#     tolerance : float
#         Distance tolerance for considering atoms as duplicates.

#     Returns
#     -------
#     Tuple[List[str], NDArray[np.float64]]
#         Unique atom symbols and coordinates.
#     """
#     if len(symbols) == 0:
#         return symbols, coords

#     unique_symbols = [symbols[0]]
#     unique_coords = [coords[0]]

#     for i in range(1, len(symbols)):
#         is_duplicate = False
#         for j in range(len(unique_coords)):
#             dist = np.linalg.norm(coords[i] - unique_coords[j])
#             # Also check for periodic images
#             for dx in [-1, 0, 1]:
#                 for dy in [-1, 0, 1]:
#                     for dz in [-1, 0, 1]:
#                         shifted = coords[i] + np.array([dx, dy, dz])
#                         dist = np.linalg.norm(shifted - unique_coords[j])
#                         if dist < tolerance:
#                             is_duplicate = True
#                             break
#                     if is_duplicate:
#                         break
#                 if is_duplicate:
#                     break
#             if is_duplicate:
#                 break

#         if not is_duplicate:
#             unique_symbols.append(symbols[i])
#             unique_coords.append(coords[i])

#     return unique_symbols, np.array(unique_coords)
