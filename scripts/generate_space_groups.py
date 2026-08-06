"""
Generate space_groups_all.json with all 564 crystallographic space group settings.

Requires: gemmi, spglib, pyxtal (install with pip install gemmi spglib pyxtal)
Output:   src/crystalatte/core/space_groups_all.json

JSON structure:
  - 564 entries keyed by normalized Hall symbol (e.g. "-P 2ybc") for every setting
  - 230 additional entries keyed by str(number) (e.g. "14") pointing to the standard
    setting data — for backward-compatible number-only lookups

Wyckoff positions:
  - Standard settings (230): pyxtal (correct site symmetry labels)
  - Alternative settings (334): 12x12x12 grid scan (site_symmetry = "?")
"""

import itertools
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

try:
    import gemmi
    import spglib
    from pyxtal.symmetry import Group, Wyckoff_position
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nRun: pip install gemmi spglib pyxtal")


# ---------------------------------------------------------------------------
# Lattice constraints by crystal system
# ---------------------------------------------------------------------------

LATTICE_CONSTRAINTS = {
    "triclinic":    {"a": None, "b": None, "c": None, "alpha": None, "beta": None, "gamma": None},
    "monoclinic":   {"a": None, "b": None, "c": None, "alpha": 90,   "beta": None, "gamma": 90},
    "orthorhombic": {"a": None, "b": None, "c": None, "alpha": 90,   "beta": 90,   "gamma": 90},
    "tetragonal":   {"a": None, "b": None, "c": None, "alpha": 90,   "beta": 90,   "gamma": 90},
    "trigonal":     {"a": None, "b": None, "c": None, "alpha": 90,   "beta": 90,   "gamma": 120},
    "hexagonal":    {"a": None, "b": None, "c": None, "alpha": 90,   "beta": 90,   "gamma": 120},
    "cubic":        {"a": None, "b": None, "c": None, "alpha": 90,   "beta": 90,   "gamma": 90},
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def normalize_hall(hall: str) -> str:
    return hall.strip()


def get_lattice_system(crystal_system: str, centring: str) -> str:
    if crystal_system == "trigonal":
        return "rhombohedral" if centring == "R" else "hexagonal"
    return crystal_system


def parse_site_symmetry(wp_str: str) -> str:
    m = re.search(r"with site symmetry\s+(\S+)", wp_str)
    return m.group(1) if m else "?"


def clean_coord(x: float) -> float:
    return round(float(x) % 1, 8)


def parse_symop_to_matrix(op_str: str):
    """Parse an xyz-string symmetry operation into (R, t*12) integer matrices."""
    R = np.zeros((3, 3), dtype=np.int16)
    t = np.zeros(3, dtype=np.int16)
    _op = op_str.replace(" ", "").split(",")
    for i, comp in enumerate(_op):
        sign = 1
        j = 0
        while j < len(comp):
            if comp[j] in "+-":
                sign = 1 if comp[j] == "+" else -1
                j += 1
            elif comp[j] in "xyz":
                axis = "xyz".index(comp[j])
                R[i, axis] = sign
                j += 1
            else:
                num_str = ""
                while j < len(comp) and (comp[j].isdigit() or comp[j] in "/."):
                    num_str += comp[j]
                    j += 1
                if num_str:
                    t[i] += int(Fraction(num_str) * 12) * sign
                    sign = 1
    return R, t


# ---------------------------------------------------------------------------
# Wyckoff: pyxtal (standard settings)
# ---------------------------------------------------------------------------

def get_wyckoff_pyxtal(n: int, general_mult: int) -> dict:
    g = Group(n)
    result = {}
    for sublist in g.wyckoffs_organized:
        for wp in sublist:
            if wp.multiplicity >= general_mult:
                continue
            letter = wp.letter
            wp_full = Wyckoff_position.from_group_and_letter(n, letter)
            site_symm = parse_site_symmetry(str(wp_full))
            coords = [
                [clean_coord(c) for c in op.translation_vector]
                for op in wp_full.ops
            ]
            result[letter] = {
                "multiplicity": wp.multiplicity,
                "site_symmetry": site_symm,
                "coordinates": coords,
            }
    return result


# ---------------------------------------------------------------------------
# Wyckoff: grid scan (alternative settings)
# ---------------------------------------------------------------------------

def find_special_positions_grid(sym_op_strings: list[str]) -> dict:
    """
    Compute special Wyckoff positions for any space group setting by scanning a
    12x12x12 fractional coordinate grid. All crystallographic fractions (1/2,
    1/3, 1/4, 1/6, 1/12) are exact multiples of 1/12, so this grid is complete.
    """
    matrices = [parse_symop_to_matrix(op) for op in sym_op_strings]
    n_ops = len(matrices)

    # All 1728 grid points as integer (i,j,k) in [0,12)
    ijk = np.array(list(itertools.product(range(12), repeat=3)), dtype=np.int16)

    # For each sym_op, compute image of every grid point; encode as base-12 integer
    all_encoded = np.zeros((n_ops, 1728), dtype=np.int32)
    for k, (R, t) in enumerate(matrices):
        img = (R.astype(np.int32) @ ijk.T).T + t.astype(np.int32)
        img = img % 12
        all_encoded[k] = img[:, 0] * 144 + img[:, 1] * 12 + img[:, 2]

    # Find orbits: group grid points by their frozenset of images
    images_T = all_encoded.T  # (1728, n_ops)
    orbits: dict[frozenset, tuple] = {}
    for pt_idx in range(1728):
        imgs = frozenset(int(x) for x in images_T[pt_idx])
        mult = len(imgs)
        if mult < n_ops and imgs not in orbits:
            orbits[imgs] = mult

    # Assign Wyckoff letters a, b, c, ... in ascending multiplicity order
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sorted_orbits = sorted(orbits.items(), key=lambda item: item[1])
    result = {}
    for idx, (imgs_frozenset, mult) in enumerate(sorted_orbits):
        letter = letters[idx] if idx < len(letters) else f"wp{idx}"
        coords = []
        for encoded in sorted(imgs_frozenset):
            ci = int(encoded) // 144
            cj = (int(encoded) % 144) // 12
            ck = int(encoded) % 12
            coords.append([ci / 12, cj / 12, ck / 12])
        coords.sort()
        result[letter] = {
            "multiplicity": mult,
            "site_symmetry": "?",
            "coordinates": coords,
        }
    return result


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def build_spglib_lookup() -> tuple[dict, dict, dict]:
    """
    Returns:
      sg_to_schoen: SG number -> Schoenflies symbol (from first/standard Hall)
      sg_to_int_short: SG number -> international_short (e.g. "P21")
      standard_hall_set: set of normalized Hall symbols for the 230 standard settings
    """
    sg_to_schoen: dict[int, str] = {}
    sg_to_int_short: dict[int, str] = {}
    for hall_num in range(1, 531):
        sg_type = spglib.get_spacegroup_type(hall_num)
        n = sg_type.number
        if n not in sg_to_schoen:
            sg_to_schoen[n] = sg_type.schoenflies
            sg_to_int_short[n] = sg_type.international_short

    standard_hall_set = {
        normalize_hall(gemmi.find_spacegroup_by_number(n).hall)
        for n in range(1, 231)
    }
    return sg_to_schoen, sg_to_int_short, standard_hall_set


def build_entry(sg_gem, sg_to_schoen: dict, sg_to_int_short: dict,
                standard_hall_set: set) -> dict:
    n = sg_gem.number
    hall = normalize_hall(sg_gem.hall)
    is_standard = hall in standard_hall_set

    sym_ops = [op.triplet() for op in sg_gem.operations()]
    crystal_sys = sg_gem.crystal_system_str()
    centring = sg_gem.centring_type()
    lattice_sys = get_lattice_system(crystal_sys, centring)

    if is_standard:
        special_wps = get_wyckoff_pyxtal(n, len(sym_ops))
    else:
        special_wps = find_special_positions_grid(sym_ops)

    return {
        "number": n,
        "symbol": {
            "Hermann-Mauguin": sg_gem.xhm(),
            "Schoenflies": sg_to_schoen.get(n, "?"),
            "PDB": sg_gem.xhm(),
            "Hall": hall,
            "short": sg_gem.short_name(),
        },
        "point_group": sg_gem.point_group_hm(),
        "crystal_system": crystal_sys,
        "lattice_system": lattice_sys,
        "sym_ops": sym_ops,
        "special_wyckoff_positions": special_wps,
        "lattice_constraints": LATTICE_CONSTRAINTS[crystal_sys],
    }


def main():
    out_path = (
        Path(__file__).parent.parent
        / "src" / "crystalatte" / "core" / "space_groups_all.json"
    )

    print("Building spglib and gemmi lookup tables...", flush=True)
    sg_to_schoen, sg_to_int_short, standard_hall_set = build_spglib_lookup()

    result: dict[str, dict] = {}
    standard_by_number: dict[int, str] = {}  # SG number -> Hall key
    failed: list[tuple] = []

    all_sgs = list(gemmi.spacegroup_table())
    print(f"Processing {len(all_sgs)} settings across 230 space groups...", flush=True)

    for sg_gem in all_sgs:
        hall = normalize_hall(sg_gem.hall)
        n = sg_gem.number
        try:
            entry = build_entry(sg_gem, sg_to_schoen, sg_to_int_short, standard_hall_set)
            result[hall] = entry
            # Track the standard setting for each SG number
            if hall in standard_hall_set and n not in standard_by_number:
                standard_by_number[n] = hall
            print(f"  {hall[:30]:30s} (SG {n})", end="\r", flush=True)
        except Exception as e:
            print(f"\n  FAILED: hall={hall!r} SG={n}: {e}", flush=True)
            failed.append((hall, n, str(e)))

    # Add number-string aliases for standard settings (backward compat)
    print(f"\nAdding {len(standard_by_number)} number-keyed aliases...", flush=True)
    for n in range(1, 231):
        if n in standard_by_number:
            result[str(n)] = result[standard_by_number[n]]

    total_hall = len([k for k in result if not k.lstrip("-").replace(" ", "").isdigit() or " " in k])
    print(f"\nGenerated {len(result)} total entries "
          f"({len(result) - 230} Hall-symbol keys + 230 number aliases).", flush=True)

    if failed:
        print(f"Failed ({len(failed)}): {[(h, n) for h, n, _ in failed[:5]]}", flush=True)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Written to {out_path}", flush=True)
    import os
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
