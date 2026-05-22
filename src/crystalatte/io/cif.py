import re
from typing import Generator

import numpy as np
from CifFile import ReadCif

from ..core.crystal import Crystal
from ..core.multimer import Monomer, ASU
from ..core.space_group import SpaceGroup


def _clean_block(block):
    """Clean the data block by removing parentheses and converting to floats. NEEDS TO BE CLEANED, IM TOO TIRED RN"""
    cleaned_block = {}
    for key, value in block.items():
        if isinstance(value, list):
            cleaned_block[key] = []
            for item in value:
                if isinstance(item, str):
                    cleaned_item = re.sub(r"\([^)]*\)", "", item)
                    try:
                        cleaned_block[key].append(float(cleaned_item))
                    except ValueError:
                        cleaned_block[key].append(cleaned_item)
                else:
                    cleaned_block[key].append(item)
        elif isinstance(value, str):
            cleaned_value = re.sub(r"\([^)]*\)", "", value)
            try:
                cleaned_block[key] = float(cleaned_value)
            except ValueError:
                cleaned_block[key] = cleaned_value
        else:
            cleaned_block[key] = value

    return cleaned_block


def from_cif(cif_file: str) -> Generator[tuple[Crystal, Monomer], None, None]:
    """Parse a CIF file and extract the unit cell parameters, space group information, and atomic positions to create Crystal and Monomer objects.

    :param cif_file: The name of the CIF input file.  Should be in the same directory as the script, or include the path to the file.
    :returns: A generator that yields tuples of (Crystal, Monomer) for each data block in the CIF file. The Crystal object contains the unit cell parameters and space group information, while the Monomer object contains the atomic symbols and fractional coordinates of the atoms in the asymmetric unit. The generator allows for processing multiple data blocks in a CIF file, if present
    """

    # will check if file exists
    cif = ReadCif(cif_file)
    blocks = [item[-1] for item in cif.items()]
    for block in blocks:
        # Clean block data
        block = _clean_block(block)

        # Extract lengths (Angstroms) and angles from the CIF file.
        a = float(block["_cell_length_a"])
        b = float(block["_cell_length_b"])
        c = float(block["_cell_length_c"])
        alpha = float(block["_cell_angle_alpha"])
        beta = float(block["_cell_angle_beta"])
        gamma = float(block["_cell_angle_gamma"])

        # basic declarations
        space_group = SpaceGroup(int(block["_symmetry_int_tables_number"]))
        coords = np.array(
            [
                block["_atom_site_fract_x"],
                block["_atom_site_fract_y"],
                block["_atom_site_fract_z"],
            ]
        ).T
        if "_atom_site_type_symbol" in block:
            asu = ASU(block["_atom_site_type_symbol"], coords)
        elif "_atom_site_label" in block:
            labels = block["_atom_site_label"]
            symbols = [re.sub(r"[\d\(\)]", "", label) for label in labels]
            asu = ASU(symbols, coords)
        else:
            raise ValueError(
                "CIF file must contain either _atom_site_type_symbol or _atom_site_label to determine atomic symbols."
            )
        crystal = Crystal((a, b, c), (alpha, beta, gamma), space_group, asu)

        # as a precaution, check that the volume of the unit cell is consistent
        volume = float(block["_cell_volume"])
        if abs(crystal.volume - volume) > 0.1:
            raise ValueError(
                f"Discordant unit cell volumes:\nDeclared in CIF: {volume:.2f} A^3\nCalculated from primitive vectors: {crystal.volume:.2f} A^3"
            )

        yield crystal, crystal.get_reference()
