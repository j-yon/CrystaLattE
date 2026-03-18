import math
import re
from pathlib import Path
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


def read_cif(fNameIn):
    """Read CIF file, and extract the necessary info in the form of a
    dictionary. E.g., the value of "_cell_volume" can be found with
    data['_cell_volume'].
    """
    data = {}

    # Use the CifFile parser to extract the data. Although there might
    # be multiple data blocks, we'll only use the first one.

    # TODO: Known bug: Sometimes not all the blocks in the CIF files are
    #      read, and the following extraction fails.
    cif_file = CifFile(fNameIn)

    for db in cif_file:
        data_block = db
        break

        # Copy the x,y,z symmetry group operations. Remove the quotes
        # if there are any.
        for op_xyz in xyz:
            if op_xyz[0] == "'":
                data["_symmetry_equiv_pos_as_xyz"].append(op_xyz[1:-1])

            else:
                data["_symmetry_equiv_pos_as_xyz"].append(op_xyz)

        # Add x,y,z of the atoms to "data", but make sure to convert
        # e.g. "0.1549(8)" to "0.1549".
        data["_atom_site_label"] = data_block["_atom_site_label"]

        data["_atom_site_fract_x"] = []

        for str_x in data_block["_atom_site_fract_x"]:
            data["_atom_site_fract_x"].append(float(str_x.split("(")[0]))

        data["_atom_site_fract_y"] = []

        for str_y in data_block["_atom_site_fract_y"]:
            data["_atom_site_fract_y"].append(float(str_y.split("(")[0]))

        data["_atom_site_fract_z"] = []

        for str_z in data_block["_atom_site_fract_z"]:
            data["_atom_site_fract_z"].append(float(str_z.split("(")[0]))

    # Return the extracted data.
    return data


def from_cif(cif_file: str) -> Generator[tuple[Crystal, Monomer], None, None]:
    """Takes the name of a CIF input file and the name of a .xyz output
    file, and optionally the number of replicas of the rectangular cell in
    each direction (A, B, and C) --- if those are not given, then it
    tries to deduce them from the monomer_cutoff.  If the monomer_cutoff
    is not given, then the program automatically picks what should be a
    safe value, based on the largest nmer_cutoff and the unit cell
    dimensions.  The given or auto-computed monomer_cutoff is returned
    by the function.  The function calls Read_CIF() and passes
    the relevant information as arguments to generate an .xyz file of the
    supercell.

    Arguments:
    <str> fNameIn
        CIF input filename
    <str> fNameOut
        XYZ output filename
    <int> Na
        Number of replicas of the cartesian unit cell in `a` direction.
        Zero to auto-size.
    <int> Nb
        Number of replicas of the cartesian unit cell in `b` direction.
        Zero to auto-size.
    <int> Nc
        Number of replicas of the cartesian unit cell in `c` direction.
        Zero to auto-size.
    <float> monomer_cutoff
        Cutoff distance used if Na, Nb, or Nc are zero
        (auto-size option).  Otherwise ignored.  Set to zero
        to auto-size this from nmer_cutoff.
    <float> nmer_cutoff
        Largest of the values for dimer cutoff, trimer cutoff, etc.
        Used if auto-sizing monomer_cutoff (i.e., if monomer_cutoff == 0).
        Otherwise ignored.
    <bool> make_rect_box
        Set to TRUE if the final configuration should be in a rectangular
        shape, or in the same shape as the unit cell.  We have been always
        using TRUE for this so far.
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
        asu = ASU(block["_atom_site_type_symbol"], coords)
        crystal = Crystal(a, b, c, alpha, beta, gamma, space_group, asu)

        # as a precaution, check that the volume of the unit cell is consistent
        volume = float(block["_cell_volume"])
        if abs(crystal.volume - volume) > 0.1:
            raise ValueError(
                f"Discordant unit cell volumes:\nDeclared in CIF: {volume:.2f} A^3\nCalculated from primitive vectors: {crystal.volume:.2f} A^3"
            )

        yield crystal, crystal.get_reference()
