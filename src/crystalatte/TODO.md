# Things to consider doing in the future:
- Ensure integration with QCA (external)/other quantum chemistry software
    - see what software uses qcschema, see how hard it would be to integrate without that
- More documentation

## Things to fix:
- slight cell size difference, comes from dropping uncertainty in CIF parsing (i think)
- pyrazole, Z'>1 isnt handled correctly rn
- ice, disordered water molecules
- carbon dioxide, linear molecules dont work
- adamantane and naphthalene? not sure
    - this one seems to be a problem with v1, not v2


## Paper layour
Introduction:
- Using the MBE for lattice energy calculations
- Already have capabilities to reduce scaling of MBE calculations by removing symmetry redundancies, bu these are currently poor
    - current CLE code uses NRE/ChSEV, which is numerically unstable and inefficient due to the need to construct full multimers for comparison
    - current code also only goes to N=5, which is fine for most practices but is still a limitation
    - current code is also limited to simple homogeneous crystals with Z'<=1, many relevant crystals have Z'>1 or multiple molecules in the asymmetric unit
    - instead, inform symmetry reduction by crystallographic/molecular symmetry; these are efficient, numerically stable, and provide expanded capabilities to handle more complex crystals
- Efforts already exist to reduce scaling of MBE calculations by using crystal symmetry, but these are limited to translational symmetry
    - refer to Xantheas paper
    - this code still leaves duplicates because it doesnt consider any symmetry from the space group
    - we are not aware of any work that reduces to completely unique multimers using only crystallographic/molecular symmetry
- This work presents a new method to reduce the scaling of MBE calculations by using crystallographic and molecular symmetry

Theory:
- MBE:
    - MBE theory, how it works, how it is used to calculate lattice energies
    - Include MBE CLE for cocrystals/crystals with Z'>1
- Symmetry reduction:
    - A multimer can be represented by a set of symmetry operations, one for each monomer in the multimer, that map the reference monomer to its new position in the multimer
    - Two multimers are equivalent if there is a global mathematical operation that maps one to the other (provide mathematical definition)
        - We can use the symmetry operations of the crystal to determine if two multimers are equivalent
    - For a molecular crystal, this operation must fall into the space group of the crystal, otherwise it wouldn't be a valid operation (it wouldnt leave the crystal invariant)
    - We remove the global frame of the operation by considering the operation from a relative perspective


