from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray


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

    symbols: list[str]
    frac_coords: NDArray[np.float64]
    cart_coords: NDArray[np.float64]
    cell_index: tuple[int, int, int]
    symop_index: int
    molecule_index: int
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
class Multimer:
    """Represents a unique multimer pairing of N monomers.

    Attributes:
    monomers: List[Monomer]
        List of monomers in the multimer, with the reference monomer first.
    distance : float
        Geometric mean of center-to-center distances between monomers in Angstroms.
    symop_index_a : int
        Symmetry operation index for monomer A.
    symop_index_b : int
        Symmetry operation index for monomer B.
    cell_index_b : Tuple[int, int, int]
        Unit cell index of monomer B relative to A.
    multiplicity : int
        Number of symmetry-equivalent copies of this multimer type.
    """

    monomers: list[Monomer]
    distance: float
    symop_indices: list[int]
    cell_indices: list[tuple[int, int, int]]
    multiplicity: int = 1

    def to_molecule(self) -> qcel.models.Molecule:
        """Convert dimer to QCElemental Molecule object."""
        symbols = [monomer.symbols for monomer in self.monomers]
        coords = [monomer.cart_coords for monomer in self.monomers]
        coords = np.vstack(coords)
        cart_bohr = coords / qcel.constants.bohr2angstroms

        # TODO: does this return a fragmented molecule?
        return qcel.models.Molecule(
            symbols=symbols,
            geometry=cart_bohr.flatten(),
        )

    def get_molecules(self) -> list[qcel.models.Molecule]:
        """Return the monomers as separate QCElemental Molecules."""
        return [monomer.to_molecule() for monomer in self.monomers]


def _get_covalent_radius(symbol: str) -> float:
    """Get covalent radius for an element in Angstroms.

    Parameters:
    symbol : str
        Element symbol.

    Returns:
    float
        Covalent radius in Angstroms.
    """
    # qcelemental returns covalent radii in Bohr
    radius_bohr = qcel.covalentradii.get(symbol)
    return radius_bohr * qcel.constants.bohr2angstroms


def _build_bond_graph(
    symbols: list[str],
    cart_coords: NDArray[np.float64],
    bond_tolerance: float = 0.4,
) -> list[list[int]]:
    """Build adjacency list for atoms based on covalent bonding.

    Two atoms are considered bonded if their distance is less than
    the sum of their covalent radii plus a tolerance.

    Parameters:
    symbols : List[str]
        Atom symbols.
    cart_coords : NDArray[np.float64]
        Cartesian coordinates, shape (N, 3).
    bond_tolerance : float, default=0.4
        Tolerance in Angstroms added to sum of covalent radii.

    Returns:
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


def _find_connected_components(adj: list[list[int]]) -> list[list[int]]:
    """Find connected components in a graph using BFS.

    Parameters:
    adj : List[List[int]]
        Adjacency list representation of the graph.

    Returns:
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
    asu_symbols: list[str],
    asu_frac_coords: list[NDArray[np.float64]],
    bond_tolerance: float = 0.4,
) -> list[Monomer]:
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

    # Create Monomer objects for molecules in the central cell
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
            mol = Monomer(
                symbols=mol_symbols,
                frac_coords=mol_frac,
                cart_coords=mol_cart,
                cell_index=(0, 0, 0),
                centroid_frac=centroid_frac,
                centroid_cart=centroid_cart,
                molecule_index=mol_idx,
            )
            molecules.append(mol)
            mol_idx += 1

    return molecules


def generate_molecular_dimers(
    filepath: str | Path,
    radius: float,
    bond_tolerance: float = 0.4,
    distance_tolerance: float = 0.01,
) -> tuple[list[Multimer], Crystal, list[Multimer]]:
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
                    if (
                        cell_idx == (0, 0, 0)
                        and mol.molecule_index == ref_molecule.molecule_index
                    ):
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
    dimers: list[MolecularDimer],
    crystal: Crystal,
    tolerance: float = 0.01,
) -> list[MolecularDimer]:
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
) -> list[qcel.models.Molecule]:
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
) -> tuple[list[DimerPair], Crystal, Monomer]:
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
    monomers: list[Monomer],
    tolerance: float = 0.01,
) -> list[Monomer]:
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
    dimers: list[DimerPair],
    crystal: Crystal,
    tolerance: float = 0.01,
) -> list[DimerPair]:
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
