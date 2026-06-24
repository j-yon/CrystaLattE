"""Chemistry-based validation helpers — **NOT** part of the dedup path.

These reproduce the original CrystaLattE NRE + Coulomb-matrix-sorted-
eigenvalues oracle so the geometric deduper's output can be cross-checked
against the chemistry equivalence definition the old code used. The dedup
itself stays purely geometric (see `crystalatte.work.generate`) because the
chemistry oracle has known reliability issues for higher orders: NRE summation
precision degrades for N > 2, and CHSEV becomes noisy at large R.

The `_nre` / `_chsev` functions imported below are byte-for-byte copies of the
originals in `crystalatte.work.crystalatte.nre` / `.chemical_space`, evaluated
on the raw stacked atomic geometry in Bohr — no `qcel.models.Molecule`
round-trip, so the comparison is free of qcel's reorientation noise.
"""

from __future__ import annotations

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray

from ..core.multimer import Multimer


def _nre(elem, geom):
    """Takes two Numpy arrays, one with the element symbols and one with
    coordinates of a set of atoms and returns a float number with the
    computed nuclear repulsion energy. Exact copy of the original CLE code.
    """

    Z = np.array([qcel.periodictable.to_Z(e) for e in elem])
    dist = np.sqrt(
        np.sum((geom[:, np.newaxis, :] - geom[np.newaxis, :, :]) ** 2, axis=2)
    )
    np.fill_diagonal(dist, np.inf)
    oodist = np.reciprocal(dist)
    nre = np.einsum("x,xy,y", Z, oodist, Z, optimize=True) / 2.0

    return nre


def _chsev(elem, geom):
    """Takes the element symbols and coordinates of a set of atoms and
    computes the chemical space matrix, eigenvalues, and eigenvectors
    of the chemical system and returns a list with the sorted
    eigenvalues. Exact copy of the original CLE code.

    Arguments:
    <numpy.ndarray> elems
        Array of 1 column with the atomic symbols of each atom.
    <numpy.ndarray> geoms
        Array of 3 columns with the coordinates of the system.

    Returns:
    <numpy.ndarray> chem_spc_eigen_values
        Array of x numbers with the sorted eigenvalues of the molecular
        system, where x is the number of atoms in the system.
    """

    Z = np.array([qcel.periodictable.to_Z(e) for e in elem])
    dist = np.sqrt(
        np.sum((geom[:, np.newaxis, :] - geom[np.newaxis, :, :]) ** 2, axis=2)
    )

    np.fill_diagonal(dist, np.inf)
    oodist = np.reciprocal(dist)

    # Fill the diagonal with the special polynomial from:
    # DOI: 10.1103/PhysRevLett.108.058301
    M = np.einsum("x,xy,y->xy", Z, oodist, Z, optimize=True)
    np.fill_diagonal(M, (Z**2.4) / 2.0)

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eig(M)

    # Eigenvalues must be in list to use 'sort'
    sorted_eigenvalues = list(eigenvalues)

    # Sort the eigenvalues in order of "decreasing absolute value".
    # This first sort is done to guarantee the same sort in the case
    # that two eigenvalues are the same magnitude but different sign.
    sorted_eigenvalues.sort(key=lambda x: -x)
    sorted_eigenvalues.sort(key=lambda x: -abs(x))

    # Cast back to a NumPy array.
    chem_spc_eigen_values = np.array(sorted_eigenvalues)

    return chem_spc_eigen_values


def _nre_chsev(multimer: Multimer) -> tuple[float, NDArray]:
    """Return `(NRE, sorted Coulomb-matrix eigenvalues)` for a multimer."""
    symbols: list[str] = []
    cart_list: list[NDArray] = []
    for mon in multimer.monomers:
        symbols.extend(mon.symbols)
        cart_list.append(mon.cart_coords)
    bohr = np.vstack(cart_list) / qcel.constants.bohr2angstroms
    return _nre(symbols, bohr), _chsev(symbols, bohr)


def count_chemistry_classes(
    multimers: list[Multimer], atol: float = 1e-9
) -> tuple[int, list[list[int]]]:
    """Group `multimers` into chemistry-equivalence classes via (NRE, CHSEV)
    comparison at tolerance `atol` (the original CrystaLattE used 1e-8).
    Returns `(num_classes, partition)` where `partition[k]` lists the multimer
    indices in class `k`.
    """
    fingerprints = [_nre_chsev(m) for m in multimers]
    classes: list[list[int]] = []
    reps: list[tuple[float, NDArray]] = []
    for i, (nre_i, chsev_i) in enumerate(fingerprints):
        for k, (nre_k, chsev_k) in enumerate(reps):
            if abs(nre_i - nre_k) < atol and np.allclose(
                chsev_i, chsev_k, atol=atol, rtol=0.0
            ):
                classes[k].append(i)
                break
        else:
            classes.append([i])
            reps.append((nre_i, chsev_i))
    return len(classes), classes
