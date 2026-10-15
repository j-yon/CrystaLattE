from __future__ import annotations

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray

from ..core.multimer import Multimer


def _nre(elem: list[str], geom: NDArray) -> float:
    """Compute the nuclear repulsion energy of a multimer
    (Exact copy of the original CLE code)

    :param elem: List of atomic symbols of each atom
    :param geom: Array of the coordinates of the system
    :returns: Nuclear repulsion energy of the system
    """
    Z = np.array([qcel.periodictable.to_Z(e) for e in elem])
    dist = np.sqrt(
        np.sum((geom[:, np.newaxis, :] - geom[np.newaxis, :, :]) ** 2, axis=2)
    )
    np.fill_diagonal(dist, np.inf)
    oodist = np.reciprocal(dist)
    nre = np.einsum("x,xy,y", Z, oodist, Z, optimize=True) / 2.0

    return nre


def _chsev(elem: list[str], geom: NDArray) -> NDArray:
    """Diagonalize the chemical space matrix of a multimer 
    (Exact copy of the original CLE code)

    :param elem: List of atomic symbols of each atom
    :param geom: Array of the coordinates of the system
    :returns: Sorted eigenvalues of the chemical space matrix
    """

    Z = np.array([qcel.periodictable.to_Z(e) for e in elem])
    dist = np.sqrt(
        np.sum((geom[:, np.newaxis, :] - geom[np.newaxis, :, :]) ** 2, axis=2)
    )

    np.fill_diagonal(dist, np.inf)
    oodist = np.reciprocal(dist)

    # fill the diagonal with the special polynomial from DOI: 10.1103/PhysRevLett.108.058301
    M = np.einsum("x,xy,y->xy", Z, oodist, Z, optimize=True)
    np.fill_diagonal(M, (Z**2.4) / 2.0)

    # solve the eigenvalue problem
    eigenvalues, _ = np.linalg.eig(M)

    # eigenvalues must be in list to use 'sort'
    sorted_eigenvalues = list(eigenvalues)

    # sort the eigenvalues in order of "decreasing absolute value"
    sorted_eigenvalues.sort(key=lambda x: -x)
    sorted_eigenvalues.sort(key=lambda x: -abs(x))

    # cast back to a NumPy array
    chem_spc_eigen_values = np.array(sorted_eigenvalues)

    return chem_spc_eigen_values


def _nre_chsev(multimer: Multimer) -> tuple[float, NDArray]:
    """Return `(NRE, sorted Coulomb-matrix eigenvalues)` for a multimer

    :param multimer: :class:`Multimer` object
    :returns: Nuclear repulsion energy and sorted eigenvalues of the chemical space matrix
    """
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
    """Group :class:`Multimer` instances into chemistry-equivalence classes via (NRE, CHSEV) comparison
    (The original CLE code has atol = 1e-8)

    :param multimers: List of :class:`Multimer` objects to classify.
    :param atol: Absolute tolerance for comparing NRE and CHSEV values.
    :returns: Tuple of number of classes and a list of lists containing indices of partitioned multimers
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
